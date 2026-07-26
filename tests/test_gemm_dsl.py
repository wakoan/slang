"""The portable DSL GEMM: one Python source, correct on Metal and on wgpu.

Prefill's fast path is MPSMatrixMultiplication, which exists only on Apple. This
kernel is what every other backend uses instead, so the bar is that it agrees
with MPS — not merely that it runs.

Shapes cover the ones prefill actually issues (K=1536 gate/up, narrow-N o-proj)
plus deliberately ragged ones, because the tile is 32x32x16 and every real
matmul in E2B is a clean multiple; a bug in the edge guards would never show on
model shapes alone.
"""
import numpy as np
import pytest

from py_shader_lang_wgpu import translate
from gemma4_150.kernels_dsl import gemm_tiled, gemm_groups

# (M, N, K) — the last two are ragged on purpose
SHAPES = [(32, 32, 16), (16, 1536, 1536), (64, 256, 512), (17, 33, 48), (5, 7, 16)]


def reference(a, b, alpha):
    """C = alpha * A @ B^T in float64, from the f16 inputs the GPU sees."""
    return alpha * (a.astype(np.float64) @ b.astype(np.float64).T)


def _inputs(M, N, K, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((N, K)).astype(np.float16)
    return a, b


def _params(M, N, K, alpha):
    return np.array([M, N, K, np.float32(alpha).view(np.uint32)], dtype=np.uint32)


def _strides(lda, ldb, ldc, tr):
    return np.array([lda, ldb, ldc, tr], dtype=np.uint32)


def run_metal(a, b, alpha, tr=True, lda=None, ldb=None, ldc=None, M=None, N=None, K=None):
    Metal = pytest.importorskip("Metal")
    dev = Metal.MTLCreateSystemDefaultDevice()
    if dev is None:
        pytest.skip("no Metal device")
    M = M or a.shape[0]
    K = K or a.shape[1]
    N = N or (b.shape[0] if tr else b.shape[1])
    src = translate(gemm_tiled, target="msl")
    lib, err = dev.newLibraryWithSource_options_error_(src, None, None)
    assert lib is not None, f"MSL compile failed: {err}"
    pso, err = dev.newComputePipelineStateWithFunction_error_(
        lib.newFunctionWithName_("gemm_tiled"), None)
    assert pso is not None, f"pipeline failed: {err}"

    opt = Metal.MTLResourceStorageModeShared
    ba = dev.newBufferWithBytes_length_options_(a.tobytes(), a.nbytes, opt)
    bb = dev.newBufferWithBytes_length_options_(b.tobytes(), b.nbytes, opt)
    bc = dev.newBufferWithLength_options_(M * N * 2, opt)
    bp = dev.newBufferWithBytes_length_options_(_params(M, N, K, alpha).tobytes(), 16, opt)
    bs = dev.newBufferWithBytes_length_options_(
        _strides(lda or K, ldb or (K if tr else N), ldc or N, 1 if tr else 0).tobytes(), 16, opt)

    cb = dev.newCommandQueue().commandBuffer()
    enc = cb.computeCommandEncoder()
    enc.setComputePipelineState_(pso)
    for i, buf in enumerate((ba, bb, bc, bp, bs)):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    gx, gy = gemm_groups(M, N)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(gx, gy, 1), Metal.MTLSizeMake(16, 16, 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return np.frombuffer(bc.contents().as_buffer(M * N * 2), dtype=np.float16).reshape(M, N)


def run_wgpu(a, b, alpha, tr=True, lda=None, ldb=None, ldc=None, M=None, N=None, K=None):
    wgpu = pytest.importorskip("wgpu")
    M = M or a.shape[0]
    K = K or a.shape[1]
    N = N or (b.shape[0] if tr else b.shape[1])
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if "shader-f16" not in adapter.features:
        pytest.skip("adapter lacks shader-f16")
    dev = adapter.request_device_sync(required_features=["shader-f16"])
    src = translate(gemm_tiled).replace("enable subgroups;\n", "")
    mod = dev.create_shader_module(code=src)

    U = wgpu.BufferUsage
    # WebGPU requires storage binding sizes to be 4-byte aligned; an f16 buffer
    # with an odd element count is not, so the HOST pads. Metal has no such rule,
    # which is exactly the kind of divergence a shared kernel source pushes into
    # the runner rather than eliminating.
    pad4 = lambda n: (n + 3) // 4 * 4
    ba = dev.create_buffer_with_data(
        data=a.tobytes().ljust(pad4(a.nbytes), b"\0"), usage=U.STORAGE)
    bb = dev.create_buffer_with_data(
        data=b.tobytes().ljust(pad4(b.nbytes), b"\0"), usage=U.STORAGE)
    bc = dev.create_buffer(size=pad4(M * N * 2), usage=U.STORAGE | U.COPY_SRC)
    bp = dev.create_buffer_with_data(data=_params(M, N, K, alpha).tobytes(), usage=U.UNIFORM)
    bs = dev.create_buffer_with_data(data=_strides(
        lda or K, ldb or (K if tr else N), ldc or N, 1 if tr else 0).tobytes(), usage=U.UNIFORM)

    pipe = dev.create_compute_pipeline(
        layout=wgpu.enums.AutoLayoutMode.auto,
        compute={"module": mod, "entry_point": "gemm_tiled"})
    bg = dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
        {"binding": i, "resource": {"buffer": buf, "offset": 0, "size": buf.size}}
        for i, buf in enumerate((ba, bb, bc, bp, bs))])
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, bg)
    gx, gy = gemm_groups(M, N)
    cp.dispatch_workgroups(gx, gy, 1)
    cp.end()
    dev.queue.submit([enc.finish()])
    raw = dev.queue.read_buffer(bc)[: M * N * 2]      # drop the alignment padding
    return np.frombuffer(raw, dtype=np.float16).reshape(M, N)


def _check(got, a, b, alpha, K):
    ref = reference(a, b, alpha)
    denom = max(np.abs(ref).max(), 1e-6)
    rel = np.abs(got.astype(np.float64) - ref).max() / denom
    # f16 output over a K-long f32 accumulation; the tolerance scales with K
    assert rel < 3e-3, f"max rel err {rel:.2e} (K={K})"


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_metal_matches_numpy(M, N, K):
    a, b = _inputs(M, N, K)
    _check(run_metal(a, b, 1.0), a, b, 1.0, K)


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_wgpu_matches_numpy(M, N, K):
    a, b = _inputs(M, N, K)
    _check(run_wgpu(a, b, 1.0), a, b, 1.0, K)


def test_alpha_is_applied():
    a, b = _inputs(32, 32, 16)
    _check(run_metal(a, b, 0.375), a, b, 0.375, 16)


def test_backends_agree():
    """Metal and wgpu run the SAME Python source; they must land together."""
    a, b = _inputs(64, 256, 512, seed=3)
    m, w = run_metal(a, b, 1.0).astype(np.float64), run_wgpu(a, b, 1.0).astype(np.float64)
    rel = np.abs(m - w).max() / max(np.abs(m).max(), 1e-6)
    assert rel < 1e-6, f"backends disagree by {rel:.2e}"


def test_matches_mps():
    """Against the library it replaces, on the dominant prefill shape."""
    mps = pytest.importorskip("MetalPerformanceShaders")
    Metal = pytest.importorskip("Metal")
    from gemma4_150.metal_runner import _SHARED
    dev = Metal.MTLCreateSystemDefaultDevice()
    M, N, K = 64, 1536, 1536
    a, b = _inputs(M, N, K, seed=7)

    ba = dev.newBufferWithBytes_length_options_(a.tobytes(), a.nbytes, _SHARED)
    bb = dev.newBufferWithBytes_length_options_(b.tobytes(), b.nbytes, _SHARED)
    bc = dev.newBufferWithLength_options_(M * N * 2, _SHARED)
    F16 = mps.MPSDataTypeFloat16

    def mat(buf, rows, cols):
        d = mps.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
            rows, cols, cols * 2, F16)
        return mps.MPSMatrix.alloc().initWithBuffer_descriptor_(buf, d)

    mm = mps.MPSMatrixMultiplication.alloc()
    mm = mm.initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
        dev, False, True, M, N, K, 1.0, 0.0)
    cb = dev.newCommandQueue().commandBuffer()
    mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
        cb, mat(ba, M, K), mat(bb, N, K), mat(bc, M, N))
    cb.commit()
    cb.waitUntilCompleted()
    ref_mps = np.frombuffer(bc.contents().as_buffer(M * N * 2), dtype=np.float16).reshape(M, N)

    got = run_metal(a, b, 1.0)
    denom = max(np.abs(ref_mps.astype(np.float64)).max(), 1e-6)
    rel = np.abs(got.astype(np.float64) - ref_mps.astype(np.float64)).max() / denom
    assert rel < 3e-3, f"DSL GEMM vs MPS: max rel err {rel:.2e}"


def test_non_transposed():
    """tr=0: attention's P @ V, where B is [K, N] rather than [N, K]."""
    rng = np.random.default_rng(11)
    M, N, K = 64, 128, 96
    a = rng.standard_normal((M, K)).astype(np.float16)
    bt = rng.standard_normal((K, N)).astype(np.float16)      # [K, N]
    ref = a.astype(np.float64) @ bt.astype(np.float64)
    for run in (run_metal, run_wgpu):
        got = run(a, bt, 1.0, tr=False, M=M, N=N, K=K)
        rel = np.abs(got.astype(np.float64) - ref).max() / max(np.abs(ref).max(), 1e-6)
        assert rel < 3e-3, f"{run.__name__}: max rel err {rel:.2e}"


def test_row_stride_windows_a_wider_buffer():
    """ldc > N: write a tile into a wider buffer, as the padded score matrix does."""
    rng = np.random.default_rng(12)
    M, N, K, LDC = 32, 30, 32, 64
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((N, K)).astype(np.float16)
    ref = a.astype(np.float64) @ b.astype(np.float64).T

    Metal = pytest.importorskip("Metal")
    dev = Metal.MTLCreateSystemDefaultDevice()
    src = translate(gemm_tiled, target="msl")
    lib, _ = dev.newLibraryWithSource_options_error_(src, None, None)
    pso, _ = dev.newComputePipelineStateWithFunction_error_(
        lib.newFunctionWithName_("gemm_tiled"), None)
    opt = Metal.MTLResourceStorageModeShared
    ba = dev.newBufferWithBytes_length_options_(a.tobytes(), a.nbytes, opt)
    bb = dev.newBufferWithBytes_length_options_(b.tobytes(), b.nbytes, opt)
    bc = dev.newBufferWithLength_options_(M * LDC * 2, opt)
    bp = dev.newBufferWithBytes_length_options_(_params(M, N, K, 1.0).tobytes(), 16, opt)
    bs = dev.newBufferWithBytes_length_options_(_strides(K, K, LDC, 1).tobytes(), 16, opt)
    cb = dev.newCommandQueue().commandBuffer()
    enc = cb.computeCommandEncoder()
    enc.setComputePipelineState_(pso)
    for i, buf in enumerate((ba, bb, bc, bp, bs)):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    gx, gy = gemm_groups(M, N)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(gx, gy, 1), Metal.MTLSizeMake(16, 16, 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()

    full = np.frombuffer(bc.contents().as_buffer(M * LDC * 2), dtype=np.float16).reshape(M, LDC)
    rel = np.abs(full[:, :N].astype(np.float64) - ref).max() / max(np.abs(ref).max(), 1e-6)
    assert rel < 3e-3, f"strided result wrong: {rel:.2e}"
    # the padding columns must be left alone, not scribbled on
    assert np.count_nonzero(full[:, N:]) == 0, "wrote past N into the row padding"
