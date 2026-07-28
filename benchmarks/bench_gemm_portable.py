"""The portable DSL GEMM on Metal vs wgpu — one kernel source, both backends.

    python benchmarks/bench_gemm_portable.py [M N K]

This exists to make ONE open question reproducible in a single command:
`kernels_dsl.gemm_tiled` runs ~4x slower under wgpu-native than under native
Metal, and that gap is the whole of the batched-prefill difference (564 vs 145
tok/s at 1024 tokens; see gemma4_150/prefill_wgpu.py). Same Python source, same
GPU, same shape — so the cost is in the translation or the runtime, not the
algorithm.

Already falsified as the cause: the f16 workgroup tiles. Pass --f32-tiles to
re-run that control; it measures within noise of the f16 version, which is why
the suspicion moved to naga's bounds-check clamps on the 64 workgroup-array
reads per k-tile (the MSL emitter inserts none).

Worth running under Dawn before concluding anything about "portable GEMM cost":
Dawn already beats wgpu-native by ~20% on decode with these same kernels, so
the browser may not share the penalty at all.

Default shape is gate/up at prefill batch: M=1024, N=12288, K=1536.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run from anywhere

from py_shader_lang_wgpu import translate
from gemma4_150.kernels_dsl import gemm_tiled, gemm_groups

REP = 20            # dispatches per timed run, all in one command buffer
ROUNDS = 3          # min-of-N; the GPU clock ramps, so never trust a single run
WARM = 5


def _uniforms(M, N, K, alpha=1.0):
    p = np.array([M, N, K, np.float32(alpha).view(np.uint32)], np.uint32)
    s = np.array([K, K, N, 1], np.uint32)      # lda, ldb, ldc, transposed
    return p, s


def bench_wgpu(M, N, K):
    import wgpu
    ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    feats = [f for f in ("subgroup", "shader-f16") if f in ad.features]
    dev = ad.request_device_sync(
        required_features=feats,
        required_limits={"max-buffer-size": 1 << 31,
                         "max-storage-buffer-binding-size": 1 << 31})
    q = dev.queue
    mod = dev.create_shader_module(code=translate(gemm_tiled, workgroup_size=(16, 16, 1)))
    pipe = dev.create_compute_pipeline(
        layout="auto", compute={"module": mod, "entry_point": "gemm_tiled"})

    S = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    A = dev.create_buffer(size=M * K * 2, usage=S)
    B = dev.create_buffer(size=N * K * 2, usage=S)
    C = dev.create_buffer(size=M * N * 2, usage=S | wgpu.BufferUsage.COPY_SRC)
    p, s = _uniforms(M, N, K)
    U = wgpu.BufferUsage.UNIFORM
    u1 = dev.create_buffer_with_data(data=p.tobytes(), usage=U)
    u2 = dev.create_buffer_with_data(data=s.tobytes(), usage=U)
    bg = dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
        {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
        for i, b in enumerate([A, B, C, u1, u2])])
    gx, gy = gemm_groups(M, N)

    def run(n):
        enc = dev.create_command_encoder()
        for _ in range(n):
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(gx, gy, 1)
            cp.end()
        q.submit([enc.finish()])
        # map a byte back to force completion before the timer stops
        rb = dev.create_buffer(size=4, usage=wgpu.BufferUsage.MAP_READ
                               | wgpu.BufferUsage.COPY_DST)
        e2 = dev.create_command_encoder()
        e2.copy_buffer_to_buffer(C, 0, rb, 0, 4)
        q.submit([e2.finish()])
        rb.map_sync(wgpu.MapMode.READ)
        rb.unmap()

    return _time(run)


def bench_metal(M, N, K):
    import Metal
    dev = Metal.MTLCreateSystemDefaultDevice()
    q = dev.newCommandQueue()
    src = translate(gemm_tiled, target="msl", workgroup_size=(16, 16, 1))
    lib, err = dev.newLibraryWithSource_options_error_(src, None, None)
    if lib is None:
        raise RuntimeError(f"MSL compile failed: {err}")
    pso, err = dev.newComputePipelineStateWithFunction_error_(
        lib.newFunctionWithName_("gemm_tiled"), None)
    if pso is None:
        raise RuntimeError(f"pipeline failed: {err}")

    shared = Metal.MTLResourceStorageModeShared
    bufs = [dev.newBufferWithLength_options_(n, shared)
            for n in (M * K * 2, N * K * 2, M * N * 2)]
    for arr in _uniforms(M, N, K):
        b = dev.newBufferWithLength_options_(arr.nbytes, shared)
        b.contents().as_buffer(arr.nbytes)[:] = arr.tobytes()
        bufs.append(b)
    gx, gy = gemm_groups(M, N)
    grid, tg = Metal.MTLSizeMake(gx, gy, 1), Metal.MTLSizeMake(16, 16, 1)

    def run(n):
        cb = q.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        for i, b in enumerate(bufs):
            enc.setBuffer_offset_atIndex_(b, 0, i)
        for _ in range(n):
            enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

    return _time(run)


def _time(run):
    run(WARM)                       # ramp the GPU clock; cold reads ~2.8x slow
    return min((_once(run) for _ in range(ROUNDS)))


def _once(run):
    t0 = time.time()
    run(REP)
    return time.time() - t0


def main():
    M, N, K = (int(x) for x in (sys.argv[1:4] or (1024, 12288, 1536)))
    flop = 2 * M * N * K * REP
    print(f"gemm_tiled  M={M} N={N} K={K}  ({REP} dispatches/run, min of {ROUNDS})\n")
    print(f"{'backend':<16}{'ms/gemm':>10}{'TFLOP/s':>10}")
    got = {}
    for name, fn in (("native Metal", bench_metal), ("wgpu-native", bench_wgpu)):
        try:
            t = fn(M, N, K)
        except Exception as e:                       # a backend may be absent
            print(f"{name:<16}{'SKIP':>10}  {type(e).__name__}: {e}")
            continue
        got[name] = flop / t / 1e12
        print(f"{name:<16}{t / REP * 1000:10.2f}{got[name]:10.2f}")
    if len(got) == 2:
        a, b = got["native Metal"], got["wgpu-native"]
        print(f"\nMetal is {a / b:.2f}x faster on the same kernel source.")


if __name__ == "__main__":
    main()
