"""GPU tests: run each generated WGSL kernel via wgpu, compare to numpy.

Skipped entirely when no WebGPU adapter is available.
"""

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")

from gemma3 import kernels as K
from gemma3.reference import rms_norm, gelu_tanh, apply_rope


@pytest.fixture(scope="module")
def device():
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        feats = [f for f in ("shader-f16", "subgroup") if f in adapter.features]
        dev = adapter.request_device_sync(required_features=feats)
        dev._has_f16 = "shader-f16" in feats
        dev._has_sg = "subgroup" in feats
        return dev
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no WebGPU adapter: {exc}")


def run_kernel(device, kern, buffers, dispatch):
    """Run one @kernel. buffers: list of (ndarray, writable). Returns final contents."""
    gpu_bufs = []
    for arr, writable in buffers:
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        buf = device.create_buffer_with_data(data=arr.tobytes(), usage=usage)
        gpu_bufs.append(buf)

    code = kern.wgsl
    try:
        module = device.create_shader_module(code=code)
    except Exception:
        if "enable subgroups;" not in code:
            raise
        # naga compat: ops supported, directive not yet
        module = device.create_shader_module(code=code.replace("enable subgroups;", ""))
    entries = []
    for i, (arr, writable) in enumerate(buffers):
        btype = (
            wgpu.BufferBindingType.storage
            if writable
            else wgpu.BufferBindingType.read_only_storage
        )
        entries.append({
            "binding": i,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {"type": btype},
        })
    layout = device.create_bind_group_layout(entries=entries)
    bind_group = device.create_bind_group(
        layout=layout,
        entries=[
            {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
            for i, b in enumerate(gpu_bufs)
        ],
    )
    pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[layout]),
        compute={"module": module, "entry_point": kern.__name__},
    )
    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bind_group)
    cp.dispatch_workgroups(*dispatch)
    cp.end()
    device.queue.submit([enc.finish()])

    out = []
    for (arr, _), buf in zip(buffers, gpu_bufs):
        raw = device.queue.read_buffer(buf)
        out.append(np.frombuffer(raw, dtype=arr.dtype).reshape(arr.shape).copy())
    return out


def u32(*vals):
    return np.array(vals, dtype=np.uint32)


rng = np.random.default_rng(42)


class TestEmbedScale:
    def test_matches_numpy(self, device):
        vocab, hidden = 100, 64
        table = rng.standard_normal((vocab, hidden), dtype=np.float32)
        tok = u32(7)
        out = np.zeros(hidden, dtype=np.float32)
        res = run_kernel(
            device, K.embed_scale,
            [(tok, False), (table, False), (out, True), (u32(hidden), False)],
            ((hidden + 63) // 64, 1, 1),
        )
        expected = table[7] * np.sqrt(np.float32(hidden))
        np.testing.assert_allclose(res[2], expected, rtol=1e-5)


class TestRmsNorm:
    def test_single_row(self, device):
        n = 640
        x = rng.standard_normal(n, dtype=np.float32)
        w = rng.standard_normal(n, dtype=np.float32) * 0.1
        out = np.zeros(n, dtype=np.float32)
        res = run_kernel(
            device, K.rmsnorm,
            [(x, False), (w, False), (out, True), (u32(1, n), False)],
            ((n + 63) // 64, 1, 1),
        )
        np.testing.assert_allclose(res[2], rms_norm(x, w), rtol=1e-4, atol=1e-5)

    def test_multi_row_per_head(self, device):
        rows, cols = 4, 256
        x = rng.standard_normal((rows, cols), dtype=np.float32)
        w = rng.standard_normal(cols, dtype=np.float32) * 0.1
        out = np.zeros((rows, cols), dtype=np.float32)
        res = run_kernel(
            device, K.rmsnorm,
            [(x, False), (w, False), (out, True), (u32(rows, cols), False)],
            ((rows * cols + 63) // 64, 1, 1),
        )
        np.testing.assert_allclose(res[2], rms_norm(x, w), rtol=1e-4, atol=1e-5)


class TestMatvec:
    def test_matches_numpy(self, device):
        n_out, n_in = 512, 640
        W = rng.standard_normal((n_out, n_in), dtype=np.float32) * 0.05
        x = rng.standard_normal(n_in, dtype=np.float32)
        y = np.zeros(n_out, dtype=np.float32)
        res = run_kernel(
            device, K.matvec,
            [(W, False), (x, False), (y, True), (u32(n_out, n_in), False)],
            ((n_out + 63) // 64, 1, 1),
        )
        np.testing.assert_allclose(res[2], W @ x, rtol=1e-4, atol=1e-4)


class TestRope:
    @pytest.mark.parametrize("theta,pos", [(10_000.0, 0), (10_000.0, 17), (1_000_000.0, 100)])
    def test_matches_reference(self, device, theta, pos):
        n_heads, head_dim = 4, 256
        x = rng.standard_normal((n_heads, head_dim), dtype=np.float32)
        fp = np.array([theta, float(pos)], dtype=np.float32)
        res = run_kernel(
            device, K.rope,
            [(x.copy(), True), (fp, False), (u32(n_heads, head_dim), False)],
            ((n_heads * head_dim // 2 + 63) // 64, 1, 1),
        )
        np.testing.assert_allclose(res[0], apply_rope(x, pos, theta), rtol=1e-4, atol=1e-4)


class TestKvAppend:
    def test_writes_at_position(self, device):
        vec_len, max_seq, pos = 256, 16, 5
        src = rng.standard_normal(vec_len, dtype=np.float32)
        cache = np.zeros((max_seq, vec_len), dtype=np.float32)
        res = run_kernel(
            device, K.kv_append,
            [(src, False), (cache, True), (u32(vec_len, pos), False)],
            ((vec_len + 63) // 64, 1, 1),
        )
        assert np.allclose(res[1][pos], src)
        assert np.allclose(res[1][:pos], 0) and np.allclose(res[1][pos + 1:], 0)


class TestAttention:
    def test_full_pipeline_matches_numpy(self, device):
        n_heads, head_dim, max_seq = 4, 256, 32
        kv_len, start = 10, 0
        q = rng.standard_normal((n_heads, head_dim), dtype=np.float32)
        k_cache = np.zeros((max_seq, head_dim), dtype=np.float32)
        v_cache = np.zeros((max_seq, head_dim), dtype=np.float32)
        k_cache[:kv_len] = rng.standard_normal((kv_len, head_dim), dtype=np.float32)
        v_cache[:kv_len] = rng.standard_normal((kv_len, head_dim), dtype=np.float32)
        scores = np.zeros((n_heads, max_seq), dtype=np.float32)

        res = run_kernel(
            device, K.attn_scores,
            [(q, False), (k_cache, False), (scores, True),
             (u32(n_heads, head_dim, kv_len, start, max_seq), False)],
            ((kv_len + 7) // 8, (n_heads + 7) // 8, 1),
        )
        scores = res[2]
        res = run_kernel(
            device, K.attn_softmax,
            [(scores, True), (u32(n_heads, kv_len, max_seq), False)],
            ((n_heads + 3) // 4, 1, 1),
        )
        scores = res[0]
        out = np.zeros((n_heads, head_dim), dtype=np.float32)
        res = run_kernel(
            device, K.attn_output,
            [(scores, False), (v_cache, False), (out, True),
             (u32(n_heads, head_dim, kv_len, max_seq), False)],
            ((n_heads * head_dim + 63) // 64, 1, 1),
        )

        # numpy reference
        ref_scores = (q @ k_cache[:kv_len].T) / np.sqrt(np.float32(head_dim))
        ref_scores -= ref_scores.max(axis=-1, keepdims=True)
        probs = np.exp(ref_scores)
        probs /= probs.sum(axis=-1, keepdims=True)
        expected = probs @ v_cache[:kv_len]
        np.testing.assert_allclose(res[2], expected, rtol=1e-4, atol=1e-4)

    def test_sliding_window_masks_prefix(self, device):
        n_heads, head_dim, max_seq = 2, 8, 16
        kv_len, start = 12, 4  # positions 0-3 outside the window
        q = rng.standard_normal((n_heads, head_dim), dtype=np.float32)
        k_cache = rng.standard_normal((max_seq, head_dim), dtype=np.float32)
        scores = np.zeros((n_heads, max_seq), dtype=np.float32)
        res = run_kernel(
            device, K.attn_scores,
            [(q, False), (k_cache, False), (scores, True),
             (u32(n_heads, head_dim, kv_len, start, max_seq), False)],
            ((kv_len + 7) // 8, (n_heads + 7) // 8, 1),
        )
        assert (res[2][:, :start] == -1e9).all()
        assert np.isfinite(res[2][:, start:kv_len]).all()


class TestGeglu:
    def test_matches_numpy(self, device):
        n = 2048
        gate = rng.standard_normal(n, dtype=np.float32)
        up = rng.standard_normal(n, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        res = run_kernel(
            device, K.geglu,
            [(gate, False), (up, False), (out, True), (u32(n), False)],
            ((n + 63) // 64, 1, 1),
        )
        np.testing.assert_allclose(res[2], gelu_tanh(gate) * up, rtol=1e-4, atol=1e-5)

    def test_large_gate_values_no_nan(self, device):
        # Regression: Metal fast-math tanh overflows to NaN for large args.
        # Real Gemma activations reach |gate| > 11 (inner > 44).
        gate = np.array([-500.0, -50.0, -12.0, 11.3, 50.0, 500.0, 0.0, 1.0],
                        dtype=np.float32)
        up = np.ones_like(gate)
        out = np.zeros_like(gate)
        res = run_kernel(
            device, K.geglu,
            [(gate, False), (up, False), (out, True), (u32(len(gate)), False)],
            (1, 1, 1),
        )
        assert np.isfinite(res[2]).all()
        np.testing.assert_allclose(res[2], gelu_tanh(gate) * up, rtol=1e-4, atol=1e-5)


class TestAddVec:
    def test_matches_numpy(self, device):
        n = 640
        a = rng.standard_normal(n, dtype=np.float32)
        b = rng.standard_normal(n, dtype=np.float32)
        res = run_kernel(
            device, K.add_vec,
            [(a.copy(), True), (b, False), (u32(n), False)],
            ((n + 63) // 64, 1, 1),
        )
        np.testing.assert_allclose(res[0], a + b, rtol=1e-6)


class TestMatvecWg:
    @pytest.mark.parametrize("n_out,n_in", [(640, 2048), (256, 640), (2048, 640), (100, 100)])
    def test_matches_numpy(self, device, n_out, n_in):
        W = rng.standard_normal((n_out, n_in), dtype=np.float32) * 0.05
        x = rng.standard_normal(n_in, dtype=np.float32)
        y = np.zeros(n_out, dtype=np.float32)
        res = run_kernel(
            device, K.matvec_wg,
            [(W, False), (x, False), (y, True), (u32(n_out, n_in), False)],
            (n_out, 1, 1),  # one workgroup per row
        )
        np.testing.assert_allclose(res[2], W @ x, rtol=1e-4, atol=1e-4)

    def test_matches_plain_matvec(self, device):
        n_out, n_in = 640, 1024
        W = rng.standard_normal((n_out, n_in), dtype=np.float32) * 0.05
        x = rng.standard_normal(n_in, dtype=np.float32)
        y1 = np.zeros(n_out, dtype=np.float32)
        y2 = np.zeros(n_out, dtype=np.float32)
        r1 = run_kernel(device, K.matvec,
                        [(W, False), (x, False), (y1, True), (u32(n_out, n_in), False)],
                        ((n_out + 63) // 64, 1, 1))
        r2 = run_kernel(device, K.matvec_wg,
                        [(W, False), (x, False), (y2, True), (u32(n_out, n_in), False)],
                        (n_out, 1, 1))
        np.testing.assert_allclose(r1[2], r2[2], rtol=1e-4, atol=1e-4)


class TestRmsNormWg:
    @pytest.mark.parametrize("rows,cols", [(1, 640), (4, 256), (1, 256), (3, 100)])
    def test_matches_numpy(self, device, rows, cols):
        x = rng.standard_normal((rows, cols), dtype=np.float32)
        w = rng.standard_normal(cols, dtype=np.float32) * 0.1
        out = np.zeros((rows, cols), dtype=np.float32)
        res = run_kernel(
            device, K.rmsnorm_wg,
            [(x, False), (w, False), (out, True), (u32(rows, cols), False)],
            (rows, 1, 1),  # one workgroup per row
        )
        np.testing.assert_allclose(res[2], rms_norm(x, w), rtol=1e-4, atol=1e-5)


class TestF16Kernels:
    @pytest.fixture(autouse=True)
    def _need_f16(self, device):
        if not getattr(device, "_has_f16", False):
            pytest.skip("shader-f16 not supported")

    def test_matvec_f16_matches_numpy(self, device):
        n_out, n_in = 512, 640
        W = (rng.standard_normal((n_out, n_in), dtype=np.float32) * 0.05).astype(np.float16)
        x = rng.standard_normal(n_in, dtype=np.float32)
        y = np.zeros(n_out, dtype=np.float32)
        res = run_kernel(
            device, K.matvec_f16,
            [(W, False), (x, False), (y, True), (u32(n_out, n_in), False)],
            ((n_out + 63) // 64, 1, 1),
        )
        np.testing.assert_allclose(res[2], W.astype(np.float32) @ x, rtol=1e-4, atol=1e-4)

    def test_matvec_wg_f16_matches_numpy(self, device):
        n_out, n_in = 640, 2048
        W = (rng.standard_normal((n_out, n_in), dtype=np.float32) * 0.05).astype(np.float16)
        x = rng.standard_normal(n_in, dtype=np.float32)
        y = np.zeros(n_out, dtype=np.float32)
        res = run_kernel(
            device, K.matvec_wg_f16,
            [(W, False), (x, False), (y, True), (u32(n_out, n_in), False)],
            (n_out, 1, 1),
        )
        np.testing.assert_allclose(res[2], W.astype(np.float32) @ x, rtol=1e-4, atol=1e-4)

    def test_embed_scale_f16(self, device):
        vocab, hidden = 50, 64
        table = (rng.standard_normal((vocab, hidden), dtype=np.float32)).astype(np.float16)
        tok = u32(9)
        out = np.zeros(hidden, dtype=np.float32)
        res = run_kernel(
            device, K.embed_scale_f16,
            [(tok, False), (table, False), (out, True), (u32(hidden), False)],
            ((hidden + 63) // 64, 1, 1),
        )
        expected = table[9].astype(np.float32) * np.sqrt(np.float32(hidden))
        np.testing.assert_allclose(res[2], expected, rtol=1e-5)


class TestRmsNormAdd:
    def test_matches_numpy(self, device):
        n = 640
        x_in = rng.standard_normal(n, dtype=np.float32)
        w = rng.standard_normal(n, dtype=np.float32) * 0.1
        residual = rng.standard_normal(n, dtype=np.float32)
        res = run_kernel(
            device, K.rmsnorm_add_wg,
            [(x_in, False), (w, False), (residual.copy(), True), (u32(1, n), False)],
            (1, 1, 1),
        )
        np.testing.assert_allclose(res[2], residual + rms_norm(x_in, w),
                                   rtol=1e-4, atol=1e-5)


class TestAttentionFused:
    @pytest.mark.parametrize("kv_len,start", [(10, 0), (100, 0), (100, 37), (1, 0)])
    def test_matches_numpy(self, device, kv_len, start):
        n_heads, head_dim, max_seq = 4, 256, 128
        q = rng.standard_normal((n_heads, head_dim), dtype=np.float32)
        k_cache = rng.standard_normal((max_seq, head_dim), dtype=np.float32)
        v_cache = rng.standard_normal((max_seq, head_dim), dtype=np.float32)
        scores = np.zeros((n_heads, max_seq), dtype=np.float32)
        out = np.zeros((n_heads, head_dim), dtype=np.float32)
        res = run_kernel(
            device, K.attention_fused,
            [(q, False), (k_cache, False), (v_cache, False), (scores, True),
             (out, True), (u32(n_heads, head_dim, kv_len, start, max_seq), False)],
            (n_heads, 1, 1),
        )
        # numpy reference over the visible window [start, kv_len)
        keys = k_cache[start:kv_len]
        vals = v_cache[start:kv_len]
        ref_scores = (q @ keys.T) / np.sqrt(np.float32(head_dim))
        ref_scores -= ref_scores.max(axis=-1, keepdims=True)
        probs = np.exp(ref_scores)
        probs /= probs.sum(axis=-1, keepdims=True)
        expected = probs @ vals
        np.testing.assert_allclose(res[4], expected, rtol=1e-4, atol=1e-4)

    def test_matches_three_pass_pipeline(self, device):
        n_heads, head_dim, max_seq = 4, 256, 64
        kv_len, start = 33, 5
        q = rng.standard_normal((n_heads, head_dim), dtype=np.float32)
        k_cache = rng.standard_normal((max_seq, head_dim), dtype=np.float32)
        v_cache = rng.standard_normal((max_seq, head_dim), dtype=np.float32)
        dims5 = u32(n_heads, head_dim, kv_len, start, max_seq)

        # fused
        res_f = run_kernel(
            device, K.attention_fused,
            [(q, False), (k_cache, False), (v_cache, False),
             (np.zeros((n_heads, max_seq), np.float32), True),
             (np.zeros((n_heads, head_dim), np.float32), True), (dims5, False)],
            (n_heads, 1, 1),
        )
        # three-pass
        r = run_kernel(device, K.attn_scores,
                       [(q, False), (k_cache, False),
                        (np.zeros((n_heads, max_seq), np.float32), True), (dims5, False)],
                       ((kv_len + 7) // 8, 1, 1))
        r = run_kernel(device, K.attn_softmax,
                       [(r[2], True), (u32(n_heads, kv_len, max_seq), False)], (1, 1, 1))
        r = run_kernel(device, K.attn_output,
                       [(r[0], False), (v_cache, False),
                        (np.zeros((n_heads, head_dim), np.float32), True),
                        (u32(n_heads, head_dim, kv_len, max_seq), False)],
                       ((n_heads * head_dim + 63) // 64, 1, 1))
        np.testing.assert_allclose(res_f[4], r[2], rtol=1e-4, atol=1e-4)


class TestPackedMatvec:
    @pytest.mark.parametrize("n_out,n_in", [(512, 640), (640, 2048), (100, 64)])
    def test_matvec_wg_packed(self, device, n_out, n_in):
        W16 = (rng.standard_normal((n_out, n_in), dtype=np.float32) * 0.05).astype(np.float16)
        Wu32 = W16.tobytes()
        w_view = np.frombuffer(Wu32, dtype=np.uint32).reshape(n_out, n_in // 2)
        x = rng.standard_normal(n_in, dtype=np.float32)
        y = np.zeros(n_out, dtype=np.float32)
        res = run_kernel(
            device, K.matvec_wg_packed,
            [(w_view, False), (x, False), (y, True), (u32(n_out, n_in), False)],
            (n_out, 1, 1),
        )
        np.testing.assert_allclose(res[2], W16.astype(np.float32) @ x, rtol=1e-4, atol=1e-4)

    def test_matvec_packed(self, device):
        n_out, n_in = 512, 640
        W16 = (rng.standard_normal((n_out, n_in), dtype=np.float32) * 0.05).astype(np.float16)
        w_view = np.frombuffer(W16.tobytes(), dtype=np.uint32).reshape(n_out, n_in // 2)
        x = rng.standard_normal(n_in, dtype=np.float32)
        y = np.zeros(n_out, dtype=np.float32)
        res = run_kernel(
            device, K.matvec_packed,
            [(w_view, False), (x, False), (y, True), (u32(n_out, n_in), False)],
            ((n_out + 63) // 64, 1, 1),
        )
        np.testing.assert_allclose(res[2], W16.astype(np.float32) @ x, rtol=1e-4, atol=1e-4)


class TestSubgroupKernels:
    @pytest.fixture(autouse=True)
    def _need_sg(self, device):
        if not getattr(device, "_has_sg", False):
            pytest.skip("subgroup feature not supported")

    def test_rmsnorm_wg_sg(self, device):
        rows, cols = 4, 256
        x = rng.standard_normal((rows, cols), dtype=np.float32)
        w = rng.standard_normal(cols, dtype=np.float32) * 0.1
        out = np.zeros((rows, cols), dtype=np.float32)
        res = run_kernel(device, K.rmsnorm_wg_sg,
                         [(x, False), (w, False), (out, True), (u32(rows, cols), False)],
                         (rows, 1, 1))
        np.testing.assert_allclose(res[2], rms_norm(x, w), rtol=1e-4, atol=1e-5)

    def test_rmsnorm_add_wg_sg(self, device):
        n = 640
        x_in = rng.standard_normal(n, dtype=np.float32)
        w = rng.standard_normal(n, dtype=np.float32) * 0.1
        residual = rng.standard_normal(n, dtype=np.float32)
        res = run_kernel(device, K.rmsnorm_add_wg_sg,
                         [(x_in, False), (w, False), (residual.copy(), True), (u32(1, n), False)],
                         (1, 1, 1))
        np.testing.assert_allclose(res[2], residual + rms_norm(x_in, w), rtol=1e-4, atol=1e-5)

    def test_matvec_wg_packed_sg(self, device):
        n_out, n_in = 640, 2048
        W16 = (rng.standard_normal((n_out, n_in), dtype=np.float32) * 0.05).astype(np.float16)
        w_view = np.frombuffer(W16.tobytes(), dtype=np.uint32).reshape(n_out, n_in // 2)
        x = rng.standard_normal(n_in, dtype=np.float32)
        y = np.zeros(n_out, dtype=np.float32)
        res = run_kernel(device, K.matvec_wg_packed_sg,
                         [(w_view, False), (x, False), (y, True), (u32(n_out, n_in), False)],
                         (n_out, 1, 1))
        np.testing.assert_allclose(res[2], W16.astype(np.float32) @ x, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("kv_len,start", [(10, 0), (100, 37), (1, 0)])
    def test_attention_fused_sg(self, device, kv_len, start):
        n_heads, head_dim, max_seq = 4, 256, 128
        q = rng.standard_normal((n_heads, head_dim), dtype=np.float32)
        k_cache = rng.standard_normal((max_seq, head_dim), dtype=np.float32)
        v_cache = rng.standard_normal((max_seq, head_dim), dtype=np.float32)
        res = run_kernel(
            device, K.attention_fused_sg,
            [(q, False), (k_cache, False), (v_cache, False),
             (np.zeros((n_heads, max_seq), np.float32), True),
             (np.zeros((n_heads, head_dim), np.float32), True),
             (u32(n_heads, head_dim, kv_len, start, max_seq), False)],
            (n_heads, 1, 1),
        )
        keys, vals = k_cache[start:kv_len], v_cache[start:kv_len]
        sc = (q @ keys.T) / np.sqrt(np.float32(head_dim))
        sc -= sc.max(axis=-1, keepdims=True)
        pr = np.exp(sc); pr /= pr.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(res[4], pr @ vals, rtol=1e-4, atol=1e-4)
