"""GPU tests for the Gemma 4 kernels: each dispatch compared to numpy.

Reuses the wgpu harness from test_gemma3_kernels_gpu; skipped without an
adapter. No model weights needed — all inputs are synthetic.
"""

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")

from test_gemma3_kernels_gpu import device, run_kernel, u32  # noqa: F401

from gemma4 import kernels as K4
from gemma4.reference import apply_rope_partial, rms_norm_noscale, softcap

rng = np.random.default_rng(1234)


def f32a(*vals):
    return np.array(vals, dtype=np.float32)


class TestRopePl:
    def test_full_cutoff_matches_reference(self, device):
        n_heads, hd = 4, 256
        x = rng.standard_normal((n_heads, hd), dtype=np.float32)
        res = run_kernel(
            device, K4.rope_pl,
            [(x, True), (f32a(1e4, 9.0), False), (u32(n_heads, hd, hd // 2), False)],
            ((n_heads * hd // 2 + 63) // 64, 1, 1),
        )
        expected = apply_rope_partial(x, 9, 1e4, cutoff=hd // 2)
        np.testing.assert_allclose(res[0], expected, rtol=1e-4, atol=1e-5)

    def test_partial_cutoff_hd512(self, device):
        n_heads, hd, cutoff = 8, 512, 64
        x = rng.standard_normal((n_heads, hd), dtype=np.float32)
        res = run_kernel(
            device, K4.rope_pl,
            [(x, True), (f32a(1e6, 100.0), False), (u32(n_heads, hd, cutoff), False)],
            ((n_heads * hd // 2 + 63) // 64, 1, 1),
        )
        expected = apply_rope_partial(x, 100, 1e6, cutoff=cutoff)
        np.testing.assert_allclose(res[0], expected, rtol=1e-4, atol=1e-5)
        # pairs beyond the cutoff are untouched
        half = hd // 2
        np.testing.assert_array_equal(res[0][:, cutoff:half], x[:, cutoff:half])
        np.testing.assert_array_equal(res[0][:, half + cutoff:], x[:, half + cutoff:])


def _np_attention_noscale(q, k, v, start):
    scores = q @ k.T  # scaling = 1.0
    if start > 0:
        scores[:, :start] = -1e9
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs @ v


class TestAttentionFusedG4:
    @pytest.mark.parametrize("hd,start", [(256, 0), (512, 0), (256, 3), (512, 5)])
    def test_matches_numpy(self, device, hd, start):
        n_heads, max_seq, kv_len = 8, 16, 11
        q = rng.standard_normal((n_heads, hd), dtype=np.float32)
        k = rng.standard_normal((max_seq, hd), dtype=np.float32)
        v = rng.standard_normal((max_seq, hd), dtype=np.float32)
        scores = np.zeros((n_heads, max_seq), dtype=np.float32)
        out = np.zeros((n_heads, hd), dtype=np.float32)
        res = run_kernel(
            device, K4.attention_fused_g4,
            [(q, False), (k, False), (v, False), (scores, True), (out, True),
             (u32(n_heads, hd, kv_len, start, max_seq), False)],
            (n_heads, 1, 1),
        )
        expected = _np_attention_noscale(q, k[:kv_len], v[:kv_len], start)
        np.testing.assert_allclose(res[4], expected, rtol=1e-4, atol=1e-5)


class TestRmsNormNoScale:
    @pytest.mark.parametrize("n_rows,row_len", [(1, 512), (8, 256), (35, 256)])
    def test_matches_numpy(self, device, n_rows, row_len):
        x = rng.standard_normal((n_rows, row_len), dtype=np.float32)
        out = np.zeros_like(x)
        res = run_kernel(
            device, K4.rmsnorm_ns_wg,
            [(x, False), (out, True), (u32(n_rows, row_len), False)],
            (n_rows, 1, 1),
        )
        np.testing.assert_allclose(res[1], rms_norm_noscale(x), rtol=1e-4, atol=1e-5)


class TestSoftcap:
    def test_matches_numpy(self, device):
        n = 1000
        x = rng.standard_normal(n, dtype=np.float32) * 50
        res = run_kernel(
            device, K4.softcap,
            [(x, True), (f32a(30.0), False), (u32(n), False)],
            ((n + 63) // 64, 1, 1),
        )
        np.testing.assert_allclose(res[0], softcap(x, 30.0), rtol=1e-5, atol=1e-6)
        assert np.abs(res[0]).max() < 30.0


class TestAddScale:
    def test_matches_numpy(self, device):
        n = 1536
        branch = rng.standard_normal(n, dtype=np.float32)
        residual = rng.standard_normal(n, dtype=np.float32)
        scale = 0.0186
        res = run_kernel(
            device, K4.add_scale,
            [(branch, False), (residual.copy(), True), (f32a(scale), False), (u32(n), False)],
            ((n + 63) // 64, 1, 1),
        )
        np.testing.assert_allclose(
            res[1], (residual + branch) * np.float32(scale), rtol=1e-5, atol=1e-7)


class TestCombineScaled:
    def test_matches_numpy(self, device):
        n = 35 * 256
        a = rng.standard_normal(n, dtype=np.float32)
        b = rng.standard_normal(n, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        scale = 2.0 ** -0.5
        res = run_kernel(
            device, K4.combine_scaled,
            [(a, False), (b, False), (out, True), (f32a(scale), False), (u32(n), False)],
            ((n + 63) // 64, 1, 1),
        )
        np.testing.assert_allclose(res[2], (a + b) * np.float32(scale), rtol=1e-5, atol=1e-6)
