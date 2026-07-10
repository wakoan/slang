"""Metal-backend tests: simd kernels vs numpy, and end-to-end Gemma 3.

Skipped when metalgpu or the model weights are unavailable.
"""

from pathlib import Path

import numpy as np
import pytest

metalgpu = pytest.importorskip("metalgpu")
from metalgpu import MetalSize

from gemma3.kernels_metal import METAL_KERNELS
from gemma3.reference import rms_norm

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gemma-3-270m-it"

rng = np.random.default_rng(7)


@pytest.fixture(scope="module")
def inst():
    try:
        i = metalgpu.Interface()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Metal unavailable: {exc}")
    return i


def run_metal(inst, kern, n_threads, arrays):
    """arrays: list of (ndarray | ('out', n, dtype)). Returns all contents."""
    inst.load_shader_from_string(kern.msl)
    bufs = []
    for a in arrays:
        if isinstance(a, tuple):
            _, n, dtype = a
            bufs.append(inst.create_buffer(n, "float" if dtype == np.float32 else "uint"))
        else:
            bufs.append(inst.array_to_buffer(np.ascontiguousarray(a)))
    inst.run_function(MetalSize(int(n_threads), 1, 1), bufs, kern.__name__)
    out = [np.array(b.contents).copy() for b in bufs]
    for b in bufs:
        b.release()
        b.release = lambda: None
    return out


class TestMetalKernels:
    def test_matvec_simd_packed(self, inst):
        n_out, n_in = 640, 2048
        W16 = (rng.standard_normal((n_out, n_in)).astype(np.float32) * 0.05).astype(np.float16)
        Wp = np.frombuffer(W16.tobytes(), dtype=np.uint32)
        x = rng.standard_normal(n_in).astype(np.float32)
        res = run_metal(inst, METAL_KERNELS["matvec_simd_packed"], n_out * 32,
                        [Wp, x, ("out", n_out, np.float32),
                         np.array([n_out, n_in], np.uint32)])
        np.testing.assert_allclose(res[2], W16.astype(np.float32) @ x,
                                   rtol=1e-4, atol=1e-4)

    def test_rmsnorm_simd(self, inst):
        rows, cols = 4, 256
        x = rng.standard_normal((rows, cols)).astype(np.float32)
        w = (rng.standard_normal(cols) * 0.1).astype(np.float32)
        res = run_metal(inst, METAL_KERNELS["rmsnorm_simd"], rows * 32,
                        [x, w, ("out", rows * cols, np.float32),
                         np.array([rows, cols], np.uint32)])
        np.testing.assert_allclose(res[2].reshape(rows, cols), rms_norm(x, w),
                                   rtol=1e-4, atol=1e-5)

    def test_rmsnorm_add_simd(self, inst):
        n = 640
        x_in = rng.standard_normal(n).astype(np.float32)
        w = (rng.standard_normal(n) * 0.1).astype(np.float32)
        residual = rng.standard_normal(n).astype(np.float32)
        res = run_metal(inst, METAL_KERNELS["rmsnorm_add_simd"], 32,
                        [x_in, w, residual.copy(), np.array([1, n], np.uint32)])
        np.testing.assert_allclose(res[2], residual + rms_norm(x_in, w),
                                   rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("kv_len,start", [(10, 0), (100, 37)])
    def test_attention_simd(self, inst, kv_len, start):
        nh, hd, ms = 4, 256, 128
        q = rng.standard_normal((nh, hd)).astype(np.float32)
        kc = rng.standard_normal((ms, hd)).astype(np.float32)
        vc = rng.standard_normal((ms, hd)).astype(np.float32)
        res = run_metal(inst, METAL_KERNELS["attention_simd"], nh * 32,
                        [q, kc, vc, ("out", nh * ms, np.float32),
                         ("out", nh * hd, np.float32),
                         np.array([nh, hd, kv_len, start, ms], np.uint32)])
        sc = (q @ kc[start:kv_len].T) / np.sqrt(np.float32(hd))
        sc -= sc.max(-1, keepdims=True)
        pr = np.exp(sc)
        pr /= pr.sum(-1, keepdims=True)
        np.testing.assert_allclose(res[4].reshape(nh, hd), pr @ vc[start:kv_len],
                                   rtol=1e-4, atol=1e-4)

    def test_embed_scale_packed(self, inst):
        vocab, hidden = 50, 64
        tbl16 = rng.standard_normal((vocab, hidden)).astype(np.float16)
        res = run_metal(inst, METAL_KERNELS["embed_scale_packed"], hidden // 2,
                        [np.array([7], np.uint32),
                         np.frombuffer(tbl16.tobytes(), dtype=np.uint32),
                         ("out", hidden, np.float32),
                         np.array([hidden], np.uint32)])
        want = tbl16[7].astype(np.float32) * np.sqrt(np.float32(hidden))
        np.testing.assert_allclose(res[2], want, rtol=1e-5)


@pytest.mark.skipif(not (MODEL_DIR / "model.safetensors").exists(),
                    reason="weights not downloaded")
class TestMetalEndToEnd:
    def test_generates_paris_and_matches_wgpu(self):
        from tokenizers import Tokenizer
        from gemma3.loader import load_model
        from gemma3.runner_metal import GemmaMetal
        from gemma3.runner import GemmaGPU

        cfg, weights = load_model(MODEL_DIR)
        tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        text = ("<start_of_turn>user\nWhat is the capital of France?"
                "<end_of_turn>\n<start_of_turn>model\n")
        ids = [cfg.bos_token_id] + tok.encode(text, add_special_tokens=False).ids

        metal = GemmaMetal(cfg, weights, max_seq=128)
        m_out = metal.generate(ids, max_new_tokens=16)
        metal.close()
        assert "Paris" in tok.decode(m_out)
        assert m_out[-1] in cfg.eos_token_ids

        # both backends quantize weights to f16 → greedy tokens should agree
        wgpu_gpu = GemmaGPU(cfg, weights, max_seq=128, dtype="f16")
        w_out = wgpu_gpu.generate(ids, max_new_tokens=16)
        assert m_out == w_out
