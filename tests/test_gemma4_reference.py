"""Gemma 4 E2B numpy reference decoder tests.

Math helpers are tested synthetically (always run); forward-pass tests need
models/gemma-4-E2B/ and are skipped otherwise. The greedy-generation gate is
marked slow (~1 min: the reference streams 9.2GB of bf16 weights per token).
"""

from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B"

needs_weights = pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").exists(),
    reason="Gemma 4 E2B weights not downloaded",
)


class TestMathHelpers:
    def test_rms_norm_scales_by_w_not_1_plus_w(self):
        from gemma4.reference import rms_norm

        x = np.random.default_rng(0).standard_normal(64).astype(np.float32)
        w = np.full(64, 2.0, dtype=np.float32)
        got = rms_norm(x, w)
        expect = x / np.sqrt(np.mean(x * x) + 1e-6) * 2.0
        np.testing.assert_allclose(got, expect, rtol=1e-6)

    def test_rms_norm_noscale(self):
        from gemma4.reference import rms_norm, rms_norm_noscale

        x = np.random.default_rng(1).standard_normal(64).astype(np.float32)
        np.testing.assert_allclose(
            rms_norm_noscale(x), rms_norm(x, np.ones(64, np.float32)), rtol=1e-6)

    def test_rope_partial_full_cutoff_matches_gemma3_rope(self):
        from gemma3.reference import apply_rope
        from gemma4.reference import apply_rope_partial

        x = np.random.default_rng(2).standard_normal((4, 256)).astype(np.float32)
        got = apply_rope_partial(x, position=7, theta=1e4, cutoff=128)
        np.testing.assert_allclose(got, apply_rope(x, 7, 1e4), rtol=1e-5, atol=1e-6)

    def test_rope_partial_identity_beyond_cutoff(self):
        from gemma4.reference import apply_rope_partial

        x = np.random.default_rng(3).standard_normal((8, 512)).astype(np.float32)
        got = apply_rope_partial(x, position=100, theta=1e6, cutoff=64)
        half = 256
        # pairs (i, i+half) for i >= cutoff have inv_freq 0 -> identity
        np.testing.assert_array_equal(got[:, 64:half], x[:, 64:half])
        np.testing.assert_array_equal(got[:, half + 64:], x[:, half + 64:])
        assert not np.allclose(got[:, :64], x[:, :64])  # rotated pairs did move

    def test_softcap_bounds_and_argmax_invariance(self):
        from gemma4.reference import softcap

        x = np.random.default_rng(4).standard_normal(1000).astype(np.float32) * 50
        y = softcap(x, 30.0)
        assert np.abs(y).max() < 30.0
        assert y.argmax() == x.argmax()


@pytest.fixture(scope="module")
def state():
    from gemma4.loader import load_model
    from gemma4.reference import ReferenceGemma4

    cfg, idx = load_model(MODEL_DIR)
    ref = ReferenceGemma4(cfg, idx, max_seq=8)
    logits = ref.forward(cfg.bos_token_id, 0)
    return cfg, ref, logits


@needs_weights
class TestForwardStructure:
    def test_logits_finite_and_softcapped(self, state):
        cfg, _, logits = state
        assert logits.shape == (cfg.vocab_size,)
        assert np.isfinite(logits).all()
        assert np.abs(logits).max() < cfg.final_logit_softcapping

    def test_kv_cache_layout(self, state):
        cfg, ref, _ = state
        # shared layers (15+) own no cache; sources 13/14 are written
        assert all(ref.k_cache[L] is None for L in range(15, 35))
        assert all(ref.k_cache[L] is not None for L in range(15))
        assert ref.k_cache[0].shape[1] == 256   # sliding head_dim
        assert ref.k_cache[4].shape[1] == 512   # full-attention head_dim
        assert np.abs(ref.k_cache[13][0]).max() > 0
        assert np.abs(ref.v_cache[14][0]).max() > 0
        assert np.abs(ref.k_cache[13][1:]).max() == 0  # only pos 0 written

    def test_collect_hidden_shapes(self, state):
        cfg, ref, _ = state
        logits, hiddens = ref.forward(cfg.bos_token_id, 1, collect_hidden=True)
        assert len(hiddens) == cfg.num_layers + 1  # embed + every layer
        assert all(h.shape == (cfg.hidden_size,) for h in hiddens)


@needs_weights
@pytest.mark.slow
class TestGreedyGeneration:
    def test_factual_completion(self):
        from tokenizers import Tokenizer

        from gemma4.loader import load_model
        from gemma4.reference import ReferenceGemma4

        tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        cfg, idx = load_model(MODEL_DIR)
        ref = ReferenceGemma4(cfg, idx, max_seq=32)

        ids = tok.encode("The capital of France is").ids  # BOS added by tokenizer
        assert ids[0] == cfg.bos_token_id
        for pos, t in enumerate(ids[:-1]):
            ref.forward(t, pos)
        out, t_id = [], ids[-1]
        for pos in range(len(ids) - 1, len(ids) + 7):
            t_id = int(ref.forward(t_id, pos).argmax())
            if t_id in cfg.eos_token_ids:
                break
            out.append(t_id)
        assert "Paris" in tok.decode(out)
