"""Gemma 4 E2B lazy loader + config vs the real checkpoint header.

Requires models/gemma-4-E2B/ (downloaded weights); skipped otherwise.
"""

from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").exists(),
    reason="Gemma 4 E2B weights not downloaded",
)

FULL_LAYERS = {4, 9, 14, 19, 24, 29, 34}


@pytest.fixture(scope="module")
def model():
    from gemma4.loader import load_model

    return load_model(MODEL_DIR)


class TestConfig:
    def test_core_dims(self, model):
        cfg, _ = model
        assert cfg.hidden_size == 1536
        assert cfg.num_layers == 35
        assert cfg.num_heads == 8
        assert cfg.vocab_size == 262144
        assert cfg.sliding_window == 512
        assert cfg.final_logit_softcapping == 30.0
        assert cfg.hidden_size_per_layer_input == 256
        assert cfg.bos_token_id == 2
        assert cfg.eos_token_ids == (1,)

    def test_layer_pattern_and_dims(self, model):
        cfg, _ = model
        assert len(cfg.layers) == 35
        for spec in cfg.layers:
            full = spec.index in FULL_LAYERS
            assert spec.sliding == (not full)
            assert spec.head_dim == (512 if full else 256)
            assert spec.q_dim == 8 * spec.head_dim
            assert spec.intermediate == (12288 if spec.index >= 15 else 6144)
            assert spec.rope_theta == (1e6 if full else 1e4)
            # p-RoPE: full layers rotate only the first 64 frequency pairs
            assert spec.rope_cutoff == (64 if full else 128)

    def test_kv_sharing_map(self, model):
        cfg, _ = model
        for spec in cfg.layers:
            if spec.index < 15:
                assert spec.kv_source == spec.index
                assert not spec.kv_shared
            else:
                assert spec.kv_source == (14 if not spec.sliding else 13)
                assert spec.kv_shared
        # the sources themselves are the last non-shared layer of each type
        assert cfg.layers[13].sliding and not cfg.layers[13].kv_shared
        assert not cfg.layers[14].sliding and not cfg.layers[14].kv_shared


class TestIndex:
    def test_shapes(self, model):
        cfg, idx = model
        assert idx.shape("model.language_model.embed_tokens.weight") == (262144, 1536)
        assert idx.shape("model.language_model.embed_tokens_per_layer.weight") == (262144, 8960)
        assert idx.shape(cfg.t(0, "layer_scalar")) == (1,)
        # v_norm is parameter-free: no weight tensor exists for it
        assert not any(".self_attn.v_norm" in n for n in idx.names())

    def test_raw_is_zero_copy_view(self, model):
        _, idx = model
        raw = idx.raw("model.language_model.norm.weight")
        assert raw.dtype == np.uint16 and raw.shape == (1536,)
        assert not raw.flags.owndata  # view into the memmap, no copy

    def test_bf16_to_f32_bit_exact(self, model):
        cfg, idx = model
        name = cfg.t(0, "input_layernorm.weight")
        raw = idx.raw(name)
        expect = (raw.astype(np.uint32) << 16).view(np.float32)
        got = idx.tensor(name)
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, expect)
        assert np.isfinite(got).all()

    def test_f16_roundtrip(self, model):
        cfg, idx = model
        name = cfg.t(0, "self_attn.q_norm.weight")
        f32 = idx.tensor(name)
        f16 = idx.tensor(name, dtype="f16")
        assert f16.dtype == np.float16
        np.testing.assert_allclose(f16.astype(np.float32), f32, rtol=1e-3, atol=1e-3)

    def test_ple_row_gather(self, model):
        from gemma4.loader import bf16_to_f32

        _, idx = model
        table = idx.raw("model.language_model.embed_tokens_per_layer.weight")
        row = bf16_to_f32(table[12345]).reshape(35, 256)
        assert np.isfinite(row).all()
        assert 0 < np.abs(row).max() < 1e3

    def test_layer_scalars_match_known_values(self, model):
        cfg, idx = model
        assert idx.tensor(cfg.t(0, "layer_scalar"))[0] == pytest.approx(0.0186, abs=2e-3)
        assert idx.tensor(cfg.t(20, "layer_scalar"))[0] == pytest.approx(0.535, abs=5e-2)
