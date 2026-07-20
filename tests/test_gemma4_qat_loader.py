"""Gemma 4 QAT loader: int2/4/8 unpack + dequant.

Unpack tests are synthetic (always run); dequant-vs-checkpoint tests need
models/gemma-4-E2B-qat/ and are skipped otherwise.
"""

from pathlib import Path

import numpy as np
import pytest

from gemma4.qat_loader import _module_bits, unpack_int2, unpack_int4

QAT_DIR = Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B-qat"


class TestUnpack:
    def test_int4_low_nibble_first_offset_8(self):
        # byte 0x?? -> [(low & 0xF) - 8, (high >> 4) - 8]
        packed = np.array([0x80, 0xF0, 0x08, 0x7F], dtype=np.uint8)
        got = unpack_int4(packed)
        # 0x80: low=0->-8, high=8->0 ; 0xF0: low=0->-8, high=15->7
        # 0x08: low=8->0,  high=0->-8; 0x7F: low=15->7, high=7->-1
        np.testing.assert_array_equal(got, [-8, 0, -8, 7, 0, -8, 7, -1])
        assert got.dtype == np.int8

    def test_int2_four_per_byte_offset_2(self):
        # 0b11_10_01_00 = 0xE4: v0=0->-2, v1=1->-1, v2=2->0, v3=3->1
        packed = np.array([0xE4], dtype=np.uint8)
        np.testing.assert_array_equal(unpack_int2(packed), [-2, -1, 0, 1])

    def test_shapes(self):
        assert unpack_int4(np.zeros((5, 4), np.uint8)).shape == (5, 8)
        assert unpack_int2(np.zeros((5, 4), np.uint8)).shape == (5, 16)


class TestBitRules:
    def test_module_bits(self):
        b = _module_bits
        assert b("language_model.layers.0.mlp.gate_proj") == 4
        assert b("language_model.layers.14.mlp.gate_proj") == 4
        assert b("language_model.layers.15.mlp.gate_proj") == 2
        assert b("language_model.layers.20.mlp.down_proj") == 2
        assert b("language_model.layers.7.self_attn.q_proj") == 4
        assert b("language_model.layers.3.per_layer_input_gate") == 8
        assert b("language_model.embed_tokens") == 2
        assert b("language_model.embed_tokens_per_layer") == 4


@pytest.fixture(scope="module")
def qat():
    from gemma4.qat_loader import load_qat
    return load_qat(QAT_DIR)


@pytest.mark.skipif(not (QAT_DIR / "model.safetensors").exists(),
                    reason="QAT checkpoint not downloaded")
class TestDequantCheckpoint:
    def test_dequant_shapes_and_finite(self, qat):
        for module, shape in [
            ("language_model.layers.0.self_attn.q_proj", (2048, 1536)),
            ("language_model.layers.0.mlp.gate_proj", (6144, 1536)),
            ("language_model.layers.20.mlp.down_proj", (1536, 12288)),
            ("language_model.layers.5.per_layer_input_gate", (256, 1536)),
        ]:
            w = qat.dequant_weight(module)
            assert w.shape == shape and w.dtype == np.float32
            assert np.isfinite(w).all() and np.abs(w).max() > 0

    def test_dequant_correlates_with_base_bf16(self, qat):
        # QAT derives from gemma-4-E2B-it; our bf16 is the base — different
        # fine-tunes, so only a moderate positive correlation is expected,
        # but a wrong unpack/sign would give ~0. This guards the formula.
        from gemma4.loader import SafetensorsIndex

        base = SafetensorsIndex(
            QAT_DIR.parent / "gemma-4-E2B" / "model.safetensors")
        if not (QAT_DIR.parent / "gemma-4-E2B" / "model.safetensors").exists():
            pytest.skip("base bf16 checkpoint not present")
        m = "language_model.layers.0.mlp.gate_proj"
        dq = qat.dequant_weight(m).ravel()
        ref = base.tensor("model." + m + ".weight").ravel()
        assert np.corrcoef(dq, ref)[0, 1] > 0.5
