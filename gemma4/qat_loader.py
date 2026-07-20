"""Loader for the Gemma 4 E2B QAT-mobile checkpoint (int2/4/8 packed weights).

`google/gemma-4-E2B-it-qat-mobile-transformers` — a 2.46GB quantization-aware
checkpoint (vs the 9.8GB bf16). Each linear stores a packed `weight` (U8 for
2/4-bit, I8 for 8-bit) plus a per-output-row f32 `weight_scale`; embeddings
store `embedding_quantized` + per-row (per-block) `embedding_scale`. Dequant
math mirrors transformers `integrations/gemma_quant.py` exactly.

Weight-only inference: dequant weights to f32, use f32 activations (SRQ
activation scales are skipped — the QAT weights carry the quantization).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .loader import PREFIX, SafetensorsIndex


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    """U8 [..., n/2] -> int8 [..., n]: low nibble first, values shifted -8."""
    packed = packed.astype(np.uint8)
    low = (packed & 0x0F).astype(np.int16) - 8
    high = (packed >> 4).astype(np.int16) - 8
    out = np.stack([low, high], axis=-1).reshape(*packed.shape[:-1], -1)
    return out.astype(np.int8)


def unpack_int2(packed: np.ndarray) -> np.ndarray:
    """U8 [..., n/4] -> int8 [..., n]: 4 values/byte low-first, shifted -2."""
    packed = packed.astype(np.uint8)
    vs = [((packed >> sh) & 0x03).astype(np.int16) - 2 for sh in (0, 2, 4, 6)]
    out = np.stack(vs, axis=-1).reshape(*packed.shape[:-1], -1)
    return out.astype(np.int8)


def unpack_bits(packed: np.ndarray, bits: int) -> np.ndarray:
    if bits == 2:
        return unpack_int2(packed)
    if bits == 4:
        return unpack_int4(packed)
    if bits == 8:
        return packed.astype(np.int8)
    raise ValueError(f"unsupported bits {bits}")


#: (regex on the tensor name without the "model." prefix) -> num_bits.
#: Order matters — first match wins, mirroring the config's dict order.
_BIT_RULES = [
    (r"^lm_head$", 2),
    (r"language_model\.embed_tokens$", 2),
    (r"language_model\.embed_tokens_per_layer$", 4),
    (r"language_model\.layers\.(\d|1[0-4])\.mlp\.", 4),
    (r"language_model\.layers\.\d+\.mlp\.", 2),
    (r"language_model\.layers\.\d+\.per_layer_input_gate$", 8),
    (r"language_model\.layers\.\d+\.per_layer_projection$", 8),
    (r"language_model\.layers\.\d+\.self_attn\.", 4),
]


def _module_bits(module: str) -> int:
    """num_bits for a module name like 'language_model.layers.0.mlp.gate_proj'."""
    for pattern, bits in _BIT_RULES:
        if re.search(pattern, module):
            return bits
    raise KeyError(f"no quant rule for module {module!r}")


class QATIndex:
    """Lazy view of the QAT checkpoint with int2/4/8 dequantization."""

    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        self.idx = SafetensorsIndex(model_dir / "model.safetensors")
        self.cfg = json.loads((model_dir / "config.json").read_text())
        self._names = set(self.idx.names())

    def has(self, name: str) -> bool:
        return name in self._names

    def is_quantized(self, module: str) -> bool:
        # quantized linears carry a per-row weight_scale; bf16 ones don't
        return f"model.{module}.weight_scale" in self._names

    # -- linear weights --------------------------------------------------- #

    def packed_weight(self, module: str) -> tuple[np.ndarray, np.ndarray, int]:
        """Raw packed weight (uint8 view), per-row f32 scale, and num_bits."""
        bits = _module_bits(module)
        packed = self.idx.raw(f"model.{module}.weight")  # U8 or I8, [n_out, packed_in]
        scale = self.idx.tensor(f"model.{module}.weight_scale")  # [n_out, 1] f32
        return np.ascontiguousarray(packed), np.ascontiguousarray(scale), bits

    def dequant_weight(self, module: str) -> np.ndarray:
        """Full f32 weight [n_out, n_in] (for reference / verification)."""
        packed, scale, bits = self.packed_weight(module)
        ints = unpack_bits(packed, bits).astype(np.float32)
        return ints * scale  # per-row scale broadcasts over n_in

    # -- embeddings ------------------------------------------------------- #

    def dequant_embedding_row(self, table: str, token_id: int,
                              bits: int, n_cols: int) -> np.ndarray:
        """Dequant a single row of a quantized embedding table (no embed_scale)."""
        packed = self.idx.raw(f"model.{table}.embedding_quantized")[token_id]
        scale = self.idx.tensor(f"model.{table}.embedding_scale")[token_id]  # [n_blocks]
        ints = unpack_bits(packed, bits).astype(np.float32)  # [n_cols]
        block = n_cols // scale.shape[-1]
        return ints * np.repeat(scale, block)

    def dequant_embedding_full(self, table: str, bits: int, n_cols: int) -> np.ndarray:
        """Full dequantized embedding table [vocab, n_cols] (for the tied lm_head)."""
        packed = self.idx.raw(f"model.{table}.embedding_quantized")
        scale = self.idx.tensor(f"model.{table}.embedding_scale")  # [vocab, n_blocks]
        ints = unpack_bits(packed, bits).astype(np.float32)
        block = n_cols // scale.shape[-1]
        return ints * np.repeat(scale, block, axis=-1)

    def activation_scales_all_zero(self) -> bool:
        """True if every *activation* scale is 0 (weight-only == exact)."""
        for name in self._names:
            if name.endswith("_activation_scale"):
                if np.any(np.asarray(self.idx.tensor(name)) != 0.0):
                    return False
        return True


def load_qat(model_dir: str | Path) -> QATIndex:
    return QATIndex(model_dir)
