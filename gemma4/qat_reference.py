"""Numpy weight-only reference for the QAT checkpoint.

Reuses ReferenceGemma4 with dequantized weights and quantized embed/PLE
gathers overridden — the exact math the QAT GPU runner implements (no SRQ).
Verification oracle: dequant streams ~2.5GB, so it is slow (test-only).
"""

from __future__ import annotations

import numpy as np

from .loader import PREFIX, Gemma4Config
from .qat_loader import QATIndex
from .reference import ReferenceGemma4


class QATReference(ReferenceGemma4):
    def __init__(self, config: Gemma4Config, qat: QATIndex,
                 max_seq: int = 1024, ple: bool = True) -> None:
        super().__init__(config, qat.idx, max_seq=max_seq, ple=ple)
        self.qat = qat
        self._embed_full: np.ndarray | None = None  # lazily dequant tied lm_head

    def _w(self, name: str) -> np.ndarray:
        if name == PREFIX + "embed_tokens.weight":  # tied lm_head (2-bit)
            if self._embed_full is None:
                self._embed_full = self.qat.dequant_embedding_full(
                    "language_model.embed_tokens", 2, self.cfg.hidden_size)
            return self._embed_full
        module = name[len("model."):-len(".weight")]
        if self.qat.is_quantized(module):
            return self.qat.dequant_weight(module)
        return self.qat.idx.tensor(name)

    def _embed_row(self, token_id: int) -> np.ndarray:
        return self.qat.dequant_embedding_row(
            "language_model.embed_tokens", token_id, 2, self.cfg.hidden_size)

    def _ple_table_row(self, token_id: int) -> np.ndarray:
        cfg = self.cfg
        n, d = cfg.num_layers, cfg.hidden_size_per_layer_input
        row = self.qat.dequant_embedding_row(
            "language_model.embed_tokens_per_layer", token_id, 4, n * d)
        return row.reshape(n, d)
