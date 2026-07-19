"""Lazy safetensors index + Gemma 4 E2B config.

The E2B checkpoint is 9.8GB bf16; eagerly materializing it as f32 (the gemma3
loader strategy) would need ~20GB RAM. `SafetensorsIndex` parses the header
once, keeps the file memmap'd, and converts individual tensors on demand.
The 4.7GB `embed_tokens_per_layer` table is consumed via `raw()` (a zero-copy
uint16 view) with per-token row gathers, never converted whole.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gemma3.loader import _bf16_to_f32 as bf16_to_f32

PREFIX = "model.language_model."


class SafetensorsIndex:
    """Header-only view of a safetensors file; tensors are sliced on demand."""

    def __init__(self, path: str | Path) -> None:
        self._data = np.memmap(path, dtype=np.uint8, mode="r")
        (header_len,) = struct.unpack("<Q", bytes(self._data[:8]))
        header = json.loads(bytes(self._data[8 : 8 + header_len]))
        header.pop("__metadata__", None)
        self._base = 8 + header_len
        self._info: dict[str, dict] = header

    def names(self) -> list[str]:
        return list(self._info)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._info[name]["shape"])

    def raw(self, name: str) -> np.ndarray:
        """Zero-copy view of the stored bytes (uint16 for BF16/F16 tensors)."""
        info = self._info[name]
        start, end = info["data_offsets"]
        buf = self._data[self._base + start : self._base + end]
        dtype = {"BF16": np.uint16, "F16": np.uint16, "F32": np.float32}[info["dtype"]]
        return buf.view(dtype).reshape(info["shape"])

    def tensor(self, name: str, dtype: str = "f32") -> np.ndarray:
        """Materialize one tensor as f32 (exact from bf16) or f16."""
        kind = self._info[name]["dtype"]
        if kind == "BF16":
            arr = bf16_to_f32(self.raw(name))
        elif kind == "F32":
            arr = self.raw(name).copy()
        elif kind == "F16":
            arr = self.raw(name).view(np.float16).astype(np.float32)
        else:
            raise ValueError(f"Unsupported dtype {kind} for tensor {name}")
        if dtype == "f16":
            return arr.astype(np.float16)
        if dtype != "f32":
            raise ValueError(f"Unsupported target dtype {dtype}")
        return np.ascontiguousarray(arr)


@dataclass(frozen=True)
class LayerSpec:
    index: int
    sliding: bool
    head_dim: int       # 256 sliding / 512 full (from q_norm shape)
    q_dim: int          # num_heads * head_dim
    intermediate: int   # 6144 layers 0-14 / 12288 layers 15-34
    rope_theta: float   # 10k sliding / 1e6 full
    rope_cutoff: int    # active frequency pairs; head_dim//2 = full rotation
    kv_source: int      # own index, or 13/14 for the KV-shared layers 15+

    @property
    def kv_shared(self) -> bool:
        return self.kv_source != self.index


class Gemma4Config:
    """Gemma 4 text config; per-layer dims are read from the checkpoint header."""

    def __init__(self, config_path: str | Path, index: SafetensorsIndex) -> None:
        cfg = json.loads(Path(config_path).read_text())["text_config"]
        self.hidden_size: int = cfg["hidden_size"]
        self.num_layers: int = cfg["num_hidden_layers"]
        self.num_heads: int = cfg["num_attention_heads"]
        self.num_kv_heads: int = cfg["num_key_value_heads"]
        self.vocab_size: int = cfg["vocab_size"]
        self.rms_norm_eps: float = cfg["rms_norm_eps"]
        self.sliding_window: int = cfg["sliding_window"]
        self.final_logit_softcapping: float = cfg["final_logit_softcapping"]
        self.hidden_size_per_layer_input: int = cfg["hidden_size_per_layer_input"]
        self.bos_token_id: int = cfg["bos_token_id"]
        self.eos_token_ids: tuple[int, ...] = (cfg["eos_token_id"],)
        if self.num_kv_heads != 1:
            raise ValueError("Kernels assume num_key_value_heads == 1")

        rope = cfg["rope_parameters"]
        layer_types: list[str] = cfg["layer_types"]
        first_shared = self.num_layers - cfg["num_kv_shared_layers"]
        specs = []
        for L, kind in enumerate(layer_types):
            sliding = kind == "sliding_attention"
            head_dim = index.shape(self.t(L, "self_attn.q_norm.weight"))[0]
            q_dim = index.shape(self.t(L, "self_attn.q_proj.weight"))[0]
            inter = index.shape(self.t(L, "mlp.gate_proj.weight"))[0]
            rotary = rope[kind].get("partial_rotary_factor", 1.0)
            if L < first_shared:
                kv_source = L
            else:  # last non-shared layer of the same attention type
                kv_source = max(s for s in range(first_shared) if layer_types[s] == kind)
            specs.append(LayerSpec(
                index=L, sliding=sliding, head_dim=head_dim, q_dim=q_dim,
                intermediate=inter, rope_theta=rope[kind]["rope_theta"],
                rope_cutoff=int(rotary * head_dim / 2), kv_source=kv_source,
            ))
        self.layers: list[LayerSpec] = specs
        self.embed_scale: float = math.sqrt(self.hidden_size)
        self.ple_scale: float = math.sqrt(self.hidden_size_per_layer_input)

    @staticmethod
    def t(layer: int, sub: str) -> str:
        return f"{PREFIX}layers.{layer}.{sub}"


def load_model(model_dir: str | Path) -> tuple[Gemma4Config, SafetensorsIndex]:
    model_dir = Path(model_dir)
    index = SafetensorsIndex(model_dir / "model.safetensors")
    return Gemma4Config(model_dir / "config.json", index), index
