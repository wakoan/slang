"""Minimal safetensors loader: BF16 → float32 numpy, no torch required."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Upcast a uint16 view of bf16 data to float32 (bf16 = top 16 bits of f32)."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def load_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Load all tensors from a safetensors file as float32 numpy arrays."""
    path = Path(path)
    data = np.memmap(path, dtype=np.uint8, mode="r")
    (header_len,) = struct.unpack("<Q", bytes(data[:8]))
    header = json.loads(bytes(data[8 : 8 + header_len]))
    base = 8 + header_len

    tensors: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        buf = data[base + start : base + end]
        shape = info["shape"]
        dtype = info["dtype"]
        if dtype == "BF16":
            arr = _bf16_to_f32(buf.view(np.uint16)).reshape(shape)
        elif dtype == "F32":
            arr = buf.view(np.float32).reshape(shape).copy()
        elif dtype == "F16":
            arr = buf.view(np.float16).astype(np.float32).reshape(shape)
        else:
            raise ValueError(f"Unsupported dtype {dtype} for tensor {name}")
        tensors[name] = np.ascontiguousarray(arr, dtype=np.float32)
    return tensors


class GemmaConfig:
    """Gemma 3 text config loaded from config.json."""

    def __init__(self, path: str | Path) -> None:
        cfg = json.loads(Path(path).read_text())
        self.hidden_size: int = cfg["hidden_size"]
        self.num_layers: int = cfg["num_hidden_layers"]
        self.num_heads: int = cfg["num_attention_heads"]
        self.num_kv_heads: int = cfg["num_key_value_heads"]
        self.head_dim: int = cfg["head_dim"]
        self.intermediate_size: int = cfg["intermediate_size"]
        self.vocab_size: int = cfg["vocab_size"]
        self.rms_norm_eps: float = cfg["rms_norm_eps"]
        self.rope_theta_global: float = cfg["rope_theta"]
        self.rope_theta_local: float = cfg["rope_local_base_freq"]
        self.sliding_window: int = cfg["sliding_window"]
        self.query_pre_attn_scalar: float = cfg["query_pre_attn_scalar"]
        # "sliding_attention" | "full_attention" per layer
        self.layer_types: list[str] = cfg["layer_types"]
        self.eos_token_ids: tuple[int, ...] = (1, 106)  # <eos>, <end_of_turn>
        self.bos_token_id: int = cfg["bos_token_id"]
        if self.num_kv_heads != 1:
            raise ValueError("Kernels assume num_key_value_heads == 1")


def load_model(model_dir: str | Path) -> tuple[GemmaConfig, dict[str, np.ndarray]]:
    model_dir = Path(model_dir)
    config = GemmaConfig(model_dir / "config.json")
    weights = load_safetensors(model_dir / "model.safetensors")
    return config, weights
