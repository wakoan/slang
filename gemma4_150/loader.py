"""Standalone QAT checkpoint loader for the gemma4_150 fast-path port.

Reads the google/gemma-4-E2B-it-qat-mobile-transformers safetensors directly
(no project dependencies). Exposes, per language-model linear:
  - packed sub-byte weight codes (uint8, low-bits-first),
  - per-row weight_scale (f32),
  - scalar input_activation_scale / output_activation_scale (the SRQ scales),
and the raw bf16 norm weights. Matches the layouts the webml reference kernels
expect. This is the data foundation for the SRQ + fused-kernel 150 tok/s port.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B-qat"
LM = "model.language_model."

def _bits(layer: int, module: str) -> int:
    """Weight bit-width. Attention 4-bit; PLE 8-bit; MLP 4-bit on L0-14,
    2-bit on the double-wide L15-34 (matches the QAT checkpoint)."""
    if "self_attn" in module:
        return 4
    if module.startswith("per_layer"):
        return 8
    if "mlp" in module:
        return 4 if layer < 15 else 2
    raise KeyError(module)


class Ckpt:
    def __init__(self, model_dir: str | Path = MODEL) -> None:
        self.path = Path(model_dir) / "model.safetensors"
        with open(self.path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(hlen))
            self.data_start = 8 + hlen
        self.cfg = json.loads((Path(model_dir) / "config.json").read_text())

    def _raw(self, name: str) -> np.ndarray:
        info = self.header[name]
        a, b = info["data_offsets"]
        dt = info["dtype"]
        npdt = {"F32": np.float32, "U8": np.uint8, "I8": np.int8,
                "BF16": np.uint16, "F16": np.float16}[dt]
        with open(self.path, "rb") as f:
            f.seek(self.data_start + a)
            buf = f.read(b - a)
        return np.frombuffer(buf, npdt).reshape(info["shape"])

    def f32(self, name: str) -> np.ndarray:
        info = self.header[name]
        if info["dtype"] == "BF16":
            u16 = self._raw(name).astype(np.uint32)
            return (u16 << 16).view(np.float32).reshape(info["shape"])
        return self._raw(name).astype(np.float32)

    def linear(self, layer: int, module: str) -> dict:
        """module e.g. 'self_attn.k_proj' or 'mlp.gate_proj'."""
        p = f"{LM}layers.{layer}.{module}."
        bits = _bits(layer, module)
        packed = self._raw(p + "weight")          # uint8 [n_out, n_in*bits/8]
        wscale = self.f32(p + "weight_scale").reshape(-1)          # [n_out]
        in_s = float(self.f32(p + "input_activation_scale"))       # scalar
        out_s = float(self.f32(p + "output_activation_scale"))     # scalar
        n_out = packed.shape[0]
        n_in = packed.shape[1] * 8 // bits
        return {"packed": packed, "wscale": wscale, "in_scale": in_s,
                "out_scale": out_s, "bits": bits, "n_out": n_out, "n_in": n_in}

    def codes(self, lin: dict) -> np.ndarray:
        """Unpack packed weight to unsigned integer codes [n_out, n_in]
        (low-bits-first). Actual weight = (code - ZP), ZP = 2^(bits-1)."""
        packed, bits = lin["packed"], lin["bits"]
        n_out, n_in = lin["n_out"], lin["n_in"]
        per = 8 // bits
        mask = (1 << bits) - 1
        flat = packed.reshape(n_out, -1)
        out = np.zeros((n_out, n_in), np.int32)
        for j in range(per):
            out[:, j::per] = (flat >> (bits * j)) & mask
        return out


if __name__ == "__main__":
    c = Ckpt()
    for mod in ["self_attn.k_proj", "mlp.gate_proj", "mlp.down_proj"]:
        L = c.linear(4, mod)
        print(f"L4 {mod}: bits={L['bits']} n_out={L['n_out']} n_in={L['n_in']} "
              f"in_s={L['in_scale']:.5g} out_s={L['out_scale']:.5g} "
              f"wscale[:3]={L['wscale'][:3].round(5)}")
