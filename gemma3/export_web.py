"""Export Gemma weights for the browser: weights.bin + manifest.json.

Packs tensors in the exact GPU layout the runners use, so the browser
slices one ArrayBuffer into GPU buffers with zero format logic:
- matmul weights: QKV / gate-up concatenated, f16 pairs packed in u32
- norm weights: f32
- embedding table: f16-packed u32 (shared by embed lookup and logits)

Usage:
    python -m gemma3.export_web [model_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .loader import load_model

_ALIGN = 256


def _pack_f16_u32(arr: np.ndarray) -> np.ndarray:
    return np.frombuffer(arr.astype(np.float16).tobytes(), dtype=np.uint32)


def export_web(model_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    model_dir = Path(model_dir)
    out_dir = Path(out_dir) if out_dir else model_dir / "web"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, w = load_model(model_dir)

    tensors: list[tuple[str, np.ndarray, str]] = []  # (name, data, dtype tag)
    tensors.append(("embed", _pack_f16_u32(w["model.embed_tokens.weight"]), "u32"))
    tensors.append(("final_norm", w["model.norm.weight"].astype(np.float32), "f32"))
    for L in range(cfg.num_layers):
        a = f"model.layers.{L}.self_attn."
        m = f"model.layers.{L}.mlp."
        p = f"model.layers.{L}."
        qkv = np.concatenate([w[a + "q_proj.weight"], w[a + "k_proj.weight"],
                              w[a + "v_proj.weight"]])
        gateup = np.concatenate([w[m + "gate_proj.weight"], w[m + "up_proj.weight"]])
        tensors += [
            (f"L{L}.qkv", _pack_f16_u32(qkv), "u32"),
            (f"L{L}.o", _pack_f16_u32(w[a + "o_proj.weight"]), "u32"),
            (f"L{L}.gateup", _pack_f16_u32(gateup), "u32"),
            (f"L{L}.down", _pack_f16_u32(w[m + "down_proj.weight"]), "u32"),
            (f"L{L}.q_norm", w[a + "q_norm.weight"].astype(np.float32), "f32"),
            (f"L{L}.k_norm", w[a + "k_norm.weight"].astype(np.float32), "f32"),
            (f"L{L}.norm_in", w[p + "input_layernorm.weight"].astype(np.float32), "f32"),
            (f"L{L}.norm_pa", w[p + "post_attention_layernorm.weight"].astype(np.float32), "f32"),
            (f"L{L}.norm_pf", w[p + "pre_feedforward_layernorm.weight"].astype(np.float32), "f32"),
            (f"L{L}.norm_pff", w[p + "post_feedforward_layernorm.weight"].astype(np.float32), "f32"),
        ]

    manifest_tensors = []
    offset = 0
    blob_path = out_dir / "weights.bin"
    with open(blob_path, "wb") as f:
        for name, data, dtype in tensors:
            pad = (-offset) % _ALIGN
            if pad:
                f.write(b"\x00" * pad)
                offset += pad
            raw = data.tobytes()
            manifest_tensors.append({
                "name": name,
                "offset": offset,
                "byteLength": len(raw),
                "dtype": dtype,
            })
            f.write(raw)
            offset += len(raw)

    manifest = {
        "totalBytes": offset,
        "tensors": manifest_tensors,
        "config": {
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "head_dim": cfg.head_dim,
            "intermediate_size": cfg.intermediate_size,
            "vocab_size": cfg.vocab_size,
            "sliding_window": cfg.sliding_window,
            "layer_types": cfg.layer_types,
            "rope_theta_local": cfg.rope_theta_local,
            "rope_theta_global": cfg.rope_theta_global,
            "bos_token_id": cfg.bos_token_id,
            "eos_token_ids": list(cfg.eos_token_ids),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    print(f"wrote {blob_path} ({offset / 1e6:.0f} MB) and manifest.json")
    return out_dir


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "models" / "gemma-3-270m-it"
    export_web(model_dir)
