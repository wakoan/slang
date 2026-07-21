"""Export Gemma 4 E2B QAT weights for the browser: weights.bin + manifest.json.

Packs the int2/4/8 QAT tensors in the exact GPU layout Gemma4QATGPU uploads, so
the browser slices one ArrayBuffer into GPU buffers with no format logic:
  - dq2/dq4 linears: packed sub-byte weights (u32-viewed) + per-row f32 scale
  - 8-bit / unquantized linears: dequant to f16, packed as u32 (read via
    unpack2x16float — no shader-f16 feature needed)
  - 2-bit embed table (tied logits) and 4-bit PLE table: raw packed bytes
  - norms: (w - 1) f32 (runner uploads w-1 to reuse the (1+w) kernels)
  - per-layer learned scalars: f32

The manifest carries per-linear metadata (kind / n_out / n_in) and the full
per-layer spec so app.js can pick the right kernel + grid and replicate the
KV-share aliasing. Mirrors gemma4/qat_runner.py::_load_weights.

Usage:
    python -m gemma4.export_qat_gendemo [model_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .loader import PREFIX, Gemma4Config
from .qat_loader import load_qat

_ALIGN = 256


def export_qat_gendemo(model_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    model_dir = Path(model_dir)
    out_dir = Path(out_dir) if out_dir else model_dir / "gendemo"
    out_dir.mkdir(parents=True, exist_ok=True)

    qat = load_qat(model_dir)
    cfg = Gemma4Config(Path(model_dir) / "config.json", qat.idx)

    tensors: list[tuple[str, bytes]] = []

    def add(name: str, data: bytes) -> str:
        tensors.append((name, data))
        return name

    def norm_w(full_name: str) -> bytes:
        return (qat.idx.tensor(full_name).astype(np.float32) - 1.0).tobytes()

    linears: dict[str, dict] = {}

    def add_linear(key: str, module: str) -> None:
        """Mirror qat_runner._load_linear, emitting tensors + manifest meta."""
        packed, scale, bits = qat.packed_weight(module)
        n_out = int(packed.shape[0])
        if bits in (2, 4):
            per = 8 // bits
            n_in = int(packed.shape[1] * per)
            wname = add(key + ".w", np.ascontiguousarray(packed).tobytes())
            sname = add(key + ".s",
                        np.ascontiguousarray(scale.ravel(), np.float32).tobytes())
            linears[key] = {"kind": "dq4" if bits == 4 else "dq2",
                            "n_out": n_out, "n_in": n_in, "w": wname, "scale": sname}
        else:  # 8-bit -> f16 packed u32 (base vec4 matvec)
            w32 = qat.dequant_weight(module)
            n_in = int(w32.shape[1])
            wname = add(key + ".w", w32.astype(np.float16).tobytes())
            linears[key] = {"kind": "f16", "n_out": n_out, "n_in": n_in,
                            "w": wname, "scale": None}

    # --- global tensors ---
    add("norm", norm_w(PREFIX + "norm.weight"))
    add("ple_proj_norm", norm_w(PREFIX + "per_layer_projection_norm.weight"))
    pmp = qat.idx.tensor(PREFIX + "per_layer_model_projection.weight").astype(np.float32)
    pmp = pmp * np.float32(cfg.hidden_size ** -0.5)  # fold ctx scale in
    add("ple_model_proj", pmp.astype(np.float16).tobytes())
    ple_model_proj_nout = int(pmp.shape[0])

    add("embed", np.ascontiguousarray(
        qat.idx.raw(PREFIX + "embed_tokens.embedding_quantized")).tobytes())
    add("embed_scale", np.ascontiguousarray(
        qat.idx.tensor(PREFIX + "embed_tokens.embedding_scale").ravel(), np.float32).tobytes())
    add("ple_table", np.ascontiguousarray(
        qat.idx.raw(PREFIX + "embed_tokens_per_layer.embedding_quantized")).tobytes())
    add("ple_table_scale", np.ascontiguousarray(
        qat.idx.tensor(PREFIX + "embed_tokens_per_layer.embedding_scale"), np.float32).tobytes())

    # --- per-layer ---
    layer_specs = []
    for spec in cfg.layers:
        p = f"L{spec.index}."
        full = f"{PREFIX}layers.{spec.index}."
        for suffix, tag in (
            ("input_layernorm.weight", "norm_in"),
            ("post_attention_layernorm.weight", "norm_pa"),
            ("pre_feedforward_layernorm.weight", "norm_pf"),
            ("post_feedforward_layernorm.weight", "norm_pff"),
            ("self_attn.q_norm.weight", "q_norm"),
            ("post_per_layer_input_norm.weight", "ple_norm"),
        ):
            add(p + tag, norm_w(full + suffix))
        add(p + "layer_scalar",
            qat.idx.tensor(full + "layer_scalar").astype(np.float32).tobytes())

        m = f"language_model.layers.{spec.index}."
        add_linear(p + "q", m + "self_attn.q_proj")
        add_linear(p + "o", m + "self_attn.o_proj")
        add_linear(p + "gate", m + "mlp.gate_proj")
        add_linear(p + "up", m + "mlp.up_proj")
        add_linear(p + "down", m + "mlp.down_proj")
        add_linear(p + "ple_gate", m + "per_layer_input_gate")
        add_linear(p + "ple_proj", m + "per_layer_projection")
        if not spec.kv_shared:
            add(p + "k_norm", norm_w(full + "self_attn.k_norm.weight"))
            add_linear(p + "k", m + "self_attn.k_proj")
            add_linear(p + "v", m + "self_attn.v_proj")

        layer_specs.append({
            "index": spec.index, "sliding": spec.sliding, "head_dim": spec.head_dim,
            "q_dim": spec.q_dim, "intermediate": spec.intermediate,
            "rope_theta": spec.rope_theta, "rope_cutoff": spec.rope_cutoff,
            "kv_source": spec.kv_source, "kv_shared": spec.kv_shared,
        })

    # --- write blob ---
    manifest_tensors = []
    offset = 0
    blob_path = out_dir / "weights.bin"
    with open(blob_path, "wb") as f:
        for name, raw in tensors:
            pad = (-offset) % _ALIGN
            if pad:
                f.write(b"\x00" * pad)
                offset += pad
            manifest_tensors.append({"name": name, "offset": offset,
                                     "byteLength": len(raw)})
            f.write(raw)
            offset += len(raw)

    manifest = {
        "totalBytes": offset,
        "tensors": manifest_tensors,
        "linears": linears,
        "layers": layer_specs,
        "config": {
            "hidden_size": cfg.hidden_size,
            "num_heads": cfg.num_heads,
            "num_layers": cfg.num_layers,
            "ple_hidden": cfg.hidden_size_per_layer_input,
            "vocab_size": cfg.vocab_size,
            "sliding_window": cfg.sliding_window,
            "softcap": cfg.final_logit_softcapping,
            "embed_scale": cfg.embed_scale,
            "ple_scale": cfg.ple_scale,
            "ple_model_proj_nout": ple_model_proj_nout,
            "eos_token_ids": list(cfg.eos_token_ids),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    print(f"wrote {blob_path} ({offset / 1e6:.0f} MB), "
          f"{len(tensors)} tensors, manifest.json")
    return out_dir


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B-qat"
    export_qat_gendemo(md)
