"""Weight export for the gemma4_150 GPU runner.

Packs every weight into the exact layout its validated reference kernel expects,
into one weights.bin + manifest.json. All quantized weights are the checkpoint's
packed bytes reinterpreted as u32 row-major (verified == kernel nibble layout in
stages 1-10); block-major logits are reshape+transpose; int8 PLE codes are
(i8+128) as u8; the dense per-layer projection is cast to f16. Scalar activation
scales and per-layer structure live in the manifest.

    python -m gemma4_150.export            # writes models/gemma-4-E2B-qat/g4_150/
    python -m gemma4_150.export --verify   # + byte-check a sample vs the loader

The runner (next stage) reads the manifest, slices weights.bin, and drives the
fused per-layer dispatch chain; correctness gate is argmax==reference.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from gemma4_150.loader import Ckpt, LM

GELU_C = 0.7978845608028654


def gelu_lut(out_scale: float) -> np.ndarray:
    g = (np.arange(256) - 128).astype(np.float64) * out_scale
    return (0.5 * g * (1.0 + np.tanh(GELU_C * (g + 0.044715 * g ** 3)))).astype(np.float32)


class Writer:
    """Appends tensors to weights.bin (4-byte aligned) and records offsets."""

    def __init__(self, path: Path):
        self.f = open(path, "wb")
        self.pos = 0
        self.tensors: dict[str, dict] = {}

    def put(self, name: str, arr: np.ndarray, dtype: str) -> None:
        assert name not in self.tensors, name
        b = np.ascontiguousarray(arr).tobytes()
        self.f.write(b)
        self.tensors[name] = {"off": self.pos, "len": len(b), "dtype": dtype,
                              "shape": list(arr.shape)}
        self.pos += len(b)
        pad = (-len(b)) % 4
        if pad:
            self.f.write(b"\x00" * pad)
            self.pos += pad

    def close(self):
        self.f.close()


def u32(packed_u8: np.ndarray) -> np.ndarray:
    """Checkpoint packed u8 rows -> u32 row-major (direct reinterpret)."""
    return np.frombuffer(np.ascontiguousarray(packed_u8).tobytes(), np.uint32)


def export(model_dir=None, verify=False):
    c = Ckpt() if model_dir is None else Ckpt(model_dir)
    cfg = json.loads((Path(c.path).parent / "config.json").read_text())["text_config"]
    H = cfg["hidden_size"]
    nL = cfg["num_hidden_layers"]
    types = cfg["layer_types"]
    first_shared = nL - cfg["num_kv_shared_layers"]
    rope_p = cfg["rope_parameters"]

    out_dir = Path(c.path).parent / "g4_150"
    out_dir.mkdir(exist_ok=True)
    w = Writer(out_dir / "weights.bin")

    def f32(n):
        return c.f32(n).astype(np.float32)

    def qlin(L, mod):
        lin = c.linear(L, mod)
        return u32(lin["packed"]), lin["wscale"].astype(np.float32), lin

    # ---- global ----
    w.put("embed_q", c._raw(LM + "embed_tokens.embedding_quantized"), "u8")
    w.put("embed_scale", f32(LM + "embed_tokens.embedding_scale").reshape(-1), "f32")
    w.put("ple_q", c._raw(LM + "embed_tokens_per_layer.embedding_quantized"), "u8")
    w.put("ple_scale", f32(LM + "embed_tokens_per_layer.embedding_scale").reshape(-1), "f32")

    lh = c._raw("lm_head.weight")                       # u8 [vocab,384]
    lh32 = np.frombuffer(np.ascontiguousarray(lh).tobytes(), np.uint32).reshape(-1, 24, 4)
    w.put("lmhead_blk", lh32.transpose(1, 0, 2).copy(), "u32")   # [24,vocab,4]
    w.put("lmhead_scale", f32("lm_head.weight_scale").reshape(-1), "f32")

    w.put("pl_model_proj", f32(LM + "per_layer_model_projection.weight").astype(np.float16), "f16")
    w.put("final_norm", f32(LM + "norm.weight"), "f32")
    w.put("pl_proj_norm", f32(LM + "per_layer_projection_norm.weight"), "f32")

    # ---- per-layer ----
    layers = []
    for L in range(nL):
        p = f"{LM}layers.{L}."
        kind = types[L]
        shared = L >= first_shared
        hd = c.header[p + "self_attn.q_norm.weight"]["shape"][0]
        qd = c.header[p + "self_attn.q_proj.weight"]["shape"][0]
        inter = c.header[p + "mlp.gate_proj.weight"]["shape"][0]
        rf = rope_p[kind].get("partial_rotary_factor", 1.0)
        src = L if not shared else max(s for s in range(first_shared) if types[s] == kind)
        nxt = f"{LM}layers.{L + 1}.input_layernorm.weight" if L + 1 < nL else LM + "norm.weight"

        # norms
        w.put(f"L{L}.in_norm", f32(p + "input_layernorm.weight"), "f32")
        w.put(f"L{L}.q_norm", f32(p + "self_attn.q_norm.weight"), "f32")
        w.put(f"L{L}.down_nw", f32(p + "post_feedforward_layernorm.weight"), "f32")
        w.put(f"L{L}.o_w12", np.concatenate([
            f32(p + "post_attention_layernorm.weight"),
            f32(p + "pre_feedforward_layernorm.weight")]), "f32")
        # kernel 77: [post_ple_norm | next-layer in_norm | layer_scalar]
        w.put(f"L{L}.pleproj_w12s", np.concatenate([
            f32(p + "post_per_layer_input_norm.weight"), f32(nxt),
            f32(p + "layer_scalar").reshape(-1)]), "f32")
        if not shared:
            w.put(f"L{L}.k_norm", f32(p + "self_attn.k_norm.weight"), "f32")

        # attention weights
        qb, qs, ql = qlin(L, "self_attn.q_proj")
        w.put(f"L{L}.q_bits", qb, "u32")
        ob, os_, ol = qlin(L, "self_attn.o_proj")
        w.put(f"L{L}.o_bits", ob, "u32")
        w.put(f"L{L}.o_scale", os_, "f32")
        scales = {"qkv_in": ql["in_scale"], "q_out": ql["out_scale"],
                  "o_in": ol["in_scale"], "o_out": ol["out_scale"]}
        if not shared:
            kb, ks, kl = qlin(L, "self_attn.k_proj")
            vb, vs, vl = qlin(L, "self_attn.v_proj")
            w.put(f"L{L}.k_bits", kb, "u32")
            w.put(f"L{L}.v_bits", vb, "u32")
            w.put(f"L{L}.qkv_scales", np.concatenate([qs, ks, vs]), "f32")
            scales.update(k_out=kl["out_scale"], v_out=vl["out_scale"])
        else:
            w.put(f"L{L}.q_scale", qs, "f32")

        # MLP weights
        for m, tag in [("mlp.gate_proj", "gate"), ("mlp.up_proj", "up"), ("mlp.down_proj", "down")]:
            b, s, l = qlin(L, m)
            w.put(f"L{L}.{tag}_bits", b, "u32")
            w.put(f"L{L}.{tag}_scale", s, "f32")
            scales[f"{tag}_in"], scales[f"{tag}_out"] = l["in_scale"], l["out_scale"]
        w.put(f"L{L}.gelu_gate", gelu_lut(scales["gate_out"]), "f32")

        # PLE int8 gate + projection
        for m, tag, nout in [("per_layer_input_gate", "plegate", 256),
                             ("per_layer_projection", "pleproj", H)]:
            wi8 = c._raw(p + m + ".weight")
            biased = (wi8.astype(np.int16) + 128).astype(np.uint8)
            w.put(f"L{L}.{tag}_codes", u32(biased), "u32")
            w.put(f"L{L}.{tag}_rowscale", f32(p + m + ".weight_scale").reshape(-1), "f32")
            scales[f"{tag}_in"] = float(c.f32(p + m + ".input_activation_scale"))
            scales[f"{tag}_out"] = float(c.f32(p + m + ".output_activation_scale"))
        w.put(f"L{L}.gelu_plegate", gelu_lut(scales["plegate_out"]), "f32")

        layers.append({
            "index": L, "sliding": kind == "sliding_attention", "head_dim": hd,
            "q_dim": qd, "intermediate": inter, "shared": shared, "kv_source": src,
            "rope_theta": rope_p[kind]["rope_theta"], "rope_cutoff": int(rf * hd / 2),
            "scales": {k: float(v) for k, v in scales.items()},
        })

    w.close()
    manifest = {
        "config": {"H": H, "nL": nL, "nH": cfg["num_attention_heads"],
                   "vocab": cfg["vocab_size"], "window": cfg["sliding_window"],
                   "softcap": cfg["final_logit_softcapping"],
                   "ple_d": cfg["hidden_size_per_layer_input"],
                   "embed_scale": math.sqrt(H),
                   "ple_scale": math.sqrt(cfg["hidden_size_per_layer_input"]),
                   "bos": cfg["bos_token_id"], "eos": cfg["eos_token_id"]},
        "layers": layers, "tensors": w.tensors,
        "bytes": w.pos,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    print(f"exported {len(w.tensors)} tensors, {w.pos/1e9:.2f}GB -> {out_dir}")

    if verify:
        _verify(c, out_dir, manifest)
    return out_dir


def _verify(c, out_dir, manifest):
    """Byte-check a sample of exported tensors against a fresh loader derivation."""
    blob = np.memmap(out_dir / "weights.bin", np.uint8, "r")

    def get(name):
        t = manifest["tensors"][name]
        return np.asarray(blob[t["off"]:t["off"] + t["len"]])

    checks = []
    # global embed
    checks.append(("embed_q", get("embed_q"),
                   np.ascontiguousarray(c._raw(LM + "embed_tokens.embedding_quantized")).view(np.uint8).reshape(-1)))
    # L3 o_proj bits (validated shape in stage 4)
    o = c.linear(3, "self_attn.o_proj")
    checks.append(("L3.o_bits", get("L3.o_bits").view(np.uint32),
                   u32(o["packed"])))
    # L3 ple gate biased codes (stage 9)
    g = c._raw(LM + "layers.3.per_layer_input_gate.weight")
    biased = (g.astype(np.int16) + 128).astype(np.uint8)
    checks.append(("L3.plegate_codes", get("L3.plegate_codes").view(np.uint32),
                   u32(biased)))
    # lm_head block-major col 0 (stage 6 layout)
    lh = np.frombuffer(np.ascontiguousarray(c._raw("lm_head.weight")).tobytes(), np.uint32).reshape(-1, 24, 4)
    checks.append(("lmhead_blk", get("lmhead_blk").view(np.uint32),
                   lh.transpose(1, 0, 2).reshape(-1)))
    ok = True
    for name, a, b in checks:
        match = a.size == b.size and bool((a == b.reshape(-1)).all())
        ok &= match
        print(f"  verify {name:20s} {'OK' if match else 'MISMATCH'} ({a.size} elems)")
    print("VERIFY", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    export(verify=ap.parse_args().verify)
