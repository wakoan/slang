"""Standalone numpy SRQ reference for the gemma4_150 fast-path port.

Replicates EXACTLY what the webml fused kernels compute: the Gemma 4 E2B QAT
forward with Static Range Quantization (SRQ) of activations at every quantized
linear. Uniform rule per quantized linear:

    y = srq(dequant_weight @ srq(x, in_scale), out_scale)

(srq is a no-op when the scale is 0). Norms / residual adds / gelu / attention
softmax run in f32 on the srq'd values, exactly as the kernels do. This is the
correctness oracle for the integrated GPU runner AND the end-to-end proof that
the SRQ recipe stays coherent (QAT-trained checkpoint -> should answer well).

Structure mirrors gemma4/reference.py (verified correct weight-only) plus SRQ.
Slow (streams + dequantizes weights per token) — a test oracle, not a runner.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from gemma4_150.loader import Ckpt, LM

EPS = 1e-6


def srq(x, s):
    return np.clip(np.round(x / s), -128.0, 127.0) * s if s != 0.0 else x


def gelu_tanh(v):
    return 0.5 * v * (1.0 + np.tanh(0.7978845608028654 * (v + 0.044715 * v ** 3)))


def rms_norm(x, w, eps=EPS):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rms_norm_noscale(x, eps=EPS):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def rope(x, pos, theta, cutoff):
    """split-half RoPE on [n_heads, head_dim]; pairs >= cutoff are identity."""
    n, hd = x.shape
    half = hd // 2
    inv = 1.0 / theta ** (2.0 * np.arange(half, dtype=np.float32) / hd)
    inv[cutoff:] = 0.0
    ang = np.float32(pos) * inv
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    a, b = x[:, :half], x[:, half:]
    return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)


class Layer:
    def __init__(self, i, sliding, head_dim, q_dim, theta, cutoff, kv_source):
        self.i, self.sliding, self.head_dim, self.q_dim = i, sliding, head_dim, q_dim
        self.theta, self.cutoff, self.kv_source = theta, cutoff, kv_source
        self.kv_shared = kv_source != i


class ReferenceSRQ:
    def __init__(self, model_dir=None):
        self.c = Ckpt() if model_dir is None else Ckpt(model_dir)
        cfg = json.loads((Path(self.c.path).parent / "config.json").read_text())["text_config"]
        self.H = cfg["hidden_size"]
        self.nL = cfg["num_hidden_layers"]
        self.nH = cfg["num_attention_heads"]
        self.vocab = cfg["vocab_size"]
        self.window = cfg["sliding_window"]
        self.softcap = cfg["final_logit_softcapping"]
        self.ple_d = cfg["hidden_size_per_layer_input"]
        self.embed_scale = math.sqrt(self.H)
        self.ple_scale = math.sqrt(self.ple_d)
        rope_p, types = cfg["rope_parameters"], cfg["layer_types"]
        first_shared = self.nL - cfg["num_kv_shared_layers"]
        self.layers = []
        for L, kind in enumerate(types):
            sliding = kind == "sliding_attention"
            hd = self.c.header[f"{LM}layers.{L}.self_attn.q_norm.weight"]["shape"][0]
            qd = self.c.header[f"{LM}layers.{L}.self_attn.q_proj.weight"]["shape"][0]
            rf = rope_p[kind].get("partial_rotary_factor", 1.0)
            src = L if L < first_shared else max(s for s in range(first_shared) if types[s] == kind)
            self.layers.append(Layer(L, sliding, hd, qd, rope_p[kind]["rope_theta"],
                                     int(rf * hd / 2), src))
        self.kc = [None if l.kv_shared else np.zeros((0, l.head_dim), np.float32) for l in self.layers]
        self.vc = [None if l.kv_shared else np.zeros((0, l.head_dim), np.float32) for l in self.layers]

    # --- weight access ---
    def _bf16(self, name):
        return self.c.f32(name)

    def _qlin(self, x, L, mod):
        lin = self.c.linear(L, mod)
        xq = srq(x, lin["in_scale"])
        zp = 1 << (lin["bits"] - 1)
        W = (self.c.codes(lin).astype(np.float32) - zp) * lin["wscale"][:, None]
        return srq(W @ xq, lin["out_scale"]), lin

    def _ilin_raw(self, x, L, mod):
        """int8 PLE linear -> (srq'd output, out_scale) ; W = i8 * row_scale."""
        p = f"{LM}layers.{L}.{mod}."
        wi8 = self.c._raw(p + "weight").astype(np.float32)
        ws = self.c.f32(p + "weight_scale").reshape(-1)
        in_s = float(self.c.f32(p + "input_activation_scale"))
        out_s = float(self.c.f32(p + "output_activation_scale"))
        y = (wi8 * ws[:, None]) @ srq(x, in_s)
        return srq(y, out_s), out_s

    # --- embeddings ---
    def embed(self, tid):
        packed = self.c._raw(LM + "embed_tokens.embedding_quantized")[tid]
        u = np.frombuffer(np.ascontiguousarray(packed).tobytes(), np.uint32)  # [96]
        codes = np.zeros(self.H, np.int32)
        for v in range(16):
            codes[v::16] = (u >> (2 * v)) & 3
        sc = float(self.c.f32(LM + "embed_tokens.embedding_scale").reshape(-1)[tid])
        return (self.embed_scale * sc * (codes - 2)).astype(np.float32)

    def ple_table(self, tid):
        packed = self.c._raw(LM + "embed_tokens_per_layer.embedding_quantized")[tid]
        u = np.frombuffer(np.ascontiguousarray(packed).tobytes(), np.uint32)  # [1120]
        codes = np.zeros(self.nL * self.ple_d, np.int32)
        for v in range(8):
            codes[v::8] = (u >> (4 * v)) & 15
        sc = self.c.f32(LM + "embed_tokens_per_layer.embedding_scale")[tid]  # [35]
        g = np.repeat(sc, self.ple_d)
        return (self.ple_scale * g * (codes - 8)).astype(np.float32).reshape(self.nL, self.ple_d)

    def ple_input(self, tid, x):
        ple = self.ple_table(tid)                                    # [35,256]
        proj = self._bf16(LM + "per_layer_model_projection.weight") @ x   # dense bf16, no srq
        ctx = proj.reshape(self.nL, self.ple_d) * np.float32(self.H ** -0.5)
        ctx = rms_norm(ctx, self._bf16(LM + "per_layer_projection_norm.weight"))
        return (ctx + ple) * np.float32(2.0 ** -0.5)

    # --- one decoder layer ---
    def layer(self, x, L, pos, ple_in):
        spec = self.layers[L]
        p = f"{LM}layers.{L}."
        nH, hd = self.nH, spec.head_dim

        # attention (sandwich norm)
        res = x
        h = rms_norm(x, self._bf16(p + "input_layernorm.weight"))
        q = self._qlin(h, L, "self_attn.q_proj")[0].reshape(nH, hd)
        q = rms_norm(q, self._bf16(p + "self_attn.q_norm.weight"))
        q = rope(q, pos, spec.theta, spec.cutoff)
        if not spec.kv_shared:
            k = self._qlin(h, L, "self_attn.k_proj")[0].reshape(1, hd)
            v = self._qlin(h, L, "self_attn.v_proj")[0].reshape(1, hd)
            k = rope(rms_norm(k, self._bf16(p + "self_attn.k_norm.weight")), pos, spec.theta, spec.cutoff)
            v = rms_norm_noscale(v)
            self.kc[L] = np.concatenate([self.kc[L], k], 0)
            self.vc[L] = np.concatenate([self.vc[L], v], 0)
        keys, vals = self.kc[spec.kv_source], self.vc[spec.kv_source]
        kv_len = pos + 1
        start = max(0, kv_len - self.window) if spec.sliding else 0
        scores = q @ keys[:kv_len].T                                 # scaling 1.0
        if start > 0:
            scores[:, :start] = -1e9
        scores = scores - scores.max(-1, keepdims=True)
        probs = np.exp(scores); probs /= probs.sum(-1, keepdims=True)
        attn = (probs @ vals[:kv_len]).reshape(-1)                   # [q_dim]
        attn = srq(attn, self.c.linear(L, "self_attn.o_proj")["in_scale"])   # kernel 101 OUT_Q
        o = self._qlin(attn, L, "self_attn.o_proj")[0]
        x = res + rms_norm(o, self._bf16(p + "post_attention_layernorm.weight"))

        # MLP (sandwich norm). gate/up/down activations are f16 in the kernels
        # (kernel 73 emits y2 = f16(srq(f16(pre_ffn_norm), gate_in)); 74/75 read
        # vec4<f16>), so the oracle must round through f16 too.
        res = x
        h = rms_norm(x, self._bf16(p + "pre_feedforward_layernorm.weight"))
        y2 = srq(h.astype(np.float16).astype(np.float32),
                 self.c.linear(L, "mlp.gate_proj")["in_scale"]).astype(np.float16).astype(np.float32)
        gate = self._qlin(y2, L, "mlp.gate_proj")[0]
        up = self._qlin(y2, L, "mlp.up_proj")[0]
        geglu = srq(gelu_tanh(gate) * up, self.c.linear(L, "mlp.down_proj")["in_scale"])
        down = self._qlin(geglu, L, "mlp.down_proj")[0]
        x = res + rms_norm(down, self._bf16(p + "post_feedforward_layernorm.weight"))

        # per-layer embedding block
        res = x
        gate_out, g_lin = self._ilin_raw(x, L, "per_layer_input_gate")
        g = gelu_tanh(gate_out) * ple_in
        proj = self._ilin_raw(g, L, "per_layer_projection")[0]
        h = rms_norm(proj, self._bf16(p + "post_per_layer_input_norm.weight"))
        return (res + h) * float(self._bf16(p + "layer_scalar")[0])

    def forward(self, tid, pos):
        x = self.embed(tid)
        ple_in = self.ple_input(tid, x)
        for L in range(self.nL):
            x = self.layer(x, L, pos, ple_in[L])
        x = rms_norm(x, self._bf16(LM + "norm.weight"))
        # separate lm_head (2-bit), input_activation_scale 0 (weight-only, no srq)
        packed = self.c._raw("lm_head.weight")
        u = np.frombuffer(np.ascontiguousarray(packed).tobytes(), np.uint32).reshape(self.vocab, 96)
        codes = np.zeros((self.vocab, self.H), np.int32)
        for v in range(16):
            codes[:, v::16] = (u >> (2 * v)) & 3
        ws = self.c.f32("lm_head.weight_scale").reshape(-1)
        logits = ((codes.astype(np.float32) - 2) * ws[:, None]) @ x
        return self.softcap * np.tanh(logits / self.softcap)
