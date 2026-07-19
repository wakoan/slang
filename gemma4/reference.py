"""Pure-numpy Gemma 4 E2B reference decoder, streaming weights from the
lazy safetensors index.

Mirrors HF Gemma4ForCausalLM in float32, one token position at a time with a
KV cache — the exact computation the GPU kernels implement, for verification.
Weights are converted bf16→f32 at use time and dropped after (a full-f32
weight dict would need ~20GB); expect seconds per token — this is a test
oracle, not a runner.
"""

from __future__ import annotations

import numpy as np

from gemma3.reference import gelu_tanh
from .loader import PREFIX, Gemma4Config, SafetensorsIndex, bf16_to_f32


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Gemma 4 RMSNorm: x/rms * w — NOT Gemma 3's x/rms * (1 + w).

    Gemma4RMSNorm stores the scale directly (init torch.ones); Gemma 3 stored
    w - 1 and applied (1 + w). Feeding Gemma 4 weights through the Gemma 3
    convention mis-scales every norm and produces garbage output.
    """
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * weight


def rms_norm_noscale(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Parameter-free RMSNorm (Gemma 4 v_norm, with_scale=False): x / rms(x)."""
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps)


def apply_rope_partial(x: np.ndarray, position: int, theta: float,
                       cutoff: int) -> np.ndarray:
    """rotate_half RoPE on [n_heads, head_dim]; pairs >= cutoff are identity.

    Gemma 4 p-RoPE: full-attention layers rotate only the first
    `partial_rotary_factor * head_dim / 2` frequency pairs (inv_freq = 0
    beyond). cutoff == head_dim//2 is a standard full rotation.
    """
    n_heads, head_dim = x.shape
    half = head_dim // 2
    i = np.arange(half, dtype=np.float32)
    inv_freq = 1.0 / theta ** (2.0 * i / head_dim)
    inv_freq[cutoff:] = 0.0
    angle = np.float32(position) * inv_freq
    c, s = np.cos(angle, dtype=np.float32), np.sin(angle, dtype=np.float32)
    a, b = x[:, :half], x[:, half:]
    return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)


def softcap(x: np.ndarray, cap: float) -> np.ndarray:
    return cap * np.tanh(x / cap)


class ReferenceGemma4:
    """Single-position forward pass with per-layer KV cache and KV sharing."""

    def __init__(self, config: Gemma4Config, index: SafetensorsIndex,
                 max_seq: int = 1024, ple: bool = True) -> None:
        self.cfg = config
        self.idx = index
        self.max_seq = max_seq
        self.ple = ple  # False isolates the non-PLE pipeline (Stage 4a gate)
        # caches only for layers that own K/V; shared layers alias by index
        self.k_cache: list[np.ndarray | None] = [
            None if s.kv_shared else np.zeros((max_seq, s.head_dim), np.float32)
            for s in config.layers
        ]
        self.v_cache: list[np.ndarray | None] = [
            None if s.kv_shared else np.zeros((max_seq, s.head_dim), np.float32)
            for s in config.layers
        ]

    def _w(self, name: str) -> np.ndarray:
        return self.idx.tensor(name)

    def _ple_input(self, token_id: int, x: np.ndarray) -> np.ndarray:
        """Per-layer input [num_layers, 256], computed once per token."""
        cfg = self.cfg
        n, d = cfg.num_layers, cfg.hidden_size_per_layer_input
        table = self.idx.raw(PREFIX + "embed_tokens_per_layer.weight")
        ple = bf16_to_f32(table[token_id]).reshape(n, d) * np.float32(cfg.ple_scale)
        ctx = self._w(PREFIX + "per_layer_model_projection.weight") @ x
        ctx = ctx.reshape(n, d) * np.float32(cfg.hidden_size ** -0.5)
        ctx = rms_norm(ctx, self._w(PREFIX + "per_layer_projection_norm.weight"),
                       cfg.rms_norm_eps)
        return (ctx + ple) * np.float32(2.0 ** -0.5)

    def _layer(self, x: np.ndarray, layer: int, pos: int,
               ple_in: np.ndarray | None) -> np.ndarray:
        cfg, w = self.cfg, self._w
        spec = cfg.layers[layer]
        p = f"{PREFIX}layers.{layer}."
        n_heads, hd = cfg.num_heads, spec.head_dim

        # --- attention block (sandwich norm) ---
        residual = x
        h = rms_norm(x, w(p + "input_layernorm.weight"), cfg.rms_norm_eps)

        q = (w(p + "self_attn.q_proj.weight") @ h).reshape(n_heads, hd)
        q = rms_norm(q, w(p + "self_attn.q_norm.weight"), cfg.rms_norm_eps)
        q = apply_rope_partial(q, pos, spec.rope_theta, spec.rope_cutoff)

        if not spec.kv_shared:
            k = (w(p + "self_attn.k_proj.weight") @ h).reshape(1, hd)
            v = (w(p + "self_attn.v_proj.weight") @ h).reshape(1, hd)
            k = rms_norm(k, w(p + "self_attn.k_norm.weight"), cfg.rms_norm_eps)
            k = apply_rope_partial(k, pos, spec.rope_theta, spec.rope_cutoff)
            v = rms_norm_noscale(v, cfg.rms_norm_eps)
            self.k_cache[layer][pos] = k[0]
            self.v_cache[layer][pos] = v[0]

        kv_len = pos + 1
        start = max(0, kv_len - cfg.sliding_window) if spec.sliding else 0

        keys = self.k_cache[spec.kv_source][:kv_len]   # [kv_len, hd]
        scores = q @ keys.T                            # scaling = 1.0
        if start > 0:
            scores[:, :start] = -1e9
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=-1, keepdims=True)

        vals = self.v_cache[spec.kv_source][:kv_len]
        attn = probs @ vals                            # [n_heads, hd]
        attn_out = w(p + "self_attn.o_proj.weight") @ attn.reshape(-1)

        h = rms_norm(attn_out, w(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps)
        x = residual + h

        # --- MLP block (sandwich norm) ---
        residual = x
        h = rms_norm(x, w(p + "pre_feedforward_layernorm.weight"), cfg.rms_norm_eps)
        gate = w(p + "mlp.gate_proj.weight") @ h
        up = w(p + "mlp.up_proj.weight") @ h
        mlp_out = w(p + "mlp.down_proj.weight") @ (gelu_tanh(gate) * up)
        h = rms_norm(mlp_out, w(p + "post_feedforward_layernorm.weight"), cfg.rms_norm_eps)
        x = residual + h

        if ple_in is None:
            return x

        # --- per-layer embedding block ---
        residual = x
        g = gelu_tanh(w(p + "per_layer_input_gate.weight") @ x) * ple_in
        h = rms_norm(w(p + "per_layer_projection.weight") @ g,
                     w(p + "post_per_layer_input_norm.weight"), cfg.rms_norm_eps)
        x = residual + h
        return x * w(p + "layer_scalar")[0]

    def forward(self, token_id: int, pos: int,
                collect_hidden: bool = False) -> np.ndarray | tuple:
        """Process one token at `pos`; returns logits (and per-layer hiddens)."""
        cfg = self.cfg
        embed_raw = self.idx.raw(PREFIX + "embed_tokens.weight")
        x = bf16_to_f32(embed_raw[token_id]) * np.float32(cfg.embed_scale)
        ple_in = self._ple_input(token_id, x) if self.ple else None
        hiddens = [x.copy()]
        for layer in range(cfg.num_layers):
            x = self._layer(x, layer, pos,
                            ple_in[layer] if ple_in is not None else None)
            if collect_hidden:
                hiddens.append(x.copy())
        x = rms_norm(x, self._w(PREFIX + "norm.weight"), cfg.rms_norm_eps)
        logits = self._w(PREFIX + "embed_tokens.weight") @ x  # tied lm_head
        logits = softcap(logits, cfg.final_logit_softcapping)
        if collect_hidden:
            return logits, hiddens
        return logits
