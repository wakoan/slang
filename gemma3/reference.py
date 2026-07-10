"""Pure-numpy Gemma 3 reference decoder.

Mirrors HF Gemma3ForCausalLM in float32, one token position at a time with
a KV cache — the exact computation the GPU kernels implement, for
verification.
"""

from __future__ import annotations

import numpy as np

from .loader import GemmaConfig


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Gemma-style RMSNorm over the last axis: x/rms * (1 + w)."""
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * (1.0 + weight)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def apply_rope(x: np.ndarray, position: int, theta: float) -> np.ndarray:
    """rotate_half RoPE on x of shape [n_heads, head_dim]."""
    n_heads, head_dim = x.shape
    half = head_dim // 2
    i = np.arange(half, dtype=np.float32)
    inv_freq = 1.0 / theta ** (2.0 * i / head_dim)
    angle = np.float32(position) * inv_freq
    c, s = np.cos(angle, dtype=np.float32), np.sin(angle, dtype=np.float32)
    a, b = x[:, :half], x[:, half:]
    return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)


class ReferenceGemma:
    """Single-position forward pass with per-layer KV cache."""

    def __init__(self, config: GemmaConfig, weights: dict[str, np.ndarray],
                 max_seq: int = 1024) -> None:
        self.cfg = config
        self.w = weights
        self.max_seq = max_seq
        hd = config.head_dim
        self.k_cache = np.zeros((config.num_layers, max_seq, hd), dtype=np.float32)
        self.v_cache = np.zeros((config.num_layers, max_seq, hd), dtype=np.float32)

    def _layer(self, x: np.ndarray, layer: int, pos: int) -> np.ndarray:
        cfg, w = self.cfg, self.w
        p = f"model.layers.{layer}."
        n_heads, hd = cfg.num_heads, cfg.head_dim
        is_sliding = cfg.layer_types[layer] == "sliding_attention"
        theta = cfg.rope_theta_local if is_sliding else cfg.rope_theta_global

        # --- attention block (sandwich norm) ---
        residual = x
        h = rms_norm(x, w[p + "input_layernorm.weight"], cfg.rms_norm_eps)

        q = (w[p + "self_attn.q_proj.weight"] @ h).reshape(n_heads, hd)
        k = (w[p + "self_attn.k_proj.weight"] @ h).reshape(1, hd)
        v = (w[p + "self_attn.v_proj.weight"] @ h).reshape(1, hd)

        # QK-norm, then RoPE
        q = rms_norm(q, w[p + "self_attn.q_norm.weight"], cfg.rms_norm_eps)
        k = rms_norm(k, w[p + "self_attn.k_norm.weight"], cfg.rms_norm_eps)
        q = apply_rope(q, pos, theta)
        k = apply_rope(k, pos, theta)

        self.k_cache[layer, pos] = k[0]
        self.v_cache[layer, pos] = v[0]

        kv_len = pos + 1
        start = max(0, kv_len - cfg.sliding_window) if is_sliding else 0

        scale = cfg.query_pre_attn_scalar ** -0.5
        keys = self.k_cache[layer, :kv_len]          # [kv_len, hd]
        scores = (q @ keys.T) * scale                # [n_heads, kv_len]
        if start > 0:
            scores[:, :start] = -1e9
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=-1, keepdims=True)

        vals = self.v_cache[layer, :kv_len]          # [kv_len, hd]
        attn = probs @ vals                          # [n_heads, hd]
        attn_out = w[p + "self_attn.o_proj.weight"] @ attn.reshape(-1)

        h = rms_norm(attn_out, w[p + "post_attention_layernorm.weight"], cfg.rms_norm_eps)
        x = residual + h

        # --- MLP block (sandwich norm) ---
        residual = x
        h = rms_norm(x, w[p + "pre_feedforward_layernorm.weight"], cfg.rms_norm_eps)
        gate = w[p + "mlp.gate_proj.weight"] @ h
        up = w[p + "mlp.up_proj.weight"] @ h
        mlp_out = w[p + "mlp.down_proj.weight"] @ (gelu_tanh(gate) * up)
        h = rms_norm(mlp_out, w[p + "post_feedforward_layernorm.weight"], cfg.rms_norm_eps)
        return residual + h

    def forward(self, token_id: int, pos: int,
                collect_hidden: bool = False) -> np.ndarray | tuple:
        """Process one token at `pos`; returns logits (and per-layer hiddens)."""
        cfg, w = self.cfg, self.w
        x = w["model.embed_tokens.weight"][token_id] * np.float32(
            np.sqrt(cfg.hidden_size)
        )
        hiddens = [x.copy()]
        for layer in range(cfg.num_layers):
            x = self._layer(x, layer, pos)
            if collect_hidden:
                hiddens.append(x.copy())
        x = rms_norm(x, w["model.norm.weight"], cfg.rms_norm_eps)
        logits = w["model.embed_tokens.weight"] @ x  # tied lm_head
        if collect_hidden:
            return logits, hiddens
        return logits
