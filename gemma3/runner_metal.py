"""Metal-native runner for Gemma 3 via the metalgpu package.

Executes the same model as runner.py but through Metal directly, using the
MSL output of the DSL (kern.msl). Because metalgpu only exposes 1-D
dispatches with uncontrollable threadgroup sizes, all reductions run at
simdgroup scope (kernels_metal.py); elementwise kernels are shared with
the wgpu backend.

Buffers are Metal shared-storage: per-step parameters are plain numpy
writes into buffer.contents, and logits are read back the same way after
a wait_for_completion.
"""

from __future__ import annotations

import numpy as np

import metalgpu
from metalgpu import MetalSize

from .kernels import KERNELS
from .kernels_metal import METAL_KERNELS
from .loader import GemmaConfig

_SHARED_ELEMENTWISE = ("rope", "kv_append", "geglu")


def _combined_msl() -> str:
    """All Metal-backend kernels in one MSL library (deduped headers)."""
    kerns = list(METAL_KERNELS.values()) + [KERNELS[n] for n in _SHARED_ELEMENTWISE]
    bodies = []
    for kern in kerns:
        lines = [l for l in kern.msl.splitlines()
                 if not l.startswith("#include") and not l.startswith("using namespace")]
        bodies.append("\n".join(lines))
    return "#include <metal_stdlib>\nusing namespace metal;\n" + "\n".join(bodies)


def _pack_f16(arr: np.ndarray) -> np.ndarray:
    """f32 matrix → u32 array of packed f16 pairs."""
    f16 = arr.astype(np.float16)
    return np.frombuffer(f16.tobytes(), dtype=np.uint32)


class GemmaMetal:
    def __init__(self, config: GemmaConfig, weights: dict[str, np.ndarray],
                 max_seq: int = 1024) -> None:
        self.cfg = config
        self.max_seq = max_seq
        self.inst = metalgpu.Interface()
        self.inst.load_shader_from_string(_combined_msl())
        self._buffers = []  # for ordered cleanup

        cfg = config
        h, hd, nh, inter = (cfg.hidden_size, cfg.head_dim, cfg.num_heads,
                            cfg.intermediate_size)
        q_dim = nh * hd

        def ubuf(arr) -> object:
            b = self.inst.array_to_buffer(np.ascontiguousarray(arr))
            self._buffers.append(b)
            return b

        def fbuf(n) -> object:
            b = self.inst.create_buffer(int(n), "float")
            self._buffers.append(b)
            return b

        # --- weights (matmuls f16-packed as u32, norms f32) ---
        self.w = {}
        for name, arr in weights.items():
            if name.endswith("_proj.weight") or name == "model.embed_tokens.weight":
                self.w[name] = ubuf(_pack_f16(arr))
            elif name.endswith("norm.weight") or "layernorm" in name:
                self.w[name] = ubuf(arr.astype(np.float32))
        # (any tensor not matched above is unused by the text decoder)

        # --- scratch ---
        self.b = {
            "x": fbuf(h), "xn": fbuf(h),
            "q": fbuf(q_dim), "qn": fbuf(q_dim), "k": fbuf(hd), "kn": fbuf(hd),
            "v": fbuf(hd),
            "scores": fbuf(nh * max_seq), "attn": fbuf(q_dim), "attn_proj": fbuf(h),
            "gate": fbuf(inter), "up": fbuf(inter), "ffh": fbuf(inter),
            "mlp_out": fbuf(h),
            "logits": fbuf(cfg.vocab_size),
            "token": ubuf(np.zeros(1, np.uint32)),
        }
        self.k_cache = [fbuf(max_seq * hd) for _ in range(cfg.num_layers)]
        self.v_cache = [fbuf(max_seq * hd) for _ in range(cfg.num_layers)]

        # --- dims (static + per-step dynamic, mutated via .contents) ---
        u32a = lambda *v: np.array(v, dtype=np.uint32)
        self.d = {
            "embed": ubuf(u32a(h)),
            "norm_h": ubuf(u32a(1, h)),
            "norm_q": ubuf(u32a(nh, hd)),
            "norm_k": ubuf(u32a(1, hd)),
            "mv_q": ubuf(u32a(q_dim, h)),
            "mv_kv": ubuf(u32a(hd, h)),
            "mv_o": ubuf(u32a(h, q_dim)),
            "mv_ff": ubuf(u32a(inter, h)),
            "mv_down": ubuf(u32a(h, inter)),
            "mv_logits": ubuf(u32a(cfg.vocab_size, h)),
            "rope_q": ubuf(u32a(nh, hd)),
            "rope_k": ubuf(u32a(1, hd)),
            "geglu": ubuf(u32a(inter)),
            "kv_append": ubuf(u32a(hd, 0)),
            "sc_slide": ubuf(u32a(nh, hd, 1, 0, max_seq)),
            "sc_full": ubuf(u32a(nh, hd, 1, 0, max_seq)),
            "rope_local": ubuf(np.array([cfg.rope_theta_local, 0.0], np.float32)),
            "rope_global": ubuf(np.array([cfg.rope_theta_global, 0.0], np.float32)),
        }

    def close(self) -> None:
        """Release buffers before the Interface is torn down (metalgpu's
        destructor order otherwise segfaults at interpreter exit)."""
        for b in self._buffers:
            try:
                b.release()
            except Exception:
                pass
            # Buffer.__del__ calls release() again; neutralize the instance
            # method so gc-time teardown can't double-release.
            b.release = lambda: None
        self._buffers.clear()

    # ------------------------------------------------------------------ #

    def _write_step_params(self, token_id: int, pos: int) -> None:
        cfg = self.cfg
        kv_len = pos + 1
        start = max(0, kv_len - cfg.sliding_window)
        self.b["token"].contents[0] = token_id
        self.d["kv_append"].contents[1] = pos
        sc = self.d["sc_slide"].contents
        sc[2], sc[3] = kv_len, start
        self.d["sc_full"].contents[2] = kv_len
        self.d["rope_local"].contents[1] = float(pos)
        self.d["rope_global"].contents[1] = float(pos)

    def step(self, token_id: int, pos: int, want_logits: bool = True):
        if pos >= self.max_seq:
            raise ValueError(f"position {pos} exceeds max_seq {self.max_seq}")
        cfg, b, d, w = self.cfg, self.b, self.d, self.w
        nh, hd, h, inter = (cfg.num_heads, cfg.head_dim, cfg.hidden_size,
                            cfg.intermediate_size)
        self._write_step_params(token_id, pos)
        calls: list[tuple[int, list, str]] = []

        calls.append((h // 2, [b["token"], w["model.embed_tokens.weight"],
                               b["x"], d["embed"]], "embed_scale_packed"))
        for L in range(cfg.num_layers):
            p = f"model.layers.{L}."
            sliding = cfg.layer_types[L] == "sliding_attention"
            rope_f = d["rope_local"] if sliding else d["rope_global"]
            sc_d = d["sc_slide"] if sliding else d["sc_full"]
            mv = "matvec_simd_packed"
            calls += [
                (32, [b["x"], w[p + "input_layernorm.weight"], b["xn"],
                      d["norm_h"]], "rmsnorm_simd"),
                (nh * hd * 32, [w[p + "self_attn.q_proj.weight"], b["xn"],
                                b["q"], d["mv_q"]], mv),
                (hd * 32, [w[p + "self_attn.k_proj.weight"], b["xn"],
                           b["k"], d["mv_kv"]], mv),
                (hd * 32, [w[p + "self_attn.v_proj.weight"], b["xn"],
                           b["v"], d["mv_kv"]], mv),
                (nh * 32, [b["q"], w[p + "self_attn.q_norm.weight"], b["qn"],
                           d["norm_q"]], "rmsnorm_simd"),
                (32, [b["k"], w[p + "self_attn.k_norm.weight"], b["kn"],
                      d["norm_k"]], "rmsnorm_simd"),
                (nh * hd // 2, [b["qn"], rope_f, d["rope_q"]], "rope"),
                (hd // 2, [b["kn"], rope_f, d["rope_k"]], "rope"),
                (hd, [b["kn"], self.k_cache[L], d["kv_append"]], "kv_append"),
                (hd, [b["v"], self.v_cache[L], d["kv_append"]], "kv_append"),
                (nh * 32, [b["qn"], self.k_cache[L], self.v_cache[L], b["scores"],
                           b["attn"], sc_d], "attention_simd"),
                (h * 32, [w[p + "self_attn.o_proj.weight"], b["attn"],
                          b["attn_proj"], d["mv_o"]], mv),
                (32, [b["attn_proj"], w[p + "post_attention_layernorm.weight"],
                      b["x"], d["norm_h"]], "rmsnorm_add_simd"),
                (32, [b["x"], w[p + "pre_feedforward_layernorm.weight"], b["xn"],
                      d["norm_h"]], "rmsnorm_simd"),
                (inter * 32, [w[p + "mlp.gate_proj.weight"], b["xn"], b["gate"],
                              d["mv_ff"]], mv),
                (inter * 32, [w[p + "mlp.up_proj.weight"], b["xn"], b["up"],
                              d["mv_ff"]], mv),
                (inter, [b["gate"], b["up"], b["ffh"], d["geglu"]], "geglu"),
                (h * 32, [w[p + "mlp.down_proj.weight"], b["ffh"], b["mlp_out"],
                          d["mv_down"]], mv),
                (32, [b["mlp_out"], w[p + "post_feedforward_layernorm.weight"],
                      b["x"], d["norm_h"]], "rmsnorm_add_simd"),
            ]
        if want_logits:
            calls += [
                (32, [b["x"], w["model.norm.weight"], b["xn"], d["norm_h"]],
                 "rmsnorm_simd"),
                (cfg.vocab_size * 32, [w["model.embed_tokens.weight"], b["xn"],
                                       b["logits"], d["mv_logits"]],
                 "matvec_simd_packed"),
            ]

        for i, (n_threads, bufs, fn) in enumerate(calls):
            last = i == len(calls) - 1
            self.inst.run_function(MetalSize(int(n_threads), 1, 1), bufs, fn,
                                   wait_for_completion=last)

        if want_logits:
            return np.array(b["logits"].contents, dtype=np.float32)
        return None

    def generate(self, prompt_ids: list[int], max_new_tokens: int = 64,
                 on_token=None) -> list[int]:
        """Greedy generation (CPU argmax over shared-memory logits)."""
        logits = None
        for pos, tid in enumerate(prompt_ids):
            logits = self.step(tid, pos, want_logits=pos == len(prompt_ids) - 1)

        out: list[int] = []
        pos = len(prompt_ids)
        for _ in range(max_new_tokens):
            next_id = int(np.argmax(logits))
            out.append(next_id)
            if on_token is not None:
                on_token(next_id)
            if next_id in self.cfg.eos_token_ids:
                break
            if pos >= self.max_seq:
                break
            logits = self.step(next_id, pos)
            pos += 1
        return out
