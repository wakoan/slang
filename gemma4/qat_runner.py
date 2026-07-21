"""wgpu runner for the Gemma 4 E2B QAT-mobile checkpoint (int2/4/8 weights).

Fork of gemma4/runner.py that reads packed sub-byte weights and dequantizes
in the matmul kernels. Same decoder algorithm and control flow; the changes
are all in weight loading and which matvec fires per module:
  - attention (4-bit), MLP (4-bit L0-14 / 2-bit L15-34): matvec_dq4 / matvec_dq2
  - PLE gate/proj (8-bit) + per_layer_model_projection (unquantized): dequant
    to f16 on load, base vec4 f16 matvec
  - embed_tokens (2-bit, tied lm_head): qat_embed_2bit gather + matvec_dq2 logits
  - embed_tokens_per_layer (4-bit PLE table, ~1.17GB one buffer): qat_ple_gather_4bit
Norms/activations stay f32. Greedy decode is GPU-resident. Weight-only:
SRQ activation scales are skipped (the QAT weights carry the quantization).
"""

from __future__ import annotations

import re
import time
from collections import defaultdict

import numpy as np
import wgpu

from . import qat_kernels as K
from .loader import PREFIX, Gemma4Config
from .qat_loader import QATIndex

_NORM_SUFFIXES = (
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "post_per_layer_input_norm.weight",
)


def _binding_access(wgsl: str) -> list[bool]:
    out = {}
    for m in re.finditer(r"@binding\((\d+)\) var<storage, (read_write|read)>", wgsl):
        out[int(m.group(1))] = m.group(2) == "read_write"
    return [out[i] for i in sorted(out)]


class Gemma4QATGPU:
    def __init__(self, config: Gemma4Config, qat: QATIndex,
                 max_seq: int = 1024, profile: bool = False) -> None:
        self.cfg = config
        self.qat = qat
        self.max_seq = max_seq
        self.profile = profile

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if "shader-f16" not in adapter.features:
            raise RuntimeError("QAT runner needs shader-f16 (8-bit/unquantized modules)")
        vocab, h = config.vocab_size, config.hidden_size
        ple_dim = config.num_layers * config.hidden_size_per_layer_input  # 8960
        # largest single buffer is the 4-bit PLE table: vocab * ple_dim/2 bytes
        ple_table_bytes = vocab * (ple_dim // 2)  # ~1.17GB
        big = max(ple_table_bytes, 1 << 28)
        limits = {"max-buffer-size": big, "max-storage-buffer-binding-size": big}
        features = ["shader-f16"]
        if profile:
            if "timestamp-query" not in adapter.features:
                raise RuntimeError("profile=True requires timestamp-query")
            features.append("timestamp-query")
        self.device = adapter.request_device_sync(
            required_limits=limits, required_features=features)
        self._MAX_Q = 4096
        if profile:
            self._qs = self.device.create_query_set(
                type=wgpu.QueryType.timestamp, count=self._MAX_Q)
            self._q_resolve = self.device.create_buffer(
                size=self._MAX_Q * 8,
                usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
            self._prof_kernels: dict[str, list] = defaultdict(lambda: [0, 0, 0])
        self._prof_phases: dict[str, float] = defaultdict(float)
        self._prof_steps: list[tuple[int, float, bool]] = []

        # pipelines
        self._pipelines: dict[str, tuple] = {}
        for name, kern in K.KERNELS.items():
            module = self.device.create_shader_module(code=kern.wgsl)
            access = _binding_access(kern.wgsl)
            entries = [
                {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.storage if rw
                            else wgpu.BufferBindingType.read_only_storage}}
                for i, rw in enumerate(access)
            ]
            layout = self.device.create_bind_group_layout(entries=entries)
            self._pipelines[name] = (self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[layout]),
                compute={"module": module, "entry_point": name}), layout)

        self._st = wgpu.BufferUsage.STORAGE
        self._pending = 0
        self._load_weights()
        self._make_buffers()
        self._make_dims()
        self._bg_cache: dict[tuple, object] = {}
        self._build_bind_groups()

    # -- upload helpers --------------------------------------------------- #

    def _upload(self, data: bytes):
        buf = self.device.create_buffer_with_data(data=data, usage=self._st)
        self._pending += len(data)
        if self._pending >= (1 << 28):
            self.device.queue.submit([])
            self._pending = 0
        return buf

    def _norm_w(self, name: str) -> bytes:
        arr = self.qat.idx.tensor(name).astype(np.float32) - 1.0  # (1+w) kernels
        return arr.tobytes()

    def _load_linear(self, module: str) -> dict:
        """Returns {kind, w, scale?, n_out, n_in}. kind in dq4/dq2/f16."""
        packed, scale, bits = self.qat.packed_weight(module)
        n_out = packed.shape[0]
        if bits in (2, 4):
            per = 8 // bits  # weights per byte
            n_in = packed.shape[1] * per
            w = self._upload(np.ascontiguousarray(packed).tobytes())  # u32-viewed
            s = self._upload(np.ascontiguousarray(scale.ravel(), np.float32).tobytes())
            return {"kind": "dq4" if bits == 4 else "dq2", "w": w, "scale": s,
                    "n_out": n_out, "n_in": n_in}
        # 8-bit -> f16 (base vec4 matvec, no scale binding)
        w32 = self.qat.dequant_weight(module)
        n_in = w32.shape[1]
        w = self._upload(w32.astype(np.float16).tobytes())
        return {"kind": "f16", "w": w, "scale": None, "n_out": n_out, "n_in": n_in}

    def _load_weights(self) -> None:
        cfg, qat, up = self.cfg, self.qat, self._upload
        self.lin: dict[str, dict] = {}
        self.w: dict[str, object] = {}  # norm / scalar buffers

        # global unquantized-ish tensors
        self.w["norm"] = up(self._norm_w(PREFIX + "norm.weight"))
        self.w["ple_proj_norm"] = up(self._norm_w(PREFIX + "per_layer_projection_norm.weight"))
        pmp = qat.idx.tensor(PREFIX + "per_layer_model_projection.weight").astype(np.float32)
        pmp = pmp * np.float32(cfg.hidden_size ** -0.5)  # fold ctx scale in
        self.w["ple_model_proj"] = up(pmp.astype(np.float16).tobytes())
        self._ple_model_proj_nout = pmp.shape[0]

        # quantized embed (2-bit, tied lm_head) + PLE table (4-bit)
        et = qat.idx.raw(PREFIX + "embed_tokens.embedding_quantized")  # U8 [vocab, hidden/16*... ]
        self.w["embed"] = up(np.ascontiguousarray(et).tobytes())
        self.w["embed_scale"] = up(np.ascontiguousarray(
            qat.idx.tensor(PREFIX + "embed_tokens.embedding_scale").ravel(), np.float32).tobytes())
        pt = qat.idx.raw(PREFIX + "embed_tokens_per_layer.embedding_quantized")  # U8
        self.w["ple_table"] = up(np.ascontiguousarray(pt).tobytes())
        self.w["ple_table_scale"] = up(np.ascontiguousarray(
            qat.idx.tensor(PREFIX + "embed_tokens_per_layer.embedding_scale"), np.float32).tobytes())

        for spec in cfg.layers:
            p = f"{PREFIX}layers.{spec.index}."
            for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight",
                           "pre_feedforward_layernorm.weight",
                           "post_feedforward_layernorm.weight",
                           "self_attn.q_norm.weight", "post_per_layer_input_norm.weight"):
                self.w[p + suffix] = up(self._norm_w(p + suffix))
            self.w[p + "layer_scalar"] = up(
                qat.idx.tensor(p + "layer_scalar").astype(np.float32).tobytes())
            m = f"language_model.layers.{spec.index}."
            self.lin[p + "q"] = self._load_linear(m + "self_attn.q_proj")
            self.lin[p + "o"] = self._load_linear(m + "self_attn.o_proj")
            self.lin[p + "gate"] = self._load_linear(m + "mlp.gate_proj")
            self.lin[p + "up"] = self._load_linear(m + "mlp.up_proj")
            self.lin[p + "down"] = self._load_linear(m + "mlp.down_proj")
            self.lin[p + "ple_gate"] = self._load_linear(m + "per_layer_input_gate")
            self.lin[p + "ple_proj"] = self._load_linear(m + "per_layer_projection")
            if not spec.kv_shared:
                self.w[p + "self_attn.k_norm.weight"] = up(
                    self._norm_w(p + "self_attn.k_norm.weight"))
                self.lin[p + "k"] = self._load_linear(m + "self_attn.k_proj")
                self.lin[p + "v"] = self._load_linear(m + "self_attn.v_proj")
        self.device.queue.submit([])  # flush tail

    def _make_buffers(self) -> None:
        cfg = self.cfg
        h, nh = cfg.hidden_size, cfg.num_heads
        hd_max = max(s.head_dim for s in cfg.layers)
        q_max = max(s.q_dim for s in cfg.layers)
        inter_max = max(s.intermediate for s in cfg.layers)
        ple_h = cfg.hidden_size_per_layer_input
        ple_n = cfg.num_layers * ple_h
        rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        upd = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST

        def fb(n):
            return self.device.create_buffer(size=n * 4, usage=rw)

        self.b = {
            "x": fb(h), "xn": fb(h), "q": fb(q_max), "qn": fb(q_max),
            "k": fb(hd_max), "kn": fb(hd_max), "v": fb(hd_max), "vn": fb(hd_max),
            "scores": fb(nh * self.max_seq), "attn": fb(q_max), "attn_proj": fb(h),
            "gate": fb(inter_max), "up": fb(inter_max), "ffh": fb(inter_max),
            "mlp_out": fb(h), "logits": fb(cfg.vocab_size),
            "ple_ctx": fb(ple_n), "ple_ctx_n": fb(ple_n), "ple_in": fb(ple_n),
            "ple_g": fb(ple_h), "ple_h": fb(ple_h), "ple_proj": fb(h),
        }
        self.b["token"] = self.device.create_buffer(
            size=4, usage=upd | wgpu.BufferUsage.COPY_SRC)
        self.b["ple_tok"] = self.device.create_buffer(size=ple_n * 4, usage=upd)
        N = 128
        self.b["part_val"] = fb(N)
        self.b["part_idx"] = fb(N)
        self.b["out_tokens"] = fb(self.max_seq)
        self.b["counter"] = self.device.create_buffer(size=4, usage=upd)
        self.b["pos"] = self.device.create_buffer(size=4, usage=upd)
        self._n_argmax = N
        self.k_cache = {s.index: fb(self.max_seq * s.head_dim)
                        for s in cfg.layers if not s.kv_shared}
        self.v_cache = {s.index: fb(self.max_seq * s.head_dim)
                        for s in cfg.layers if not s.kv_shared}

    def _make_dims(self) -> None:
        cfg = self.cfg
        h, nh = cfg.hidden_size, cfg.num_heads
        ple_h = cfg.hidden_size_per_layer_input
        ple_n = cfg.num_layers * ple_h

        def ub(*v):
            return self.device.create_buffer_with_data(
                data=np.array(v, np.uint32).tobytes(), usage=self._st)

        def fb(*v):
            return self.device.create_buffer_with_data(
                data=np.array(v, np.float32).tobytes(), usage=self._st)

        def dyn(n):
            return self.device.create_buffer(
                size=n * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)

        hd_s = next(s.head_dim for s in cfg.layers if s.sliding)
        hd_f = next(s.head_dim for s in cfg.layers if not s.sliding)
        self.d = {
            "embed": ub(h), "norm_h": ub(1, h), "geglu_ple": ub(ple_h),
            "norm_ple_rows": ub(cfg.num_layers, ple_h), "combine": ub(ple_n),
            "ple_gather": ub(ple_n, ple_h, cfg.num_layers),
            "mv_ple_ctx": ub(ple_n, h), "softcap": ub(cfg.vocab_size),
            "argmax1": ub(cfg.vocab_size, 128), "argmax2": ub(128),
            "setup_cfg": ub(cfg.sliding_window),
            "kv_append_s": dyn(2), "kv_append_f": dyn(2),
            "scores_sliding": dyn(5), "scores_full": dyn(5),
            "rope_local": dyn(2), "rope_global": dyn(2),
        }
        for tag, hd in (("s", hd_s), ("f", hd_f)):
            cutoff = next(s.rope_cutoff for s in cfg.layers if s.head_dim == hd)
            self.d[f"norm_q_{tag}"] = ub(nh, hd)
            self.d[f"norm_k_{tag}"] = ub(1, hd)
            self.d[f"rope_q_{tag}"] = ub(nh, hd, cutoff)
            self.d[f"rope_k_{tag}"] = ub(1, hd, cutoff)
        self.fp = {
            "softcap": fb(cfg.final_logit_softcapping), "combine": fb(2.0 ** -0.5),
            "embed_scale": fb(cfg.embed_scale), "ple_scale": fb(cfg.ple_scale),
        }
        # dims for each linear shape [n_out, n_in]
        self._mvdims: dict[tuple, object] = {}
        for rec in self.lin.values():
            key = (rec["n_out"], rec["n_in"])
            if key not in self._mvdims:
                self._mvdims[key] = ub(*key)
        self._mvdims[(self._ple_model_proj_nout, h)] = ub(self._ple_model_proj_nout, h)

        # tied 2-bit logits: full vocab in one output-blocked (8 rows/wg) dispatch
        assert cfg.vocab_size % 8 == 0
        self._mvdims.setdefault((cfg.vocab_size, h), ub(cfg.vocab_size, h))

    # -- matvec dispatch -------------------------------------------------- #

    def _mv_kernel(self, rec: dict) -> str:
        # scalar dq beats vec4-with-dynamic-index (wv[c] spills on Apple GPUs)
        return {"dq4": "matvec_dq4_blk2", "dq2": "matvec_dq2",
                "f16": "matvec_wg_packed_v4"}[rec["kind"]]

    def _gateup_kernel(self, rec: dict) -> str:
        # dq2 (wide 2-bit layers) uses the output-blocked variant (4 rows/wg).
        return "mv_gateup_geglu_dq2_blk8" if rec["kind"] == "dq2" else "mv_gateup_geglu_dq4"

    def _gateup_grid(self, rec: dict) -> int:
        return rec["n_out"] // 8 if rec["kind"] == "dq2" else rec["n_out"]

    def _gateup_bg(self, gate: dict, up: dict):
        # Fused gate+up+geglu. gate/up share kind (both 2-bit L15-34, 4-bit L0-14).
        assert gate["kind"] == up["kind"] and gate["kind"] in ("dq2", "dq4")
        dims = self._mvdims[(gate["n_out"], gate["n_in"])]
        return self._bg(self._gateup_kernel(gate), gate["w"], up["w"], self.b["xn"],
                        gate["scale"], up["scale"], self.b["ffh"], dims)

    def _mv_bg(self, rec: dict, x_buf, y_buf, wview=None, sview=None, dims=None):
        w = wview if wview is not None else rec["w"]
        dims = dims if dims is not None else self._mvdims[(rec["n_out"], rec["n_in"])]
        if rec["kind"] == "f16":
            return self._bg("matvec_wg_packed_v4", w, x_buf, y_buf, dims)
        s = sview if sview is not None else rec["scale"]
        return self._bg(self._mv_kernel(rec), w, x_buf, s, y_buf, dims)

    def _bg(self, kern, *buffers):
        def norm(b):
            return b if isinstance(b, tuple) else (b, 0, b.size)
        views = [norm(b) for b in buffers]
        key = (kern, *((id(b), o, sz) for b, o, sz in views))
        if key not in self._bg_cache:
            _, layout = self._pipelines[kern]
            self._bg_cache[key] = self.device.create_bind_group(
                layout=layout,
                entries=[{"binding": i, "resource": {"buffer": b, "offset": o, "size": sz}}
                         for i, (b, o, sz) in enumerate(views)])
        return self._bg_cache[key]

    def _build_bind_groups(self) -> None:
        cfg, b, d, w = self.cfg, self.b, self.d, self.w
        ple_h = cfg.hidden_size_per_layer_input
        # embed gather (2-bit) + tied logits (2-bit dq matvec, chunked)
        self.bg_embed = self._bg("qat_embed_2bit", b["token"], w["embed"],
                                 w["embed_scale"], b["x"], self.fp["embed_scale"], d["embed"])
        self.bg_final_norm = self._bg("rmsnorm_wg", b["x"], w["norm"], b["xn"], d["norm_h"])
        self.bg_logits = self._bg(
            "matvec_dq2_blk16", w["embed"], b["xn"], w["embed_scale"], b["logits"],
            self._mvdims[(cfg.vocab_size, cfg.hidden_size)])
        self.bg_softcap = self._bg("softcap", b["logits"], self.fp["softcap"], d["softcap"])
        self.bg_argmax1 = self._bg("argmax_stage1", b["logits"], b["part_val"],
                                   b["part_idx"], d["argmax1"])
        self.bg_argmax2 = self._bg("argmax_stage2", b["part_val"], b["part_idx"],
                                   b["token"], b["out_tokens"], b["counter"], d["argmax2"])
        self.bg_setup = self._bg("step_setup_g4", b["pos"], d["kv_append_s"],
                                 d["kv_append_f"], d["scores_sliding"], d["scores_full"],
                                 d["rope_local"], d["rope_global"], d["setup_cfg"])
        # PLE per-token input
        self.bg_ple_gather = self._bg("qat_ple_gather_4bit", b["token"], w["ple_table"],
                                      w["ple_table_scale"], b["ple_tok"],
                                      self.fp["ple_scale"], d["ple_gather"])
        pmp = {"kind": "f16", "w": w["ple_model_proj"], "scale": None,
               "n_out": self._ple_model_proj_nout, "n_in": cfg.hidden_size}
        self.bg_ple_ctx = self._mv_bg(pmp, b["x"], b["ple_ctx"])
        self.bg_ple_ctx_norm = self._bg("rmsnorm_wg", b["ple_ctx"], w["ple_proj_norm"],
                                        b["ple_ctx_n"], d["norm_ple_rows"])
        self.bg_ple_combine = self._bg("combine_scaled", b["ple_ctx_n"], b["ple_tok"],
                                       b["ple_in"], self.fp["combine"], d["combine"])
        ple_bytes = ple_h * 4

        self.layer_bgs = []
        for spec in cfg.layers:
            L, hd, nh = spec.index, spec.head_dim, cfg.num_heads
            p = f"{PREFIX}layers.{L}."
            tag = "s" if spec.sliding else "f"
            rope_f = d["rope_local"] if spec.sliding else d["rope_global"]
            scores_d = d["scores_sliding"] if spec.sliding else d["scores_full"]
            bg = {
                "norm1": self._bg("rmsnorm_wg", b["x"], w[p + "input_layernorm.weight"],
                                  b["xn"], d["norm_h"]),
                "q": self._mv_bg(self.lin[p + "q"], b["xn"], b["q"]),
                "qnorm": self._bg("rmsnorm_wg", b["q"], w[p + "self_attn.q_norm.weight"],
                                  b["qn"], d[f"norm_q_{tag}"]),
                "rope_q": self._bg("rope_pl", b["qn"], rope_f, d[f"rope_q_{tag}"]),
                "attn": self._bg("attention_fused_g4", b["qn"], self.k_cache[spec.kv_source],
                                 self.v_cache[spec.kv_source], b["scores"], b["attn"], scores_d),
                "o": self._mv_bg(self.lin[p + "o"], b["attn"], b["attn_proj"]),
                "norm_pa_pf": self._bg("rmsnorm_add_norm_wg", b["attn_proj"],
                                       w[p + "post_attention_layernorm.weight"],
                                       w[p + "pre_feedforward_layernorm.weight"],
                                       b["x"], b["xn"], d["norm_h"]),
                "gateup": self._gateup_bg(self.lin[p + "gate"], self.lin[p + "up"]),
                "down": (self._bg(
                    "matvec_dq2_blk2", self.lin[p + "down"]["w"], b["ffh"],
                    self.lin[p + "down"]["scale"], b["mlp_out"],
                    self._mvdims[(self.lin[p + "down"]["n_out"], self.lin[p + "down"]["n_in"])])
                    if self.lin[p + "down"]["kind"] == "dq2"
                    else self._mv_bg(self.lin[p + "down"], b["ffh"], b["mlp_out"])),
                # post-FFN residual add-norm: x += rmsnorm(mlp_out) * (1 + w)
                "norm_pff_add": self._bg("rmsnorm_add_wg", b["mlp_out"],
                                         w[p + "post_feedforward_layernorm.weight"],
                                         b["x"], d["norm_h"]),
                # PLE block: fused gate matmul + geglu (up operand = ple_in slice)
                "ple_gateup": self._bg(
                    "mv_geglu_f16", self.lin[p + "ple_gate"]["w"], b["x"],
                    (b["ple_in"], L * ple_bytes, ple_bytes), b["ple_h"],
                    self._mvdims[(self.lin[p + "ple_gate"]["n_out"],
                                  self.lin[p + "ple_gate"]["n_in"])]),
                "ple_proj": self._mv_bg(self.lin[p + "ple_proj"], b["ple_h"], b["ple_proj"]),
                "ple_norm_add": self._bg("rmsnorm_add_scale_wg", b["ple_proj"],
                                         w[p + "post_per_layer_input_norm.weight"],
                                         b["x"], w[p + "layer_scalar"], d["norm_h"]),
            }
            if not spec.kv_shared:
                bg["k"] = self._mv_bg(self.lin[p + "k"], b["xn"], b["k"])
                bg["knorm"] = self._bg("rmsnorm_wg", b["k"], w[p + "self_attn.k_norm.weight"],
                                       b["kn"], d[f"norm_k_{tag}"])
                bg["rope_k"] = self._bg("rope_pl", b["kn"], rope_f, d[f"rope_k_{tag}"])
                bg["v"] = self._mv_bg(self.lin[p + "v"], b["xn"], b["v"])
                bg["vnorm"] = self._bg("rmsnorm_ns_wg", b["v"], b["vn"], d[f"norm_k_{tag}"])
                bg["app_k"] = self._bg("kv_append", b["kn"], self.k_cache[L], d[f"kv_append_{tag}"])
                bg["app_v"] = self._bg("kv_append", b["vn"], self.v_cache[L], d[f"kv_append_{tag}"])
            self.layer_bgs.append(bg)

    def _geglu_dims(self, inter):
        key = ("geglu", inter)
        if key not in self._bg_cache:
            self._bg_cache[key] = self.device.create_buffer_with_data(
                data=np.array([inter], np.uint32).tobytes(), usage=self._st)
        return self._bg_cache[key]

    # -- forward ---------------------------------------------------------- #

    def _write_step_params(self, token_id, pos):
        cfg, q = self.cfg, self.device.queue
        kv_len = pos + 1
        start = max(0, kv_len - cfg.sliding_window)
        nh, ms = cfg.num_heads, self.max_seq
        hd_s = next(s.head_dim for s in cfg.layers if s.sliding)
        hd_f = next(s.head_dim for s in cfg.layers if not s.sliding)
        theta_s = next(s.rope_theta for s in cfg.layers if s.sliding)
        theta_f = next(s.rope_theta for s in cfg.layers if not s.sliding)

        def u(buf, *v):
            q.write_buffer(buf, 0, np.array(v, np.uint32).tobytes())

        u(self.b["token"], token_id)
        u(self.d["kv_append_s"], hd_s, pos)
        u(self.d["kv_append_f"], hd_f, pos)
        u(self.d["scores_sliding"], nh, hd_s, kv_len, start, ms)
        u(self.d["scores_full"], nh, hd_f, kv_len, 0, ms)
        q.write_buffer(self.d["rope_local"], 0, np.array([theta_s, pos], np.float32).tobytes())
        q.write_buffer(self.d["rope_global"], 0, np.array([theta_f, pos], np.float32).tobytes())

    def _encode(self, run, argmax):
        cfg = self.cfg
        h, nh = cfg.hidden_size, cfg.num_heads
        ple_h = cfg.hidden_size_per_layer_input
        ple_n = cfg.num_layers * ple_h

        def mv(rec, bg, label):
            k = self._mv_kernel(rec)
            n = rec["n_out"] // 2 if k.endswith("_blk2") else rec["n_out"]
            run(k, bg, n, label=label, grid=(n, 1, 1))

        run("qat_embed_2bit", self.bg_embed, h, label="embed")
        run("matvec_wg_packed_v4", self.bg_ple_ctx, self._ple_model_proj_nout,
            label="mv_ple_ctx", grid=(self._ple_model_proj_nout, 1, 1))
        run("rmsnorm_wg", self.bg_ple_ctx_norm, ple_n, label="norm_ple_ctx",
            grid=(cfg.num_layers, 1, 1))
        run("qat_ple_gather_4bit", self.bg_ple_gather, ple_n, label="ple_gather")
        run("combine_scaled", self.bg_ple_combine, ple_n, label="ple_combine")

        for spec in cfg.layers:
            bg = self.layer_bgs[spec.index]
            hd = spec.head_dim
            run("rmsnorm_wg", bg["norm1"], h, label="norm_input", grid=(1, 1, 1))
            mv(self.lin[f"{PREFIX}layers.{spec.index}.q"], bg["q"], "mv_q")
            run("rmsnorm_wg", bg["qnorm"], spec.q_dim, label="qk_norm", grid=(nh, 1, 1))
            run("rope_pl", bg["rope_q"], nh * hd // 2, label="rope")
            if not spec.kv_shared:
                p = f"{PREFIX}layers.{spec.index}."
                mv(self.lin[p + "k"], bg["k"], "mv_kv")
                run("rmsnorm_wg", bg["knorm"], hd, label="qk_norm", grid=(1, 1, 1))
                run("rope_pl", bg["rope_k"], hd // 2, label="rope")
                mv(self.lin[p + "v"], bg["v"], "mv_kv")
                run("rmsnorm_ns_wg", bg["vnorm"], hd, label="v_norm", grid=(1, 1, 1))
                run("kv_append", bg["app_k"], hd, label="kv_append")
                run("kv_append", bg["app_v"], hd, label="kv_append")
            run("attention_fused_g4", bg["attn"], nh * 64, label="attn", grid=(nh, 1, 1))
            mv(self.lin[f"{PREFIX}layers.{spec.index}.o"], bg["o"], "mv_o")
            run("rmsnorm_add_norm_wg", bg["norm_pa_pf"], h, label="norm_add_norm", grid=(1, 1, 1))
            gate_rec = self.lin[f"{PREFIX}layers.{spec.index}.gate"]
            run(self._gateup_kernel(gate_rec), bg["gateup"], 0, label="mv_gateup",
                grid=(self._gateup_grid(gate_rec), 1, 1))
            down_rec = self.lin[f"{PREFIX}layers.{spec.index}.down"]
            if down_rec["kind"] == "dq2":
                run("matvec_dq2_blk2", bg["down"], down_rec["n_out"] // 2,
                    label="mv_down", grid=(down_rec["n_out"] // 2, 1, 1))
            else:
                mv(down_rec, bg["down"], "mv_down")
            run("rmsnorm_add_wg", bg["norm_pff_add"], h, label="norm_post_add", grid=(1, 1, 1))
            run("mv_geglu_f16", bg["ple_gateup"], ple_h, label="mv_ple", grid=(ple_h, 1, 1))
            mv(self.lin[f"{PREFIX}layers.{spec.index}.ple_proj"], bg["ple_proj"], "mv_ple")
            run("rmsnorm_add_scale_wg", bg["ple_norm_add"], h, label="norm_ple_add", grid=(1, 1, 1))

        run("rmsnorm_wg", self.bg_final_norm, h, label="norm_final", grid=(1, 1, 1))
        run("matvec_dq2_blk16", self.bg_logits, cfg.vocab_size // 16, label="mv_logits",
            grid=(cfg.vocab_size // 16, 1, 1))
        if argmax:
            run("argmax_stage1", self.bg_argmax1, self._n_argmax * 64,
                label="argmax", grid=(self._n_argmax, 1, 1))
            run("argmax_stage2", self.bg_argmax2, 64, label="argmax", grid=(1, 1, 1))
        else:
            run("softcap", self.bg_softcap, cfg.vocab_size, label="softcap")

    def step(self, token_id, pos, argmax=False):
        self._write_step_params(token_id, pos)
        enc = self.device.create_command_encoder()
        cp = None if self.profile else enc.begin_compute_pass()
        labels = []

        def run(name, bg, n, wg=64, label=None, grid=None):
            nonlocal cp
            pipe, _ = self._pipelines[name]
            if self.profile:
                i = len(labels) * 2
                cp = enc.begin_compute_pass(timestamp_writes={
                    "query_set": self._qs, "beginning_of_pass_write_index": i,
                    "end_of_pass_write_index": i + 1})
                labels.append((label or name, n))
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(*(grid or ((n + wg - 1) // wg, 1, 1)))
            if self.profile:
                cp.end()

        t0 = time.perf_counter()
        self._encode(run, argmax)
        if self.profile:
            enc.resolve_query_set(self._qs, 0, len(labels) * 2, self._q_resolve, 0)
        else:
            cp.end()
        self.device.queue.submit([enc.finish()])
        out = None
        if argmax:
            out = int(np.frombuffer(self.device.queue.read_buffer(self.b["token"]), np.uint32)[0])
        else:
            out = np.frombuffer(self.device.queue.read_buffer(self.b["logits"]), np.float32).copy()
        if self.profile:
            raw = self.device.queue.read_buffer(self._q_resolve)
            ts = np.frombuffer(raw, np.uint64)[: len(labels) * 2]
            for j, (lab, n) in enumerate(labels):
                rec = self._prof_kernels[lab]
                rec[0] += 1
                rec[1] += int(ts[2 * j + 1] - ts[2 * j])
                rec[2] = n
        self._prof_steps.append((pos + 1, time.perf_counter() - t0, True))
        return out

    def read_hidden(self):
        return np.frombuffer(self.device.queue.read_buffer(self.b["x"]), np.float32).copy()

    def _encode_chunk(self, k: int):
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()

        def run(name, bg, n, wg=64, label=None, grid=None):
            pipe, _ = self._pipelines[name]
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(*(grid or ((n + wg - 1) // wg, 1, 1)))

        for _ in range(k):
            run("step_setup_g4", self.bg_setup, 1, grid=(1, 1, 1))
            self._encode(run, argmax=True)
        cp.end()
        return enc.finish()

    def _generate_resident(self, prompt_ids, max_new_tokens, on_token):
        """GPU-resident greedy decode: step params, PLE gather, argmax feedback
        stay on-device; the CPU reads out_tokens once per chunk (EOS check)."""
        q = self.device.queue
        for pos, tid in enumerate(prompt_ids[:-1]):
            self._write_step_params(tid, pos)
            self.step(tid, pos, argmax=False)  # prefill (logits unused)
        start = len(prompt_ids) - 1
        self._write_step_params(prompt_ids[-1], start)
        q.write_buffer(self.b["pos"], 0, np.array([start], np.uint32).tobytes())
        q.write_buffer(self.b["counter"], 0, np.zeros(1, np.uint32).tobytes())

        budget = min(max_new_tokens, self.max_seq - start)
        sizes, c, left = [], min(8, 64), budget
        while left > 0:
            k = min(c, left)
            sizes.append(k)
            left -= k
            c = min(c * 2, 64)

        out, emitted, produced = [], 0, 0
        cmd = self._encode_chunk(sizes[0])
        t0 = time.perf_counter()
        q.submit([cmd])
        for i, k in enumerate(sizes):
            nxt = self._encode_chunk(sizes[i + 1]) if i + 1 < len(sizes) else None
            produced += k
            toks = np.frombuffer(q.read_buffer(self.b["out_tokens"], size=produced * 4), np.uint32)
            self._prof_steps.append((start + produced, (time.perf_counter() - t0) / k, True))
            t0 = time.perf_counter()
            stop = False
            for tid in toks[emitted:]:
                tid = int(tid)
                out.append(tid)
                if on_token:
                    on_token(tid)
                emitted += 1
                if tid in self.cfg.eos_token_ids:
                    stop = True
                    break
            if stop or nxt is None:
                break
            q.submit([nxt])
        return out

    def generate(self, prompt_ids, max_new_tokens=64, on_token=None, resident=True):
        """Greedy decode. Resident keeps the whole loop on-GPU (chunked)."""
        if resident and not self.profile:
            return self._generate_resident(prompt_ids, max_new_tokens, on_token)
        q = self.device.queue
        q.write_buffer(self.b["counter"], 0, np.zeros(1, np.uint32).tobytes())
        nxt = None
        for pos, tid in enumerate(prompt_ids):
            last = pos == len(prompt_ids) - 1
            r = self.step(tid, pos, argmax=last)
            if last:
                nxt = r
        out, pos = [], len(prompt_ids)
        while len(out) < max_new_tokens:
            out.append(nxt)
            if on_token:
                on_token(nxt)
            if nxt in self.cfg.eos_token_ids or pos >= self.max_seq:
                break
            nxt = self.step(nxt, pos, argmax=True)
            pos += 1
        return out
