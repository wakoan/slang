"""wgpu runner for Gemma 4 E2B: full decoder forward pass on the GPU.

Structural fork of gemma3/runner.py. Differences:
- weights stream from the lazy SafetensorsIndex and upload one tensor at a
  time (never a full-model host copy);
- scaled norm weights upload as (w - 1) so the Gemma 3 (1 + w) kernels
  realise Gemma 4's direct-w convention unchanged;
- per-layer dims (head_dim 256/512, intermediate 6144/12288) select between
  two static dims-buffer families;
- KV sharing: only layers 0-14 own caches; layers 15+ bind layer 13/14's
  buffers (flat [max_seq, hd] caches + score-time start offset make the
  aliasing exact) and skip k/v work entirely;
- Per-Layer Embeddings: per-token 35KB gather from the CPU-side mmap'd
  4.7GB table + a small GPU pipeline (gate → geglu → proj → norm →
  scaled residual);
- decode is CPU-driven per-step (the PLE gather depends on the generated
  token); GPU-resident decode is a later optimization.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict

import numpy as np
import wgpu

from . import kernels as K
from .loader import PREFIX, Gemma4Config, SafetensorsIndex, bf16_to_f32

_MAX_QUERIES = 4096  # timestamps per step (2 per dispatch)

#: scaled-norm weights: uploaded as (w - 1) to reuse the (1 + w) kernels
_NORM_SUFFIXES = (
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "post_per_layer_input_norm.weight",
)


def _binding_access(wgsl: str) -> list[bool]:
    """Per binding index: True if read_write, from the generated WGSL."""
    out = {}
    for m in re.finditer(r"@binding\((\d+)\) var<storage, (read_write|read)>", wgsl):
        out[int(m.group(1))] = m.group(2) == "read_write"
    return [out[i] for i in sorted(out)]


class Gemma4GPU:
    #: weights stored f16 (packed-u32 matvec loads); norms/activations stay f32
    _F16_SUFFIXES = (
        "embed_tokens.weight", "qkv_proj.weight", "q_proj.weight",
        "o_proj.weight", "gateup_proj.weight", "down_proj.weight",
        "per_layer_input_gate.weight", "per_layer_projection.weight",
        "per_layer_model_projection.weight",
    )

    def __init__(self, config: Gemma4Config, index: SafetensorsIndex,
                 max_seq: int = 1024, profile: bool = False,
                 dtype: str = "f16", ple: bool = True,
                 resident: bool = True) -> None:
        self.cfg = config
        self.idx = index
        self.max_seq = max_seq
        self.profile = profile

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if dtype == "f16" and "shader-f16" not in adapter.features:
            dtype = "f32"  # graceful fallback
        self.dtype = dtype
        self.ple = ple  # False isolates the non-PLE pipeline for verification
        # GPU-resident greedy decode needs the PLE table on-GPU as f16
        # (+4.7GB) and the f16 gather kernel; f32 mode stays CPU-driven.
        self.resident = resident and ple and dtype == "f16"
        embed_bytes = config.vocab_size * config.hidden_size * 4
        half_table = (config.vocab_size // 2) * config.num_layers \
            * config.hidden_size_per_layer_input * 2
        big = max(embed_bytes, half_table if self.resident else 0, 1 << 28)
        limits = {
            "max-buffer-size": big,
            "max-storage-buffer-binding-size": big,
        }
        features = []
        self._sg_feature = "subgroup" in adapter.features
        if self._sg_feature:
            features.append("subgroup")
        if dtype == "f16":
            features.append("shader-f16")
        if profile:
            if "timestamp-query" not in adapter.features:
                raise RuntimeError("profile=True requires the timestamp-query feature")
            features.append("timestamp-query")
        self.device = adapter.request_device_sync(
            required_limits=limits, required_features=features)

        if profile:
            self._qs = self.device.create_query_set(
                type=wgpu.QueryType.timestamp, count=_MAX_QUERIES)
            self._q_resolve = self.device.create_buffer(
                size=_MAX_QUERIES * 8,
                usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
            self._prof_kernels: dict[str, list] = defaultdict(lambda: [0, 0, 0])
        self._prof_phases: dict[str, float] = defaultdict(float)
        self._prof_steps: list[tuple[int, float, bool]] = []

        # --- pipelines (one per kernel) ---
        self._pipelines: dict[str, tuple] = {}
        for name, kern in K.KERNELS.items():
            if name.endswith("_f16") and self.dtype != "f16":
                continue  # f16 shaders need the shader-f16 feature
            if name.endswith("_sg") and not self._sg_feature:
                continue  # subgroup shaders need the subgroup feature
            module = self._shader_module(kern.wgsl)
            access = _binding_access(kern.wgsl)
            entries = [
                {
                    "binding": i,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": wgpu.BufferBindingType.storage if rw
                        else wgpu.BufferBindingType.read_only_storage
                    },
                }
                for i, rw in enumerate(access)
            ]
            layout = self.device.create_bind_group_layout(entries=entries)
            pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[layout]),
                compute={"module": module, "entry_point": name},
            )
            self._pipelines[name] = (pipeline, layout)

        # Measured on M4 Pro (resident decode, 200 tokens): the 32-thread
        # _sg kernels are a net LOSS for E2B's long rows (31.5 vs 33.0
        # tok/s; norm_post_add 36 vs 20µs, mv_down 174 vs 145µs) — unlike
        # gemma3's 640-wide rows where they won. Keep them off; the
        # variants stay in the registry for future selective use.
        self.use_subgroups = False

        # resolved kernel names per call-site family
        f16 = self.dtype == "f16"
        sg = "_sg" if self.use_subgroups else ""
        self._kn = {
            "embed_scale": "embed_scale_f16" if f16 else "embed_scale",
            "matvec_wg": ("matvec_wg_packed" + sg) if f16 else "matvec_wg",
            "rmsnorm_wg": "rmsnorm_wg" + sg,
            "rmsnorm_add_wg": "rmsnorm_add_wg" + sg,
            "attention_fused_g4": "attention_fused_g4" + sg,
        }

        # --- weight buffers (streamed: one host tensor alive at a time) ---
        st = wgpu.BufferUsage.STORAGE
        cfg = config
        h = cfg.hidden_size

        def wtensor(name: str) -> np.ndarray:
            arr = index.tensor(name)
            if name.endswith(_NORM_SUFFIXES) or name == PREFIX + "norm.weight" \
                    or name == PREFIX + "per_layer_projection_norm.weight":
                arr = arr - 1.0  # (1 + w) kernels -> direct-w convention
            if name == PREFIX + "per_layer_model_projection.weight":
                arr = arr * np.float32(h ** -0.5)  # fold the ctx scale in
            return arr

        def wdata(name: str, arr: np.ndarray) -> bytes:
            if f16 and name.endswith(self._F16_SUFFIXES):
                return arr.astype(np.float16).tobytes()
            return arr.tobytes()

        # mapped-at-creation uploads are staged until a queue submit; wgpu
        # silently drops the pending copies if gigabytes accumulate unflushed
        # (empirically ~9GB kills ALL of them). Flush every 256MB.
        pending = 0

        def upload(data: bytes) -> object:
            nonlocal pending
            buf = self.device.create_buffer_with_data(data=data, usage=st)
            pending += len(data)
            if pending >= (1 << 28):
                self.device.queue.submit([])
                pending = 0
            return buf

        def wbuf(name: str) -> object:
            return upload(wdata(name, wtensor(name)))

        self._wbuf: dict[str, object] = {}
        w = self._wbuf
        for name in (PREFIX + "embed_tokens.weight", PREFIX + "norm.weight",
                     PREFIX + "per_layer_model_projection.weight",
                     PREFIX + "per_layer_projection_norm.weight"):
            w[name] = wbuf(name)

        self._fp_scalar: list[object] = []  # per-layer learned layer_scalar
        for spec in cfg.layers:
            p = f"{PREFIX}layers.{spec.index}."
            for suffix in ("input_layernorm.weight", "post_attention_layernorm.weight",
                           "pre_feedforward_layernorm.weight",
                           "post_feedforward_layernorm.weight",
                           "self_attn.q_norm.weight", "self_attn.o_proj.weight",
                           "mlp.down_proj.weight"):
                w[p + suffix] = wbuf(p + suffix)
            gu = np.concatenate([wtensor(p + "mlp.gate_proj.weight"),
                                 wtensor(p + "mlp.up_proj.weight")])
            w[p + "mlp.gateup_proj.weight"] = upload(
                wdata(p + "mlp.gateup_proj.weight", gu))
            del gu
            if spec.kv_shared:
                w[p + "self_attn.q_proj.weight"] = wbuf(p + "self_attn.q_proj.weight")
            else:
                qkv = np.concatenate([wtensor(p + "self_attn.q_proj.weight"),
                                      wtensor(p + "self_attn.k_proj.weight"),
                                      wtensor(p + "self_attn.v_proj.weight")])
                w[p + "self_attn.qkv_proj.weight"] = upload(
                    wdata(p + "self_attn.qkv_proj.weight", qkv))
                del qkv
                w[p + "self_attn.k_norm.weight"] = wbuf(p + "self_attn.k_norm.weight")
            if ple:
                for suffix in ("per_layer_input_gate.weight",
                               "per_layer_projection.weight",
                               "post_per_layer_input_norm.weight"):
                    w[p + suffix] = wbuf(p + suffix)
            self._fp_scalar.append(upload(
                index.tensor(p + "layer_scalar").tobytes()))
        self.device.queue.submit([])  # flush the last partial batch

        #: CPU-side mmap view of the 4.7GB PLE table for per-token gathers
        self._ple_raw = index.raw(PREFIX + "embed_tokens_per_layer.weight")

        if self.resident:
            # PLE table on-GPU as f16 for the resident decode chain. 4.7GB
            # exceeds the 4.29GB max storage binding, so it ships as two
            # half-tables; ple_gather_f16 picks the half by token range.
            split = cfg.vocab_size // 2
            row_len = cfg.num_layers * cfg.hidden_size_per_layer_input
            self._ple_split = split
            self._ple_table = []
            for r0, r1 in ((0, split), (split, cfg.vocab_size)):
                buf = self.device.create_buffer(
                    size=(r1 - r0) * row_len * 2,
                    usage=st | wgpu.BufferUsage.COPY_DST)
                CH = 8192  # rows per upload chunk (~147MB f16)
                for r in range(r0, r1, CH):
                    rows = bf16_to_f32(self._ple_raw[r:min(r + CH, r1)])
                    self.device.queue.write_buffer(
                        buf, (r - r0) * row_len * 2,
                        rows.astype(np.float16).tobytes())
                    self.device.queue.submit([])  # flush staging per chunk
                self._ple_table.append(buf)

        # --- scratch buffers (sized at per-family maxima) ---
        nh = cfg.num_heads
        hd_max = max(s.head_dim for s in cfg.layers)          # 512
        q_max = max(s.q_dim for s in cfg.layers)              # 4096
        inter_max = max(s.intermediate for s in cfg.layers)   # 12288
        ple_h = cfg.hidden_size_per_layer_input               # 256
        ple_n = cfg.num_layers * ple_h                        # 8960
        rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        upd = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST

        def fbuf(n):
            return self.device.create_buffer(size=n * 4, usage=rw)

        self.b = {
            "x": fbuf(h), "xn": fbuf(h),
            "qkv": fbuf(q_max + 2 * hd_max), "qn": fbuf(q_max),
            "kn": fbuf(hd_max), "vn": fbuf(hd_max),
            "scores": fbuf(nh * max_seq), "attn": fbuf(q_max), "attn_proj": fbuf(h),
            "gateup": fbuf(2 * inter_max), "ffh": fbuf(inter_max),
            "mlp_out": fbuf(h),
            "logits": fbuf(cfg.vocab_size),
            "ple_ctx": fbuf(ple_n), "ple_ctx_n": fbuf(ple_n), "ple_in": fbuf(ple_n),
            "ple_g": fbuf(ple_h), "ple_h": fbuf(ple_h),
            "ple_proj": fbuf(h), "ple_norm": fbuf(h),
        }
        self.b["token"] = self.device.create_buffer(
            size=4, usage=upd | wgpu.BufferUsage.COPY_SRC)
        self.b["ple_tok"] = self.device.create_buffer(size=ple_n * 4, usage=upd)
        # GPU argmax state: greedy decode reads back a 4-byte token instead
        # of the 1MB logits (softcap is argmax-invariant and skipped)
        N_ARGMAX_WGS = 128
        self.b["part_val"] = fbuf(N_ARGMAX_WGS)
        self.b["part_idx"] = fbuf(N_ARGMAX_WGS)
        self.b["out_tokens"] = fbuf(max_seq)
        self.b["counter"] = self.device.create_buffer(size=4, usage=upd)
        self.b["pos"] = self.device.create_buffer(size=4, usage=upd)
        self._n_argmax_wgs = N_ARGMAX_WGS
        # KV caches only for layers that own K/V; shared layers alias these
        self.k_cache = {s.index: fbuf(max_seq * s.head_dim)
                        for s in cfg.layers if not s.kv_shared}
        self.v_cache = {s.index: fbuf(max_seq * s.head_dim)
                        for s in cfg.layers if not s.kv_shared}

        # --- dims / fparams buffers ---
        def ubuf(*vals):
            return self.device.create_buffer_with_data(
                data=np.array(vals, dtype=np.uint32).tobytes(), usage=st)

        def fubuf(*vals):
            return self.device.create_buffer_with_data(
                data=np.array(vals, dtype=np.float32).tobytes(), usage=st)

        def dynbuf(n):
            return self.device.create_buffer(size=n * 4, usage=upd)

        hd_s = next(s.head_dim for s in cfg.layers if s.sliding)       # 256
        hd_f = next(s.head_dim for s in cfg.layers if not s.sliding)   # 512
        self.d = {
            "embed": ubuf(h),
            "norm_h": ubuf(1, h),
            "geglu_ple": ubuf(ple_h),
            "add": ubuf(h),
            "mv_ple_ctx": ubuf(ple_n, h),
            "norm_ple_rows": ubuf(cfg.num_layers, ple_h),
            "combine": ubuf(ple_n),
            "mv_ple_gate": ubuf(ple_h, h),
            "mv_ple_proj": ubuf(h, ple_h),
            "softcap": ubuf(cfg.vocab_size),
            "argmax1": ubuf(cfg.vocab_size, 128),
            "argmax2": ubuf(128),
            "setup_cfg": ubuf(cfg.sliding_window),
            # dynamic (rewritten each step)
            "kv_append_s": dynbuf(2), "kv_append_f": dynbuf(2),
            "scores_sliding": dynbuf(5), "scores_full": dynbuf(5),
            "rope_local": dynbuf(2), "rope_global": dynbuf(2),
        }
        for tag, hd in (("s", hd_s), ("f", hd_f)):
            q_dim = nh * hd
            cutoff = next(s.rope_cutoff for s in cfg.layers
                          if s.head_dim == hd)
            self.d[f"norm_q_{tag}"] = ubuf(nh, hd)
            self.d[f"norm_k_{tag}"] = ubuf(1, hd)
            self.d[f"rope_q_{tag}"] = ubuf(nh, hd, cutoff)
            self.d[f"rope_k_{tag}"] = ubuf(1, hd, cutoff)
            self.d[f"mv_qkv_{tag}"] = ubuf(q_dim + 2 * hd, h)
            self.d[f"mv_q_{tag}"] = ubuf(q_dim, h)
            self.d[f"mv_o_{tag}"] = ubuf(h, q_dim)
        for tag, inter in (("n", 6144), ("w", 12288)):
            self.d[f"mv_gateup_{tag}"] = ubuf(2 * inter, h)
            self.d[f"mv_down_{tag}"] = ubuf(h, inter)
            self.d[f"geglu_{tag}"] = ubuf(inter)
        self.fp = {
            "softcap": fubuf(cfg.final_logit_softcapping),
            "combine": fubuf(2.0 ** -0.5),
            "ple_scale": fubuf(cfg.ple_scale),
        }
        if self.resident:
            self.d["ple_gather"] = ubuf(
                cfg.num_layers * cfg.hidden_size_per_layer_input, self._ple_split)

        LC = 32768  # logits matvec: chunked under the 65535-workgroup cap
        self._logits_chunks = [(s, min(LC, cfg.vocab_size - s))
                               for s in range(0, cfg.vocab_size, LC)]
        for _, rows in self._logits_chunks:
            self.d.setdefault(f"mv_logits_{rows}", ubuf(rows, h))

        # --- bind groups ---
        self._bg_cache: dict[tuple, object] = {}
        self._build_bind_groups()

    # ------------------------------------------------------------------ #

    def _shader_module(self, code: str):
        """Compat shim: current naga rejects `enable subgroups;` while
        supporting the ops themselves — retry without the directive."""
        try:
            return self.device.create_shader_module(code=code)
        except Exception:
            if "enable subgroups;" in code:
                return self.device.create_shader_module(
                    code=code.replace("enable subgroups;", ""))
            raise

    def _probe_subgroup_size(self) -> int:
        buf = self.device.create_buffer(
            size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        pipeline, layout = self._pipelines["probe_sg"]
        bg = self.device.create_bind_group(
            layout=layout,
            entries=[{"binding": 0, "resource": {"buffer": buf, "offset": 0, "size": 4}}])
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(1, 1, 1)
        cp.end()
        self.device.queue.submit([enc.finish()])
        return int(np.frombuffer(self.device.queue.read_buffer(buf), dtype=np.uint32)[0])

    def _bg(self, kern_name: str, *buffers):
        """buffers: GPUBuffer or (GPUBuffer, offset_bytes, size_bytes) tuples."""
        def norm(b):
            return b if isinstance(b, tuple) else (b, 0, b.size)

        views = [norm(b) for b in buffers]
        key = (kern_name, *((id(b), o, sz) for b, o, sz in views))
        if key not in self._bg_cache:
            _, layout = self._pipelines[kern_name]
            self._bg_cache[key] = self.device.create_bind_group(
                layout=layout,
                entries=[
                    {"binding": i, "resource": {"buffer": b, "offset": o, "size": sz}}
                    for i, (b, o, sz) in enumerate(views)
                ],
            )
        return self._bg_cache[key]

    def _build_bind_groups(self) -> None:
        cfg, b, d, w = self.cfg, self.b, self.d, self._wbuf
        emb = w[PREFIX + "embed_tokens.weight"]
        self.bg_embed = self._bg(self._kn["embed_scale"], b["token"], emb, b["x"], d["embed"])
        self.bg_final_norm = self._bg(
            self._kn["rmsnorm_wg"], b["x"], w[PREFIX + "norm.weight"], b["xn"], d["norm_h"])
        row_bytes = cfg.hidden_size * (2 if self.dtype == "f16" else 4)
        self.bg_logits_chunks = [
            (rows, self._bg(self._kn["matvec_wg"],
                            (emb, start * row_bytes, rows * row_bytes),
                            b["xn"],
                            (b["logits"], start * 4, rows * 4),
                            d[f"mv_logits_{rows}"]))
            for start, rows in self._logits_chunks
        ]
        self.bg_softcap = self._bg(
            "softcap", b["logits"], self.fp["softcap"], d["softcap"])
        self.bg_argmax1 = self._bg(
            "argmax_stage1", b["logits"], b["part_val"], b["part_idx"], d["argmax1"])
        self.bg_argmax2 = self._bg(
            "argmax_stage2", b["part_val"], b["part_idx"], b["token"],
            b["out_tokens"], b["counter"], d["argmax2"])
        if self.resident:
            self.bg_setup = self._bg(
                "step_setup_g4", b["pos"], d["kv_append_s"], d["kv_append_f"],
                d["scores_sliding"], d["scores_full"], d["rope_local"],
                d["rope_global"], d["setup_cfg"])
            self.bg_ple_gather = self._bg(
                "ple_gather_f16", b["token"], self._ple_table[0],
                self._ple_table[1], b["ple_tok"], self.fp["ple_scale"],
                d["ple_gather"])

        if self.ple:
            self.bg_ple_ctx = self._bg(
                self._kn["matvec_wg"], w[PREFIX + "per_layer_model_projection.weight"],
                b["x"], b["ple_ctx"], d["mv_ple_ctx"])
            self.bg_ple_ctx_norm = self._bg(
                self._kn["rmsnorm_wg"], b["ple_ctx"],
                w[PREFIX + "per_layer_projection_norm.weight"],
                b["ple_ctx_n"], d["norm_ple_rows"])
            self.bg_ple_combine = self._bg(
                "combine_scaled", b["ple_ctx_n"], b["ple_tok"], b["ple_in"],
                self.fp["combine"], d["combine"])

        ple_bytes = cfg.hidden_size_per_layer_input * 4
        self.layer_bgs = []
        for spec in cfg.layers:
            L, hd, nh = spec.index, spec.head_dim, cfg.num_heads
            p = f"{PREFIX}layers.{L}."
            tag = "s" if spec.sliding else "f"
            wtag = "w" if spec.intermediate == 12288 else "n"
            inter = spec.intermediate
            rope_f = d["rope_local"] if spec.sliding else d["rope_global"]
            scores_d = d["scores_sliding"] if spec.sliding else d["scores_full"]
            q_bytes = spec.q_dim * 4
            kv_bytes = hd * 4
            qkv = b["qkv"]
            gu = b["gateup"]
            bg = {
                "norm1": self._bg(self._kn["rmsnorm_wg"], b["x"], w[p + "input_layernorm.weight"],
                                  b["xn"], d["norm_h"]),
                "qnorm": self._bg(self._kn["rmsnorm_wg"], (qkv, 0, q_bytes),
                                  w[p + "self_attn.q_norm.weight"],
                                  b["qn"], d[f"norm_q_{tag}"]),
                "rope_q": self._bg("rope_pl", b["qn"], rope_f, d[f"rope_q_{tag}"]),
                "attn": self._bg(self._kn["attention_fused_g4"], b["qn"],
                                 self.k_cache[spec.kv_source],
                                 self.v_cache[spec.kv_source],
                                 b["scores"], b["attn"], scores_d),
                "mv_o": self._bg(self._kn["matvec_wg"], w[p + "self_attn.o_proj.weight"],
                                 b["attn"], b["attn_proj"], d[f"mv_o_{tag}"]),
                "norm_pa_add": self._bg(self._kn["rmsnorm_add_wg"], b["attn_proj"],
                                        w[p + "post_attention_layernorm.weight"],
                                        b["x"], d["norm_h"]),
                "norm_pf": self._bg(self._kn["rmsnorm_wg"], b["x"],
                                    w[p + "pre_feedforward_layernorm.weight"],
                                    b["xn"], d["norm_h"]),
                "mv_gateup": self._bg(self._kn["matvec_wg"], w[p + "mlp.gateup_proj.weight"],
                                      b["xn"], gu, d[f"mv_gateup_{wtag}"]),
                "geglu": self._bg("geglu", (gu, 0, inter * 4), (gu, inter * 4, inter * 4),
                                  b["ffh"], d[f"geglu_{wtag}"]),
                "mv_down": self._bg(self._kn["matvec_wg"], w[p + "mlp.down_proj.weight"],
                                    b["ffh"], b["mlp_out"], d[f"mv_down_{wtag}"]),
                "norm_pff_add": self._bg(self._kn["rmsnorm_add_wg"], b["mlp_out"],
                                         w[p + "post_feedforward_layernorm.weight"],
                                         b["x"], d["norm_h"]),
            }
            if spec.kv_shared:
                bg["mv_q"] = self._bg(self._kn["matvec_wg"], w[p + "self_attn.q_proj.weight"],
                                      b["xn"], (qkv, 0, q_bytes), d[f"mv_q_{tag}"])
            else:
                bg["mv_qkv"] = self._bg(self._kn["matvec_wg"], w[p + "self_attn.qkv_proj.weight"],
                                        b["xn"], qkv, d[f"mv_qkv_{tag}"])
                bg["knorm"] = self._bg(self._kn["rmsnorm_wg"], (qkv, q_bytes, kv_bytes),
                                       w[p + "self_attn.k_norm.weight"],
                                       b["kn"], d[f"norm_k_{tag}"])
                bg["vnorm"] = self._bg("rmsnorm_ns_wg",
                                       (qkv, q_bytes + kv_bytes, kv_bytes),
                                       b["vn"], d[f"norm_k_{tag}"])
                bg["rope_k"] = self._bg("rope_pl", b["kn"], rope_f, d[f"rope_k_{tag}"])
                bg["app_k"] = self._bg("kv_append", b["kn"], self.k_cache[L],
                                       d[f"kv_append_{tag}"])
                bg["app_v"] = self._bg("kv_append", b["vn"], self.v_cache[L],
                                       d[f"kv_append_{tag}"])
            if self.ple:
                bg["mv_ple_gate"] = self._bg(
                    self._kn["matvec_wg"], w[p + "per_layer_input_gate.weight"],
                    b["x"], b["ple_g"], d["mv_ple_gate"])
                bg["geglu_ple"] = self._bg(
                    "geglu", b["ple_g"], (b["ple_in"], L * ple_bytes, ple_bytes),
                    b["ple_h"], d["geglu_ple"])
                bg["mv_ple_proj"] = self._bg(
                    self._kn["matvec_wg"], w[p + "per_layer_projection.weight"],
                    b["ple_h"], b["ple_proj"], d["mv_ple_proj"])
                bg["norm_ple"] = self._bg(
                    self._kn["rmsnorm_wg"], b["ple_proj"],
                    w[p + "post_per_layer_input_norm.weight"],
                    b["ple_norm"], d["norm_h"])
                bg["add_scale"] = self._bg(
                    "add_scale", b["ple_norm"], b["x"], self._fp_scalar[L], d["add"])
            self.layer_bgs.append(bg)

    # ------------------------------------------------------------------ #

    def _write_step_params(self, token_id: int, pos: int) -> None:
        cfg, q = self.cfg, self.device.queue
        kv_len = pos + 1
        start = max(0, kv_len - cfg.sliding_window)
        nh, ms = cfg.num_heads, self.max_seq
        hd_s = next(s.head_dim for s in cfg.layers if s.sliding)
        hd_f = next(s.head_dim for s in cfg.layers if not s.sliding)
        theta_s = next(s.rope_theta for s in cfg.layers if s.sliding)
        theta_f = next(s.rope_theta for s in cfg.layers if not s.sliding)

        def u32(buf, *vals):
            q.write_buffer(buf, 0, np.array(vals, dtype=np.uint32).tobytes())

        u32(self.b["token"], token_id)
        u32(self.d["kv_append_s"], hd_s, pos)
        u32(self.d["kv_append_f"], hd_f, pos)
        u32(self.d["scores_sliding"], nh, hd_s, kv_len, start, ms)
        u32(self.d["scores_full"], nh, hd_f, kv_len, 0, ms)
        q.write_buffer(self.d["rope_local"], 0,
                       np.array([theta_s, pos], dtype=np.float32).tobytes())
        q.write_buffer(self.d["rope_global"], 0,
                       np.array([theta_f, pos], dtype=np.float32).tobytes())
        if self.ple:
            row = bf16_to_f32(self._ple_raw[token_id]) * np.float32(cfg.ple_scale)
            q.write_buffer(self.b["ple_tok"], 0, row.tobytes())

    def _encode_forward(self, run, want_logits: bool, argmax: bool = False) -> None:
        """Encode one full decoder forward pass via the given run() callback."""
        cfg = self.cfg
        nh, h = cfg.num_heads, cfg.hidden_size
        ple_h = cfg.hidden_size_per_layer_input
        ple_n = cfg.num_layers * ple_h
        run(self._kn["embed_scale"], self.bg_embed, h, label="embed")
        if self.ple:
            run(self._kn["matvec_wg"], self.bg_ple_ctx, ple_n, label="mv_ple_ctx",
                grid=(ple_n, 1, 1))
            run(self._kn["rmsnorm_wg"], self.bg_ple_ctx_norm, ple_n, label="norm_ple_ctx",
                grid=(cfg.num_layers, 1, 1))
            run("combine_scaled", self.bg_ple_combine, ple_n, label="ple_combine")
        for spec in cfg.layers:
            bg = self.layer_bgs[spec.index]
            hd, q_dim, inter = spec.head_dim, spec.q_dim, spec.intermediate
            run(self._kn["rmsnorm_wg"], bg["norm1"], h, label="norm_input", grid=(1, 1, 1))
            if spec.kv_shared:
                run(self._kn["matvec_wg"], bg["mv_q"], q_dim, label="mv_qkv",
                    grid=(q_dim, 1, 1))
            else:
                run(self._kn["matvec_wg"], bg["mv_qkv"], q_dim + 2 * hd, label="mv_qkv",
                    grid=(q_dim + 2 * hd, 1, 1))
            run(self._kn["rmsnorm_wg"], bg["qnorm"], q_dim, label="qk_norm", grid=(nh, 1, 1))
            run("rope_pl", bg["rope_q"], nh * hd // 2, label="rope")
            if not spec.kv_shared:
                run(self._kn["rmsnorm_wg"], bg["knorm"], hd, label="qk_norm", grid=(1, 1, 1))
                run("rmsnorm_ns_wg", bg["vnorm"], hd, label="v_norm", grid=(1, 1, 1))
                run("rope_pl", bg["rope_k"], hd // 2, label="rope")
                run("kv_append", bg["app_k"], hd, label="kv_append")
                run("kv_append", bg["app_v"], hd, label="kv_append")
            run(self._kn["attention_fused_g4"], bg["attn"], nh * 64, label="attn_fused",
                grid=(nh, 1, 1))
            run(self._kn["matvec_wg"], bg["mv_o"], h, label="mv_o", grid=(h, 1, 1))
            run(self._kn["rmsnorm_add_wg"], bg["norm_pa_add"], h, label="norm_post_add",
                grid=(1, 1, 1))
            run(self._kn["rmsnorm_wg"], bg["norm_pf"], h, label="norm_pre_ff", grid=(1, 1, 1))
            run(self._kn["matvec_wg"], bg["mv_gateup"], 2 * inter, label="mv_gateup",
                grid=(2 * inter, 1, 1))
            run("geglu", bg["geglu"], inter, label="geglu")
            run(self._kn["matvec_wg"], bg["mv_down"], h, label="mv_down", grid=(h, 1, 1))
            run(self._kn["rmsnorm_add_wg"], bg["norm_pff_add"], h, label="norm_post_add",
                grid=(1, 1, 1))
            if self.ple:
                run(self._kn["matvec_wg"], bg["mv_ple_gate"], ple_h, label="mv_ple",
                    grid=(ple_h, 1, 1))
                run("geglu", bg["geglu_ple"], ple_h, label="geglu")
                run(self._kn["matvec_wg"], bg["mv_ple_proj"], h, label="mv_ple",
                    grid=(h, 1, 1))
                run(self._kn["rmsnorm_wg"], bg["norm_ple"], h, label="norm_ple", grid=(1, 1, 1))
                run("add_scale", bg["add_scale"], h, label="add_scale")

        if want_logits or argmax:
            run(self._kn["rmsnorm_wg"], self.bg_final_norm, h, label="norm_final", grid=(1, 1, 1))
            for rows, bg in self.bg_logits_chunks:
                run(self._kn["matvec_wg"], bg, rows, label="mv_logits", grid=(rows, 1, 1))
        if argmax:  # softcap is argmax-invariant — skip it on this path
            run("argmax_stage1", self.bg_argmax1, self._n_argmax_wgs * 64,
                label="argmax", grid=(self._n_argmax_wgs, 1, 1))
            run("argmax_stage2", self.bg_argmax2, 64, label="argmax", grid=(1, 1, 1))
        elif want_logits:
            run("softcap", self.bg_softcap, cfg.vocab_size, label="softcap")

    def step(self, token_id: int, pos: int, want_logits: bool = True,
             argmax: bool = False) -> np.ndarray | int | None:
        """Run one decoder forward pass on the GPU for `token_id` at `pos`.

        argmax=True runs the two-stage GPU argmax and returns the winning
        token id (a 4-byte readback instead of the 1MB logits).
        """
        if pos >= self.max_seq:
            raise ValueError(f"position {pos} exceeds max_seq {self.max_seq}")
        kv_len = pos + 1
        t_step = time.perf_counter()
        self._write_step_params(token_id, pos)
        t0 = time.perf_counter()

        enc = self.device.create_command_encoder()
        cp = None if self.profile else enc.begin_compute_pass()
        pass_labels: list[tuple[str, int]] = []  # (label, n_threads)

        def run(name, bg, n_threads, wg=64, label=None, grid=None):
            nonlocal cp
            pipeline, _ = self._pipelines[name]
            if self.profile:
                i = len(pass_labels) * 2
                cp = enc.begin_compute_pass(timestamp_writes={
                    "query_set": self._qs,
                    "beginning_of_pass_write_index": i,
                    "end_of_pass_write_index": i + 1,
                })
                pass_labels.append((label or name, n_threads))
            cp.set_pipeline(pipeline)
            cp.set_bind_group(0, bg)
            if grid is not None:
                cp.dispatch_workgroups(*grid)
            else:
                cp.dispatch_workgroups((n_threads + wg - 1) // wg, 1, 1)
            if self.profile:
                cp.end()

        self._encode_forward(run, want_logits and not argmax, argmax)

        if self.profile:
            enc.resolve_query_set(self._qs, 0, len(pass_labels) * 2,
                                  self._q_resolve, 0)
        else:
            cp.end()
        t1 = time.perf_counter()
        self.device.queue.submit([enc.finish()])
        t2 = time.perf_counter()

        logits = None
        if argmax:
            raw = self.device.queue.read_buffer(self.b["token"])
            logits = int(np.frombuffer(raw, dtype=np.uint32)[0])
        elif want_logits:
            raw = self.device.queue.read_buffer(self.b["logits"])
            logits = np.frombuffer(raw, dtype=np.float32).copy()
        t3 = time.perf_counter()

        if self.profile:
            raw = self.device.queue.read_buffer(self._q_resolve)
            ts = np.frombuffer(raw, dtype=np.uint64)[: len(pass_labels) * 2]
            for j, (label, n_threads) in enumerate(pass_labels):
                dur = int(ts[2 * j + 1] - ts[2 * j])
                rec = self._prof_kernels[label]
                rec[0] += 1
                rec[1] += dur
                rec[2] = n_threads
        ph = self._prof_phases
        ph["params_write"] += t0 - t_step
        ph["encode"] += t1 - t0
        ph["submit"] += t2 - t1
        ph["logits_readback"] += t3 - t2
        self._prof_steps.append((kv_len, time.perf_counter() - t_step, want_logits))
        return logits

    def read_hidden(self) -> np.ndarray:
        """Read back the current hidden state x (for verification)."""
        raw = self.device.queue.read_buffer(self.b["x"])
        return np.frombuffer(raw, dtype=np.float32).copy()

    def profile_report(self) -> str:
        """Human-readable profiling summary (requires profile=True)."""
        if not self.profile:
            raise RuntimeError("construct with profile=True to collect timings")
        if not self._prof_steps:
            return "no steps profiled"
        total_ns = sum(rec[1] for rec in self._prof_kernels.values())
        lines = [f"== GPU time by call-site ({len(self._prof_steps)} steps) =="]
        lines.append(f"{'call-site':<14}{'count':>7}{'mean µs':>9}"
                     f"{'total ms':>10}{'share':>7}{'threads':>9}")
        for label, (count, ns, threads) in sorted(
                self._prof_kernels.items(), key=lambda kv: -kv[1][1]):
            share = 100.0 * ns / total_ns if total_ns else 0.0
            lines.append(f"{label:<14}{count:>7}{ns / count / 1e3:>9.1f}"
                         f"{ns / 1e6:>10.2f}{share:>6.1f}%{threads:>9}")
        wall = sum(s[1] for s in self._prof_steps)
        lines.append(f"GPU busy: {total_ns / 1e6:.1f} ms | step wall: "
                     f"{wall * 1e3:.1f} ms")
        lines.append("== CPU phases (totals) ==")
        for phase, sec in sorted(self._prof_phases.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {phase:<32}{sec * 1e3:>9.1f} ms")
        wbytes = sum(b.size for b in self._wbuf.values())
        lines.append(f"weights on GPU: {wbytes / 1e9:.2f} GB")
        return "\n".join(lines)

    @staticmethod
    def _pick(logits: np.ndarray, temperature: float, top_k: int,
              rng: np.random.Generator) -> int:
        if temperature <= 0.0:
            return int(np.argmax(logits))
        scaled = logits.astype(np.float64) / temperature
        if top_k > 0:
            kth = np.partition(scaled, -top_k)[-top_k]
            scaled = np.where(scaled < kth, -np.inf, scaled)
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        return int(rng.choice(len(probs), p=probs))

    def _generate_resident(self, prompt_ids: list[int], max_new_tokens: int,
                           on_token, chunk: int = 64) -> list[int]:
        """Greedy decode fully on-GPU: step params, PLE gather, token
        feedback, and argmax never leave the device; the CPU checks for
        EOS once per chunk."""
        q = self.device.queue
        # prefill all but the last prompt token (no logits needed)
        for pos, tid in enumerate(prompt_ids[:-1]):
            self.step(tid, pos, want_logits=False)

        # seed resident state: the last prompt token produces generation[0].
        # _write_step_params also fills the static dims fields (head dims,
        # rope thetas, max_seq) that step_setup_g4 never touches.
        start_pos = len(prompt_ids) - 1
        self._write_step_params(prompt_ids[-1], start_pos)
        q.write_buffer(self.b["pos"], 0,
                       np.array([start_pos], dtype=np.uint32).tobytes())
        q.write_buffer(self.b["counter"], 0, np.zeros(1, dtype=np.uint32).tobytes())

        cfg = self.cfg
        emitted = 0
        out: list[int] = []
        produced = 0
        budget = min(max_new_tokens, self.max_seq - start_pos)

        def encode_chunk(k: int):
            enc = self.device.create_command_encoder()
            cp = enc.begin_compute_pass()

            def run(name, bg, n_threads, wg=64, label=None, grid=None):
                pipeline, _ = self._pipelines[name]
                cp.set_pipeline(pipeline)
                cp.set_bind_group(0, bg)
                if grid is not None:
                    cp.dispatch_workgroups(*grid)
                else:
                    cp.dispatch_workgroups((n_threads + wg - 1) // wg, 1, 1)

            ple_n = cfg.num_layers * cfg.hidden_size_per_layer_input
            for _ in range(k):
                run("step_setup_g4", self.bg_setup, 1, grid=(1, 1, 1))
                run("ple_gather_f16", self.bg_ple_gather, ple_n)
                self._encode_forward(run, want_logits=False, argmax=True)
            cp.end()
            return enc.finish()

        # Chunk sizes ramp 8→chunk so short answers don't pay for a full
        # chunk of speculative forward passes. Encoding is pipelined: while
        # the GPU runs chunk N, the CPU encodes chunk N+1.
        sizes: list[int] = []
        c, left = min(8, chunk), budget
        while left > 0:
            k = min(c, left)
            sizes.append(k)
            left -= k
            c = min(c * 2, chunk)

        t0 = time.perf_counter()
        cmd = encode_chunk(sizes[0])
        t1 = time.perf_counter()
        self._prof_phases["encode"] += t1 - t0
        q.submit([cmd])

        for i, k in enumerate(sizes):
            t1 = time.perf_counter()
            next_cmd = encode_chunk(sizes[i + 1]) if i + 1 < len(sizes) else None
            t2 = time.perf_counter()
            produced += k
            raw = q.read_buffer(self.b["out_tokens"], size=produced * 4)
            toks = np.frombuffer(raw, dtype=np.uint32)
            t3 = time.perf_counter()
            ph = self._prof_phases
            ph["encode"] += t2 - t1
            ph["chunk_submit_readback"] += t3 - t2
            self._prof_steps.append((start_pos + produced, (t3 - t1) / k, True))

            stop = False
            for tid in toks[emitted:]:
                tid = int(tid)
                out.append(tid)
                if on_token is not None:
                    on_token(tid)
                emitted += 1
                if tid in cfg.eos_token_ids:
                    stop = True
                    break
            if stop or next_cmd is None:
                break
            q.submit([next_cmd])
        return out

    def generate(self, prompt_ids: list[int], max_new_tokens: int = 64,
                 on_token=None, temperature: float = 0.0, top_k: int = 64,
                 seed: int | None = None) -> list[int]:
        """Generate token ids (without prompt). temperature<=0 → greedy.

        CPU-driven per-step decode: the PLE gather for each generated token
        happens host-side, so tokens round-trip through the CPU. Greedy
        decode picks on-GPU (two-stage argmax) and reads back 4 bytes.
        """
        if temperature <= 0.0 and self.resident and not self.profile:
            # fully GPU-resident chunks (per-kernel profiling needs the
            # per-step path, so profile mode falls through)
            return self._generate_resident(prompt_ids, max_new_tokens, on_token)

        if temperature <= 0.0:
            self.device.queue.write_buffer(
                self.b["counter"], 0, np.zeros(1, dtype=np.uint32).tobytes())
            next_id = None
            for pos, tid in enumerate(prompt_ids):
                last = pos == len(prompt_ids) - 1
                res = self.step(tid, pos, want_logits=False, argmax=last)
                if last:
                    next_id = res
            out: list[int] = []
            pos = len(prompt_ids)
            while len(out) < max_new_tokens:
                out.append(next_id)
                if on_token is not None:
                    on_token(next_id)
                if next_id in self.cfg.eos_token_ids or pos >= self.max_seq:
                    break
                next_id = self.step(next_id, pos, want_logits=False, argmax=True)
                pos += 1
            return out

        rng = np.random.default_rng(seed)
        logits = None
        for pos, tid in enumerate(prompt_ids):
            last = pos == len(prompt_ids) - 1
            logits = self.step(tid, pos, want_logits=last)

        out: list[int] = []
        pos = len(prompt_ids)
        for _ in range(max_new_tokens):
            next_id = self._pick(logits, temperature, top_k, rng)
            out.append(next_id)
            if on_token is not None:
                on_token(next_id)
            if next_id in self.cfg.eos_token_ids:
                break
            if pos >= self.max_seq:
                break  # KV cache full — stop cleanly
            logits = self.step(next_id, pos)
            pos += 1
        return out
