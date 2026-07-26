"""GPU runner for the gemma4_150 fast-path port (wgpu-py).

Reads the exported manifest + weights.bin, uploads every buffer, and drives the
webml reference fused SRQ kernels (templated per layer shape) to decode Gemma 4
E2B. Correctness gate: argmax == gemma4_150.reference (numpy SRQ oracle).

wgpu-native/naga compiles the reference kernels once the `enable subgroups;`
directive is stripped (the ops are supported; same shim as gemma4/runner.py).

This is built up stage by stage; each stage validates a GPU intermediate against
reference.py before the next is wired.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import wgpu

REF_DIR = Path(__file__).resolve().parent.parent / "reference" / "webml_gemma4_kernels"
_BIND = re.compile(r"@group\(0\)\s*@binding\((\d+)\)\s*var<(storage|uniform)(?:,\s*(read|read_write))?>")


@lru_cache(maxsize=None)
def _kernel(name: str) -> str:
    return (REF_DIR / f"{name}.wgsl").read_text()


_ENTRY = re.compile(r"@compute[\s\S]*?fn\s+(\w+)\s*\(")


def _entry(wgsl: str) -> str:
    """Entry-point name. The captured kernels all use `main`; DSL-generated ones
    use the Python function name, so read it out rather than assuming."""
    m = _ENTRY.search(wgsl)
    return m.group(1) if m else "main"


def _access(wgsl: str):
    """Ordered per-binding kind: 'r' read-only storage, 'w' read_write, 'u' uniform."""
    slots = {}
    for m in _BIND.finditer(wgsl):
        b, cls, acc = int(m.group(1)), m.group(2), m.group(3)
        slots[b] = "u" if cls == "uniform" else ("w" if acc == "read_write" else "r")
    return [slots[i] for i in range(len(slots))]


class G4Runner:
    def __init__(self, model_dir=None):
        base = Path(model_dir) if model_dir else \
            Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B-qat"
        self.dir = base / "g4_150"
        self.man = json.loads((self.dir / "manifest.json").read_text())
        self.cfg = self.man["config"]
        self.blob = np.memmap(self.dir / "weights.bin", np.uint8, "r")

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        feats = [f for f in ("subgroup", "shader-f16") if f in adapter.features]
        big = 1 << 31   # 2GB: covers the 1.17GB ple_q buffer
        self.device = adapter.request_device_sync(
            required_features=feats,
            required_limits={"max-buffer-size": big, "max-storage-buffer-binding-size": big,
                             "max-storage-buffers-per-shader-stage": 10})
        self.queue = self.device.queue
        self._pipes: dict[str, tuple] = {}
        self._bufs: dict[str, object] = {}
        self._upload_all()

    # ---- buffers ----
    def _upload_all(self):
        st = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        pending = 0
        for name, t in self.man["tensors"].items():
            data = bytes(self.blob[t["off"]:t["off"] + t["len"]])
            self._bufs[name] = self.device.create_buffer_with_data(data=data, usage=st)
            pending += t["len"]
            if pending >= (1 << 28):        # flush every 256MB (staging pitfall)
                self.queue.submit([])
                pending = 0
        self.queue.submit([])

    def _tmp(self, nbytes, src=True):
        u = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        if src:
            u |= wgpu.BufferUsage.COPY_SRC
        return self.device.create_buffer(size=max(nbytes, 4), usage=u)

    def _uniform(self, arr: np.ndarray):
        return self.device.create_buffer_with_data(
            data=arr.tobytes(), usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    # --- persistent pools: buffers are stable objects so bind groups cache (resident path) ---
    _pool: dict = None
    _unis: dict = None
    _bgcache: dict = None

    def _scratch(self, name, nbytes):
        if self._pool is None:
            self._pool = {}
        b = self._pool.get(name)
        if b is None or b.size < nbytes:
            b = self._tmp(max(nbytes, 4))
            self._pool[name] = b
        return b

    def _uni(self, name, arr):
        if self._unis is None:
            self._unis = {}
        data = np.ascontiguousarray(arr).tobytes()
        b = self._unis.get(name)
        if b is None or b.size < len(data):
            # STORAGE as well as UNIFORM: a couple of the DSL kernels take their
            # 8-word params struct as a read-only storage binding (vec4 uniforms
            # only hold four words), and the same buffer serves both.
            b = self.device.create_buffer(
                size=max(len(data), 16),
                usage=(wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.STORAGE
                       | wgpu.BufferUsage.COPY_DST))
            self._unis[name] = b
        self.queue.write_buffer(b, 0, data)
        return b

    # ---- pipelines (templated: patch baked consts before compile) ----
    def _shader(self, code: str):
        try:
            return self.device.create_shader_module(code=code)
        except Exception:
            return self.device.create_shader_module(code=code.replace("enable subgroups;", ""))

    def _pipe(self, key: str, code: str):
        if key not in self._pipes:
            acc = _access(code)
            entries = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": wgpu.BufferBindingType.uniform if a == "u"
                                   else wgpu.BufferBindingType.storage if a == "w"
                                   else wgpu.BufferBindingType.read_only_storage}}
                       for i, a in enumerate(acc)]
            layout = self.device.create_bind_group_layout(entries=entries)
            pipe = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[layout]),
                compute={"module": self._shader(code), "entry_point": _entry(code)})
            self._pipes[key] = (pipe, layout)
        return self._pipes[key]

    _enc = None

    def dispatch(self, key, code, buffers, grid, bgkey=None):
        pipe, layout = self._pipe(key, code)
        if bgkey is not None and self._bgcache is None:
            self._bgcache = {}
        bg = self._bgcache.get(bgkey) if bgkey is not None else None
        if bg is None:
            bg = self.device.create_bind_group(layout=layout, entries=[
                {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
                for i, b in enumerate(buffers)])
            if bgkey is not None:
                self._bgcache[bgkey] = bg
        enc = self._enc or self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(*grid)
        cp.end()
        if self._enc is None:
            self.queue.submit([enc.finish()])

    def read(self, buf, nbytes) -> np.ndarray:
        rb = self.device.create_buffer(
            size=nbytes, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
        enc = self.device.create_command_encoder()
        enc.copy_buffer_to_buffer(buf, 0, rb, 0, nbytes)
        self.queue.submit([enc.finish()])
        rb.map_sync(wgpu.MapMode.READ)
        out = np.frombuffer(rb.read_mapped(), np.uint8).copy()
        rb.unmap()
        return out

    def _ids_buf(self, token_id: int):
        b = self._scratch("ids", 4)
        self.queue.write_buffer(b, 0, np.array([token_id], np.uint32).tobytes())
        return b

    # ---- stage 1: embed gather (kernel 00) ----
    def embed(self, token_id: int, out=None, ids=None) -> object:
        H = self.cfg["H"]
        y = out if out is not None else self._scratch("hidden", H * 4)
        idb = ids if ids is not None else self._ids_buf(token_id)   # resident: read a GPU token
        par = self._uni("embed_par", np.array([1, 0, 0, 0], np.uint32))
        self.dispatch("embed", self._k("00_main", "embed_00", (64, 1, 1)),
                      [idb, self._bufs["embed_q"], self._bufs["embed_scale"], y, par],
                      (1, 1, 1), bgkey=("embed", id(y), id(idb)))
        return y

    # ---- stage 2: PLE input (kernel 68 proj + kernel 01 gather + combine) ----
    _COMBINE = """
const D:u32=256u; const HINV:f32=%.9ef; const EPS:f32=1e-6; const RS2:f32=0.7071067811865476;
@group(0) @binding(0) var<storage, read> ctx: array<f32>;
@group(0) @binding(1) var<storage, read> ple: array<f32>;
@group(0) @binding(2) var<storage, read> nw: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
var<workgroup> red: array<f32, D>;
@compute @workgroup_size(D,1,1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x; let tid = lid.x; let base = row*D + tid;
  let c = ctx[base]*HINV;
  red[tid] = c*c; workgroupBarrier();
  var s:u32 = D/2u; loop { if (s==0u){break;} if (tid<s){red[tid]=red[tid]+red[tid+s];} s=s/2u; workgroupBarrier(); }
  let rms = inverseSqrt(red[0]/f32(D)+EPS);
  outp[base] = (c*rms*nw[tid] + ple[base])*RS2;
}"""

    def ple_input(self, token_id: int, embed_buf, ids=None) -> object:
        H, nL, d = self.cfg["H"], self.cfg["nL"], self.cfg["ple_d"]
        ctx = self._scratch("ctx", nL * d * 4)
        ple = self._scratch("plegath", nL * d * 4)
        idb = ids if ids is not None else self._ids_buf(token_id)
        par0 = self._uni("ple_par0", np.array([0, 0, 0, 0], np.float32))   # kernel 68 inScale/outScale
        seq1 = self._uni("ple_seq1", np.array([1, 0, 0, 0], np.uint32))    # kernel 01 seq=1
        self.dispatch("proj68", self._k("68_reduce", "proj_68", (32, 1, 1)),
                      [embed_buf, self._bufs["pl_model_proj"], ctx, par0], (nL * d // 8, 1, 1),
                      bgkey=("proj68", id(embed_buf)))
        self.dispatch("plegather", self._k("01_main", "plegather_01", (64, 1, 1)),
                      [idb, self._bufs["ple_q"], self._bufs["ple_scale"], ple, seq1],
                      (1, 1, 1), bgkey=("plegather", id(idb)))
        out = self._scratch("ple", nL * d * 4)
        code = (self._k(None, "combine", (256, 1, 1)) if self.USE_DSL
                else self._COMBINE % (H ** -0.5))
        self.dispatch("combine", code, [ctx, ple, self._bufs["pl_proj_norm"], out], (nL, 1, 1),
                      bgkey=("combine",))
        return out

    # ---- stage 3: one decoder layer ----
    # Fused k/v cache write (replaces reference k71 + a separate v-norm): k gets
    # weighted RMSNorm + split-half RoPE, v gets scale-free RMSNorm, both -> cache.
    _KVNORM = """
const HD:u32=%du; const HALF:u32=%du; const EPS:f32=1e-6;
@group(0) @binding(0) var<storage, read> ink: array<f32>;
@group(0) @binding(1) var<storage, read> inv: array<f32>;
@group(0) @binding(2) var<storage, read> knorm: array<f32>;
@group(0) @binding(3) var<storage, read> cosT: array<f32>;
@group(0) @binding(4) var<storage, read> sinT: array<f32>;
@group(0) @binding(5) var<storage, read_write> kcache: array<f32>;
@group(0) @binding(6) var<storage, read_write> vcache: array<f32>;
@group(0) @binding(7) var<uniform> p: vec4<u32>;   // p.x = dstOffset
var<workgroup> rk: array<f32, HD>;
var<workgroup> rv: array<f32, HD>;
@compute @workgroup_size(HD,1,1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let ko = ink[tid]; let vo = inv[tid];
  rk[tid] = ko*ko; rv[tid] = vo*vo; workgroupBarrier();
  var s:u32 = HD/2u; loop { if(s==0u){break;} if(tid<s){rk[tid]=rk[tid]+rk[tid+s]; rv[tid]=rv[tid]+rv[tid+s];} s=s/2u; workgroupBarrier(); }
  let rmsk = inverseSqrt(rk[0]/f32(HD)+EPS);
  let rmsv = inverseSqrt(rv[0]/f32(HD)+EPS);
  vcache[p.x + tid] = vo * rmsv;
  if (tid < HALF) {
    let n0 = ink[tid]*rmsk*knorm[tid];
    let n1 = ink[tid+HALF]*rmsk*knorm[tid+HALF];
    let c = cosT[tid]; let sn = sinT[tid];
    kcache[p.x + tid] = n0*c - n1*sn;
    kcache[p.x + tid + HALF] = n1*c + n0*sn;
  }
}"""

    def _write_step_uniforms(self, pos):
        """Write the per-token dynamic uniforms ONCE per head-dim type (not per layer):
        rope cos/sin, attention params (pos), and the k/v-cache dstOffset."""
        nH, win = self.cfg["nH"], self.cfg["window"]
        if not hasattr(self, "_rope_cfgs"):
            self._rope_cfgs = {}
            for s in self.man["layers"]:
                self._rope_cfgs[s["head_dim"]] = (s["rope_theta"], s["rope_cutoff"], s["sliding"])
        for hd, (theta, cutoff, sliding) in self._rope_cfgs.items():
            half = hd // 2
            inv = 1.0 / theta ** (np.arange(half, dtype=np.float64) / half)
            inv[cutoff:] = 0.0
            ang = pos * inv
            self.queue.write_buffer(self._scratch(f"rcos{half}", half * 4), 0, np.cos(ang).astype(np.float32).tobytes())
            self.queue.write_buffer(self._scratch(f"rsin{half}", half * 4), 0, np.sin(ang).astype(np.float32).tobytes())
            self._uni(f"parA_{hd}", np.array([1, pos + 1, pos, nH, 1, win if sliding else 0, 0, 0], np.uint32))
            self._uni(f"pkv_{hd}", np.array([pos * hd, 0, 0, 0], np.uint32))

    # Kernel source: "dsl" compiles from gemma4_150/kernels_dsl.py (the same
    # Python source the Metal runner uses); "ref" uses the captured webml WGSL.
    USE_DSL = os.environ.get("G4_KERNEL_SOURCE", "dsl") != "ref"

    def _k(self, ref, dsl, threads=None, consts=None, patch=None):
        """WGSL for a kernel, from the DSL when ported, else the captured file."""
        if self.USE_DSL:
            from gemma4_150 import kernels_dsl
            from py_shader_lang_wgpu import translate
            fn = getattr(kernels_dsl, dsl, None)
            if fn is not None:
                return translate(fn, workgroup_size=threads, consts=consts)
        code = _kernel(ref)
        return self._patch(code, **patch) if patch else code

    @lru_cache(maxsize=None)
    def _patch(self, code, **consts):
        for k, val in consts.items():
            code = re.sub(rf"const {k}: (u32|f32) = [^;]+;",
                          lambda m: f"const {k}: {m.group(1)} = {val}"
                          + ("u" if m.group(1) == "u32" else "") + ";", code)
        return code

    def _uni_static(self, name, arr):
        """Persistent uniform written once (scale params that don't change per token)."""
        if self._unis is None:
            self._unis = {}
        if name not in self._unis:
            return self._uni(name, arr)
        return self._unis[name]

    def setup_caches(self, max_seq=2048):
        self.max_seq = max_seq
        self.kc, self.vc = {}, {}
        for s in self.man["layers"]:
            if not s["shared"]:
                hd = s["head_dim"]
                self.kc[s["index"]] = self._tmp(max_seq * hd * 4, src=False)
                self.vc[s["index"]] = self._tmp(max_seq * hd * 4, src=False)
        # pre-size pooled scratch at maxima so bind groups never reallocate
        H, d, nL, V = self.cfg["H"], self.cfg["ple_d"], self.cfg["nL"], self.cfg["vocab"]
        for name, n in [("hidden", H * 4), ("a", H * 4), ("outq", 4096 * 4), ("attn", 4096 * 4),
                        ("outk", 512 * 4), ("outv", 512 * 4), ("dummy", 512 * 4),
                        ("y2", H * 4), ("y2n", H * 4), ("geglu", 12288 * 2), ("gate", d * 4),
                        ("ctx", nL * d * 4), ("plegath", nL * d * 4), ("ple", nL * d * 4),
                        ("normed", H * 4), ("logits", V * 4), ("cv", 256 * 4), ("ci", 256 * 4),
                        ("suma", 4), ("sum2", 4), ("sum2n", 4), ("sa", 4), ("ids", 4)]:
            self._scratch(name, n)
        # atomic scratch: WebGPU zero-inits + the kernels self-reset the ticket, so zero once
        for name, n in [("pp73", (H + 1) * 4), ("pp75", (H + 1) * 4), ("pp77", (H + 1) * 4),
                        ("partials256", (8 * 32 * 258 + 8) * 4), ("partials512", (8 * 32 * 514 + 8) * 4)]:
            bz = self._scratch(name, n)
            self.queue.write_buffer(bz, 0, np.zeros(n // 4, np.uint32).tobytes())

    def layer(self, L, pos, hidden, ple_buf, ple_off):
        """Run decoder layer L in place on `hidden` (f32 [H])."""
        s = self.man["layers"][L]
        sc = s["scales"]
        H, nH = self.cfg["H"], self.cfg["nH"]
        hd, qd, inter = s["head_dim"], s["q_dim"], s["intermediate"]
        half, cutoff, shared = hd // 2, s["rope_cutoff"], s["shared"]
        kc, vc = self.kc[s["kv_source"]], self.vc[s["kv_source"]]
        b = self._bufs
        cb, sb = self._scratch(f"rcos{half}", half * 4), self._scratch(f"rsin{half}", half * 4)  # written per-token
        hk = (L, id(hidden), id(ple_buf))    # bind-group cache suffix (hidden/ple vary by caller)

        outq, dummy = self._scratch("outq", qd * 4), self._scratch("dummy", hd * 4)
        outk, outv = self._scratch("outk", hd * 4), self._scratch("outv", hd * 4)
        attn = self._scratch("attn", qd * 4)
        y2, sum2 = self._scratch("y2", H * 4), self._scratch("sum2", 4)
        geglu, gate = self._scratch("geglu", inter * 2), self._scratch("gate", self.cfg["ple_d"] * 4)
        y2n, sum2n = self._scratch("y2n", H * 4), self._scratch("sum2n", 4)
        pp73, pp75, pp77 = self._scratch("pp73", (H + 1) * 4), self._scratch("pp75", (H + 1) * 4), self._scratch("pp77", (H + 1) * 4)
        # sliding (hd=256) and full (hd=512) attention put their self-reset counter at
        # different offsets; a full layer's value writes would clobber a sliding counter,
        # so keep separate partials buffers per head-dim.
        partials = self._scratch(f"partials{hd}", (8 * 32 * (hd + 2) + 8) * 4)

        # 69: input norm + srq -> a[H], sum_a. Only layer 0 needs it; layers 1+ reuse the
        # PREVIOUS layer's k77 outputs (y2n/sum2n), which already = srq(this-layer input
        # norm(x), this-layer qkv_in) + its sum — the reference's cross-layer fusion.
        if L == 0:
            a, suma = self._scratch("a", H * 4), self._scratch("suma", 4)
            self.dispatch("k69", self._k("69_sg_sum", "rmssrq_69", (256, 1, 1)),
                          [hidden, b[f"L{L}.in_norm"], a, suma,
                           self._uni_static(f"p69_{L}", np.array([1, 0, np.float32(sc["qkv_in"]).view(np.uint32), 0], np.uint32))],
                          (1, 1, 1), bgkey=("k69",)+hk)
        else:
            a, suma = self._scratch("y2n", H * 4), self._scratch("sum2n", 4)
        # 70: qkv (shared: q-only). KV head_dim == q head_dim -> patch KV_OUT/KV_WGS too.
        total = qd // 2 + hd
        k70 = self._k("70_srq", "qkv_70", (32, 1, 1),
                      consts={"Q_OUT": qd, "KV_OUT": hd, "Q_WGS": qd // 2,
                              "KV_WGS": hd // 2, "TOTAL_WGS": total, "GRID_X": total},
                      patch={"Q_OUT": qd, "Q_WGS": qd // 2, "KV_OUT": hd,
                             "KV_WGS": hd // 2, "TOTAL_WGS": total, "GRID_X": total})
        par70 = self._uni_static(f"p70_{L}", np.array([sc["q_out"], sc.get("k_out", 0), sc.get("v_out", 0), 0], np.float32))
        if shared:
            self.dispatch(f"k70_{qd}_{hd}", k70,
                          [a, b[f"L{L}.q_bits"], b[f"L{L}.q_bits"], b[f"L{L}.q_bits"],
                           b[f"L{L}.q_scale"], suma, outq, dummy, dummy, par70], (qd // 2, 1, 1), bgkey=("k70",)+hk)
        else:
            self.dispatch(f"k70_{qd}_{hd}", k70,
                          [a, b[f"L{L}.q_bits"], b[f"L{L}.k_bits"], b[f"L{L}.v_bits"],
                           b[f"L{L}.qkv_scales"], suma, outq, outk, outv, par70], (total, 1, 1), bgkey=("k70",)+hk)
            # fused k-norm+rope + v-norm -> caches (one dispatch; dstOffset shared per type)
            kvn = (self._k(None, "kvnorm", (hd, 1, 1), consts={"HD": hd, "HALF": half})
                   if self.USE_DSL else self._KVNORM % (hd, half))
            self.dispatch(f"kvnorm_{hd}", kvn,
                          [outk, outv, b[f"L{L}.k_norm"], cb, sb, kc, vc,
                           self._unis[f"pkv_{hd}"]], (1, 1, 1), bgkey=("kvn",)+hk)
        # attention (patch HEAD_DIM/HALF/OUT_Q); parA has pos -> dynamic
        att = self._k("101_srq", "attn_101", (256, 1, 1),
                      consts={"HEAD_DIM": hd, "HALF_DIM": half, "HD4": hd // 4,
                              "J_GROUPS": 256 // (hd // 4),
                              "PP_COUNTER_BASE": 8 * 32 * (hd + 2), "OUT_Q": sc["o_in"]},
                      patch={"HEAD_DIM": hd, "HALF_DIM": half})
        if not self.USE_DSL:
            att = att.replace("const OUT_Q: f32 = 0.014886821620166302;",
                              f"const OUT_Q: f32 = {sc['o_in']!r};")
        self.dispatch(f"att_{hd}_{sc['o_in']}", att,
                      [outq, b[f"L{L}.q_norm"], cb, sb, kc, vc, partials, attn, self._unis[f"parA_{hd}"]],
                      (nH, 32, 1), bgkey=("att",)+hk)
        # 73: o-proj + post-attn norm-add + pre-ffn norm
        k73 = self._k("73_sg_sum", "oproj_73", (256, 1, 1), consts={"WPR": qd // 8},
                      patch={"IN_FEATURES": qd, "WORDS_PER_ROW": qd // 8})
        par73 = self._uni_static(f"p73_{L}", np.array([sc["o_out"], sc["gate_in"], 0, 0], np.float32))
        self.dispatch(f"k73_{qd}", k73, [attn, b[f"L{L}.o_bits"], b[f"L{L}.o_scale"], pp73,
                      hidden, b[f"L{L}.o_w12"], y2, sum2, par73], (192, 1, 1), bgkey=("k73",)+hk)
        # 74/95: gate/up geglu (4-bit kernel 74 / 2-bit double-wide kernel 95)
        gu_kern, gu_grid = ("74_sg_sum", 768) if inter == 6144 else ("95_sg_sum", 3072)
        par74 = self._uni_static(f"p74_{L}", np.array([sc["gate_out"], sc["up_out"], sc["down_in"], 0], np.float32))
        gu_src = self._k(gu_kern, "gateup_74" if inter == 6144 else "gateup_95", (64, 1, 1))
        self.dispatch(f"gu_{inter}", gu_src, [y2, b[f"L{L}.gate_bits"], b[f"L{L}.gate_scale"],
                      b[f"L{L}.up_bits"], b[f"L{L}.up_scale"], sum2, geglu, b[f"L{L}.gelu_gate"], par74],
                      (gu_grid, 1, 1), bgkey=("gu",)+hk)
        # 75/96: down + post-ffn norm-add (4-bit -> 75, 2-bit -> 96)
        down_kern = "75_srq" if inter == 6144 else "96_srq"
        par75 = self._uni_static(f"p75_{L}", np.array([sc["down_in"], sc["down_out"], 0, 0], np.float32))
        down_src = self._k(down_kern, "down_75" if inter == 6144 else "down_96", (256, 1, 1))
        self.dispatch(f"down_{inter}", down_src, [geglu, b[f"L{L}.down_bits"], pp75,
                      b[f"L{L}.down_scale"], hidden, b[f"L{L}.down_nw"], par75], (H // 4, 1, 1), bgkey=("down",)+hk)
        # 76: PLE input gate (reads its 256-slice at ple_off; static param)
        parP = np.zeros(4, np.uint32)
        parP[:2] = np.array([sc["plegate_in"], sc["plegate_out"]], np.float32).view(np.uint32)
        parP[2] = ple_off
        self.dispatch("k76", self._k("76_reduce", "plegate_76", (32, 1, 1)), [hidden, b[f"L{L}.plegate_codes"],
                      b[f"L{L}.plegate_rowscale"], ple_buf, gate, b[f"L{L}.gelu_plegate"],
                      self._uni_static(f"p76_{L}", parP)], (self.cfg["ple_d"], 1, 1), bgkey=("k76",)+hk)
        # 77: PLE proj + residual*layer_scalar + next-layer norm
        nxt_in = self.man["layers"][L + 1]["scales"]["qkv_in"] if L + 1 < self.cfg["nL"] else 0.0
        par77 = self._uni_static(f"p77_{L}", np.array([nxt_in, sc["pleproj_in"], sc["pleproj_out"], 0], np.float32))
        self.dispatch("k77", self._k("77_sg_sum", "pleproj_77", (256, 1, 1)), [gate, b[f"L{L}.pleproj_codes"],
                      b[f"L{L}.pleproj_rowscale"], pp77, hidden, b[f"L{L}.pleproj_w12s"],
                      y2n, sum2n, par77], (96, 1, 1), bgkey=("k77",)+hk)
        return hidden

    # ---- stage 4: full forward ----
    def forward(self, token_id: int, pos: int, hidden=None, argmax=None,
                ids_buf=None, gen_ids=None, step=0):
        H, nL, d, vocab = self.cfg["H"], self.cfg["nL"], self.cfg["ple_d"], self.cfg["vocab"]
        if not hasattr(self, "kc"):
            self.setup_caches()
        hidden = hidden if hidden is not None else self._scratch("hidden", H * 4)
        logits = self._scratch("logits", vocab * 4)
        self._write_step_uniforms(pos)                     # per-type rope/params, once per token
        self._enc = self.device.create_command_encoder()   # batch the whole forward into one submit
        self.embed(token_id, out=hidden, ids=ids_buf)
        ple = self.ple_input(token_id, hidden, ids=ids_buf)
        for L in range(nL):
            self.layer(L, pos, hidden, ple, L * d)
        # final norm (kernel 69, srq passthrough inScale=0) -> normed
        normed, sa = self._scratch("normed", H * 4), self._scratch("sa", 4)
        self.dispatch("finalnorm", self._k("69_sg_sum", "rmssrq_69", (256, 1, 1)),
                      [hidden, self._bufs["final_norm"], normed, sa,
                       self._uni_static("pfn", np.array([1, 0, 0, 0], np.uint32))], (1, 1, 1),
                      bgkey=("finalnorm", id(hidden)))
        # logits (kernel 33), lm_head act scales are 0 -> weight-only
        self.dispatch("logits", self._k("33_srq", "logits_33", (128, 1, 1)),
                      [normed, self._bufs["lmhead_blk"], self._bufs["lmhead_scale"], logits,
                       self._uni_static("plog", np.array([0, 0, 0, 0], np.float32))], (vocab // 128, 1, 1),
                      bgkey=("logits",))
        if argmax:
            # GPU argmax (softcap is monotonic -> skipped): 34 -> 256 candidates -> 35 -> token
            cv, ci = self._scratch("cv", 256 * 4), self._scratch("ci", 256 * 4)
            self.dispatch("amax1", self._k("34_main", "argmax1_34", (256, 1, 1)), [logits, cv, ci], (256, 1, 1), bgkey=("amax1",))
            self.dispatch("amax2", self._k("35_main", "argmax2_35", (256, 1, 1)), [cv, ci, argmax], (1, 1, 1), bgkey=("amax2", id(argmax)))
            if gen_ids is not None:      # resident: append the token to the output buffer on-GPU
                self._enc.copy_buffer_to_buffer(argmax, 0, gen_ids, step * 4, 4)
        self.queue.submit([self._enc.finish()])
        self._enc = None
        return logits

    def generate_resident(self, ids, n_new=40, eos=1, chunk=16):
        """GPU-resident greedy decode: the argmax token feeds back into cur_tok on
        the GPU, so no CPU sync per token — the GPU runs async while the CPU records
        ahead. EOS is checked once per `chunk` tokens."""
        import time
        self.setup_caches()
        hidden = self._scratch("hidden", self.cfg["H"] * 4)
        cur = self._scratch("cur_tok", 4)             # the on-GPU current token (fed back)
        gen = self._scratch("gen_ids", n_new * 4)
        pos = 0
        for t in ids[:-1]:                            # prefill (CPU-known tokens)
            self.queue.write_buffer(cur, 0, np.array([t], np.uint32).tobytes())
            self.forward(int(t), pos, hidden, ids_buf=cur); pos += 1
        self.queue.write_buffer(cur, 0, np.array([ids[-1]], np.uint32).tobytes())
        out = []
        t0 = time.time()
        step = 0
        while step < n_new:
            for _ in range(min(chunk, n_new - step)):
                # cur holds the token; forward reads it, argmax writes the next into cur
                self.forward(0, pos, hidden, argmax=cur, ids_buf=cur, gen_ids=gen, step=step)
                pos += 1; step += 1
            got = self.read(gen, step * 4).view(np.uint32)   # one sync per chunk
            new = [int(x) for x in got[len(out):step]]
            for tok in new:
                if tok == eos:
                    dt = time.time() - t0
                    return out, step / dt if dt else 0.0
                out.append(tok)
        dt = time.time() - t0
        return out, step / dt if dt else 0.0

    def logits_np(self, token_id, pos, hidden=None):
        lg = self.forward(token_id, pos, hidden)
        return self.read(lg, self.cfg["vocab"] * 4).view(np.float32)

    def generate(self, ids, n_new=20, eos=1):
        """Greedy decode. Returns (new_ids, decode_tok_per_s)."""
        import time
        self.setup_caches()
        hidden = self._scratch("hidden", self.cfg["H"] * 4)
        pos = 0
        for t in ids[:-1]:                       # prefill all but last
            self.forward(t, pos, hidden); pos += 1
        cur = ids[-1]
        out = []
        t0 = time.time()
        for _ in range(n_new):
            # CPU argmax: in this CPU-driven loop the per-token sync dominates, so the
            # 1MB readback beats GPU argmax's 2 extra dispatches (GPU argmax is for the
            # future resident loop, where the token feeds back without a CPU sync).
            lg = self.read(self.forward(cur, pos, hidden), self.cfg["vocab"] * 4).view(np.float32)
            cur = int(np.argmax(lg)); pos += 1
            if cur == eos:
                break
            out.append(cur)
        dt = time.time() - t0
        return out, len(out) / dt if dt else 0.0


if __name__ == "__main__":
    import sys
    from gemma4_150.reference import ReferenceSRQ
    r = G4Runner()
    ref = ReferenceSRQ()
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    eb = r.embed(tid)
    g = r.read(eb, r.cfg["H"] * 4).view(np.float32)
    e = ref.embed(tid)
    print(f"embed[{tid}]: maxAbsDiff={np.abs(g - e).max():.3e}")

    pb = r.ple_input(tid, eb)
    nL, d = r.cfg["nL"], r.cfg["ple_d"]
    gp = r.read(pb, nL * d * 4).view(np.float32)
    ep = ref.ple_input(tid, e).reshape(-1)
    print(f"ple_input[{tid}]: maxAbsDiff={np.abs(gp - ep).max():.3e}")

    # stage 3: layer 0
    H = r.cfg["H"]
    hidden = r._tmp(H * 4)
    r.embed(tid, out=hidden)
    hd0 = r.man["layers"][0]["head_dim"]
    kc = r._tmp(8 * hd0 * 4); vc = r._tmp(8 * hd0 * 4)
    ple_slice = r._tmp(d * 4)
    enc = r.device.create_command_encoder()
    enc.copy_buffer_to_buffer(pb, 0, ple_slice, 0, d * 4)   # layer 0's 256-slice
    r.queue.submit([enc.finish()])
    r.setup_caches()
    r.layer(0, 0, hidden, pb, 0)
    gh = r.read(hidden, H * 4).view(np.float32)
    rh = ref.layer(e.copy(), 0, 0, ref.ple_input(tid, e)[0])
    print(f"layer0[{tid}]: maxAbsDiff={np.abs(gh - rh).max():.3e}")

    # stage 4: full forward, first-token argmax vs reference
    import time
    t0 = time.time()
    glog = r.logits_np(tid, 0)
    rlog = ref.forward(tid, 0)
    ga, ra = int(np.argmax(glog)), int(np.argmax(rlog))
    print(f"forward[{tid}] ({time.time()-t0:.1f}s): gpu argmax={ga} ref argmax={ra} "
          f"{'MATCH' if ga == ra else 'MISMATCH'} | logits maxAbsDiff={np.abs(glog-rlog).max():.3e}")
