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
import re
from pathlib import Path

import numpy as np
import wgpu

REF_DIR = Path(__file__).resolve().parent.parent / "reference" / "webml_gemma4_kernels"
_BIND = re.compile(r"@group\(0\)\s*@binding\((\d+)\)\s*var<(storage|uniform)(?:,\s*(read|read_write))?>")


def _kernel(name: str) -> str:
    return (REF_DIR / f"{name}.wgsl").read_text()


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
                compute={"module": self._shader(code), "entry_point": "main"})
            self._pipes[key] = (pipe, layout)
        return self._pipes[key]

    def dispatch(self, key, code, buffers, grid):
        pipe, layout = self._pipe(key, code)
        bg = self.device.create_bind_group(layout=layout, entries=[
            {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
            for i, b in enumerate(buffers)])
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(*grid)
        cp.end()
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
        return self.device.create_buffer_with_data(
            data=np.array([token_id], np.uint32).tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)

    # ---- stage 1: embed gather (kernel 00) ----
    def embed(self, token_id: int, out=None) -> object:
        H = self.cfg["H"]
        y = out if out is not None else self._tmp(H * 4)
        par = self._uniform(np.array([1, 0, 0, 0], np.uint32))
        self.dispatch("embed", _kernel("00_main"),
                      [self._ids_buf(token_id), self._bufs["embed_q"],
                       self._bufs["embed_scale"], y, par], (1, 1, 1))
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

    def ple_input(self, token_id: int, embed_buf) -> object:
        H, nL, d = self.cfg["H"], self.cfg["nL"], self.cfg["ple_d"]
        ctx = self._tmp(nL * d * 4)
        ple = self._tmp(nL * d * 4)
        par0 = self._uniform(np.array([0, 0, 0, 0], np.float32))      # kernel 68: inScale/outScale
        seq1 = self._uniform(np.array([1, 0, 0, 0], np.uint32))       # kernel 01: seq=1
        self.dispatch("proj68", _kernel("68_reduce"),
                      [embed_buf, self._bufs["pl_model_proj"], ctx, par0], (nL * d // 8, 1, 1))
        self.dispatch("plegather", _kernel("01_main"),
                      [self._ids_buf(token_id), self._bufs["ple_q"],
                       self._bufs["ple_scale"], ple, seq1], (1, 1, 1))
        out = self._tmp(nL * d * 4)
        code = self._COMBINE % (H ** -0.5)
        self.dispatch("combine", code, [ctx, ple, self._bufs["pl_proj_norm"], out], (nL, 1, 1))
        return out

    # ---- stage 3: one decoder layer ----
    _VNORM = """
const HD:u32=%du; const EPS:f32=1e-6;
@group(0) @binding(0) var<storage, read> v: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>;   // p.x = dstOffset
var<workgroup> red: array<f32, HD>;
@compute @workgroup_size(HD,1,1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let x = v[tid];
  red[tid] = x*x; workgroupBarrier();
  var s:u32 = HD/2u; loop { if(s==0u){break;} if(tid<s){red[tid]=red[tid]+red[tid+s];} s=s/2u; workgroupBarrier(); }
  cache[p.x + tid] = x * inverseSqrt(red[0]/f32(HD)+EPS);
}"""

    def _rope(self, pos, theta, half, cutoff):
        inv = 1.0 / theta ** (np.arange(half, dtype=np.float64) / half)
        inv[cutoff:] = 0.0
        ang = pos * inv
        cb = self.device.create_buffer_with_data(
            data=np.cos(ang).astype(np.float32).tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        sb = self.device.create_buffer_with_data(
            data=np.sin(ang).astype(np.float32).tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        return cb, sb

    def _patch(self, code, **consts):
        for k, val in consts.items():
            code = re.sub(rf"const {k}: (u32|f32) = [^;]+;",
                          lambda m: f"const {k}: {m.group(1)} = {val}"
                          + ("u" if m.group(1) == "u32" else "") + ";", code)
        return code

    def setup_caches(self, max_seq=2048):
        self.max_seq = max_seq
        self.kc, self.vc = {}, {}
        for s in self.man["layers"]:
            if not s["shared"]:
                hd = s["head_dim"]
                self.kc[s["index"]] = self._tmp(max_seq * hd * 4, src=False)
                self.vc[s["index"]] = self._tmp(max_seq * hd * 4, src=False)

    def layer(self, L, pos, hidden, ple_buf, ple_off):
        """Run decoder layer L in place on `hidden` (f32 [H])."""
        s = self.man["layers"][L]
        sc = s["scales"]
        H, nH = self.cfg["H"], self.cfg["nH"]
        hd, qd, inter = s["head_dim"], s["q_dim"], s["intermediate"]
        half, cutoff, shared = hd // 2, s["rope_cutoff"], s["shared"]
        kc, vc = self.kc[s["kv_source"]], self.vc[s["kv_source"]]
        b = self._bufs
        cb, sb = self._rope(pos, s["rope_theta"], half, cutoff)

        # 69: input norm + srq -> a[H], sum_a
        a = self._tmp(H * 4); suma = self._tmp(4)
        self.dispatch("k69", _kernel("69_sg_sum"),
                      [hidden, b[f"L{L}.in_norm"], a, suma,
                       self._uniform(np.array([1, 0, np.float32(sc["qkv_in"]).view(np.uint32), 0], np.uint32))],
                      (1, 1, 1))
        # 70: qkv  (shared layers: q-only — bind q_bits x3, dispatch only the Q workgroups)
        # KV head_dim == q head_dim (256 sliding / 512 full), so patch KV_OUT/KV_WGS too.
        outq = self._tmp(qd * 4)
        dummy = self._tmp(hd * 4)
        total = qd // 2 + hd     # Q_WGS + 2*KV_WGS = qd/2 + 2*(hd/2)
        k70 = self._patch(_kernel("70_srq"), Q_OUT=qd, Q_WGS=qd // 2, KV_OUT=hd,
                          KV_WGS=hd // 2, TOTAL_WGS=total, GRID_X=total)
        par70 = self._uniform(np.array([sc["q_out"], sc.get("k_out", 0), sc.get("v_out", 0), 0], np.float32))
        if shared:
            self.dispatch(f"k70_{qd}_{hd}", k70,
                          [a, b[f"L{L}.q_bits"], b[f"L{L}.q_bits"], b[f"L{L}.q_bits"],
                           b[f"L{L}.q_scale"], suma, outq, dummy, dummy, par70], (qd // 2, 1, 1))
        else:
            outk, outv = self._tmp(hd * 4), self._tmp(hd * 4)
            self.dispatch(f"k70_{qd}_{hd}", k70,
                          [a, b[f"L{L}.q_bits"], b[f"L{L}.k_bits"], b[f"L{L}.v_bits"],
                           b[f"L{L}.qkv_scales"], suma, outq, outk, outv, par70], (total, 1, 1))
            # 71: k norm+rope -> cache ; v norm -> cache (only the KV-owning layers)
            k71 = self._patch(_kernel("71_main"), HEAD_DIM=hd, HALF_DIM=half)
            self.dispatch(f"k71_{hd}", k71,
                          [outk, b[f"L{L}.k_norm"], cb, sb, kc,
                           self._uniform(np.array([1, 1, pos * hd, 0], np.uint32))], (1, 1, 1))
            self.dispatch(f"vnorm_{hd}", self._VNORM % hd,
                          [outv, vc, self._uniform(np.array([pos * hd, 0, 0, 0], np.uint32))], (1, 1, 1))
        # attention (patch HEAD_DIM/HALF/OUT_Q from 101 template)
        att = self._patch(_kernel("101_srq"), HEAD_DIM=hd, HALF_DIM=half)
        att = att.replace("const OUT_Q: f32 = 0.014886821620166302;",
                          f"const OUT_Q: f32 = {sc['o_in']!r};")
        partials = self._tmp((8 * 32 * (hd + 2) + 8) * 4, src=False)
        self.queue.write_buffer(partials, 0, np.zeros(8 * 32 * (hd + 2) + 8, np.uint32).tobytes())
        attn = self._tmp(qd * 4)
        window = self.cfg["window"] if s["sliding"] else 0
        parA = self._uniform(np.array([1, pos + 1, pos, nH, 1, window, 0, 0], np.uint32))
        self.dispatch(f"att_{hd}_{sc['o_in']}", att,
                      [outq, b[f"L{L}.q_norm"], cb, sb, kc, vc, partials, attn, parA], (nH, 32, 1))
        # 73: o-proj + post-attn norm-add + pre-ffn norm  (updates hidden, emits y2 f16 + sum2)
        k73 = self._patch(_kernel("73_sg_sum"), IN_FEATURES=qd, WORDS_PER_ROW=qd // 8)
        pp73 = self._tmp((H + 1) * 4, src=False)
        self.queue.write_buffer(pp73, 0, np.zeros(H + 1, np.uint32).tobytes())
        y2, sum2 = self._tmp(H * 2), self._tmp(4)
        par73 = self._uniform(np.array([sc["o_out"], sc["gate_in"], 0, 0], np.float32))
        self.dispatch(f"k73_{qd}", k73, [attn, b[f"L{L}.o_bits"], b[f"L{L}.o_scale"], pp73,
                      hidden, b[f"L{L}.o_w12"], y2, sum2, par73], (192, 1, 1))
        # 74/95: gate/up geglu -> down input (f16). 4-bit MLP (inter 6144, L0-14) uses
        # kernel 74; 2-bit double-wide MLP (inter 12288, L15-34) uses kernel 95.
        gu_kern, gu_grid = ("74_sg_sum", 768) if inter == 6144 else ("95_sg_sum", 3072)
        geglu = self._tmp(inter * 2)
        par74 = self._uniform(np.array([sc["gate_out"], sc["up_out"], sc["down_in"], 0], np.float32))
        self.dispatch(f"gu_{inter}", _kernel(gu_kern), [y2, b[f"L{L}.gate_bits"], b[f"L{L}.gate_scale"],
                      b[f"L{L}.up_bits"], b[f"L{L}.up_scale"], sum2, geglu, b[f"L{L}.gelu_gate"], par74],
                      (gu_grid, 1, 1))
        # 75/96: down + post-ffn norm-add (updates hidden). 4-bit -> 75, 2-bit -> 96.
        down_kern = "75_srq" if inter == 6144 else "96_srq"
        pp75 = self._tmp((H + 1) * 4, src=False)
        self.queue.write_buffer(pp75, 0, np.zeros(H + 1, np.uint32).tobytes())
        par75 = self._uniform(np.array([sc["down_in"], sc["down_out"], 0, 0], np.float32))
        self.dispatch(f"down_{inter}", _kernel(down_kern), [geglu, b[f"L{L}.down_bits"], pp75,
                      b[f"L{L}.down_scale"], hidden, b[f"L{L}.down_nw"], par75], (H // 4, 1, 1))
        # 76: PLE input gate  (reads its 256-slice at ple_off in the shared ple buffer)
        gate = self._tmp(self.cfg["ple_d"] * 4)
        parP = np.zeros(4, np.uint32)
        parP[:2] = np.array([sc["plegate_in"], sc["plegate_out"]], np.float32).view(np.uint32)
        parP[2] = ple_off
        self.dispatch("k76", _kernel("76_reduce"), [hidden, b[f"L{L}.plegate_codes"],
                      b[f"L{L}.plegate_rowscale"], ple_buf, gate, b[f"L{L}.gelu_plegate"],
                      self._uniform(parP)], (self.cfg["ple_d"], 1, 1))
        # 77: PLE proj + residual*layer_scalar + next-layer norm -> hidden, y2_next
        pp77 = self._tmp((H + 1) * 4, src=False)
        self.queue.write_buffer(pp77, 0, np.zeros(H + 1, np.uint32).tobytes())
        y2n, sum2n = self._tmp(H * 4), self._tmp(4)
        nxt_in = self.man["layers"][L + 1]["scales"]["qkv_in"] if L + 1 < self.cfg["nL"] else 0.0
        par77 = self._uniform(np.array([nxt_in, sc["pleproj_in"], sc["pleproj_out"], 0], np.float32))
        self.dispatch("k77", _kernel("77_sg_sum"), [gate, b[f"L{L}.pleproj_codes"],
                      b[f"L{L}.pleproj_rowscale"], pp77, hidden, b[f"L{L}.pleproj_w12s"],
                      y2n, sum2n, par77], (96, 1, 1))
        return hidden

    # ---- stage 4: full forward ----
    def forward(self, token_id: int, pos: int, hidden=None):
        H, nL, d, vocab = self.cfg["H"], self.cfg["nL"], self.cfg["ple_d"], self.cfg["vocab"]
        if not hasattr(self, "kc"):
            self.setup_caches()
        hidden = hidden if hidden is not None else self._tmp(H * 4)
        self.embed(token_id, out=hidden)
        ple = self.ple_input(token_id, hidden)
        for L in range(nL):
            self.layer(L, pos, hidden, ple, L * d)
        # final norm (kernel 69, srq passthrough inScale=0) -> normed
        normed = self._tmp(H * 4); sa = self._tmp(4)
        self.dispatch("finalnorm", _kernel("69_sg_sum"),
                      [hidden, self._bufs["final_norm"], normed, sa,
                       self._uniform(np.array([1, 0, 0, 0], np.uint32))], (1, 1, 1))
        # logits (kernel 33), lm_head act scales are 0 -> weight-only
        logits = self._tmp(vocab * 4)
        self.dispatch("logits", _kernel("33_srq"),
                      [normed, self._bufs["lmhead_blk"], self._bufs["lmhead_scale"], logits,
                       self._uniform(np.array([0, 0, 0, 0], np.float32))], (vocab // 128, 1, 1))
        return logits

    def logits_np(self, token_id, pos, hidden=None):
        lg = self.forward(token_id, pos, hidden)
        return self.read(lg, self.cfg["vocab"] * 4).view(np.float32)


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
