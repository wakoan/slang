"""Gemma 4 E2B QAT decode runner in pure Python via PyObjC — Metal directly,
no wgpu. A port of swift-gemma4 / rust-gemma4: same manifest, same weights.bin,
same .metal kernels (gemma4_150/kernels_msl), same ring-buffered resident decode
and multi-turn chat.

Unlike gemma3/runner_metal.py (which uses the `metalgpu` package — 1-D dispatch
only, no offset binding), this drives Metal through PyObjC's full API, so the
2-D attention dispatch and buffer-offset binding the kernels need are available.

CLI:  python -m gemma4_150.metal_runner "prompt" [max_new]
      python -m gemma4_150.metal_runner chat
"""
from __future__ import annotations
import json, mmap, os, struct, sys, time

import Metal
from tokenizers import Tokenizer

RING = 32  # per-step uniform ring; must be >= max chunk
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GDIR = os.path.join(ROOT, "models/gemma-4-E2B-qat/g4_150")
KDIR = os.path.join(ROOT, "gemma4_150/kernels_msl")
TOKJSON = os.path.join(ROOT, "models/gemma-4-E2B-qat/tokenizer.json")
_SHARED = Metal.MTLResourceStorageModeShared


def _parse_threads(src: str) -> int:
    a = "threadsPerThreadgroup = ("
    i = src.find(a)
    if i < 0:
        return 64
    j = i + len(a)
    k = j
    while k < len(src) and src[k].isdigit():
        k += 1
    return int(src[j:k]) if k > j else 64


class Batch:
    """One command buffer, one reusable compute encoder (split around blits)."""
    __slots__ = ("r", "cb", "enc")

    def __init__(self, r: "G4"):
        self.r = r
        self.cb = r.q.commandBuffer()
        self.enc = None

    def _e(self):
        if self.enc is None:
            self.enc = self.cb.computeCommandEncoder()
        return self.enc

    def dg(self, name, bufs, groups):
        pso, tpg = self.r.kernels[name]
        e = self._e()
        e.setComputePipelineState_(pso)
        for i, (b, off) in enumerate(bufs):
            e.setBuffer_offset_atIndex_(b, off, i)
        e.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSizeMake(groups, 1, 1), tpg)

    def dg2d(self, name, bufs, width, height):
        pso, tpg = self.r.kernels[name]
        e = self._e()
        e.setComputePipelineState_(pso)
        for i, (b, off) in enumerate(bufs):
            e.setBuffer_offset_atIndex_(b, off, i)
        e.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSizeMake(width, height, 1), tpg)

    def blit(self, src, so, dst, do, n):
        if self.enc is not None:
            self.enc.endEncoding()
            self.enc = None
        bl = self.cb.blitCommandEncoder()
        bl.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(src, so, dst, do, n)
        bl.endEncoding()

    def commit_wait(self):
        if self.enc is not None:
            self.enc.endEncoding()
            self.enc = None
        self.cb.commit()
        self.cb.waitUntilCompleted()

    def commit(self):
        if self.enc is not None:
            self.enc.endEncoding()
            self.enc = None
        self.cb.commit()
        return self.cb


class G4:
    def __init__(self):
        self.dev = Metal.MTLCreateSystemDefaultDevice()
        self.q = self.dev.newCommandQueue()
        self.kernels = {}  # name -> (pipeline, threadsPerThreadgroup MTLSize)

        man = json.load(open(os.path.join(GDIR, "manifest.json")))
        cfg = man["config"]
        self.H = cfg["H"]; self.nL = cfg["nL"]; self.nH = cfg["nH"]
        self.vocab = cfg["vocab"]; self.window = cfg["window"]; self.ple_d = cfg["ple_d"]
        self.bos = cfg["bos"]; self.eos = cfg["eos"]
        self.layers = man["layers"]

        # weights.bin -> one Metal buffer per tensor
        f = open(os.path.join(GDIR, "weights.bin"), "rb")
        self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(self.mm)
        self.w = {}
        for name, t in man["tensors"].items():
            off, ln = t["off"], t["len"]
            self.w[name] = self.dev.newBufferWithBytes_length_options_(mv[off:off + ln], ln, _SHARED)

        self.pool = {}; self.uni = {}; self.kc = {}; self.vc = {}
        self._pf = None       # lazy batched-prefill engine (gemma4_150/prefill_gpu.py)
        self.mvcache = {}     # buffer-name -> writable memoryview (uniforms/cur/gen)
        self.chat_pos = 0; self.chat_carry = None

        self._compile_all()
        self._setup()
        self._build_uniforms()

    # ---- helpers ----
    def sc(self, l, k):
        return float(self.layers[l]["scales"][k])

    def sc_opt(self, l, k):
        return float(self.layers[l]["scales"].get(k, 0.0))

    def _new(self, nbytes):
        return self.dev.newBufferWithLength_options_(max(nbytes, 16), _SHARED)

    def _buf_u32(self, vals):
        data = struct.pack("<%dI" % len(vals), *vals)
        return self.dev.newBufferWithBytes_length_options_(data, len(data), _SHARED)

    def _buf_f32(self, vals):
        data = struct.pack("<%df" % len(vals), *vals)
        return self.dev.newBufferWithBytes_length_options_(data, len(data), _SHARED)

    def _mv(self, name, nbytes):
        b = self.pool[name]
        m = b.contents().as_buffer(nbytes)
        self.mvcache[name] = m
        return m

    # ---- compile: fixed kernels + per-shape variants ----
    def _compile_one(self, src, func_name, key):
        lib, err = self.dev.newLibraryWithSource_options_error_(src, None, None)
        if lib is None:
            raise RuntimeError(f"MSL compile failed for {key}: {err}")
        fn = lib.newFunctionWithName_(func_name)
        pso, err = self.dev.newComputePipelineStateWithFunction_error_(fn, None)
        if pso is None:
            raise RuntimeError(f"pipeline failed for {key}: {err}")
        self.kernels[key] = (pso, Metal.MTLSizeMake(_parse_threads(src), 1, 1))

    def _variant(self, path, name, repl):
        if name in self.kernels:
            return
        src = open(path).read()
        for a, b in repl:
            src = src.replace(a, b)
        base = os.path.splitext(os.path.basename(path))[0]
        self._compile_one(src, base, name)

    def _compile_all(self):
        for fn in os.listdir(KDIR):
            if fn.endswith(".metal"):
                base = fn[:-6]
                self._compile_one(open(os.path.join(KDIR, fn)).read(), base, base)
        kf = lambda n: os.path.join(KDIR, n + ".metal")
        seen = set()
        for l in range(self.nL):
            hd = self.layers[l]["head_dim"]; qd = self.layers[l]["q_dim"]
            total = qd // 2 + hd
            self._variant(kf("qkv_70"), f"qkv_{qd}_{hd}", [
                ("Q_OUT=2048u", f"Q_OUT={qd}u"), ("KV_OUT=256u", f"KV_OUT={hd}u"),
                ("Q_WGS=1024u", f"Q_WGS={qd // 2}u"), ("KV_WGS=128u", f"KV_WGS={hd // 2}u"),
                ("TOTAL_WGS=1280u", f"TOTAL_WGS={total}u"), ("GRID_X=1280u", f"GRID_X={total}u")])
            self._variant(kf("oproj_73"), f"oproj_{qd}", [
                ("IN_FEATURES=2048u", f"IN_FEATURES={qd}u"), ("WPR=256u", f"WPR={qd // 8}u")])
            oin = self.sc(l, "o_in")
            self._variant(kf("attn_101"), f"attn_{hd}_{l}", [
                ("HEAD_DIM=512u", f"HEAD_DIM={hd}u"), ("HALF_DIM=256u", f"HALF_DIM={hd // 2}u"),
                ("OUT_Q=0.014886821620166302f", f"OUT_Q={oin!r}f")])
            if hd not in seen:
                seen.add(hd)
                self._variant(kf("kvnorm"), f"kvnorm_{hd}", [
                    ("HD=256u", f"HD={hd}u"), ("HALF=128u", f"HALF={hd // 2}u"),
                    ("threadsPerThreadgroup = (256)", f"threadsPerThreadgroup = ({hd})")])

    def _setup(self):
        maxseq = 2048
        hd_map = {}
        for l in range(self.nL):
            hd_map[self.layers[l]["head_dim"]] = self.layers[l]["sliding"]
        self.hd_types = list(hd_map.items())

        for l in range(self.nL):
            if not self.layers[l]["shared"]:
                hd = self.layers[l]["head_dim"]
                self.kc[l] = self._new(maxseq * hd * 4)
                self.vc[l] = self._new(maxseq * hd * 4)

        def alloc(n, nbytes):
            self.pool[n] = self._new(nbytes)

        alloc("hidden", self.H * 4); alloc("cur", 4); alloc("gen", 256 * 4)
        for n, sz in [("pp73", (self.H + 1) * 4), ("pp75", (self.H + 1) * 4), ("pp77", (self.H + 1) * 4),
                      ("partials256", (8 * 32 * 258 + 8) * 4), ("partials512", (8 * 32 * 514 + 8) * 4)]:
            alloc(n, sz)
            self.pool[n].contents().as_buffer(sz)[:] = b"\x00" * sz
        for n, sz in [("normed", self.H * 4), ("logits", self.vocab * 4), ("cv", 256 * 4), ("ci", 256 * 4),
                      ("sa", 4), ("a", self.H * 4), ("suma", 4), ("outq", 4096 * 4), ("outk", 512 * 4),
                      ("outv", 512 * 4), ("dummy", 512 * 4), ("dummy2", 512 * 4), ("attn", 4096 * 4),
                      ("y2", self.H * 4), ("sum2", 4), ("geglu", 12288 * 2), ("gate", self.ple_d * 4),
                      ("y2n", self.H * 4), ("sum2n", 4), ("ctx", self.nL * self.ple_d * 4),
                      ("plegath", self.nL * self.ple_d * 4), ("ple", self.nL * self.ple_d * 4)]:
            alloc(n, sz)

        # ring of per-step position uniforms + cached writable memoryviews
        for hd, _ in self.hd_types:
            for s in range(RING):
                alloc(f"parA_{hd}_{s}", 32); self._mv(f"parA_{hd}_{s}", 32)
                alloc(f"pkv_{hd}_{s}", 16); self._mv(f"pkv_{hd}_{s}", 16)
        self._mv("cur", 4); self._mv("gen", 256 * 4)

        # rope cos/sin precompute per head-dim type
        import math
        for hd, _ in self.hd_types:
            theta, cutoff = 0.0, 0
            for l in range(self.nL):
                if self.layers[l]["head_dim"] == hd:
                    theta = self.layers[l]["rope_theta"]; cutoff = self.layers[l]["rope_cutoff"]
            half = hd // 2
            cosv = [0.0] * (maxseq * half); sinv = [0.0] * (maxseq * half)
            for pos in range(maxseq):
                for i in range(half):
                    inv = (1.0 / theta ** (i / half)) if i < cutoff else 0.0
                    ang = pos * inv
                    cosv[pos * half + i] = math.cos(ang); sinv[pos * half + i] = math.sin(ang)
            self.pool[f"rcosT_{hd}"] = self._buf_f32(cosv)
            self.pool[f"rsinT_{hd}"] = self._buf_f32(sinv)

    def _build_uniforms(self):
        f2u = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
        self.uni["z4"] = self._buf_f32([0, 0, 0, 0])
        self.uni["plog"] = self._buf_f32([0, 0, 0, 0])
        self.uni["ones1"] = self._buf_u32([1, 0, 0, 0])
        d = self.ple_d
        for l in range(self.nL):
            if l == 0:
                self.uni[f"p69_{l}"] = self._buf_u32([1, 0, f2u(self.sc(l, "qkv_in")), 0])
            self.uni[f"p70_{l}"] = self._buf_f32([self.sc(l, "q_out"), self.sc_opt(l, "k_out"), self.sc_opt(l, "v_out"), 0])
            self.uni[f"p73_{l}"] = self._buf_f32([self.sc(l, "o_out"), self.sc(l, "gate_in"), 0, 0])
            self.uni[f"p74_{l}"] = self._buf_f32([self.sc(l, "gate_out"), self.sc(l, "up_out"), self.sc(l, "down_in"), 0])
            self.uni[f"p75_{l}"] = self._buf_f32([self.sc(l, "down_in"), self.sc(l, "down_out"), 0, 0])
            self.uni[f"p76_{l}"] = self._buf_u32([f2u(self.sc(l, "plegate_in")), f2u(self.sc(l, "plegate_out")), l * d, 0])
            nxt = self.sc(l + 1, "qkv_in") if l + 1 < self.nL else 0.0
            self.uni[f"p77_{l}"] = self._buf_f32([nxt, self.sc(l, "pleproj_in"), self.sc(l, "pleproj_out"), 0])

    # ---- per-step dynamic uniforms (ring slot) ----
    def _write_step(self, pos, slot):
        nH, win = self.nH, self.window
        for hd, sliding in self.hd_types:
            struct.pack_into("<8I", self.mvcache[f"parA_{hd}_{slot}"], 0,
                             1, pos + 1, pos, nH, 1, win if sliding else 0, 0, 0)
            struct.pack_into("<4I", self.mvcache[f"pkv_{hd}_{slot}"], 0, pos * hd, 0, 0, 0)

    def set_cur(self, tok):
        struct.pack_into("<I", self.mvcache["cur"], 0, tok)

    def _read_gen(self, i):
        return struct.unpack_from("<I", self.mvcache["gen"], i * 4)[0]

    def _read_cur(self):
        return struct.unpack_from("<I", self.mvcache["cur"], 0)[0]

    # ---- forward ----
    def _layer(self, b, l, pos, slot):
        li = self.layers[l]
        hd, qd, inter = li["head_dim"], li["q_dim"], li["intermediate"]
        half = hd // 2; src = li["kv_source"]; roff = pos * half * 4
        P = self.pool; U = self.uni
        cb = P[f"rcosT_{hd}"]; sb = P[f"rsinT_{hd}"]
        kcb = self.kc[src]; vcb = self.vc[src]
        lw = lambda n: self.w[f"L{l}.{n}"]
        parA = P[f"parA_{hd}_{slot}"]; pkv = P[f"pkv_{hd}_{slot}"]; partials = P[f"partials{hd}"]

        if l == 0:
            b.dg("rmssrq_69", [(P["hidden"], 0), (lw("in_norm"), 0), (P["a"], 0), (P["suma"], 0),
                               (U[f"p69_{l}"], 0)], 1)
            a, suma = P["a"], P["suma"]
        else:
            a, suma = P["y2n"], P["sum2n"]

        par70 = U[f"p70_{l}"]
        if li["shared"]:
            b.dg(f"qkv_{qd}_{hd}", [(a, 0), (lw("q_bits"), 0), (lw("q_bits"), 0), (lw("q_bits"), 0),
                                    (lw("q_scale"), 0), (suma, 0), (P["outq"], 0), (P["dummy"], 0),
                                    (P["dummy2"], 0), (par70, 0)], qd // 2)
        else:
            b.dg(f"qkv_{qd}_{hd}", [(a, 0), (lw("q_bits"), 0), (lw("k_bits"), 0), (lw("v_bits"), 0),
                                    (lw("qkv_scales"), 0), (suma, 0), (P["outq"], 0), (P["outk"], 0),
                                    (P["outv"], 0), (par70, 0)], qd // 2 + hd)
            b.dg(f"kvnorm_{hd}", [(P["outk"], 0), (P["outv"], 0), (lw("k_norm"), 0), (cb, roff), (sb, roff),
                                  (kcb, 0), (vcb, 0), (pkv, 0)], 1)
        b.dg2d(f"attn_{hd}_{l}", [(P["outq"], 0), (lw("q_norm"), 0), (cb, roff), (sb, roff), (kcb, 0),
                                  (vcb, 0), (partials, 0), (P["attn"], 0), (parA, 0)], self.nH, 32)
        b.dg(f"oproj_{qd}", [(P["attn"], 0), (lw("o_bits"), 0), (lw("o_scale"), 0), (P["pp73"], 0),
                             (P["hidden"], 0), (lw("o_w12"), 0), (P["y2"], 0), (P["sum2"], 0),
                             (U[f"p73_{l}"], 0)], 192)
        par74 = U[f"p74_{l}"]; par75 = U[f"p75_{l}"]
        if inter == 6144:
            b.dg("gateup_74", [(P["y2"], 0), (lw("gate_bits"), 0), (lw("gate_scale"), 0), (lw("up_bits"), 0),
                               (lw("up_scale"), 0), (P["sum2"], 0), (P["geglu"], 0), (lw("gelu_gate"), 0),
                               (par74, 0)], 768)
            b.dg("down_75", [(P["geglu"], 0), (lw("down_bits"), 0), (P["pp75"], 0), (lw("down_scale"), 0),
                             (P["hidden"], 0), (lw("down_nw"), 0), (par75, 0)], 384)
        else:
            b.dg("gateup_95", [(P["y2"], 0), (lw("gate_bits"), 0), (lw("gate_scale"), 0), (lw("up_bits"), 0),
                               (lw("up_scale"), 0), (P["sum2"], 0), (P["geglu"], 0), (lw("gelu_gate"), 0),
                               (par74, 0)], 3072)
            b.dg("down_96", [(P["geglu"], 0), (lw("down_bits"), 0), (P["pp75"], 0), (lw("down_scale"), 0),
                             (P["hidden"], 0), (lw("down_nw"), 0), (par75, 0)], 384)
        b.dg("plegate_76", [(P["hidden"], 0), (lw("plegate_codes"), 0), (lw("plegate_rowscale"), 0),
                            (P["ple"], 0), (P["gate"], 0), (lw("gelu_plegate"), 0), (U[f"p76_{l}"], 0)], 256)
        b.dg("pleproj_77", [(P["gate"], 0), (lw("pleproj_codes"), 0), (lw("pleproj_rowscale"), 0),
                            (P["pp77"], 0), (P["hidden"], 0), (lw("pleproj_w12s"), 0), (P["y2n"], 0),
                            (P["sum2n"], 0), (U[f"p77_{l}"], 0)], 96)

    def _forward(self, pos, step, argmax, gen, wait, tok=None):
        # `tok` = (buffer, byte offset) holding this step's token id; defaults to
        # the GPU-written `cur`. Prefill passes the prompt buffer at pos*4 so the
        # CPU never rewrites a buffer the GPU may still be reading (embed_00 and
        # plegather_01 read ids[0] when seq=1, so an offset bind selects the token).
        tb, to = tok if tok is not None else (self.pool["cur"], 0)
        slot = step % RING
        self._write_step(pos, slot)
        b = Batch(self)
        P = self.pool; U = self.uni; W = self.w
        b.dg("embed_00", [(tb, to), (W["embed_q"], 0), (W["embed_scale"], 0), (P["hidden"], 0),
                          (U["ones1"], 0)], 1)
        b.dg("proj_68", [(P["hidden"], 0), (W["pl_model_proj"], 0), (P["ctx"], 0), (U["z4"], 0)], 1120)
        b.dg("plegather_01", [(tb, to), (W["ple_q"], 0), (W["ple_scale"], 0), (P["plegath"], 0),
                              (U["ones1"], 0)], 1)
        b.dg("combine", [(P["ctx"], 0), (P["plegath"], 0), (W["pl_proj_norm"], 0), (P["ple"], 0)], self.nL)
        for l in range(self.nL):
            self._layer(b, l, pos, slot)
        b.dg("rmssrq_69", [(P["hidden"], 0), (W["final_norm"], 0), (P["normed"], 0), (P["sa"], 0),
                           (U["ones1"], 0)], 1)
        b.dg("logits_33", [(P["normed"], 0), (W["lmhead_blk"], 0), (W["lmhead_scale"], 0), (P["logits"], 0),
                           (U["plog"], 0)], self.vocab // 128)
        if argmax:
            b.dg("argmax1_34", [(P["logits"], 0), (P["cv"], 0), (P["ci"], 0)], 256)
            b.dg("argmax2_35", [(P["cv"], 0), (P["ci"], 0), (P["cur"], 0)], 1)
            if gen:
                b.blit(P["cur"], 0, P["gen"], step * 4, 4)
        if wait:
            b.commit_wait()
            return None
        return b.commit()

    # ---- pipelined prefill ----
    def prefill(self, toks, pos, chunk=RING):
        """Run `toks` through the model from `pos`, pipelined: commit a chunk of
        forwards without waiting so CPU recording overlaps GPU execution (the
        same trick that makes decode fast). Token ids come from a GPU buffer
        bound per-step at an offset, so there is no CPU/GPU write race. Chunk is
        capped at RING so no in-flight step's uniform slot is reused. Returns the
        new position."""
        if not toks:
            return pos
        # Batched prefill (layers-outer, MPS GEMMs) is ~5.6x faster once the
        # prompt is long enough to amortize its fixed ~27ms of weight dequant.
        # It is not numerically identical to the per-token path -- f16 GEMM
        # replaces the exact int-dot -- so G4_BATCHED_PREFILL=0 opts out.
        if len(toks) >= 16 and os.environ.get("G4_BATCHED_PREFILL", "1") != "0":
            try:
                from gemma4_150.prefill_gpu import PrefillGPU
                if self._pf is None:
                    self._pf = PrefillGPU(self)
                return self._pf.prefill(toks, pos)
            except Exception as e:
                print(f"(batched prefill unavailable: {e}; using per-token)", file=sys.stderr)
        buf = self._buf_u32(list(toks))
        i = 0
        while i < len(toks):
            end = min(i + chunk, len(toks))
            last = None
            while i < end:
                last = self._forward(pos, i % RING, False, False, False, tok=(buf, i * 4))
                pos += 1; i += 1
            if last is not None:
                last.waitUntilCompleted()
        return pos

    # ---- single-shot greedy generate ----
    def generate(self, ids, n_new, chunk=32):
        """Returns (tokens, decode tok/s).

        The rate divides by `step` -- the forwards actually executed and timed --
        NOT by len(out). Decode runs a whole chunk of forwards before reading any
        of them back, so an early EOS discards up to chunk-1 tokens whose work is
        still inside dt. Dividing by the surviving tokens puts a truncated
        numerator over a full denominator and silently underreports: at a fixed
        1024-token context this reads 150 tok/s for a long answer, 77 for a
        one-sentence one and 24 for a yes/no -- same code, same context, same
        machine. A stale 125.5 in litert/RESULTS.md came from exactly this.
        chat_turn has always divided by step; generate was the odd one out."""
        pos = self.prefill(list(ids[:-1]), 0)
        self.set_cur(ids[-1])
        out = []; step = 0; cap = min(n_new, 240)
        t0 = time.time()
        while step < cap:
            end = min(step + chunk, cap); last = None
            while step < end:
                last = self._forward(pos, step, True, True, False); pos += 1; step += 1
            if last is not None:
                last.waitUntilCompleted()
            while len(out) < step:
                tk = self._read_gen(len(out))
                if tk == self.eos:
                    dt = time.time() - t0
                    return out, (step / dt if dt > 0 else 0.0)
                out.append(tk)
        dt = time.time() - t0
        return out, (step / dt if dt > 0 else 0.0)

    # ---- multi-turn chat (KV cache resident across turns) ----
    def chat_reset(self):
        self.chat_pos = 0; self.chat_carry = None

    def chat_turn(self, frame, max_new, stop, on_chunk, chunk=32):
        pre = ([self.chat_carry] if self.chat_carry is not None else []) + list(frame)
        self.chat_pos = self.prefill(pre[:-1], self.chat_pos)
        self.set_cur(pre[-1])
        prefill_end = self.chat_pos
        out = []; step = 0; stopped = False; cap = min(max_new, 240)
        t0 = time.time()
        while step < cap and not stopped and self.chat_pos < 2040:
            end = min(step + chunk, cap); last = None
            while step < end:
                last = self._forward(self.chat_pos, step, True, True, False)
                self.chat_pos += 1; step += 1
            if last is not None:
                last.waitUntilCompleted()
            i = len(out)
            while i < step:
                tk = self._read_gen(i)
                if tk in stop:
                    stopped = True; self.chat_carry = tk; self.chat_pos = prefill_end + i + 1
                    break
                out.append(tk); i += 1
            on_chunk(out)
        if not stopped:
            self.chat_carry = self._read_cur()
        dt = time.time() - t0
        return out, (step / dt if dt > 0 else 0.0)


# ---------------------------------------------------------------------------
def _chat(g, tok):
    stop = {g.eos}
    tc = tok.token_to_id("<turn|>")
    if tc is not None:
        stop.add(tc)
    g.chat_reset()
    turn = 0
    print("chat ready — type a message; /reset clears history, /quit exits.\n")
    while True:
        try:
            msg = input("you: ").strip()
        except EOFError:
            break
        if not msg:
            continue
        if msg in ("/quit", "/exit"):
            break
        if msg == "/reset":
            g.chat_reset(); turn = 0; print("(history cleared)\n"); continue
        user = f"<|turn>user\n{msg}<turn|>\n<|turn>model\n"
        if turn == 0:
            frame = [g.bos] + tok.encode(user, add_special_tokens=False).ids
        else:
            frame = tok.encode("\n" + user, add_special_tokens=False).ids
        turn += 1
        sys.stdout.write("bot: "); sys.stdout.flush()
        printed = [0]

        def on_chunk(acc):
            s = tok.decode(acc)
            if len(s) > printed[0]:
                sys.stdout.write(s[printed[0]:]); sys.stdout.flush(); printed[0] = len(s)

        out, tps = g.chat_turn(frame, 240, stop, on_chunk)
        print(f"\n({len(out)} tok, {tps:.1f} tok/s)\n")


def main():
    args = sys.argv[1:]
    prompt = args[0] if args else "The capital of France is"
    n_new = int(args[1]) if len(args) > 1 else 48
    print("loading model + compiling kernels…")
    g = G4()
    tok = Tokenizer.from_file(TOKJSON)
    if prompt == "chat":
        _chat(g, tok); return
    framed = f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
    ids = [g.bos] + tok.encode(framed, add_special_tokens=False).ids
    out, tps = g.generate(ids, n_new)
    print("prompt:", prompt)
    print("=>", tok.decode(out))
    print(f"decode: {tps:.1f} tok/s")


if __name__ == "__main__":
    main()
