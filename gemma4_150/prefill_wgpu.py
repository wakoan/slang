"""Batched prefill for the wgpu runner — the portable counterpart of prefill_gpu.py.

Same idea and the same DSL kernels: hold all S token activations resident and
iterate LAYERS OUTER, TOKENS INNER, so each weight is read once per prompt
instead of once per token and every matmul becomes a GEMM at M=S.

The one structural difference is the GEMM. prefill_gpu.py defaults to
MPSMatrixMultiplication, which is Apple-only; here every matmul is
kernels_dsl.gemm_tiled, the portable tiled GEMM. That costs about half the
throughput (2.61 vs 5.64 TFLOP/s measured) and is the reason this module exists
separately rather than as a backend flag on the Metal one.

    from gemma4_150.prefill_wgpu import PrefillWGPU
    pf = PrefillWGPU(runner)        # runner is a runner.G4Runner
    pos = pf.prefill(token_ids, 0)

Like the Metal one, this is NOT bit-identical to the per-token path: dequant +
f16 GEMM replaces the exact int-dot arithmetic, and the difference compounds
through the 15 cache-owning layers. Measured against the per-token caches at 18
tokens it is 1.2e-2 relative at layer 0 rising to 9.1e-2 at layer 14 — the Metal
path, which has been in use since it beat LiteRT, shows 1.0e-1 by the same
measure. That equivalence is the gate (tests/test_prefill_wgpu.py); expecting
bit-identity here would be expecting the wrong thing.
"""
from __future__ import annotations

import numpy as np

from py_shader_lang_wgpu import translate
from gemma4_150 import kernels_dsl
from gemma4_150.kernels_dsl import gemm_groups

BUCKET = 16
MAXSEQ = 2048


def _f32u(x):
    return np.float32(x).view(np.uint32)


class PrefillWGPU:
    def __init__(self, r):
        self.r = r
        self.cfg = r.cfg
        self.layers = r.man["layers"]
        self._S = 0
        self._src = {}
        self._svs = {}
        self._live = []

    # ---- kernel text (memoized; translating per dispatch cost 76x once) ----
    def K(self, name, threads=(256, 1, 1), **consts):
        key = (name, threads, tuple(sorted(consts.items())))
        if key not in self._src:
            self._src[key] = translate(getattr(kernels_dsl, name),
                                       workgroup_size=threads, consts=consts or None)
        return self._src[key]

    def _sv(self, L, H):
        """Layer scalar, cached. It lives past the two norm vectors in w12s, and
        reading it inside the layer loop would issue a buffer copy while the
        prefill encoder is open."""
        if L not in self._svs:
            raw = self.r.read(self.r._bufs[f"L{L}.pleproj_w12s"], (2 * H + 1) * 4)
            self._svs[L] = float(np.frombuffer(raw, np.float32)[2 * H])
        return self._svs[L]

    def _b(self, name, nbytes):
        return self.r._scratch(f"pf_{name}", nbytes)

    def _u(self, name, *words):
        """A FRESH uniform buffer per dispatch (floats passed pre-bitcast).

        Deliberately not pooled by name. queue.write_buffer is ordered against
        SUBMITS, and this module records ~500 dispatches into one encoder and
        submits once — so with a pooled buffer every write would land before any
        dispatch ran, and two dispatches sharing a name would both see the last
        value written. Allocating per dispatch makes that impossible.
        """
        import wgpu
        b = self.r.device.create_buffer_with_data(
            data=np.array(words, np.uint32).tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.STORAGE)
        self._live.append(b)
        return b

    # ---- GEMM: C[M,N] = alpha * A @ (B^T if tr else B) --------------------
    def _gemm(self, A, Wb, C, M, N, K, alpha=1.0, tr=True, acols=None, ccols=None,
              aoff=0, boff=0, coff=0, tag=""):
        gx, gy = gemm_groups(M, N)
        self.r.dispatch(
            "gemm_tiled", self.K("gemm_tiled", (16, 16, 1)),
            [(A, aoff), (Wb, boff), (C, coff),
             self._u(f"g{tag}", M, N, K, _f32u(alpha)),
             self._u(f"gs{tag}", acols or K, K if tr else N, ccols or N, 1 if tr else 0)],
            (gx, gy, 1))

    def _dq(self, bits, scale, dst, n_in, n_out, tag, soff=0):
        """Dequantize [n_out][n_in] to f16, width derived from the buffer size.

        Do NOT hardcode the width: layers 0-14 pack gate/up/down at 4 bits and
        layers 15-34 at 2, and getting it wrong is silent.

        `soff` is a BYTE offset into the row-scale buffer, and it is not
        optional decoration: on the layers that own a KV cache, q/k/v share one
        concatenated `qkv_scales`, so k starts at q_dim and v at q_dim+head_dim.
        Omitting it dequantizes k and v with q's scales, which is exactly wrong
        enough to look like a rope or cache-offset bug three stages later.
        """
        nbits = bits.size * 8 // (n_in * n_out)
        assert nbits in (2, 4, 8), f"odd width {nbits} for [{n_out}x{n_in}]"
        nwords = n_out * (n_in // (32 // nbits))
        self.r.dispatch(f"dq{nbits}", self.K("dq_f16", BITS=nbits),
                        [bits, (scale, soff), dst,
                         self._u(f"dq{tag}", n_in, n_out, 0, 0)],
                        ((nwords + 255) // 256, 1, 1))

    # ---- buffers -----------------------------------------------------------
    def _alloc(self, S):
        if S <= self._S:
            return
        c, L = self.cfg, self.layers
        H, nL, d = c["H"], c["nL"], c["ple_d"]
        qd = max(l["q_dim"] for l in L)
        hd = max(l["head_dim"] for l in L)
        it = max(l["intermediate"] for l in L)
        for n, w in [("hidden", H), ("a", H), ("q", qd), ("ctx", nL * d),
                     ("plegath", nL * d), ("ple", nL * d), ("gple", d)]:
            self._b(n, S * w * 4)
        for n, w in [("ah", H), ("qraw", qd), ("kraw", hd), ("vraw", hd),
                     ("attnh", qd), ("oraw", H), ("y2", H), ("graw", it),
                     ("uraw", it), ("geglu", it), ("draw", H), ("ctxh", nL * d),
                     ("hh", H), ("plin", d), ("gph", d), ("ppraw", H),
                     ("qf16", qd), ("attnraw", qd)]:
            self._b(n, S * w * 2)
        for n in ("suma", "sum2"):
            self._b(n, S * 4)
        self._b("scores", S * c["nH"] * MAXSEQ * 2)
        self._b("denom", S * c["nH"] * 4)
        self._b("kf16", MAXSEQ * hd * 2)
        self._b("vf16", MAXSEQ * hd * 2)
        # full-position rope tables, one pair per head-dim type
        seen = {}
        for l in L:
            seen[l["head_dim"]] = (l["rope_theta"], l["rope_cutoff"])
        for h, (theta, cutoff) in seen.items():
            half = h // 2
            i = np.arange(half, dtype=np.float64)
            inv = np.where(i < cutoff, 1.0 / theta ** (i / half), 0.0)
            ang = np.outer(np.arange(MAXSEQ, dtype=np.float64), inv)
            for nm, val in (("rcosT", np.cos(ang)), ("rsinT", np.sin(ang))):
                b = self._b(f"{nm}{h}", MAXSEQ * half * 4)
                self.r.queue.write_buffer(b, 0, val.astype(np.float32).tobytes())
        for n, sz in [("wq", qd * H), ("wk", hd * H), ("wv", hd * H), ("wo", H * qd),
                      ("wgate", it * H), ("wup", it * H), ("wdown", H * it),
                      ("wplegate", d * H), ("wpleproj", H * d)]:
            self._b(n, sz * 2)
        self._S = S

    # ---- one layer ---------------------------------------------------------
    def _layer(self, L, S, startPos):
        r, c = self.r, self.cfg
        li = self.layers[L]
        sc = li["scales"]
        H, nH, d, nL = c["H"], c["nH"], c["ple_d"], c["nL"]
        hd, qd, it = li["head_dim"], li["q_dim"], li["intermediate"]
        B = lambda n: self._b(n, 0)
        W = lambda n: r._bufs[f"L{L}.{n}"]
        src = li["kv_source"]
        kc, vc = r.kc[src], r.vc[src]
        # The runner's rcos/rsin are PER-TOKEN (pre-offset, `half` entries); the
        # batched kernels index by absolute position, so use the full tables.
        cb, sb = self._b(f"rcosT{hd}", 0), self._b(f"rsinT{hd}", 0)
        T = startPos + S
        shared = li["shared"]

        # q (and k/v where this layer owns a cache)
        r.dispatch("srqh_b", self.K("srqh_b"), [B("a"), B("ah"), self._u(f"s0{L}", 0, S*H, 0, 0)],
                   ((S*H + 255)//256, 1, 1))
        self._dq(W("q_bits"), W("q_scale") if shared else W("qkv_scales"),
                 B("wq"), H, qd, f"q{L}")
        self._gemm(B("ah"), B("wq"), B("qraw"), S, qd, H, tag=f"q{L}")
        r.dispatch("srq_b", self.K("srq_b"),
                   [B("qraw"), B("q"), self._u(f"s1{L}", _f32u(sc["q_out"]), S*qd, 0, 0)],
                   ((S*qd + 255)//256, 1, 1))
        if not shared:
            self._dq(W("k_bits"), W("qkv_scales"), B("wk"), H, hd, f"k{L}",
                     soff=qd * 4)
            self._gemm(B("ah"), B("wk"), B("kraw"), S, hd, H, tag=f"k{L}")
            self._dq(W("v_bits"), W("qkv_scales"), B("wv"), H, hd, f"v{L}",
                     soff=(qd + hd) * 4)
            self._gemm(B("ah"), B("wv"), B("vraw"), S, hd, H, tag=f"v{L}")
            r.dispatch(f"kvnormb_{hd}", self.K("kvnorm_b", (hd, 1, 1), HD=hd, HALF=hd//2),
                       [B("kraw"), B("vraw"), W("k_norm"), cb, sb, kc, vc,
                        self._u(f"kv{L}", startPos, _f32u(sc.get("k_out", 0)),
                                _f32u(sc.get("v_out", 0)), 0)], (S, 1, 1))

        # attention as two GEMMs (dense; banding needs per-block key ranges)
        win = c["window"] if li["sliding"] else 0
        Tp = (T + 7)//8*8
        r.dispatch(f"qprep_{hd}", self.K("qprep_b", (hd, 1, 1), HEAD_DIM=hd, HALF_DIM=hd//2),
                   [B("q"), W("q_norm"), cb, sb, B("qf16"),
                    self._u(f"qp{L}", startPos, nH, 0, 0)], (nH, S, 1))
        for cache, dst in ((kc, "kf16"), (vc, "vf16")):
            r.dispatch("srqh_b", self.K("srqh_b"),
                       [cache, B(dst), self._u(f"kc{L}{dst}", 0, T*hd, 0, 0)],
                       ((T*hd + 255)//256, 1, 1))
        R = S * nH
        self._gemm(B("qf16"), B("kf16"), B("scores"), R, T, hd, ccols=Tp, tag=f"s{L}")
        r.dispatch("smax_b", self.K("smax_b"),
                   [B("scores"), B("denom"),
                    self._u(f"sm{L}", startPos, nH, win, T, Tp, 0, 0, 0)], (R, 1, 1))
        self._gemm(B("scores"), B("vf16"), B("attnraw"), R, hd, T, tr=False,
                   acols=Tp, tag=f"pv{L}")
        r.dispatch("attnout_b", self.K("attnout_b"),
                   [B("attnraw"), B("denom"), B("attnh"),
                    self._u(f"ao{L}", _f32u(sc["o_in"]), R*hd, hd, 0)],
                   ((R*hd + 255)//256, 1, 1))

        # o-proj + residual add-norm + pre-FFN norm
        self._dq(W("o_bits"), W("o_scale"), B("wo"), qd, H, f"o{L}")
        self._gemm(B("attnh"), B("wo"), B("oraw"), S, H, qd, tag=f"o{L}")
        r.dispatch("rmsadd_b", self.K("rmsadd_b"),
                   [B("oraw"), W("o_w12"), B("hidden"),
                    self._u(f"ra{L}", _f32u(sc["o_out"]), _f32u(1.0), H, 0)], (S, 1, 1))
        r.dispatch("rmssrqh_b", self.K("rmssrqh_b"),
                   [B("hidden"), (W("o_w12"), H*4), B("y2"), B("sum2"),
                    self._u(f"rh{L}", _f32u(sc["gate_in"]), H, 0, 0)], (S, 1, 1))

        # MLP
        self._dq(W("gate_bits"), W("gate_scale"), B("wgate"), H, it, f"gt{L}")
        self._gemm(B("y2"), B("wgate"), B("graw"), S, it, H, tag=f"gt{L}")
        self._dq(W("up_bits"), W("up_scale"), B("wup"), H, it, f"up{L}")
        self._gemm(B("y2"), B("wup"), B("uraw"), S, it, H, tag=f"up{L}")
        r.dispatch("geglu_b", self.K("geglu_b"),
                   [B("graw"), B("uraw"), B("geglu"), W("gelu_gate"),
                    self._u(f"gg{L}", _f32u(sc["gate_out"]), _f32u(sc["up_out"]),
                            _f32u(sc["down_in"]), S*it)], ((S*it + 255)//256, 1, 1))
        self._dq(W("down_bits"), W("down_scale"), B("wdown"), it, H, f"dn{L}")
        self._gemm(B("geglu"), B("wdown"), B("draw"), S, H, it,
                   alpha=sc["down_in"], tag=f"dn{L}")
        r.dispatch("rmsadd_b", self.K("rmsadd_b"),
                   [B("draw"), W("down_nw"), B("hidden"),
                    self._u(f"rb{L}", _f32u(sc["down_out"]), _f32u(1.0), H, 0)], (S, 1, 1))

        # per-layer embedding block
        r.dispatch("srqh_b", self.K("srqh_b"),
                   [B("hidden"), B("hh"), self._u(f"s2{L}", _f32u(sc["plegate_in"]), S*H, 0, 0)],
                   ((S*H + 255)//256, 1, 1))
        self._dq(W("plegate_codes"), W("plegate_rowscale"), B("wplegate"), H, d, f"pg{L}")
        self._gemm(B("hh"), B("wplegate"), B("plin"), S, d, H, tag=f"pg{L}")
        r.dispatch("plegate_b", self.K("plegate_b"),
                   [B("plin"), B("ple"), B("gple"), W("gelu_plegate"),
                    self._u(f"pb{L}", _f32u(sc["plegate_out"]), L*d, nL*d, S*d)],
                   ((S*d + 255)//256, 1, 1))
        r.dispatch("srqh_b", self.K("srqh_b"),
                   [B("gple"), B("gph"), self._u(f"s3{L}", _f32u(sc["pleproj_in"]), S*d, 0, 0)],
                   ((S*d + 255)//256, 1, 1))
        self._dq(W("pleproj_codes"), W("pleproj_rowscale"), B("wpleproj"), d, H, f"pp{L}")
        self._gemm(B("gph"), B("wpleproj"), B("ppraw"), S, H, d, tag=f"pp{L}")
        sv = self._sv(L, H)
        r.dispatch("rmsadd_b", self.K("rmsadd_b"),
                   [B("ppraw"), W("pleproj_w12s"), B("hidden"),
                    self._u(f"rc{L}", _f32u(sc["pleproj_out"]), _f32u(sv), H, 0)], (S, 1, 1))
        nxt = self.layers[L+1]["scales"]["qkv_in"] if L+1 < nL else 0.0
        r.dispatch("rmssrq_69", self.K("rmssrq_69"),
                   [B("hidden"), (W("pleproj_w12s"), H*4), B("a"), B("suma"),
                    self._u(f"r6{L}", S, S, _f32u(nxt), 0)], (S, 1, 1))

    # ---- entry point -------------------------------------------------------
    def prefill(self, toks, startPos=0):
        r, c = self.r, self.cfg
        n = len(toks)
        if not n:
            return startPos
        S = min(max(((n + BUCKET - 1)//BUCKET)*BUCKET, BUCKET), MAXSEQ)
        self._alloc(S)
        H, nL, d = c["H"], c["nL"], c["ple_d"]
        B = lambda x: self._b(x, 0)
        ids = r._scratch("pf_ids", S*4)
        r.queue.write_buffer(ids, 0, np.array(list(toks) + [toks[-1]]*(S-n),
                                              np.uint32).tobytes())
        seq = self._u("seq", S, 0, 0, 0)

        for L in range(nL):
            self._sv(L, H)              # readbacks must not happen mid-encoder
        enc = r._enc
        r._enc = r.device.create_command_encoder()
        r.dispatch("embed_00", self.K("embed_00", (64, 1, 1)),
                   [ids, r._bufs["embed_q"], r._bufs["embed_scale"], B("hidden"), seq], (S, 1, 1))
        r.dispatch("srqh_b", self.K("srqh_b"),
                   [B("hidden"), B("ah"), self._u("p0", 0, S*H, 0, 0)], ((S*H + 255)//256, 1, 1))
        self._gemm(B("ah"), r._bufs["pl_model_proj"], B("ctxh"), S, nL*d, H, tag="pm")
        r.dispatch("srq_b", self.K("srq_b"),
                   [B("ctxh"), B("ctx"), self._u("p1", 0, S*nL*d, 0, 0)],
                   ((S*nL*d + 255)//256, 1, 1))
        r.dispatch("plegather_01", self.K("plegather_01", (64, 1, 1)),
                   [ids, r._bufs["ple_q"], r._bufs["ple_scale"], B("plegath"), seq], (S, 1, 1))
        r.dispatch("combine_b", self.K("combine_b"),
                   [B("ctx"), B("plegath"), r._bufs["pl_proj_norm"], B("ple"),
                    self._u("p2", nL, 0, 0, 0)], (nL, S, 1))
        r.dispatch("rmssrq_69", self.K("rmssrq_69"),
                   [B("hidden"), r._bufs["L0.in_norm"], B("a"), B("suma"),
                    self._u("p3", S, S, _f32u(self.layers[0]["scales"]["qkv_in"]), 0)],
                   (S, 1, 1))
        for L in range(nL):
            self._layer(L, S, startPos)
        r.queue.submit([r._enc.finish()])
        r._enc = enc
        # Drain before returning, as the Metal path's commit_wait() does. This
        # is not just tidiness: prefill is one submit of ~500 dispatches, so
        # without it the caller starts its decode timer while seconds of prefill
        # are still queued, and decode "measures" 12 tok/s instead of 87.
        r.read(B("suma"), 4)
        self._live.clear()
        return startPos + n
