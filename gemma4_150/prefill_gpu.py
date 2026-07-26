"""Batched prefill for Gemma 4 E2B QAT: the whole prompt in one pass per layer.

Decode streams one token through all 35 layers, so every weight is read once per
token — for a 1024-token prompt that is 840 GB, capping prefill at ~330 tok/s no
matter how good the kernels are. This module inverts the loop: hold all S token
activations resident and iterate LAYERS OUTER, TOKENS INNER. Then each weight is
read once for the whole prompt, and the matmuls become GEMMs at M=S where MPS
reaches 5.1-5.6 TFLOP/s against the per-token GEMV path's 1.6.

The ordering is legal because every token clears layer L before any reaches
L+1, so a later token's attention can still read an earlier token's KV.

It is also what makes dequantization free: the 2/4/8-bit weights are unpacked to
f16 scratch ONCE PER LAYER (3.75 GB total, ~27 ms) rather than once per token.

Prefill only has to leave the KV caches correct — the caller starts decoding
from the last prompt token — so there are no logits, no final norm and no argmax
here.

    from gemma4_150.prefill_gpu import PrefillGPU
    pf = PrefillGPU(g)          # g is a metal_runner.G4
    pos = pf.prefill(token_ids, 0)
"""
from __future__ import annotations

import os, struct

import Metal
import MetalPerformanceShaders as mps

from gemma4_150.metal_runner import G4, KDIR, _SHARED, Batch

HERE = os.path.dirname(os.path.abspath(__file__))
DQ = os.path.join(HERE, "prefill_research", "dq_f16.metal")
F16 = mps.MPSDataTypeFloat16


def _mps_encode(b, mm, A, B, C):
    """MPS needs its own encoder, so close any open compute encoder first."""
    if b.enc is not None:
        b.enc.endEncoding(); b.enc = None
    mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(b.cb, A, B, C)


class PrefillGPU:
    BUCKET = 16          # batch-size quantum; see prefill()
    ATTN_REPEAT = 1      # >1 dispatches attention N times: a diagnostic knob for
                         # measuring its true share by differencing (the profiler
                         # overweights small dispatches; end-to-end diffs do not)
    ATTN_GEMM = True     # False falls back to the fused attn_prefill kernel; see
                         # _attn_gemm() for why the GEMM form exists
    MAXSEQ = 2048
    # Which GEMM backs the weight matmuls:
    #   "mps" — MPSMatrixMultiplication, 5.64 TFLOP/s, Apple-only
    #   "dsl" — kernels_dsl.gemm_tiled, 2.61 TFLOP/s, runs anywhere the DSL does
    # Metal defaults to MPS because it is there and it is 2.2x faster; wgpu and
    # the browser have no equivalent and must use the DSL kernel. The kernel
    # SOURCE is shared either way — this picks an implementation, not a codebase.
    GEMM = "mps"

    def __init__(self, g: G4):
        self.g = g
        self.dev = g.dev
        self._mm = {}          # (M,N,K,alpha) -> MPSMatrixMultiplication
        self._mat = {}         # (buffer id, rows, cols) -> MPSMatrix
        self._buf = {}
        self._uni_cache = {}
        self._S = 0
        self._compile()
        self._alloc_weights()

    # ---- kernels -----------------------------------------------------------
    def _compile(self):
        g = self.g
        if "gemm_tiled" not in g.kernels:
            from gemma4_150.kernels_dsl import gemm_tiled
            g._compile_one(gemm_tiled.msl, "gemm_tiled", "gemm_tiled")
            # _compile_one parses "threadsPerThreadgroup = (N)" and builds a 1-D
            # MTLSize, so a 2-D kernel silently gets 16 threads instead of 16x16
            # — which computes a quarter of each tile and runs FASTER, i.e. it
            # looks like a win on the clock and is simply wrong.
            pso, _ = g.kernels["gemm_tiled"]
            g.kernels["gemm_tiled"] = (pso, Metal.MTLSizeMake(*gemm_tiled.workgroup_size))
        src = open(DQ).read()
        for nb in (2, 4, 8):
            if f"dq{nb}" not in g.kernels:
                g._compile_one(src.replace("BITS = 2u", f"BITS = {nb}u"), "dq_f16", f"dq{nb}")
        kf = lambda n: os.path.join(KDIR, n + ".metal")
        seen = set()
        for l in range(g.nL):
            li = g.layers[l]
            hd, oin = li["head_dim"], g.sc(l, "o_in")
            if not g._dsl_variant("attn_prefill", f"attnp_{hd}_{l}", {
                    "HEAD_DIM": hd, "HALF_DIM": hd // 2, "HD4": hd // 4,
                    "J_GROUPS": 256 // (hd // 4), "OUT_Q": oin}, 256):
                g._variant(kf("attn_prefill"), f"attnp_{hd}_{l}", [
                    ("HEAD_DIM=512u, HALF_DIM=256u", f"HEAD_DIM={hd}u, HALF_DIM={hd // 2}u"),
                    ("OUT_Q=0.014886821620166302f", f"OUT_Q={oin!r}f")])
            if hd not in seen:
                seen.add(hd)
                if not g._dsl_variant("kvnorm_b", f"kvnormb_{hd}",
                                      {"HD": hd, "HALF": hd // 2}, hd):
                    g._variant(kf("kvnorm_b"), f"kvnormb_{hd}", [
                        ("HD=256u, HALF=128u", f"HD={hd}u, HALF={hd // 2}u"),
                        ("threadsPerThreadgroup = (HD)", f"threadsPerThreadgroup = ({hd})")])
                if not g._dsl_variant("qprep_b", f"qprep_{hd}",
                                      {"HEAD_DIM": hd, "HALF_DIM": hd // 2}, hd):
                    g._variant(kf("qprep_b"), f"qprep_{hd}", [
                        ("HEAD_DIM=512u, HALF_DIM=256u", f"HEAD_DIM={hd}u, HALF_DIM={hd // 2}u"),
                        ("threadsPerThreadgroup = (512)", f"threadsPerThreadgroup = ({hd})")])

    def _new(self, n):
        return self.dev.newBufferWithLength_options_(max(n, 16), _SHARED)

    def _alloc_weights(self):
        g = self.g
        H, nL = g.H, g.nL
        qd = max(g.layers[l]["q_dim"] for l in range(nL))
        hd = max(g.layers[l]["head_dim"] for l in range(nL))
        it = max(g.layers[l]["intermediate"] for l in range(nL))
        W = self._buf
        for n, sz in [("wq", qd * H), ("wk", hd * H), ("wv", hd * H), ("wo", H * qd),
                      ("wgate", it * H), ("wup", it * H), ("wdown", H * it),
                      ("wplegate", g.ple_d * H), ("wpleproj", H * g.ple_d)]:
            W[n] = self._new(sz * 2)

    # ---- MPS plumbing ------------------------------------------------------
    def _matrix(self, buf, rows, cols):
        key = (id(buf), rows, cols)
        m = self._mat.get(key)
        if m is None:
            d = mps.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
                rows, cols, cols * 2, F16)
            m = mps.MPSMatrix.alloc().initWithBuffer_descriptor_(buf, d)
            self._mat[key] = m
        return m

    def _gemm(self, b, A, Wb, C, M, N, K, alpha=1.0, tr=True, acols=None, ccols=None):
        """C[M,N] = alpha * A[M,K] @ (W[N,K]^T if tr else W[K,N]), all f16.

        acols/ccols give A and C a row stride wider than the operation, which the
        attention GEMMs need: the score matrix is padded to a 16-byte-aligned row
        so MPS is happy, while the operation itself covers only the T real keys."""
        if self.GEMM == "dsl":
            return self._gemm_dsl(b, A, Wb, C, M, N, K, alpha, tr, acols, ccols)
        key = (M, N, K, alpha, tr)
        mm = self._mm.get(key)
        if mm is None:
            mm = mps.MPSMatrixMultiplication.alloc()
            mm = mm.initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
                self.dev, False, tr, M, N, K, alpha, 0.0)
            self._mm[key] = mm
        Bm = self._matrix(Wb, N, K) if tr else self._matrix(Wb, K, N)
        _mps_encode(b, mm, self._matrix(A, M, acols or K), Bm, self._matrix(C, M, ccols or N))

    def _gemm_dsl(self, b, A, Wb, C, M, N, K, alpha=1.0, tr=True,
                  acols=None, ccols=None):
        """The portable GEMM: gemma4_150/kernels_dsl.py, emitted for this backend.

        Covers everything MPS does here — both orientations and the row strides
        the attention score matrix needs — so selecting "dsl" needs no fallback."""
        from gemma4_150.kernels_dsl import gemm_groups
        gx, gy = gemm_groups(M, N)
        ldb = K if tr else N
        b.dg2d("gemm_tiled",
               [(A, 0), (Wb, 0), (C, 0),
                self._uni(struct.pack("<3If", M, N, K, alpha)),
                self._u32(acols or K, ldb, ccols or N, 1 if tr else 0)], gx, gy)

    def _dq(self, b, bits, scale, scale_off, dst, n_in, n_out):
        """Dequantize [n_out][n_in] to f16, deriving the bit width from the packed
        buffer's size.

        Do NOT hardcode this. The model mixes widths in a way that is easy to get
        wrong: the narrow-MLP layers (intermediate 6144, layers 0-14) pack
        gate/up/down at 4 bits, while the wide ones (12288, layers 15-34) use 2 —
        which is why there are separate gateup_74/gateup_95 and down_75/down_96
        kernels at all. Reading it off the buffer cannot drift out of sync."""
        nbits = bits.length() * 8 // (n_in * n_out)
        assert nbits in (2, 4, 8), f"odd width {nbits} for [{n_out}x{n_in}]"
        nwords = n_out * (n_in // (32 // nbits))
        b.dg(f"dq{nbits}", [(bits, 0), (scale, scale_off), (dst, 0),
                            self._u32(n_in, n_out, 0, 0)],
             (nwords + 255) // 256)

    # ---- small uniform helpers --------------------------------------------
    # All return a ready-to-bind (buffer, offset) pair, since that is the only
    # way they are ever used.
    def _uni(self, data):
        """Memoize uniform buffers on their contents.

        A prefill issues ~525 dispatches, and allocating a fresh Metal buffer for
        each one's 16-byte uniform was pure CPU overhead — measurable, because it
        is per-prefill work that does NOT shrink as the prompt grows. Almost all
        of these repeat: most depend only on (layer, S), and the byte patterns
        recur across layers and across calls. Keying on the bytes is safe because
        a uniform buffer is never mutated after creation."""
        b = self._uni_cache.get(data)
        if b is None:
            b = self.dev.newBufferWithBytes_length_options_(data, len(data), _SHARED)
            self._uni_cache[data] = b
        return (b, 0)

    def _u32(self, *v):
        return self._uni(struct.pack("<%dI" % len(v), *v))

    def _mix(self, *v):
        """Pack a uniform of mixed float/int words (float(x) picks the f32 slot)."""
        return self._uni(b"".join(
            struct.pack("<f", x) if isinstance(x, float) else struct.pack("<I", x) for x in v))

    # ---- activation arena --------------------------------------------------
    def _alloc_acts(self, S):
        if S <= self._S:
            return
        g = self.g
        H, nL, d = g.H, g.nL, g.ple_d
        qd = max(g.layers[l]["q_dim"] for l in range(g.nL))
        hd = max(g.layers[l]["head_dim"] for l in range(g.nL))
        it = max(g.layers[l]["intermediate"] for l in range(g.nL))
        B = self._buf
        f32 = [("hidden", H), ("a", H), ("y2n", H), ("q", qd), ("attn", qd),
               ("ctx", nL * d), ("plegath", nL * d), ("ple", nL * d), ("gple", d)]
        f16 = [("ah", H), ("qraw", qd), ("kraw", hd), ("vraw", hd), ("attnh", qd),
               ("oraw", H), ("y2", H), ("graw", it), ("uraw", it), ("geglu", it),
               ("draw", H), ("ctxh", nL * d), ("hh", H), ("plin", d), ("gph", d),
               ("ppraw", H), ("qf16", qd), ("attnraw", qd)]
        for n, w in f32:
            B[n] = self._new(S * w * 4)
        for n, w in f16:
            B[n] = self._new(S * w * 2)
        for n in ("suma", "sum2"):
            B[n] = self._new(S * 4)
        # GEMM-attention scratch. `scores` is the big one: S*nH rows of up to
        # MAXSEQ keys (33 MB at S=1024). k/v are converted copies of the caches.
        B["scores"] = self._new(S * g.nH * self.MAXSEQ * 2)
        B["denom"] = self._new(S * g.nH * 4)
        B["kf16"] = self._new(self.MAXSEQ * hd * 2)
        B["vf16"] = self._new(self.MAXSEQ * hd * 2)
        self._S = S

    # ---- attention as two GEMMs -------------------------------------------
    def _attn_gemm(self, b, l, S, startPos):
        """Attention for the whole prompt as QK^T -> softmax -> PV.

        attn_prefill computes the same thing with a hand-rolled online softmax,
        which never touches the matrix units: end-to-end differencing put it at
        25.8% of prefill, the largest single item left. At prefill sizes both
        products are ordinary GEMMs, and because kvHeads==1 neither needs
        batching — q's [S][h][d] layout is already [S*h][d], and every row of it
        multiplies the same K/V cache.

        The cost is arithmetic the fused kernel skipped: a dense score matrix
        computes the masked-out upper triangle (and, on sliding layers,
        everything outside the window) and throws it away. That is a ~2x FLOP
        overhead bought against a ~3x throughput gain."""
        g, B = self.g, self._buf
        li = g.layers[l]
        nH, hd, qd = g.nH, li["head_dim"], li["q_dim"]
        lw = lambda n: g.w[f"L{l}.{n}"]
        src = li["kv_source"]
        kc, vc = g.kc[src], g.vc[src]
        cb, sb = g.pool[f"rcosT_{hd}"], g.pool[f"rsinT_{hd}"]
        T = startPos + S
        Tp = (T + 7) // 8 * 8      # 16-byte-aligned row stride for MPS
        R, win = S * nH, (g.window if li["sliding"] else 0)

        b.dg2d(f"qprep_{hd}", [(B["q"], 0), (lw("q_norm"), 0), (cb, 0), (sb, 0),
                               (B["qf16"], 0), (self._u32(startPos, nH, 0, 0))], nH, S)
        # The caches are f32; MPS wants f16. Converted per layer rather than once
        # per distinct cache: kf16/vf16 are single scratch buffers, so a shared
        # cache's copy would have been overwritten by the intervening layers, and
        # the conversion is ~3 MB of traffic against the GEMM's hundreds.
        for cache, dst in ((kc, "kf16"), (vc, "vf16")):
            b.dg("srqh_b", [(cache, 0), (B[dst], 0), self._mix(0.0, T * hd, 0, 0)],
                 (T * hd + 255) // 256)
        for _ in range(self.ATTN_REPEAT):
            self._gemm(b, B["qf16"], B["kf16"], B["scores"], R, T, hd, ccols=Tp)
            b.dg("smax_b", [(B["scores"], 0), (B["denom"], 0),
                            self._u32(startPos, nH, win, T, Tp, 0, 0, 0)], R)
            self._gemm(b, B["scores"], B["vf16"], B["attnraw"], R, hd, T,
                       tr=False, acols=Tp)
        b.dg("attnout_b", [(B["attnraw"], 0), (B["denom"], 0), (B["attnh"], 0),
                           self._mix(g.sc(l, "o_in"), R * hd, hd, 0)],
             (R * hd + 255) // 256)

    # ---- one layer ---------------------------------------------------------
    def _layer(self, b, l, S, startPos):
        g, B = self.g, self._buf
        li = g.layers[l]
        H, nH, d, nL = g.H, g.nH, g.ple_d, g.nL
        hd, qd, it = li["head_dim"], li["q_dim"], li["intermediate"]
        lw = lambda n: g.w[f"L{l}.{n}"]
        sc, sco = lambda k: g.sc(l, k), lambda k: g.sc_opt(l, k)
        kc, vc = g.kc[li["kv_source"]], g.vc[li["kv_source"]]
        cb, sb = g.pool[f"rcosT_{hd}"], g.pool[f"rsinT_{hd}"]

        # ---- q (and k/v on the layers that own a cache)
        b.dg("srqh_b", [(B["a"], 0), (B["ah"], 0), (self._mix(0.0, S * H, 0, 0))],
             (S * H + 255) // 256)
        shared = li["shared"]
        qsc, qoff = (lw("q_scale"), 0) if shared else (lw("qkv_scales"), 0)
        self._dq(b, lw("q_bits"), qsc, qoff, B["wq"], H, qd)
        self._gemm(b, B["ah"], B["wq"], B["qraw"], S, qd, H)
        b.dg("srq_b", [(B["qraw"], 0), (B["q"], 0), (self._mix(sc("q_out"), S * qd, 0, 0))],
             (S * qd + 255) // 256)
        if not shared:
            self._dq(b, lw("k_bits"), lw("qkv_scales"), qd * 4, B["wk"], H, hd)
            self._gemm(b, B["ah"], B["wk"], B["kraw"], S, hd, H)
            self._dq(b, lw("v_bits"), lw("qkv_scales"), (qd + hd) * 4, B["wv"], H, hd)
            self._gemm(b, B["ah"], B["wv"], B["vraw"], S, hd, H)
            b.dg(f"kvnormb_{hd}", [(B["kraw"], 0), (B["vraw"], 0), (lw("k_norm"), 0), (cb, 0),
                                   (sb, 0), (kc, 0), (vc, 0),
                                   (self._mix(startPos, sco("k_out"), sco("v_out"), 0))], S)

        # ---- attention (causal, whole batch)
        if self.ATTN_GEMM:
            self._attn_gemm(b, l, S, startPos)          # writes B["attnh"] directly
        else:
            win = g.window if li["sliding"] else 0
            for _ in range(self.ATTN_REPEAT):
                b.dg2d(f"attnp_{hd}_{l}",
                       [(B["q"], 0), (lw("q_norm"), 0), (cb, 0), (sb, 0), (kc, 0), (vc, 0),
                        (B["attn"], 0), (self._u32(startPos, nH, 1, win, S, 0, 0, 0))], nH, S)
            b.dg("srqh_b", [(B["attn"], 0), (B["attnh"], 0), (self._mix(0.0, S * qd, 0, 0))],
                 (S * qd + 255) // 256)

        # ---- o-proj + residual add-norm + pre-FFN norm
        self._dq(b, lw("o_bits"), lw("o_scale"), 0, B["wo"], qd, H)
        self._gemm(b, B["attnh"], B["wo"], B["oraw"], S, H, qd)
        b.dg("rmsadd_b", [(B["oraw"], 0), (lw("o_w12"), 0), (B["hidden"], 0),
                          (self._mix(sc("o_out"), 1.0, H, 0))], S)
        b.dg("rmssrqh_b", [(B["hidden"], 0), (lw("o_w12"), H * 4), (B["y2"], 0), (B["sum2"], 0),
                           (self._mix(sc("gate_in"), H, 0, 0))], S)

        # ---- MLP
        self._dq(b, lw("gate_bits"), lw("gate_scale"), 0, B["wgate"], H, it)
        self._gemm(b, B["y2"], B["wgate"], B["graw"], S, it, H)
        self._dq(b, lw("up_bits"), lw("up_scale"), 0, B["wup"], H, it)
        self._gemm(b, B["y2"], B["wup"], B["uraw"], S, it, H)
        b.dg("geglu_b", [(B["graw"], 0), (B["uraw"], 0), (B["geglu"], 0), (lw("gelu_gate"), 0),
                         (self._mix(sc("gate_out"), sc("up_out"), sc("down_in"), S * it))],
             (S * it + 255) // 256)
        self._dq(b, lw("down_bits"), lw("down_scale"), 0, B["wdown"], it, H)
        self._gemm(b, B["geglu"], B["wdown"], B["draw"], S, H, it, alpha=sc("down_in"))
        b.dg("rmsadd_b", [(B["draw"], 0), (lw("down_nw"), 0), (B["hidden"], 0),
                          (self._mix(sc("down_out"), 1.0, H, 0))], S)

        # ---- per-layer embedding block
        b.dg("srqh_b", [(B["hidden"], 0), (B["hh"], 0), (self._mix(sc("plegate_in"), S * H, 0, 0))],
             (S * H + 255) // 256)
        self._dq(b, lw("plegate_codes"), lw("plegate_rowscale"), 0, B["wplegate"], H, d)
        self._gemm(b, B["hh"], B["wplegate"], B["plin"], S, d, H)
        b.dg("plegate_b", [(B["plin"], 0), (B["ple"], 0), (B["gple"], 0), (lw("gelu_plegate"), 0),
                           (self._mix(sc("plegate_out"), l * d, nL * d, S * d))],
             (S * d + 255) // 256)
        b.dg("srqh_b", [(B["gple"], 0), (B["gph"], 0), (self._mix(sc("pleproj_in"), S * d, 0, 0))],
             (S * d + 255) // 256)
        self._dq(b, lw("pleproj_codes"), lw("pleproj_rowscale"), 0, B["wpleproj"], d, H)
        self._gemm(b, B["gph"], B["wpleproj"], B["ppraw"], S, H, d)
        sv = struct.unpack_from("<f", lw("pleproj_w12s").contents().as_buffer((2 * H + 1) * 4),
                                2 * H * 4)[0]
        b.dg("rmsadd_b", [(B["ppraw"], 0), (lw("pleproj_w12s"), 0), (B["hidden"], 0),
                          (self._mix(sc("pleproj_out"), sv, H, 0))], S)
        nxt = g.sc(l + 1, "qkv_in") if l + 1 < nL else 0.0
        b.dg("rmssrq_69", [(B["hidden"], 0), (lw("pleproj_w12s"), H * 4), (B["a"], 0),
                           (B["suma"], 0), (self._mix(S, S, nxt, 0))], S)

    # ---- entry point -------------------------------------------------------
    def prefill(self, toks, startPos=0):
        """Run `toks` through the model from `startPos`, filling the KV caches.
        Returns the new position."""
        if not toks:
            return startPos
        g, B = self.g, self._buf
        n = len(toks)
        # Round the batch up to a bucket. MPS builds a pipeline per unique
        # (M,N,K,alpha), so a chat whose turns are 37, 52, 61 tokens long would
        # pay that build on every turn; bucketing makes them share one. The
        # padding repeats the last token, and its KV lands at positions
        # >= startPos+n which no real query can attend to (attention is bounded
        # by each query's own absolute position) and which the next prefill or
        # decode step overwrites.
        S = min(max(((n + self.BUCKET - 1) // self.BUCKET) * self.BUCKET, self.BUCKET), 2048)
        self._alloc_acts(S)
        ids = g._buf_u32(list(toks) + [toks[-1]] * (S - n))
        H, nL, d = g.H, g.nL, g.ple_d
        seq = self._u32(S, 0, 0, 0)

        b = Batch(g)
        b.dg("embed_00", [(ids, 0), (g.w["embed_q"], 0), (g.w["embed_scale"], 0),
                          (B["hidden"], 0), seq], S)
        # per-layer model projection: already f16, no dequant needed
        b.dg("srqh_b", [(B["hidden"], 0), (B["ah"], 0), (self._mix(0.0, S * H, 0, 0))],
             (S * H + 255) // 256)
        self._gemm(b, B["ah"], g.w["pl_model_proj"], B["ctxh"], S, nL * d, H)
        b.dg("srq_b", [(B["ctxh"], 0), (B["ctx"], 0), (self._mix(0.0, S * nL * d, 0, 0))],
             (S * nL * d + 255) // 256)
        b.dg("plegather_01", [(ids, 0), (g.w["ple_q"], 0), (g.w["ple_scale"], 0),
                              (B["plegath"], 0), seq], S)
        b.dg2d("combine_b", [(B["ctx"], 0), (B["plegath"], 0), (g.w["pl_proj_norm"], 0),
                             (B["ple"], 0), (self._u32(nL, 0, 0, 0))], nL, S)
        b.dg("rmssrq_69", [(B["hidden"], 0), (g.w["L0.in_norm"], 0), (B["a"], 0), (B["suma"], 0),
                           (self._mix(S, S, g.sc(0, "qkv_in"), 0))], S)
        for l in range(nL):
            self._layer(b, l, S, startPos)
        b.commit_wait()
        return startPos + n   # padding is scratch; only real tokens advance pos
