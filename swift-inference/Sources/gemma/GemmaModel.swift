import Foundation
import Metal

/// CPU-driven Gemma 3 decoder mirroring gemma3/runner.py's f16+subgroups
/// path: same buffer layout, same kernel sequence, one command buffer per
/// step. Apple GPUs have simdgroup width 32, so the _sg kernels are safe.
final class GemmaModel {
    let cfg: GemmaConfig
    let runner: MetalRunner
    let weights: WeightStore
    let maxSeq: Int

    private var b: [String: MTLBuffer] = [:]
    private var d: [String: MTLBuffer] = [:]
    private var kCache: [MTLBuffer] = []
    private var vCache: [MTLBuffer] = []

    init(cfg: GemmaConfig, runner: MetalRunner, weights: WeightStore, maxSeq: Int = 1024) throws {
        self.cfg = cfg
        self.runner = runner
        self.weights = weights
        self.maxSeq = maxSeq

        let h = cfg.hiddenSize, hd = cfg.headDim, nh = cfg.numHeads
        let inter = cfg.intermediateSize
        let qDim = nh * hd

        func fbuf(_ name: String, _ n: Int) throws {
            b[name] = try runner.makeBuffer(bytes: n * 4, label: name)
        }
        try fbuf("x", h); try fbuf("xn", h)
        try fbuf("qkv", qDim + 2 * hd); try fbuf("qn", qDim); try fbuf("kn", hd)
        try fbuf("scores", nh * maxSeq); try fbuf("attn", qDim); try fbuf("attn_proj", h)
        try fbuf("gateup", 2 * inter); try fbuf("ffh", inter)
        try fbuf("mlp_out", h)
        try fbuf("logits", cfg.vocabSize)
        try fbuf("token", 1)

        for _ in 0..<cfg.numLayers {
            kCache.append(try runner.makeBuffer(bytes: maxSeq * hd * 4, label: "k_cache"))
            vCache.append(try runner.makeBuffer(bytes: maxSeq * hd * 4, label: "v_cache"))
        }

        func ubuf(_ name: String, _ vals: [UInt32]) throws {
            d[name] = try runner.makeBuffer(vals, label: name)
        }
        try ubuf("embed", [UInt32(h)])
        try ubuf("norm_h", [1, UInt32(h)])
        try ubuf("norm_q", [UInt32(nh), UInt32(hd)])
        try ubuf("norm_k", [1, UInt32(hd)])
        try ubuf("mv_qkv", [UInt32(qDim + 2 * hd), UInt32(h)])
        try ubuf("mv_o", [UInt32(h), UInt32(qDim)])
        try ubuf("mv_gateup", [UInt32(2 * inter), UInt32(h)])
        try ubuf("mv_down", [UInt32(h), UInt32(inter)])
        try ubuf("mv_logits", [UInt32(cfg.vocabSize), UInt32(h)])
        try ubuf("rope_q", [UInt32(nh), UInt32(hd)])
        try ubuf("rope_k", [1, UInt32(hd)])
        try ubuf("geglu", [UInt32(inter)])
        // dynamic, rewritten each step
        try ubuf("kv_append", [UInt32(hd), 0])
        try ubuf("scores_sliding", [UInt32(nh), UInt32(hd), 0, 0, UInt32(maxSeq)])
        try ubuf("scores_full", [UInt32(nh), UInt32(hd), 0, 0, UInt32(maxSeq)])
        d["rope_local"] = try runner.makeBuffer([cfg.ropeThetaLocal, 0], label: "rope_local")
        d["rope_global"] = try runner.makeBuffer([cfg.ropeThetaGlobal, 0], label: "rope_global")
    }

    private func writeStepParams(tokenId: Int, pos: Int) {
        let kvLen = UInt32(pos + 1)
        let start = UInt32(max(0, pos + 1 - cfg.slidingWindow))
        b["token"]!.contents().assumingMemoryBound(to: UInt32.self)[0] = UInt32(tokenId)
        d["kv_append"]!.contents().assumingMemoryBound(to: UInt32.self)[1] = UInt32(pos)
        let sl = d["scores_sliding"]!.contents().assumingMemoryBound(to: UInt32.self)
        sl[2] = kvLen; sl[3] = start
        d["scores_full"]!.contents().assumingMemoryBound(to: UInt32.self)[2] = kvLen
        d["rope_local"]!.contents().assumingMemoryBound(to: Float.self)[1] = Float(pos)
        d["rope_global"]!.contents().assumingMemoryBound(to: Float.self)[1] = Float(pos)
    }

    /// One decoder forward pass; returns a pointer to logits (valid until the
    /// next step) or nil when `wantLogits` is false.
    func step(tokenId: Int, pos: Int, wantLogits: Bool = true) throws -> UnsafeBufferPointer<Float>? {
        guard pos < maxSeq else { throw RuntimeError("position \(pos) exceeds maxSeq \(maxSeq)") }
        let h = cfg.hiddenSize, hd = cfg.headDim, nh = cfg.numHeads
        let inter = cfg.intermediateSize
        let qDim = nh * hd
        let qBytes = qDim * 4, kvBytes = hd * 4
        let w = weights

        writeStepParams(tokenId: tokenId, pos: pos)

        let batch = try runner.batch()
        try batch.dispatch("embed_scale_f16",
            buffers: [b["token"]!, w["model.embed_tokens.weight"], b["x"]!, d["embed"]!],
            totalThreads: h)

        for L in 0..<cfg.numLayers {
            let p = "model.layers.\(L)."
            let a = p + "self_attn.", m = p + "mlp."
            let sliding = cfg.layerTypes[L] == "sliding_attention"
            let ropeF = d[sliding ? "rope_local" : "rope_global"]!
            let scoresD = d[sliding ? "scores_sliding" : "scores_full"]!
            let qkv = b["qkv"]!

            try batch.dispatchGroups("rmsnorm_wg_sg",
                buffers: [(b["x"]!, 0), (w[p + "input_layernorm.weight"], 0),
                          (b["xn"]!, 0), (d["norm_h"]!, 0)], groups: 1)
            try batch.dispatchGroups("matvec_wg_packed_sg",
                buffers: [(w[a + "qkv_proj.weight"], 0), (b["xn"]!, 0),
                          (qkv, 0), (d["mv_qkv"]!, 0)], groups: qDim + 2 * hd)
            try batch.dispatchGroups("rmsnorm_wg_sg",
                buffers: [(qkv, 0), (w[a + "q_norm.weight"], 0),
                          (b["qn"]!, 0), (d["norm_q"]!, 0)], groups: nh)
            try batch.dispatchGroups("rmsnorm_wg_sg",
                buffers: [(qkv, qBytes), (w[a + "k_norm.weight"], 0),
                          (b["kn"]!, 0), (d["norm_k"]!, 0)], groups: 1)
            try batch.dispatch("rope",
                buffers: [(b["qn"]!, 0), (ropeF, 0), (d["rope_q"]!, 0)],
                totalThreads: qDim / 2)
            try batch.dispatch("rope",
                buffers: [(b["kn"]!, 0), (ropeF, 0), (d["rope_k"]!, 0)],
                totalThreads: hd / 2)
            try batch.dispatch("kv_append",
                buffers: [(b["kn"]!, 0), (kCache[L], 0), (d["kv_append"]!, 0)],
                totalThreads: hd)
            try batch.dispatch("kv_append",
                buffers: [(qkv, qBytes + kvBytes), (vCache[L], 0), (d["kv_append"]!, 0)],
                totalThreads: hd)
            try batch.dispatchGroups("attention_fused_sg",
                buffers: [(b["qn"]!, 0), (kCache[L], 0), (vCache[L], 0),
                          (b["scores"]!, 0), (b["attn"]!, 0), (scoresD, 0)], groups: nh)
            try batch.dispatchGroups("matvec_wg_packed_sg",
                buffers: [(w[a + "o_proj.weight"], 0), (b["attn"]!, 0),
                          (b["attn_proj"]!, 0), (d["mv_o"]!, 0)], groups: h)
            try batch.dispatchGroups("rmsnorm_add_wg_sg",
                buffers: [(b["attn_proj"]!, 0), (w[p + "post_attention_layernorm.weight"], 0),
                          (b["x"]!, 0), (d["norm_h"]!, 0)], groups: 1)
            try batch.dispatchGroups("rmsnorm_wg_sg",
                buffers: [(b["x"]!, 0), (w[p + "pre_feedforward_layernorm.weight"], 0),
                          (b["xn"]!, 0), (d["norm_h"]!, 0)], groups: 1)
            try batch.dispatchGroups("matvec_wg_packed_sg",
                buffers: [(w[m + "gateup_proj.weight"], 0), (b["xn"]!, 0),
                          (b["gateup"]!, 0), (d["mv_gateup"]!, 0)], groups: 2 * inter)
            try batch.dispatch("geglu",
                buffers: [(b["gateup"]!, 0), (b["gateup"]!, inter * 4),
                          (b["ffh"]!, 0), (d["geglu"]!, 0)],
                totalThreads: inter)
            try batch.dispatchGroups("matvec_wg_packed_sg",
                buffers: [(w[m + "down_proj.weight"], 0), (b["ffh"]!, 0),
                          (b["mlp_out"]!, 0), (d["mv_down"]!, 0)], groups: h)
            try batch.dispatchGroups("rmsnorm_add_wg_sg",
                buffers: [(b["mlp_out"]!, 0), (w[p + "post_feedforward_layernorm.weight"], 0),
                          (b["x"]!, 0), (d["norm_h"]!, 0)], groups: 1)
        }

        if wantLogits {
            try batch.dispatchGroups("rmsnorm_wg_sg",
                buffers: [(b["x"]!, 0), (w["model.norm.weight"], 0),
                          (b["xn"]!, 0), (d["norm_h"]!, 0)], groups: 1)
            try batch.dispatch("matvec_packed",
                buffers: [w["model.embed_tokens.weight"], b["xn"]!,
                          b["logits"]!, d["mv_logits"]!],
                totalThreads: cfg.vocabSize)
        }
        batch.commitAndWait()

        guard wantLogits else { return nil }
        return UnsafeBufferPointer(
            start: b["logits"]!.contents().assumingMemoryBound(to: Float.self),
            count: cfg.vocabSize)
    }

    /// Greedy generation from prompt ids. Returns generated ids (prompt
    /// excluded) and the decode-only rate; `ignoreEOS` keeps generating past
    /// end-of-turn for benchmarking.
    func generateGreedy(promptIds: [Int], maxNewTokens: Int,
                        ignoreEOS: Bool = false) throws -> (ids: [Int], decodeTokPerSec: Double) {
        precondition(!promptIds.isEmpty)
        for (i, tok) in promptIds.dropLast().enumerated() {
            _ = try step(tokenId: tok, pos: i, wantLogits: false)
        }
        var next = promptIds.last!
        var pos = promptIds.count - 1
        var out: [Int] = []
        let t0 = Date()
        while out.count < maxNewTokens {
            let logits = try step(tokenId: next, pos: pos, wantLogits: true)!
            var best = 0
            var bestVal = -Float.infinity
            for i in 0..<logits.count where logits[i] > bestVal {
                bestVal = logits[i]; best = i
            }
            if !ignoreEOS && cfg.eosTokenIds.contains(best) { break }
            out.append(best)
            next = best
            pos += 1
        }
        let dt = Date().timeIntervalSince(t0)
        return (out, Double(out.count) / dt)
    }
}
