import Foundation
import Metal

// Full gemma4_150 decode runner in Swift/Metal — mirrors gemma4_150/runner.py.
final class G4 {
    let runner: MetalRunner
    let repo: URL, gdir: URL, kdir: URL
    let man: [String: Any], cfg: [String: Any]
    let layers: [[String: Any]]
    let weights: Data
    var tensors: [String: (off: Int, len: Int)] = [:]
    var bufs: [String: MTLBuffer] = [:]      // weights (cached)
    var pool: [String: MTLBuffer] = [:]      // scratch
    var kc: [Int: MTLBuffer] = [:], vc: [Int: MTLBuffer] = [:]
    // ---- multi-turn chat state (KV cache persists across turns) ----
    var chatPos = 0                 // next free KV slot
    var chatCarry: UInt32? = nil    // last generated token, not yet in the cache

    func C(_ k: String) -> Int { (cfg[k] as! NSNumber).intValue }
    func sc(_ l: Int, _ k: String) -> Float { Float(((layers[l]["scales"] as! [String: Any])[k] as! NSNumber).doubleValue) }
    func scOpt(_ l: Int, _ k: String) -> Float {
        guard let v = (layers[l]["scales"] as! [String: Any])[k] as? NSNumber else { return 0 }
        return Float(v.doubleValue)
    }
    func LI(_ l: Int, _ k: String) -> Int { (layers[l][k] as! NSNumber).intValue }
    func LB(_ l: Int, _ k: String) -> Bool { (layers[l][k] as! NSNumber).boolValue }

    init() throws {
        repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        gdir = repo.appendingPathComponent("models/gemma-4-E2B-qat/g4_150")
        kdir = repo.appendingPathComponent("gemma4_150/kernels_gen")  // generated from kernels_dsl.py
        man = try JSONSerialization.jsonObject(with: Data(contentsOf: gdir.appendingPathComponent("manifest.json"))) as! [String: Any]
        cfg = man["config"] as! [String: Any]
        layers = man["layers"] as! [[String: Any]]
        for (n, v) in man["tensors"] as! [String: Any] {
            let t = v as! [String: Any]; tensors[n] = (t["off"] as! Int, t["len"] as! Int)
        }
        weights = try Data(contentsOf: gdir.appendingPathComponent("weights.bin"), options: .mappedIfSafe)
        runner = try MetalRunner()
        try compileAll()
        try setup()
    }

    func W(_ name: String) -> MTLBuffer {
        if let b = bufs[name] { return b }
        let t = tensors[name]!
        let b = weights.withUnsafeBytes { raw in
            runner.device.makeBuffer(bytes: raw.baseAddress!.advanced(by: t.off), length: t.len, options: .storageModeShared)!
        }
        bufs[name] = b; return b
    }
    func S(_ name: String, _ bytes: Int) -> MTLBuffer {
        if let b = pool[name], b.length >= bytes { return b }
        let b = try! runner.makeBuffer(bytes: max(bytes, 16), label: name); pool[name] = b; return b
    }
    func u32(_ a: [UInt32]) -> MTLBuffer { try! runner.makeBuffer(a, label: "u") }
    func f32(_ a: [Float]) -> MTLBuffer { try! runner.makeBuffer(a, label: "f") }
    // cached (static) uniform buffers — created once, reused every token
    var ucache: [String: MTLBuffer] = [:]
    func UF(_ k: String, _ a: [Float]) -> MTLBuffer { if let b = ucache[k] { return b }; let b = f32(a); ucache[k] = b; return b }
    func UU(_ k: String, _ a: [UInt32]) -> MTLBuffer { if let b = ucache[k] { return b }; let b = u32(a); ucache[k] = b; return b }

    // ---- compile fixed kernels + per-shape variants ----
    func compileAll() throws {
        for f in try FileManager.default.contentsOfDirectory(atPath: kdir.path) where f.hasSuffix(".metal") {
            try runner.compileKernel(file: kdir.appendingPathComponent(f).path)
        }
        func file(_ n: String) -> String { kdir.appendingPathComponent("\(n).metal").path }
        var seenHd = Set<Int>()
        for l in 0..<C("nL") {
            let hd = LI(l, "head_dim"), qd = LI(l, "q_dim")
            let total = qd/2 + hd
            try runner.compileVariant(file: file("qkv_70"), name: "qkv_\(qd)_\(hd)", replace: [
                ("Q_OUT=2048u", "Q_OUT=\(qd)u"), ("KV_OUT=256u", "KV_OUT=\(hd)u"),
                ("Q_WGS=1024u", "Q_WGS=\(qd/2)u"), ("KV_WGS=128u", "KV_WGS=\(hd/2)u"),
                ("TOTAL_WGS=1280u", "TOTAL_WGS=\(total)u"), ("GRID_X=1280u", "GRID_X=\(total)u")])
            try runner.compileVariant(file: file("oproj_73"), name: "oproj_\(qd)", replace: [
                ("IN_FEATURES=2048u", "IN_FEATURES=\(qd)u"), ("WPR=256u", "WPR=\(qd/8)u")])
            let oin = sc(l, "o_in")
            try runner.compileVariant(file: file("attn_101"), name: "attn_\(hd)_\(l)", replace: [
                ("HEAD_DIM=512u", "HEAD_DIM=\(hd)u"), ("HALF_DIM=256u", "HALF_DIM=\(hd/2)u"),
                // HD4/J_GROUPS/PP_COUNTER_BASE derive from HEAD_DIM and must be
                // patched too. Omitting them leaves 256-wide layers running
                // 512-wide constants: full speed, wrong output.
                ("HD4=128u", "HD4=\(hd/4)u"), ("J_GROUPS=2u", "J_GROUPS=\(256/(hd/4))u"),
                ("PP_COUNTER_BASE=131584u", "PP_COUNTER_BASE=\(8*32*(hd+2))u"),
                ("OUT_Q=0.014886821620166302f", "OUT_Q=\(oin)f")])
            if !seenHd.contains(hd) {
                seenHd.insert(hd)
                try runner.compileVariant(file: file("kvnorm"), name: "kvnorm_\(hd)", replace: [
                    ("HD=256u", "HD=\(hd)u"), ("HALF=128u", "HALF=\(hd/2)u"),
                    ("threadsPerThreadgroup = (256)", "threadsPerThreadgroup = (\(hd))")])
            }
        }
    }

    func setup() throws {
        let H = C("H"), d = C("ple_d"), nL = C("nL"), maxSeq = 2048
        for l in 0..<nL where !LB(l, "shared") {
            let hd = LI(l, "head_dim")
            kc[l] = try runner.makeBuffer(bytes: maxSeq*hd*4, label: "kc\(l)")
            vc[l] = try runner.makeBuffer(bytes: maxSeq*hd*4, label: "vc\(l)")
        }
        // pre-size + zero atomics
        _ = S("hidden", H*4); _ = S("gen", 256*4)
        for (n, sz) in [("pp73", (H+1)*4), ("pp75", (H+1)*4), ("pp77", (H+1)*4),
                        ("partials256", (8*32*258+8)*4), ("partials512", (8*32*514+8)*4)] {
            let b = S(n, sz); memset(b.contents(), 0, sz)
        }
        // precompute rope cos/sin for all positions per head-dim type (bind at offset pos*half)
        var byHd: [Int: (Double, Int)] = [:]
        for l in 0..<nL { byHd[LI(l, "head_dim")] = ((layers[l]["rope_theta"] as! NSNumber).doubleValue, LI(l, "rope_cutoff")) }
        for (hd, (theta, cutoff)) in byHd {
            let half = hd/2
            var cosv = [Float](repeating: 0, count: maxSeq*half), sinv = cosv
            for pos in 0..<maxSeq { for i in 0..<half {
                let inv = i < cutoff ? 1.0 / pow(theta, Double(i)/Double(half)) : 0.0
                let ang = Double(pos) * inv
                cosv[pos*half+i] = Float(cos(ang)); sinv[pos*half+i] = Float(sin(ang))
            } }
            pool["rcosT_\(hd)"] = try runner.makeBuffer(cosv, label: "rcosT\(hd)")
            pool["rsinT_\(hd)"] = try runner.makeBuffer(sinv, label: "rsinT\(hd)")
        }
    }

    // Ring of per-step uniform buffers. Chunked decode commits many forwards
    // without waiting, so a *single* parA/pkv buffer would be overwritten by the
    // CPU before the GPU consumes it (pkv sets the K/V write slot — racing it
    // corrupts the cache). `slot = step % RING` gives each in-flight forward its
    // own copy; RING ≥ max chunk guarantees no live buffer is reused.
    static let RING = 32

    // ---- per-token dynamic uniforms (attention params + kv dstOffset per type) ----
    func writeStepUniforms(_ pos: Int, _ slot: Int) {
        let nH = C("nH"), win = C("window")
        var byHd: [Int: Bool] = [:]
        for l in 0..<C("nL") { byHd[LI(l, "head_dim")] = LB(l, "sliding") }
        for (hd, sliding) in byHd {
            S("parA_\(hd)_\(slot)", 32).contents().bindMemory(to: UInt32.self, capacity: 8)
                .update(from: [1, UInt32(pos+1), UInt32(pos), UInt32(nH), 1, UInt32(sliding ? win : 0), 0, 0], count: 8)
            S("pkv_\(hd)_\(slot)", 16).contents().bindMemory(to: UInt32.self, capacity: 4)
                .update(from: [UInt32(pos*hd), 0, 0, 0], count: 4)
        }
    }

    func embed(_ b: MetalRunner.CommandBatch, _ ids: MTLBuffer, _ idsOff: Int, _ out: MTLBuffer) throws {
        try b.dispatchGroups("embed_00", buffers: [(ids,idsOff),(W("embed_q"),0),(W("embed_scale"),0),(out,0),(UU("emb",[1,0,0,0]),0)], groups: 1)
    }
    func pleInput(_ b: MetalRunner.CommandBatch, _ ids: MTLBuffer, _ idsOff: Int, _ hidden: MTLBuffer) throws -> MTLBuffer {
        let nL = C("nL"), d = C("ple_d")
        let ctx = S("ctx", nL*d*4), ple = S("plegath", nL*d*4), out = S("ple", nL*d*4)
        try b.dispatchGroups("proj_68", buffers: [(hidden,0),(W("pl_model_proj"),0),(ctx,0),(UF("z4",[0,0,0,0]),0)], groups: 1120)
        try b.dispatchGroups("plegather_01", buffers: [(ids,idsOff),(W("ple_q"),0),(W("ple_scale"),0),(ple,0),(UU("seq1",[1,0,0,0]),0)], groups: 1)
        try b.dispatchGroups("combine", buffers: [(ctx,0),(ple,0),(W("pl_proj_norm"),0),(out,0)], groups: nL)
        return out
    }

    func layer(_ b: MetalRunner.CommandBatch, _ l: Int, _ pos: Int, _ slot: Int, _ hidden: MTLBuffer, _ ple: MTLBuffer) throws {
        let H = C("H"), nH = C("nH"), hd = LI(l,"head_dim"), qd = LI(l,"q_dim"), inter = LI(l,"intermediate")
        let half = hd/2, shared = LB(l,"shared"), src = LI(l,"kv_source"), d = C("ple_d")
        func LW(_ n: String) -> MTLBuffer { W("L\(l).\(n)") }
        let cb = pool["rcosT_\(hd)"]!, sb = pool["rsinT_\(hd)"]!, roff = pos*half*4
        let kcb = kc[src]!, vcb = vc[src]!
        let outq = S("outq", qd*4), outk = S("outk", hd*4), outv = S("outv", hd*4), dummy = S("dummy", hd*4), dummy2 = S("dummy2", hd*4)
        let attn = S("attn", qd*4), y2 = S("y2", H*4), sum2 = S("sum2", 4)
        let geglu = S("geglu", inter*2), gate = S("gate", d*4), y2n = S("y2n", H*4), sum2n = S("sum2n", 4)
        let pp73 = S("pp73",(H+1)*4), pp75 = S("pp75",(H+1)*4), pp77 = S("pp77",(H+1)*4)
        let partials = S("partials\(hd)", (8*32*(hd+2)+8)*4)

        // 69 only on layer 0; else reuse prev k77 y2n/sum2n
        var a = y2n, suma = sum2n
        if l == 0 {
            a = S("a", H*4); suma = S("suma", 4)
            var p = [UInt32(1), 0, sc(l,"qkv_in").bitPattern, 0]
            try b.dispatchGroups("rmssrq_69", buffers: [(hidden,0),(LW("in_norm"),0),(a,0),(suma,0),(UU("p69_\(l)",p),0)], groups: 1)
        }
        // 70 qkv
        let par70 = UF("p70_\(l)",[sc(l,"q_out"), scOpt(l,"k_out"), scOpt(l,"v_out"), 0])
        if shared {
            try b.dispatchGroups("qkv_\(qd)_\(hd)", buffers: [(a,0),(LW("q_bits"),0),(LW("q_bits"),0),(LW("q_bits"),0),(LW("q_scale"),0),(suma,0),(outq,0),(dummy,0),(dummy2,0),(par70,0)], groups: qd/2)
        } else {
            try b.dispatchGroups("qkv_\(qd)_\(hd)", buffers: [(a,0),(LW("q_bits"),0),(LW("k_bits"),0),(LW("v_bits"),0),(LW("qkv_scales"),0),(suma,0),(outq,0),(outk,0),(outv,0),(par70,0)], groups: qd/2+hd)
            try b.dispatchGroups("kvnorm_\(hd)", buffers: [(outk,0),(outv,0),(LW("k_norm"),0),(cb,roff),(sb,roff),(kcb,0),(vcb,0),(S("pkv_\(hd)_\(slot)",16),0)], groups: 1)
        }
        // attention (2-D: heads x chunks)
        try b.dispatchGroups2D("attn_\(hd)_\(l)", buffers: [(outq,0),(LW("q_norm"),0),(cb,roff),(sb,roff),(kcb,0),(vcb,0),(partials,0),(attn,0),(S("parA_\(hd)_\(slot)",32),0)], width: nH, height: 32)
        // 73 o-proj + norms
        try b.dispatchGroups("oproj_\(qd)", buffers: [(attn,0),(LW("o_bits"),0),(LW("o_scale"),0),(pp73,0),(hidden,0),(LW("o_w12"),0),(y2,0),(sum2,0),(UF("p73_\(l)",[sc(l,"o_out"),sc(l,"gate_in"),0,0]),0)], groups: 192)
        // 74/95 gate/up ; 75/96 down
        let par74 = UF("p74_\(l)",[sc(l,"gate_out"), sc(l,"up_out"), sc(l,"down_in"), 0])
        if inter == 6144 {
            try b.dispatchGroups("gateup_74", buffers: [(y2,0),(LW("gate_bits"),0),(LW("gate_scale"),0),(LW("up_bits"),0),(LW("up_scale"),0),(sum2,0),(geglu,0),(LW("gelu_gate"),0),(par74,0)], groups: 768)
            try b.dispatchGroups("down_75", buffers: [(geglu,0),(LW("down_bits"),0),(pp75,0),(LW("down_scale"),0),(hidden,0),(LW("down_nw"),0),(UF("p75_\(l)",[sc(l,"down_in"),sc(l,"down_out"),0,0]),0)], groups: 384)
        } else {
            try b.dispatchGroups("gateup_95", buffers: [(y2,0),(LW("gate_bits"),0),(LW("gate_scale"),0),(LW("up_bits"),0),(LW("up_scale"),0),(sum2,0),(geglu,0),(LW("gelu_gate"),0),(par74,0)], groups: 3072)
            try b.dispatchGroups("down_96", buffers: [(geglu,0),(LW("down_bits"),0),(pp75,0),(LW("down_scale"),0),(hidden,0),(LW("down_nw"),0),(UF("p75_\(l)",[sc(l,"down_in"),sc(l,"down_out"),0,0]),0)], groups: 384)
        }
        // 76 PLE gate
        var p76 = [UInt32](repeating: 0, count: 4)
        p76[0] = sc(l,"plegate_in").bitPattern; p76[1] = sc(l,"plegate_out").bitPattern; p76[2] = UInt32(l*d)
        try b.dispatchGroups("plegate_76", buffers: [(hidden,0),(LW("plegate_codes"),0),(LW("plegate_rowscale"),0),(ple,0),(gate,0),(LW("gelu_plegate"),0),(UU("p76_\(l)",p76),0)], groups: 256)
        // 77 PLE proj
        let nxt = l+1 < C("nL") ? sc(l+1,"qkv_in") : 0
        try b.dispatchGroups("pleproj_77", buffers: [(gate,0),(LW("pleproj_codes"),0),(LW("pleproj_rowscale"),0),(pp77,0),(hidden,0),(LW("pleproj_w12s"),0),(y2n,0),(sum2n,0),(UF("p77_\(l)",[nxt,sc(l,"pleproj_in"),sc(l,"pleproj_out"),0]),0)], groups: 96)
    }

    func profile(_ n: Int = 40) throws {
        let prof = try KernelProfiler(device: runner.device)
        let hidden = S("hidden", C("H")*4), cur = S("cur", 4), gen = S("gen", 256*4)
        cur.contents().bindMemory(to: UInt32.self, capacity: 1)[0] = 2
        for i in 0..<n { try forward(cur, i, hidden, argmax: cur, gen: gen, step: i%200, prof: prof) }
        print(prof.report())
    }

    @discardableResult
    func forward(_ tok: MTLBuffer, _ pos: Int, _ hidden: MTLBuffer, argmax: MTLBuffer?,
                 gen: MTLBuffer? = nil, step: Int = 0, wait: Bool = true, tokOff: Int = 0,
                 prof: KernelProfiler? = nil) throws -> MTLCommandBuffer? {
        let H = C("H"), V = C("vocab")
        let slot = step % Self.RING
        writeStepUniforms(pos, slot)
        let b = try runner.batch(profiler: prof)
        try embed(b, tok, tokOff, hidden)
        let ple = try pleInput(b, tok, tokOff, hidden)
        for l in 0..<C("nL") { try layer(b, l, pos, slot, hidden, ple) }
        let normed = S("normed", H*4)
        try b.dispatchGroups("rmssrq_69", buffers: [(hidden,0),(W("final_norm"),0),(normed,0),(S("sa",4),0),(UU("fn",[1,0,0,0]),0)], groups: 1)
        let logits = S("logits", V*4)
        try b.dispatchGroups("logits_33", buffers: [(normed,0),(W("lmhead_blk"),0),(W("lmhead_scale"),0),(logits,0),(UF("plog",[0,0,0,0]),0)], groups: V/128)
        if let am = argmax {
            let cv = S("cv", 256*4), ci = S("ci", 256*4)
            try b.dispatchGroups("argmax1_34", buffers: [(logits,0),(cv,0),(ci,0)], groups: 256)
            try b.dispatchGroups("argmax2_35", buffers: [(cv,0),(ci,0),(am,0)], groups: 1)
            if let g = gen { b.blitCopy(am, 0, g, step*4, 4) }   // record token on-GPU
        }
        if wait { b.commitAndWait(); return nil }
        return b.commit()
    }

    // Measure CPU record rate vs GPU-overlapped rate to find the bottleneck.
    func bench(_ n: Int = 128) throws {
        let hidden = S("hidden", C("H")*4), cur = S("cur", 4), gen = S("gen", 256*4)
        cur.contents().bindMemory(to: UInt32.self, capacity: 1)[0] = 2
        for p in 0..<8 { try forward(cur, p, hidden, argmax: cur, gen: gen, step: p) }   // warm
        let t0 = Date(); var last: MTLCommandBuffer?
        for i in 0..<n { last = try forward(cur, 8+i, hidden, argmax: cur, gen: gen, step: i%200, wait: false) }
        let cpu = Date().timeIntervalSince(t0)
        last?.waitUntilCompleted()
        let total = Date().timeIntervalSince(t0)
        print(String(format: "CPU-record: %.1f tok/s (%.2f ms/tok) | GPU-overlapped: %.1f tok/s (%.2f ms/tok)",
                     Double(n)/cpu, cpu/Double(n)*1000, Double(n)/total, total/Double(n)*1000))
    }

    // ---- GPU-resident greedy decode: token feeds back via `cur` on-GPU; commit
    // without per-token sync (GPU runs async), read the gen buffer once per chunk. ----
    /// Pipelined prefill: commit a chunk of forwards without waiting so CPU
    /// recording overlaps GPU execution. Token ids live in a GPU buffer bound
    /// per-step at an offset, so the CPU never rewrites a buffer the GPU may
    /// still be reading. Chunk is capped at RING so no in-flight uniform slot
    /// is reused. Returns the new position.
    @discardableResult
    func prefill(_ toks: [UInt32], _ startPos: Int, chunk: Int = RING) throws -> Int {
        if toks.isEmpty { return startPos }
        let hidden = S("hidden", C("H")*4)
        let buf = try runner.makeBuffer(toks, label: "prompt")
        var pos = startPos, i = 0
        while i < toks.count {
            let end = min(i + chunk, toks.count)
            var last: MTLCommandBuffer?
            while i < end {
                last = try forward(buf, pos, hidden, argmax: nil, step: i % Self.RING,
                                   wait: false, tokOff: i * 4)
                pos += 1; i += 1
            }
            last?.waitUntilCompleted()
        }
        return pos
    }

    func generate(_ ids: [UInt32], _ nNew: Int, eos: UInt32 = 1, chunk: Int = 32) throws -> ([UInt32], Double) {
        let hidden = S("hidden", C("H")*4), cur = S("cur", 4), gen = S("gen", 256*4)
        var pos = try prefill(Array(ids.dropLast()), 0)
        cur.contents().bindMemory(to: UInt32.self, capacity: 1)[0] = ids.last!
        var out: [UInt32] = []
        let t0 = Date()
        var step = 0
        while step < nNew {
            let end = min(step + chunk, nNew)
            var last: MTLCommandBuffer?
            while step < end { last = try forward(cur, pos, hidden, argmax: cur, gen: gen, step: step, wait: false); pos += 1; step += 1 }
            last?.waitUntilCompleted()
            let g = gen.contents().bindMemory(to: UInt32.self, capacity: 256)
            while out.count < step {
                let tk = g[out.count]
                // step, not out.count: the elapsed time covers the whole
                // dispatched chunk, so an early EOS would put a truncated
                // numerator over a full denominator and underreport decode.
                if tk == eos { return (out, Double(step) / Date().timeIntervalSince(t0)) }
                out.append(tk)
            }
        }
        return (out, Double(step) / Date().timeIntervalSince(t0))
    }

    // ---- interactive chat: keep the KV cache resident across turns so each
    // turn only prefills the NEW framing tokens, not the whole transcript. ----
    func chatReset() { chatPos = 0; chatCarry = nil }

    // `frame` = this turn's user framing tokens (chat template already applied,
    // BOS included on the first turn). Prefills them after the carried token,
    // then greedily decodes until a `stop` token or `maxNew`.
    //
    // Decode is GPU-resident and *chunked* (same engine as `generate`): a chunk
    // of forwards is committed without per-token waits so CPU command-recording
    // overlaps GPU execution — that overlap is the whole difference between the
    // ~90 tok/s per-token path and the ~170 tok/s chunked path. On `stop` we may
    // have speculatively run a few forwards past the turn end; those extra KV
    // slots are simply overwritten next turn (attention only reads [start..pos]),
    // so we fix `chatPos`/`chatCarry` back to the true turn boundary. `onChunk`
    // gets the full accepted-token list after each chunk syncs, for streaming.
    @discardableResult
    func chatTurn(_ frame: [UInt32], maxNew: Int, stop: Set<UInt32>, chunk: Int = 32,
                  onChunk: ([UInt32]) -> Void) throws -> (tokens: [UInt32], tps: Double) {
        let H = C("H")
        let hidden = S("hidden", H*4), cur = S("cur", 4), gen = S("gen", 256*4)
        let curP = cur.contents().bindMemory(to: UInt32.self, capacity: 1)
        let genP = gen.contents().bindMemory(to: UInt32.self, capacity: 256)

        // prefill everything except the last framing token (which seeds decode)
        let pre = (chatCarry.map { [$0] } ?? []) + frame
        chatPos = try prefill(Array(pre.dropLast()), chatPos)
        curP[0] = pre.last!
        let prefillEnd = chatPos            // slot where g_i lands = prefillEnd + i + 1

        var out: [UInt32] = []
        let cap = min(maxNew, 240)          // gen buffer holds 256 slots
        var step = 0, stopped = false
        let t0 = Date()                     // decode-rate timer (excludes prefill)
        while step < cap && !stopped && chatPos < 2040 {
            let end = min(step + chunk, cap)
            var last: MTLCommandBuffer?
            while step < end {
                last = try forward(cur, chatPos, hidden, argmax: cur, gen: gen, step: step, wait: false)
                chatPos += 1; step += 1
            }
            last?.waitUntilCompleted()
            var i = out.count
            while i < step {
                let tk = genP[i]
                if stop.contains(tk) {          // true turn end: rewind past the speculative tail
                    stopped = true; chatCarry = tk; chatPos = prefillEnd + i + 1; break
                }
                out.append(tk); i += 1
            }
            onChunk(out)
        }
        if !stopped { chatCarry = curP[0] }     // last produced token, uncached; chatPos already correct
        let dt = Date().timeIntervalSince(t0)
        return (out, dt > 0 ? Double(step) / dt : 0)   // step = forwards run (incl. speculative tail)
    }
}
