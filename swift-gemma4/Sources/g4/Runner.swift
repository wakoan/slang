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
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        gdir = repo.appendingPathComponent("models/gemma-4-E2B-qat/g4_150")
        kdir = repo.appendingPathComponent("swift-gemma4/kernels")
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
    }

    // ---- per-token dynamic uniforms (rope/attn/kv per head-dim type) ----
    func writeStepUniforms(_ pos: Int) {
        let nH = C("nH"), win = C("window")
        var byHd: [Int: (Double, Int, Bool)] = [:]
        for l in 0..<C("nL") {
            byHd[LI(l, "head_dim")] = ((layers[l]["rope_theta"] as! NSNumber).doubleValue, LI(l, "rope_cutoff"), LB(l, "sliding"))
        }
        for (hd, (theta, cutoff, sliding)) in byHd {
            let half = hd/2
            var cosv = [Float](repeating: 0, count: half), sinv = cosv
            for i in 0..<half {
                let inv = i < cutoff ? 1.0 / pow(theta, Double(i)/Double(half)) : 0.0
                let ang = Double(pos) * inv
                cosv[i] = Float(cos(ang)); sinv[i] = Float(sin(ang))
            }
            let cb = S("rcos\(half)", half*4), sb = S("rsin\(half)", half*4)
            cb.contents().copyMemory(from: cosv, byteCount: half*4)
            sb.contents().copyMemory(from: sinv, byteCount: half*4)
            let pA = S("parA_\(hd)", 32)
            pA.contents().bindMemory(to: UInt32.self, capacity: 8).update(from: [1, UInt32(pos+1), UInt32(pos), UInt32(nH), 1, UInt32(sliding ? win : 0), 0, 0], count: 8)
            let pkv = S("pkv_\(hd)", 16)
            pkv.contents().bindMemory(to: UInt32.self, capacity: 4).update(from: [UInt32(pos*hd), 0, 0, 0], count: 4)
        }
    }

    func embed(_ b: MetalRunner.CommandBatch, _ ids: MTLBuffer, _ out: MTLBuffer) throws {
        try b.dispatchGroups("embed_00", buffers: [(ids,0),(W("embed_q"),0),(W("embed_scale"),0),(out,0),(u32([1,0,0,0]),0)], groups: 1)
    }
    func pleInput(_ b: MetalRunner.CommandBatch, _ ids: MTLBuffer, _ hidden: MTLBuffer) throws -> MTLBuffer {
        let nL = C("nL"), d = C("ple_d")
        let ctx = S("ctx", nL*d*4), ple = S("plegath", nL*d*4), out = S("ple", nL*d*4)
        try b.dispatchGroups("proj_68", buffers: [(hidden,0),(W("pl_model_proj"),0),(ctx,0),(f32([0,0,0,0]),0)], groups: 1120)
        try b.dispatchGroups("plegather_01", buffers: [(ids,0),(W("ple_q"),0),(W("ple_scale"),0),(ple,0),(u32([1,0,0,0]),0)], groups: 1)
        try b.dispatchGroups("combine", buffers: [(ctx,0),(ple,0),(W("pl_proj_norm"),0),(out,0)], groups: nL)
        return out
    }

    func layer(_ b: MetalRunner.CommandBatch, _ l: Int, _ hidden: MTLBuffer, _ ple: MTLBuffer) throws {
        let H = C("H"), nH = C("nH"), hd = LI(l,"head_dim"), qd = LI(l,"q_dim"), inter = LI(l,"intermediate")
        let half = hd/2, shared = LB(l,"shared"), src = LI(l,"kv_source"), d = C("ple_d")
        func LW(_ n: String) -> MTLBuffer { W("L\(l).\(n)") }
        let cb = S("rcos\(half)", half*4), sb = S("rsin\(half)", half*4)
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
            try b.dispatchGroups("rmssrq_69", buffers: [(hidden,0),(LW("in_norm"),0),(a,0),(suma,0),(u32(p),0)], groups: 1)
        }
        // 70 qkv
        let par70 = f32([sc(l,"q_out"), scOpt(l,"k_out"), scOpt(l,"v_out"), 0])
        if shared {
            try b.dispatchGroups("qkv_\(qd)_\(hd)", buffers: [(a,0),(LW("q_bits"),0),(LW("q_bits"),0),(LW("q_bits"),0),(LW("q_scale"),0),(suma,0),(outq,0),(dummy,0),(dummy2,0),(par70,0)], groups: qd/2)
        } else {
            try b.dispatchGroups("qkv_\(qd)_\(hd)", buffers: [(a,0),(LW("q_bits"),0),(LW("k_bits"),0),(LW("v_bits"),0),(LW("qkv_scales"),0),(suma,0),(outq,0),(outk,0),(outv,0),(par70,0)], groups: qd/2+hd)
            try b.dispatchGroups("kvnorm_\(hd)", buffers: [(outk,0),(outv,0),(LW("k_norm"),0),(cb,0),(sb,0),(kcb,0),(vcb,0),(S("pkv_\(hd)",16),0)], groups: 1)
        }
        // attention (2-D: heads x chunks)
        try b.dispatchGroups2D("attn_\(hd)_\(l)", buffers: [(outq,0),(LW("q_norm"),0),(cb,0),(sb,0),(kcb,0),(vcb,0),(partials,0),(attn,0),(S("parA_\(hd)",32),0)], width: nH, height: 32)
        // 73 o-proj + norms
        try b.dispatchGroups("oproj_\(qd)", buffers: [(attn,0),(LW("o_bits"),0),(LW("o_scale"),0),(pp73,0),(hidden,0),(LW("o_w12"),0),(y2,0),(sum2,0),(f32([sc(l,"o_out"),sc(l,"gate_in"),0,0]),0)], groups: 192)
        // 74/95 gate/up ; 75/96 down
        let par74 = f32([sc(l,"gate_out"), sc(l,"up_out"), sc(l,"down_in"), 0])
        if inter == 6144 {
            try b.dispatchGroups("gateup_74", buffers: [(y2,0),(LW("gate_bits"),0),(LW("gate_scale"),0),(LW("up_bits"),0),(LW("up_scale"),0),(sum2,0),(geglu,0),(LW("gelu_gate"),0),(par74,0)], groups: 768)
            try b.dispatchGroups("down_75", buffers: [(geglu,0),(LW("down_bits"),0),(pp75,0),(LW("down_scale"),0),(hidden,0),(LW("down_nw"),0),(f32([sc(l,"down_in"),sc(l,"down_out"),0,0]),0)], groups: 384)
        } else {
            try b.dispatchGroups("gateup_95", buffers: [(y2,0),(LW("gate_bits"),0),(LW("gate_scale"),0),(LW("up_bits"),0),(LW("up_scale"),0),(sum2,0),(geglu,0),(LW("gelu_gate"),0),(par74,0)], groups: 3072)
            try b.dispatchGroups("down_96", buffers: [(geglu,0),(LW("down_bits"),0),(pp75,0),(LW("down_scale"),0),(hidden,0),(LW("down_nw"),0),(f32([sc(l,"down_in"),sc(l,"down_out"),0,0]),0)], groups: 384)
        }
        // 76 PLE gate
        var p76 = [UInt32](repeating: 0, count: 4)
        p76[0] = sc(l,"plegate_in").bitPattern; p76[1] = sc(l,"plegate_out").bitPattern; p76[2] = UInt32(l*d)
        try b.dispatchGroups("plegate_76", buffers: [(hidden,0),(LW("plegate_codes"),0),(LW("plegate_rowscale"),0),(ple,0),(gate,0),(LW("gelu_plegate"),0),(u32(p76),0)], groups: 256)
        // 77 PLE proj
        let nxt = l+1 < C("nL") ? sc(l+1,"qkv_in") : 0
        try b.dispatchGroups("pleproj_77", buffers: [(gate,0),(LW("pleproj_codes"),0),(LW("pleproj_rowscale"),0),(pp77,0),(hidden,0),(LW("pleproj_w12s"),0),(y2n,0),(sum2n,0),(f32([nxt,sc(l,"pleproj_in"),sc(l,"pleproj_out"),0]),0)], groups: 96)
    }

    func forward(_ tok: MTLBuffer, _ pos: Int, _ hidden: MTLBuffer, argmax: MTLBuffer?) throws {
        let H = C("H"), V = C("vocab")
        writeStepUniforms(pos)
        let b = try runner.batch()
        try embed(b, tok, hidden)
        let ple = try pleInput(b, tok, hidden)
        for l in 0..<C("nL") { try layer(b, l, hidden, ple) }
        let normed = S("normed", H*4)
        try b.dispatchGroups("rmssrq_69", buffers: [(hidden,0),(W("final_norm"),0),(normed,0),(S("sa",4),0),(u32([1,0,0,0]),0)], groups: 1)
        let logits = S("logits", V*4)
        try b.dispatchGroups("logits_33", buffers: [(normed,0),(W("lmhead_blk"),0),(W("lmhead_scale"),0),(logits,0),(f32([0,0,0,0]),0)], groups: V/128)
        if let am = argmax {
            let cv = S("cv", 256*4), ci = S("ci", 256*4)
            try b.dispatchGroups("argmax1_34", buffers: [(logits,0),(cv,0),(ci,0)], groups: 256)
            try b.dispatchGroups("argmax2_35", buffers: [(cv,0),(ci,0),(am,0)], groups: 1)
        }
        b.commitAndWait()
    }

    // ---- greedy decode with on-GPU token feedback (cur buffer) ----
    func generate(_ ids: [UInt32], _ nNew: Int, eos: UInt32 = 1) throws -> ([UInt32], Double) {
        let hidden = S("hidden", C("H")*4)
        let cur = S("cur", 4)
        var pos = 0
        for t in ids.dropLast() {
            cur.contents().bindMemory(to: UInt32.self, capacity: 1)[0] = t
            try forward(cur, pos, hidden, argmax: nil); pos += 1
        }
        cur.contents().bindMemory(to: UInt32.self, capacity: 1)[0] = ids.last!
        var out: [UInt32] = []
        let t0 = Date()
        for _ in 0..<nNew {
            try forward(cur, pos, hidden, argmax: cur); pos += 1
            let tk = cur.contents().bindMemory(to: UInt32.self, capacity: 1)[0]
            if tk == eos { break }
            out.append(tk)
        }
        return (out, out.isEmpty ? 0 : Double(out.count) / Date().timeIntervalSince(t0))
    }
}
