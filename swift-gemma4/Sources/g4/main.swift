import Foundation
import Metal

// --- locate repo + exported model ---
let repo = URL(fileURLWithPath: #filePath)          // .../swift-gemma4/Sources/g4/main.swift
    .deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()
let gdir = repo.appendingPathComponent("models/gemma-4-E2B-qat/g4_150")
let kdir = repo.appendingPathComponent("swift-gemma4/kernels")

struct Tensor { let off: Int; let len: Int }

// --- manifest ---
let manData = try Data(contentsOf: gdir.appendingPathComponent("manifest.json"))
let man = try JSONSerialization.jsonObject(with: manData) as! [String: Any]
let cfg = man["config"] as! [String: Any]
var tensors: [String: Tensor] = [:]
for (name, v) in man["tensors"] as! [String: Any] {
    let t = v as! [String: Any]
    tensors[name] = Tensor(off: t["off"] as! Int, len: t["len"] as! Int)
}
func cfgU(_ k: String) -> Int { (cfg[k] as! NSNumber).intValue }

// --- weights (mmap) ---
let weights = try Data(contentsOf: gdir.appendingPathComponent("weights.bin"), options: .mappedIfSafe)
print("model: \(tensors.count) tensors, \(weights.count / 1_000_000) MB")

let runner = try MetalRunner()
func weightBuffer(_ name: String) -> MTLBuffer {
    let t = tensors[name]!
    return weights.withUnsafeBytes { raw in
        runner.device.makeBuffer(bytes: raw.baseAddress!.advanced(by: t.off), length: t.len,
                                 options: .storageModeShared)!
    }
}

// --- compile kernels ---
for f in try FileManager.default.contentsOfDirectory(atPath: kdir.path) where f.hasSuffix(".metal") {
    try runner.compileKernel(file: kdir.appendingPathComponent(f).path)
}

// --- stage 1: embed(token 2) vs reference [-1.3005, -1.3005, 0, -1.3005] ---
let H = cfgU("H")
let embedQ = weightBuffer("embed_q")
let embedScale = weightBuffer("embed_scale")
let ids = try runner.makeBuffer([UInt32(2)], label: "ids")
let y = try runner.makeBuffer(bytes: H * 4, label: "y")
let p00 = try runner.makeBuffer([UInt32(1), 0, 0, 0], label: "p00")

let batch = try runner.batch()
try batch.dispatchGroups("embed_00", buffers: [(ids, 0), (embedQ, 0), (embedScale, 0), (y, 0), (p00, 0)],
                         groups: 1)
batch.commitAndWait()

let yp = y.contents().bindMemory(to: Float.self, capacity: H)
print("embed(2)[:4] = [\(yp[0]), \(yp[1]), \(yp[2]), \(yp[3])]")
print("expected     = [-1.3005, -1.3005, 0.0, -1.3005]")
