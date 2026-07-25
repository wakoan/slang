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

// --- stage 2: rmssrq_69 (input norm + srq, subgroup reduction) ---
// a = srq(rms_norm(embed, L0.in_norm), qkv_in=0.608803)
let inNorm = weightBuffer("L0.in_norm")
let y69 = try runner.makeBuffer(bytes: H * 4, label: "y69")
let suma = try runner.makeBuffer(bytes: 4, label: "sum_a")
let p69 = try runner.makeBuffer([UInt32(1), UInt32(0), Float(0.608803).bitPattern, UInt32(0)], label: "p69")
let b2 = try runner.batch()
try b2.dispatchGroups("rmssrq_69", buffers: [(y, 0), (inNorm, 0), (y69, 0), (suma, 0), (p69, 0)], groups: 1)
b2.commitAndWait()
let ap = y69.contents().bindMemory(to: Float.self, capacity: H)
let sp = suma.contents().bindMemory(to: Float.self, capacity: 1)
print("k69 a[:4]    = [\(ap[0]), \(ap[1]), \(ap[2]), \(ap[3])]  sum_a=\(sp[0])")
print("expected     = [-14.61128, -13.39368, 0.0, -17.0465]  sum_a=-848.672")
