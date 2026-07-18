import Foundation
import Metal

// Locate the kernels/ directory whether we run from the repo root or from
// swift-inference/.
func findKernelsDir() throws -> String {
    let fm = FileManager.default
    for candidate in ["swift-inference/kernels", "kernels"] {
        if fm.fileExists(atPath: candidate + "/matvec_simd_packed.metal") {
            return candidate
        }
    }
    throw RuntimeError("kernels/ directory not found; run from repo root or swift-inference/")
}

/// Pack an f32 array into u32s holding f16 pairs (element 2j in the low half).
func packF16Pairs(_ values: [Float]) -> [UInt32] {
    precondition(values.count % 2 == 0)
    var out = [UInt32]()
    out.reserveCapacity(values.count / 2)
    for j in stride(from: 0, to: values.count, by: 2) {
        let lo = Float16(values[j]).bitPattern
        let hi = Float16(values[j + 1]).bitPattern
        out.append(UInt32(lo) | (UInt32(hi) << 16))
    }
    return out
}

func smokeTestMatvec(_ runner: MetalRunner, kernelsDir: String) throws {
    try runner.compileKernel(file: kernelsDir + "/matvec_simd_packed.metal")

    let nOut = 64, nIn = 128
    var rng = SystemRandomNumberGenerator()
    let w = (0..<nOut * nIn).map { _ in Float.random(in: -1...1, using: &rng) }
    let x = (0..<nIn).map { _ in Float.random(in: -1...1, using: &rng) }

    let wBuf = try runner.makeBuffer(packF16Pairs(w), label: "w")
    let xBuf = try runner.makeBuffer(x, label: "x")
    let yBuf = try runner.makeBuffer(bytes: nOut * 4, label: "y")
    let dims = try runner.makeBuffer([UInt32(nOut), UInt32(nIn)], label: "dims")

    let batch = try runner.batch()
    try batch.dispatch("matvec_simd_packed",
                       buffers: [wBuf, xBuf, yBuf, dims],
                       totalThreads: nOut * 32)  // one simdgroup per row
    batch.commitAndWait()

    let y = UnsafeBufferPointer(
        start: yBuf.contents().assumingMemoryBound(to: Float.self), count: nOut)

    // CPU reference through the same f16 rounding the GPU sees.
    var maxErr: Float = 0
    for r in 0..<nOut {
        var ref: Float = 0
        for j in 0..<nIn {
            ref += Float(Float16(w[r * nIn + j])) * x[j]
        }
        maxErr = max(maxErr, abs(ref - y[r]))
    }
    guard maxErr < 1e-3 else {
        throw RuntimeError("matvec_simd_packed FAILED, max error \(maxErr)")
    }
    print("✓ matvec_simd_packed matches CPU reference (max err \(maxErr))")
}

func compileAllKernels(_ runner: MetalRunner, kernelsDir: String) throws {
    let files = try FileManager.default.contentsOfDirectory(atPath: kernelsDir)
        .filter { $0.hasSuffix(".metal") }.sorted()
    var failed: [String] = []
    for file in files {
        do {
            try runner.compileKernel(file: kernelsDir + "/" + file)
        } catch {
            failed.append("\(file): \(error)")
        }
    }
    guard failed.isEmpty else {
        throw RuntimeError("kernel compile failures:\n" + failed.joined(separator: "\n"))
    }
    print("✓ all \(files.count) kernels compile")
}

do {
    let runner = try MetalRunner()
    print("Metal device: \(runner.device.name)")
    let kernelsDir = try findKernelsDir()
    try compileAllKernels(runner, kernelsDir: kernelsDir)
    try smokeTestMatvec(runner, kernelsDir: kernelsDir)
} catch {
    print("Error: \(error)")
    exit(1)
}
