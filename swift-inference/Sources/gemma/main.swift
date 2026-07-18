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

func findModelDir() throws -> String {
    let fm = FileManager.default
    for candidate in ["models/gemma-3-270m-it", "../models/gemma-3-270m-it"] {
        if fm.fileExists(atPath: candidate + "/config.json") { return candidate }
    }
    throw RuntimeError("model dir not found; run `python -m gemma3.download` first")
}

/// Load all weights onto the GPU, then verify the embedding path end-to-end:
/// GPU embed_scale_f16 vs a CPU reference computed straight from the file.
func loadAndVerifyWeights(_ runner: MetalRunner, modelDir: String) throws -> (GemmaConfig, WeightStore) {
    let config = try GemmaConfig.load(from: modelDir + "/config.json")
    let file = try SafetensorsFile(path: modelDir + "/model.safetensors")

    let t0 = Date()
    let weights = try WeightStore(file: file, config: config, runner: runner)
    let dt = Date().timeIntervalSince(t0)
    let mb = Double(weights.totalBytes) / 1e6
    print(String(format: "✓ weights on GPU: %.0f MB in %.2fs (%.0f MB/s)", mb, dt, mb / dt))

    let h = config.hiddenSize
    let tok: UInt32 = 1000
    let tokBuf = try runner.makeBuffer([tok], label: "token")
    let xBuf = try runner.makeBuffer(bytes: h * 4, label: "x")
    let dims = try runner.makeBuffer([UInt32(h)], label: "dims")

    let batch = try runner.batch()
    try batch.dispatch("embed_scale_f16",
                       buffers: [tokBuf, weights["model.embed_tokens.weight"], xBuf, dims],
                       totalThreads: h)
    batch.commitAndWait()

    let x = UnsafeBufferPointer(start: xBuf.contents().assumingMemoryBound(to: Float.self), count: h)
    var maxErr: Float = 0
    try file.withBF16("model.embed_tokens.weight") { src in
        let scale = sqrt(Float(h))
        for i in 0..<h {
            let ref = Float(Float16(Float(bitPattern: UInt32(src[Int(tok) * h + i]) << 16))) * scale
            maxErr = max(maxErr, abs(ref - x[i]))
        }
    }
    guard maxErr < 1e-4 else {
        throw RuntimeError("embed_scale_f16 FAILED, max error \(maxErr)")
    }
    print("✓ embed_scale_f16 matches file-derived reference (max err \(maxErr))")
    return (config, weights)
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

func argValue(_ flag: String) -> String? {
    let args = CommandLine.arguments
    guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
    return args[i + 1]
}

do {
    let runner = try MetalRunner()
    print("Metal device: \(runner.device.name)")
    let kernelsDir = try findKernelsDir()
    try compileAllKernels(runner, kernelsDir: kernelsDir)

    if let idsArg = argValue("--ids") {
        // generation mode: token ids come from the Python tokenizer for now
        let promptIds = idsArg.split(separator: ",").compactMap { Int($0) }
        let maxTokens = Int(argValue("--max-tokens") ?? "32") ?? 32
        let (config, weights) = try loadAndVerifyWeights(runner, modelDir: try findModelDir())
        let model = try GemmaModel(cfg: config, runner: runner, weights: weights)

        let ignoreEOS = CommandLine.arguments.contains("--ignore-eos")
        let (out, rate): ([Int], Double)
        if CommandLine.arguments.contains("--step-mode") {
            (out, rate) = try model.generateGreedy(
                promptIds: promptIds, maxNewTokens: maxTokens, ignoreEOS: ignoreEOS)
        } else {
            let chunk = Int(argValue("--chunk") ?? "64") ?? 64
            (out, rate) = try model.generateResident(
                promptIds: promptIds, maxNewTokens: maxTokens, chunk: chunk,
                ignoreEOS: ignoreEOS)
        }
        print("ids: " + out.map(String.init).joined(separator: ","))
        print(String(format: "%d decode tokens → %.1f tok/s", out.count, rate))
    } else {
        try smokeTestMatvec(runner, kernelsDir: kernelsDir)
        _ = try loadAndVerifyWeights(runner, modelDir: try findModelDir())
    }
} catch {
    print("Error: \(error)")
    exit(1)
}
