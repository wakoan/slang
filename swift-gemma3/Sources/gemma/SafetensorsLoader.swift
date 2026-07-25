import Foundation
import Metal

struct TensorInfo {
    let dtype: String
    let shape: [Int]
    let start: Int  // absolute byte offset in file
    let end: Int

    var elementCount: Int { shape.reduce(1, *) }
}

/// Memory-mapped safetensors file with raw per-tensor access.
final class SafetensorsFile {
    private let data: Data
    let tensors: [String: TensorInfo]

    init(path: String) throws {
        data = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
        guard data.count >= 8 else { throw RuntimeError("safetensors file too small") }
        let headerLen = data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        let base = 8 + Int(headerLen)
        guard let header = try JSONSerialization.jsonObject(
            with: data.subdata(in: 8..<base)) as? [String: Any] else {
            throw RuntimeError("safetensors header is not a JSON object")
        }

        var tensors: [String: TensorInfo] = [:]
        for (name, info) in header where name != "__metadata__" {
            guard let d = info as? [String: Any],
                  let offsets = d["data_offsets"] as? [Int], offsets.count == 2,
                  let shape = d["shape"] as? [Int],
                  let dtype = d["dtype"] as? String else {
                throw RuntimeError("malformed header entry for \(name)")
            }
            tensors[name] = TensorInfo(dtype: dtype, shape: shape,
                                       start: base + offsets[0], end: base + offsets[1])
        }
        self.tensors = tensors
    }

    func info(_ name: String) throws -> TensorInfo {
        guard let t = tensors[name] else { throw RuntimeError("tensor not found: \(name)") }
        return t
    }

    /// Raw bf16 element access (Gemma weights are all BF16 on disk).
    func withBF16<R>(_ name: String, _ body: (UnsafeBufferPointer<UInt16>) throws -> R) throws -> R {
        let t = try info(name)
        guard t.dtype == "BF16" else {
            throw RuntimeError("\(name): expected BF16, got \(t.dtype)")
        }
        return try data.withUnsafeBytes { raw -> R in
            let ptr = raw.baseAddress! + t.start
            precondition(Int(bitPattern: ptr) % 2 == 0, "unaligned bf16 tensor")
            return try body(UnsafeBufferPointer(
                start: ptr.assumingMemoryBound(to: UInt16.self), count: t.elementCount))
        }
    }
}

private func bf16ToF32(_ bits: UInt16) -> Float {
    Float(bitPattern: UInt32(bits) << 16)
}

/// All model weights as MTLBuffers laid out the way the kernels expect:
/// matmul weights + embed table as f16 (halves; packed-u32 kernels read the
/// same bytes as uint pairs), norm weights as f32. QKV and gate/up are merged
/// by row-concatenation into single matrices, consumed via buffer offsets.
final class WeightStore {
    private(set) var buffers: [String: MTLBuffer] = [:]

    subscript(name: String) -> MTLBuffer {
        guard let b = buffers[name] else { fatalError("weight not loaded: \(name)") }
        return b
    }

    init(file: SafetensorsFile, config: GemmaConfig, runner: MetalRunner) throws {
        // bf16 tensor → f16 halves at byte `offset` inside `buf`
        func convertF16(_ name: String, into buf: MTLBuffer, offset: Int) throws {
            try file.withBF16(name) { src in
                let dst = (buf.contents() + offset).assumingMemoryBound(to: UInt16.self)
                for i in 0..<src.count {
                    dst[i] = Float16(bf16ToF32(src[i])).bitPattern
                }
            }
        }

        func f16Buffer(_ label: String, elements: Int) throws -> MTLBuffer {
            let b = try runner.makeBuffer(bytes: elements * 2, label: label)
            buffers[label] = b
            return b
        }

        func loadF16(_ name: String) throws {
            let buf = try f16Buffer(name, elements: try file.info(name).elementCount)
            try convertF16(name, into: buf, offset: 0)
        }

        func loadF32(_ name: String) throws {
            let count = try file.info(name).elementCount
            let buf = try runner.makeBuffer(bytes: count * 4, label: name)
            try file.withBF16(name) { src in
                let dst = buf.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<src.count { dst[i] = bf16ToF32(src[i]) }
            }
            buffers[name] = buf
        }

        try loadF16("model.embed_tokens.weight")
        try loadF32("model.norm.weight")

        for L in 0..<config.numLayers {
            let p = "model.layers.\(L)."
            let a = p + "self_attn.", m = p + "mlp."

            // merged QKV: rows [q_dim | hd | hd] × hiddenSize
            let qCount = try file.info(a + "q_proj.weight").elementCount
            let kCount = try file.info(a + "k_proj.weight").elementCount
            let vCount = try file.info(a + "v_proj.weight").elementCount
            let qkv = try f16Buffer(a + "qkv_proj.weight", elements: qCount + kCount + vCount)
            try convertF16(a + "q_proj.weight", into: qkv, offset: 0)
            try convertF16(a + "k_proj.weight", into: qkv, offset: qCount * 2)
            try convertF16(a + "v_proj.weight", into: qkv, offset: (qCount + kCount) * 2)

            // merged gate/up: rows [inter | inter] × hiddenSize
            let gCount = try file.info(m + "gate_proj.weight").elementCount
            let uCount = try file.info(m + "up_proj.weight").elementCount
            let gu = try f16Buffer(m + "gateup_proj.weight", elements: gCount + uCount)
            try convertF16(m + "gate_proj.weight", into: gu, offset: 0)
            try convertF16(m + "up_proj.weight", into: gu, offset: gCount * 2)

            try loadF16(a + "o_proj.weight")
            try loadF16(m + "down_proj.weight")

            for norm in ["input_layernorm", "post_attention_layernorm",
                         "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
                try loadF32(p + norm + ".weight")
            }
            try loadF32(a + "q_norm.weight")
            try loadF32(a + "k_norm.weight")
        }
    }

    var totalBytes: Int {
        buffers.values.reduce(0) { $0 + $1.length }
    }
}
