import Foundation
import Metal

struct RuntimeError: Error, CustomStringConvertible {
    let message: String
    init(_ message: String) { self.message = message }
    var description: String { message }
}

/// Per-kernel state: pipeline plus the threadgroup width the DSL declared.
struct Kernel {
    let pipeline: MTLComputePipelineState
    let threadsPerGroup: Int
}

final class MetalRunner {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private(set) var kernels: [String: Kernel] = [:]

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw RuntimeError("Metal device not available")
        }
        guard let queue = device.makeCommandQueue() else {
            throw RuntimeError("Failed to create command queue")
        }
        self.device = device
        self.queue = queue
    }

    /// Compile one .metal file into its own library. The kernel function is
    /// named after the file's basename; threadgroup width is parsed from the
    /// "// dispatch with threadsPerThreadgroup = (N)" comment the generator emits.
    func compileKernel(file path: String) throws {
        let name = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        let source = try String(contentsOfFile: path, encoding: .utf8)

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw RuntimeError("MSL compile failed for \(name): \(error)")
        }
        guard let function = library.makeFunction(name: name) else {
            throw RuntimeError("Function '\(name)' not found in \(path)")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        kernels[name] = Kernel(
            pipeline: pipeline,
            threadsPerGroup: Self.parseThreadsPerGroup(source) ?? 64
        )
    }

    private static func parseThreadsPerGroup(_ source: String) -> Int? {
        guard let range = source.range(of: "threadsPerThreadgroup = (") else { return nil }
        let rest = source[range.upperBound...].prefix(while: { $0.isNumber })
        return Int(rest)
    }

    func makeBuffer(bytes: Int, label: String) throws -> MTLBuffer {
        guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw RuntimeError("Buffer alloc failed: \(label) (\(bytes) bytes)")
        }
        buf.label = label
        return buf
    }

    func makeBuffer<T>(_ data: [T], label: String) throws -> MTLBuffer {
        precondition(_isPOD(T.self), "buffer element type must be POD")
        return try data.withUnsafeBytes { raw in
            guard let buf = device.makeBuffer(
                bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared
            ) else {
                throw RuntimeError("Buffer alloc failed: \(label) (\(raw.count) bytes)")
            }
            buf.label = label
            return buf
        }
    }

    /// Encodes many dispatches into one command buffer — commit once, wait
    /// once. With a profiler attached, every dispatch gets its own encoder
    /// with GPU timestamps at its boundaries (Apple GPUs sample at stage
    /// boundaries only); this serializes dispatches, so absolute totals are
    /// slightly inflated but per-kernel shares are accurate.
    final class CommandBatch {
        private let runner: MetalRunner
        private let cmdBuffer: MTLCommandBuffer
        private var encoder: MTLComputeCommandEncoder?
        private let profiler: KernelProfiler?

        fileprivate init(runner: MetalRunner, profiler: KernelProfiler?) throws {
            guard let cmdBuffer = runner.queue.makeCommandBuffer() else {
                throw RuntimeError("Failed to create command buffer")
            }
            self.runner = runner
            self.cmdBuffer = cmdBuffer
            self.profiler = profiler
        }

        private func encoderFor(label: String) throws -> MTLComputeCommandEncoder {
            if let profiler {
                encoder?.endEncoding()
                guard let (i0, i1) = profiler.nextSampleIndices(label: label) else {
                    throw RuntimeError("profiler sample buffer full")
                }
                let desc = MTLComputePassDescriptor()
                let att = desc.sampleBufferAttachments[0]!
                att.sampleBuffer = profiler.sampleBuffer
                att.startOfEncoderSampleIndex = i0
                att.endOfEncoderSampleIndex = i1
                guard let enc = cmdBuffer.makeComputeCommandEncoder(descriptor: desc) else {
                    throw RuntimeError("Failed to create profiled encoder")
                }
                encoder = enc
                return enc
            }
            if let enc = encoder { return enc }
            guard let enc = cmdBuffer.makeComputeCommandEncoder() else {
                throw RuntimeError("Failed to create command encoder")
            }
            encoder = enc
            return enc
        }

        private func encode(_ name: String, buffers: [(MTLBuffer, Int)],
                            groups: Int, label: String?) throws {
            guard let kernel = runner.kernels[name] else {
                throw RuntimeError("Kernel '\(name)' not compiled")
            }
            let enc = try encoderFor(label: label ?? name)
            enc.setComputePipelineState(kernel.pipeline)
            for (i, (buf, offset)) in buffers.enumerated() {
                enc.setBuffer(buf, offset: offset, index: i)
            }
            enc.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: kernel.threadsPerGroup, height: 1, depth: 1)
            )
        }

        /// `totalThreads` is the 1-D grid size; buffer entries may carry an
        /// offset (in bytes) — the capability metalgpu lacked.
        func dispatch(_ name: String, buffers: [(MTLBuffer, Int)], totalThreads: Int,
                      label: String? = nil) throws {
            guard let kernel = runner.kernels[name] else {
                throw RuntimeError("Kernel '\(name)' not compiled")
            }
            let tpg = kernel.threadsPerGroup
            try encode(name, buffers: buffers, groups: (totalThreads + tpg - 1) / tpg,
                       label: label)
        }

        func dispatch(_ name: String, buffers: [MTLBuffer], totalThreads: Int,
                      label: String? = nil) throws {
            try dispatch(name, buffers: buffers.map { ($0, 0) },
                         totalThreads: totalThreads, label: label)
        }

        /// Explicit threadgroup count (e.g. one workgroup per matrix row).
        func dispatchGroups(_ name: String, buffers: [(MTLBuffer, Int)], groups: Int,
                            label: String? = nil) throws {
            try encode(name, buffers: buffers, groups: groups, label: label)
        }

        func commitAndWait() {
            encoder?.endEncoding()
            encoder = nil
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
            profiler?.collect()
        }
    }

    func batch(profiler: KernelProfiler? = nil) throws -> CommandBatch {
        try CommandBatch(runner: self, profiler: profiler)
    }
}

/// Per-kernel GPU timing via counter sample buffers. One instance accumulates
/// across many batches; `report()` prints totals.
final class KernelProfiler {
    let sampleBuffer: MTLCounterSampleBuffer
    private let device: MTLDevice
    private let capacity: Int
    private var labels: [String] = []
    private(set) var totals: [String: (count: Int, ns: Double)] = [:]
    private let calibration: (cpu0: MTLTimestamp, gpu0: MTLTimestamp)

    init(device: MTLDevice, capacity: Int = 1024) throws {
        guard device.supportsCounterSampling(.atStageBoundary) else {
            throw RuntimeError("device does not support stage-boundary counter sampling")
        }
        guard let set = device.counterSets?.first(where: { $0.name == "timestamp" }) else {
            throw RuntimeError("no timestamp counter set")
        }
        let desc = MTLCounterSampleBufferDescriptor()
        desc.counterSet = set
        desc.storageMode = .shared
        desc.sampleCount = capacity * 2
        self.sampleBuffer = try device.makeCounterSampleBuffer(descriptor: desc)
        self.device = device
        self.capacity = capacity
        let ts = device.sampleTimestamps()
        self.calibration = (ts.cpu, ts.gpu)
    }

    fileprivate func nextSampleIndices(label: String) -> (Int, Int)? {
        guard labels.count < capacity else { return nil }
        let i = labels.count
        labels.append(label)
        return (2 * i, 2 * i + 1)
    }

    /// Convert GPU tick durations to ns using a fresh calibration pair.
    private func gpuTickScale() -> Double {
        let ts = device.sampleTimestamps()
        let dGpu = Double(ts.gpu - calibration.gpu0)
        return dGpu > 0 ? Double(ts.cpu - calibration.cpu0) / dGpu : 1.0
    }

    /// Resolve samples of the just-completed batch and fold into totals.
    fileprivate func collect() {
        guard !labels.isEmpty,
              let data = (try? sampleBuffer.resolveCounterRange(0..<(2 * labels.count))) ?? nil
        else {
            labels.removeAll()
            return
        }
        let scale = gpuTickScale()
        data.withUnsafeBytes { raw in
            let ts = raw.bindMemory(to: UInt64.self)
            for (i, label) in labels.enumerated() {
                let t0 = ts[2 * i], t1 = ts[2 * i + 1]
                guard t1 > t0, t0 != 0, t1 != .max else { continue }
                var rec = totals[label] ?? (0, 0)
                rec.count += 1
                rec.ns += Double(t1 - t0) * scale
                totals[label] = rec
            }
        }
        labels.removeAll()
    }

    func report() -> String {
        let grand = totals.values.reduce(0.0) { $0 + $1.ns }
        func pad(_ s: String, _ n: Int) -> String {
            s.count >= n ? s : s + String(repeating: " ", count: n - s.count)
        }
        var lines = [pad("call-site", 16) + " count  mean µs  total ms  share"]
        for (label, rec) in totals.sorted(by: { $0.value.ns > $1.value.ns }) {
            lines.append(pad(label, 16) + String(
                format: "%6d %8.1f %9.2f %5.1f%%",
                rec.count, rec.ns / Double(rec.count) / 1000,
                rec.ns / 1e6, 100 * rec.ns / grand))
        }
        lines.append(String(format: "GPU total: %.1f ms", grand / 1e6))
        return lines.joined(separator: "\n")
    }
}
