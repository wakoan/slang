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

    /// Encodes many dispatches into one command buffer — commit once, wait once.
    final class CommandBatch {
        private let runner: MetalRunner
        private let cmdBuffer: MTLCommandBuffer
        private let encoder: MTLComputeCommandEncoder

        fileprivate init(runner: MetalRunner) throws {
            guard let cmdBuffer = runner.queue.makeCommandBuffer(),
                  let encoder = cmdBuffer.makeComputeCommandEncoder() else {
                throw RuntimeError("Failed to create command buffer/encoder")
            }
            self.runner = runner
            self.cmdBuffer = cmdBuffer
            self.encoder = encoder
        }

        /// `totalThreads` is the 1-D grid size; buffer entries may carry an
        /// offset (in bytes) — the capability metalgpu lacked.
        func dispatch(_ name: String, buffers: [(MTLBuffer, Int)], totalThreads: Int) throws {
            guard let kernel = runner.kernels[name] else {
                throw RuntimeError("Kernel '\(name)' not compiled")
            }
            encoder.setComputePipelineState(kernel.pipeline)
            for (i, (buf, offset)) in buffers.enumerated() {
                encoder.setBuffer(buf, offset: offset, index: i)
            }
            let tpg = kernel.threadsPerGroup
            let groups = (totalThreads + tpg - 1) / tpg
            encoder.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1)
            )
        }

        func dispatch(_ name: String, buffers: [MTLBuffer], totalThreads: Int) throws {
            try dispatch(name, buffers: buffers.map { ($0, 0) }, totalThreads: totalThreads)
        }

        func commitAndWait() {
            encoder.endEncoding()
            cmdBuffer.commit()
            cmdBuffer.waitUntilCompleted()
        }
    }

    func batch() throws -> CommandBatch {
        try CommandBatch(runner: self)
    }
}
