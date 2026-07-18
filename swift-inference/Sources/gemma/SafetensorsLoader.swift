import Foundation

func Float16toFloat32(_ bits: UInt16) -> Float {
    // Convert IEEE 754 half-precision to single-precision
    let exponent = (bits >> 10) & 0x1F
    let mantissa = bits & 0x3FF
    let sign = bits >> 15

    if exponent == 0 {
        return sign == 0 ? 0.0 : -0.0
    } else if exponent == 31 {
        return mantissa == 0 ? (sign == 0 ? Float.infinity : -Float.infinity) : Float.nan
    }

    let f32Mantissa = UInt32(mantissa) << 13
    let f32Exponent = UInt32((Int(exponent) - 15 + 127)) << 23
    let f32Sign = UInt32(sign) << 31
    let f32Bits = f32Sign | f32Exponent | f32Mantissa
    return Float(bitPattern: f32Bits)
}

class SafetensorsLoader {
    let fileURL: URL

    init(path: String) {
        self.fileURL = URL(fileURLWithPath: path)
    }

    func load() throws -> [String: [Float]] {
        let data = try Data(contentsOf: fileURL)

        // Read header length (first 8 bytes, little-endian u64)
        var headerLen: UInt64 = 0
        _ = withUnsafeMutableBytes(of: &headerLen) { buffer in
            data.copyBytes(to: buffer, from: 0..<8)
        }

        // Parse header JSON
        let headerData = data.subdata(in: 8..<(8 + Int(headerLen)))
        guard let headerJson = try JSONSerialization.jsonObject(
            with: headerData
        ) as? [String: Any] else {
            throw LoadError("Failed to parse header")
        }

        var tensors: [String: [Float]] = [:]
        let base = 8 + Int(headerLen)

        for (name, info) in headerJson {
            if name == "__metadata__" { continue }

            guard let infoDict = info as? [String: Any],
                  let offsets = infoDict["data_offsets"] as? [Int],
                  offsets.count == 2,
                  let shape = infoDict["shape"] as? [Int],
                  let dtype = infoDict["dtype"] as? String else {
                continue
            }

            let start = offsets[0]
            let end = offsets[1]
            let bufferRange = base + start..<base + end

            let buffer = data.subdata(in: bufferRange)
            let floatData = try parseBuffer(buffer, dtype: dtype, shape: shape)
            tensors[name] = floatData
        }

        return tensors
    }

    private func parseBuffer(_ buffer: Data, dtype: String, shape: [Int]) throws -> [Float] {
        let elementCount = shape.reduce(1, *)

        if dtype == "F32" {
            var floats = [Float](repeating: 0, count: elementCount)
            buffer.withUnsafeBytes { ptr in
                guard let src = ptr.baseAddress?.assumingMemoryBound(to: Float.self) else {
                    return
                }
                memcpy(&floats, src, MemoryLayout<Float>.size * elementCount)
            }
            return floats
        } else if dtype == "F16" {
            var floats = [Float]()
            floats.reserveCapacity(elementCount)
            _ = buffer.withUnsafeBytes { ptr in
                guard let baseAddress = ptr.baseAddress else { return 0 }
                for i in 0..<elementCount {
                    let u16Ptr = baseAddress.assumingMemoryBound(to: UInt16.self)
                    let u16 = u16Ptr[i]
                    floats.append(Float16toFloat32(u16))
                }
                return 0
            }
            return floats
        } else if dtype == "BF16" {
            var floats = [Float]()
            floats.reserveCapacity(elementCount)
            let uint16Array = buffer.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: UInt16.self))
            }
            for u16 in uint16Array {
                // BF16: shift left 16 bits to reconstruct f32
                let f32Bits = UInt32(u16) << 16
                floats.append(Float(bitPattern: f32Bits))
            }
            return floats
        } else {
            throw LoadError("Unsupported dtype: \(dtype)")
        }
    }
}

enum LoadError: Error {
    case message(String)

    init(_ msg: String) {
        self = .message(msg)
    }
}
