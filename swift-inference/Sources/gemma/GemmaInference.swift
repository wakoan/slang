import Foundation

struct GemmaConfig {
    let hiddenSize: Int
    let numLayers: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let intermediateSize: Int
    let vocabSize: Int
    let rmsNormEps: Float
    let ropeThetaGlobal: Float
    let ropeThetaLocal: Float
    let slidingWindow: Int
    let queryPreAttnScalar: Float
    let layerTypes: [String]  // "sliding_attention" | "full_attention"
    let bosTokenId: Int
    let eosTokenIds: [Int]

    static func load(from path: String) throws -> GemmaConfig {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RuntimeError("config.json is not a JSON object")
        }
        func intVal(_ key: String) throws -> Int {
            guard let v = dict[key] as? Int else { throw RuntimeError("config missing \(key)") }
            return v
        }
        func floatVal(_ key: String) throws -> Float {
            guard let v = dict[key] as? Double else { throw RuntimeError("config missing \(key)") }
            return Float(v)
        }
        let numKVHeads = try intVal("num_key_value_heads")
        guard numKVHeads == 1 else {
            throw RuntimeError("Kernels assume num_key_value_heads == 1, got \(numKVHeads)")
        }
        return GemmaConfig(
            hiddenSize: try intVal("hidden_size"),
            numLayers: try intVal("num_hidden_layers"),
            numHeads: try intVal("num_attention_heads"),
            numKVHeads: numKVHeads,
            headDim: try intVal("head_dim"),
            intermediateSize: try intVal("intermediate_size"),
            vocabSize: try intVal("vocab_size"),
            rmsNormEps: try floatVal("rms_norm_eps"),
            ropeThetaGlobal: try floatVal("rope_theta"),
            ropeThetaLocal: try floatVal("rope_local_base_freq"),
            slidingWindow: try intVal("sliding_window"),
            queryPreAttnScalar: try floatVal("query_pre_attn_scalar"),
            layerTypes: dict["layer_types"] as? [String] ?? [],
            bosTokenId: try intVal("bos_token_id"),
            eosTokenIds: [1, 106]  // <eos>, <end_of_turn>
        )
    }
}
