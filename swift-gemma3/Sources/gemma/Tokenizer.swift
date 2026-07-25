import Foundation

class Tokenizer {
    // Simple BPE tokenizer stub
    // TODO: Full implementation with merge rules and subword encoding
    var vocabSize: Int = 256128
    var bosToken: Int = 2
    var eosTokens: [Int] = [1, 106]

    static func loadFromVocab(_ path: String) throws -> Tokenizer {
        let tokenizer = Tokenizer()
        // Future: parse tokenizer.json to load BPE merges and vocab
        return tokenizer
    }

    func encode(_ text: String) -> [Int] {
        // Ultra-simple: byte-level encoding
        // In production, use full BPE from tokenizer.json
        var tokens: [Int] = [bosToken]
        for byte in text.utf8 {
            if byte < 128 {
                tokens.append(Int(byte))
            }
        }
        return tokens
    }

    func decode(_ tokens: [Int]) -> String {
        // Ultra-simple reverse
        var result = ""
        for token in tokens {
            if token >= 32 && token < 127 {
                result.append(Character(UnicodeScalar(token)!))
            }
        }
        return result
    }
}
