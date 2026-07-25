import Foundation

// Persistent tokenizer co-process: one long-lived `python` running the HF
// `tokenizers` lib, spoken to over a JSON-lines protocol on stdin/stdout.
// Replaces the per-call `python -c …` spawns in main.swift so a chat session
// pays the ~0.3s interpreter+model load once instead of per token.
final class Tok {
    private let proc = Process()
    private let inPipe = Pipe(), outPipe = Pipe()
    private let inH: FileHandle, outH: FileHandle
    private var buf = Data()

    init(repo: URL, tokenizerJSON: String) throws {
        let code = """
        import sys, json
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(sys.argv[1])
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            op = r["op"]
            if op == "encode":
                out = {"ids": t.encode(r["text"], add_special_tokens=False).ids}
            elif op == "decode":
                out = {"text": t.decode([int(x) for x in r["ids"]])}
            elif op == "special":
                out = {s: t.token_to_id(s) for s in r["tokens"]}
            else:
                out = {"error": "bad op"}
            sys.stdout.write(json.dumps(out) + "\\n")
            sys.stdout.flush()
        """
        proc.executableURL = repo.appendingPathComponent("venv/bin/python")
        proc.arguments = ["-c", code, tokenizerJSON]
        proc.standardInput = inPipe
        proc.standardOutput = outPipe
        inH = inPipe.fileHandleForWriting
        outH = outPipe.fileHandleForReading
        try proc.run()
    }

    private func request(_ obj: [String: Any]) -> [String: Any] {
        let data = try! JSONSerialization.data(withJSONObject: obj)
        inH.write(data); inH.write(Data([0x0A]))
        while true {
            if let nl = buf.firstIndex(of: 0x0A) {
                let line = Data(buf[buf.startIndex..<nl])
                buf = Data(buf[buf.index(after: nl)...])
                return (try? JSONSerialization.jsonObject(with: line) as? [String: Any]) ?? [:]
            }
            let chunk = outH.availableData
            if chunk.isEmpty { fatalError("tokenizer process ended") }
            buf.append(chunk)
        }
    }

    func encode(_ text: String) -> [UInt32] {
        (request(["op": "encode", "text": text])["ids"] as? [Any] ?? [])
            .map { UInt32(truncating: $0 as! NSNumber) }
    }
    func decode(_ ids: [UInt32]) -> String {
        request(["op": "decode", "ids": ids.map { Int($0) }])["text"] as? String ?? ""
    }
    // token_to_id for special-token strings; nil when the token is absent.
    func specialIds(_ tokens: [String]) -> [String: UInt32?] {
        let m = request(["op": "special", "tokens": tokens])
        var out: [String: UInt32?] = [:]
        for tk in tokens {
            if let n = m[tk] as? NSNumber { out[tk] = UInt32(truncating: n) } else { out[tk] = UInt32?.none }
        }
        return out
    }
}
