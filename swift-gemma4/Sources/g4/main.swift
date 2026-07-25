import Foundation

let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

// tokenize / detokenize via the repo venv python (tokenizers lib, -it chat format)
func py(_ code: String, _ arg: String) -> String {
    let p = Process()
    p.executableURL = repo.appendingPathComponent("venv/bin/python")
    p.arguments = ["-c", code, arg]
    let pipe = Pipe(); p.standardOutput = pipe
    try! p.run(); p.waitUntilExit()
    return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)!
        .trimmingCharacters(in: .whitespacesAndNewlines)
}
let TOK = repo.appendingPathComponent("models/gemma-4-E2B-qat/tokenizer.json").path
func tokenize(_ s: String) -> [UInt32] {
    let code = "from tokenizers import Tokenizer; import sys; t=Tokenizer.from_file('\(TOK)'); " +
        "ids=[2]+t.encode('<|turn>user\\n'+sys.argv[1]+'<turn|>\\n<|turn>model\\n',add_special_tokens=False).ids; print(' '.join(map(str,ids)))"
    return py(code, s).split(separator: " ").map { UInt32($0)! }
}
func detokenize(_ ids: [UInt32]) -> String {
    let code = "from tokenizers import Tokenizer; import sys; t=Tokenizer.from_file('\(TOK)'); print(t.decode([int(x) for x in sys.argv[1].split()]))"
    return py(code, ids.map(String.init).joined(separator: " "))
}

let args = CommandLine.arguments
let prompt = args.count > 1 ? args[1] : "The capital of France is"
let nNew = args.count > 2 ? Int(args[2])! : 48

print("loading model + compiling kernels…")
let g4 = try G4()
if prompt == "bench" { try g4.bench(); exit(0) }
if prompt == "profile" { try g4.profile(); exit(0) }
if prompt == "chat" { try chat(g4); exit(0) }
let ids = tokenize(prompt)
let (out, tps) = try g4.generate(ids, nNew)
print("prompt: \(prompt)")
print("=> \(detokenize(out))")
print(String(format: "decode: %.1f tok/s", tps))

// ---- interactive multi-turn chat (KV cache persists across turns) ----
func chat(_ g4: G4) throws {
    let tok = try Tok(repo: repo, tokenizerJSON: TOK)
    let sp = tok.specialIds(["<turn|>", "<bos>"])
    let bos: UInt32 = (sp["<bos>"] ?? nil) ?? 2
    var stop: Set<UInt32> = [1]                       // EOS
    if let tc = (sp["<turn|>"] ?? nil) { stop.insert(tc) }

    g4.chatReset()
    var turn = 0
    print("chat ready — type a message; /reset clears history, /quit exits.\n")
    while true {
        print("you: ", terminator: ""); fflush(stdout)
        guard let line = readLine() else { break }
        let msg = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if msg.isEmpty { continue }
        if msg == "/quit" || msg == "/exit" { break }
        if msg == "/reset" { g4.chatReset(); turn = 0; print("(history cleared)\n"); continue }

        let userStr = "<|turn>user\n\(msg)<turn|>\n<|turn>model\n"
        let frame = turn == 0 ? [bos] + tok.encode(userStr) : tok.encode("\n" + userStr)
        turn += 1

        print("bot: ", terminator: ""); fflush(stdout)
        // Detokenize + print on a background queue so the GPU decode pipeline
        // (recorded on this thread) never stalls waiting on the Python IPC.
        let dq = DispatchQueue(label: "detok")
        var printed = 0
        let (out, tps) = try g4.chatTurn(frame, maxNew: 240, stop: stop) { acc in
            dq.async {
                let s = tok.decode(acc)                // re-decode for correct BPE spacing
                if s.count > printed {
                    let from = s.index(s.startIndex, offsetBy: printed)
                    print(s[from...], terminator: ""); fflush(stdout)
                    printed = s.count
                }
            }
        }
        dq.sync {}                                     // flush any queued prints
        print(String(format: "\n(%d tok, %.1f tok/s)\n", out.count, tps))
    }
}
