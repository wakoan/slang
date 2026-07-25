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
let ids = tokenize(prompt)
let (out, tps) = try g4.generate(ids, nNew)
print("prompt: \(prompt)")
print("=> \(detokenize(out))")
print(String(format: "decode: %.1f tok/s", tps))
