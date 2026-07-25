mod metal;
mod runner;

use runner::G4;
use std::io::{self, BufRead, Write};
use tokenizers::Tokenizer;

fn tok_path() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()   // rust_metal -> backends
        .parent().unwrap()   // backends -> gemma4_150
        .parent().unwrap()   // gemma4_150 -> repo root
        .join("models/gemma-4-E2B-qat/tokenizer.json")
        .to_string_lossy()
        .into_owned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prompt = args.get(1).cloned().unwrap_or_else(|| "The capital of France is".into());
    let n_new: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(48);

    eprintln!("loading model + compiling kernels…");
    let mut g4 = G4::new().expect("init");
    let tok = Tokenizer::from_file(tok_path()).expect("tokenizer");

    if prompt == "chat" {
        chat(&mut g4, &tok);
        return;
    }

    // single-shot completion in the QAT chat format
    let framed = format!("<|turn>user\n{prompt}<turn|>\n<|turn>model\n");
    let mut ids = vec![g4.bos];
    ids.extend(tok.encode(framed, false).unwrap().get_ids().iter().copied());
    let (out, tps) = g4.generate(&ids, n_new);
    println!("prompt: {prompt}");
    println!("=> {}", tok.decode(&out, true).unwrap());
    println!("decode: {tps:.1} tok/s");
}

fn chat(g4: &mut G4, tok: &Tokenizer) {
    let turn_close = tok.token_to_id("<turn|>");
    let mut stop = vec![g4.eos];
    if let Some(tc) = turn_close {
        stop.push(tc);
    }
    g4.chat_reset();
    let mut turn = 0usize;
    println!("chat ready — type a message; /reset clears history, /quit exits.\n");
    let stdin = io::stdin();
    loop {
        print!("you: ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let msg = line.trim();
        if msg.is_empty() {
            continue;
        }
        if msg == "/quit" || msg == "/exit" {
            break;
        }
        if msg == "/reset" {
            g4.chat_reset();
            turn = 0;
            println!("(history cleared)\n");
            continue;
        }

        let user = format!("<|turn>user\n{msg}<turn|>\n<|turn>model\n");
        let mut frame: Vec<u32> = Vec::new();
        if turn == 0 {
            frame.push(g4.bos);
            frame.extend(tok.encode(user, false).unwrap().get_ids().iter().copied());
        } else {
            frame.extend(tok.encode(format!("\n{user}"), false).unwrap().get_ids().iter().copied());
        }
        turn += 1;

        print!("bot: ");
        io::stdout().flush().ok();
        let mut printed = 0usize;
        let (out, tps) = g4.chat_turn(&frame, 240, &stop, |acc| {
            // native detokenize is cheap; decode-and-delta-print each chunk
            let s = tok.decode(acc, true).unwrap_or_default();
            if s.len() > printed {
                print!("{}", &s[printed..]);
                io::stdout().flush().ok();
                printed = s.len();
            }
        });
        println!("\n({} tok, {:.1} tok/s)\n", out.len(), tps);
    }
}
