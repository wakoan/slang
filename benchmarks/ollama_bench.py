"""Benchmark Ollama on Gemma 4 E2B, matched to litert/bench.py's shape.

    python benchmarks/ollama_bench.py [model] [prefill_tokens] [decode_tokens]

Third-party reference point for gemma4_150's prefill/decode numbers. Ollama's
/api/generate reports prompt_eval_* and eval_* separately, which is exactly the
prefill/decode split we want -- and it excludes model load time (reported
separately as load_duration), so the figures are comparable to ours.

Methodology, the same rules the prefill work had to learn twice (see
gemma4_150/prefill_research/README.md):

  * warm up before timing -- the first request loads the model AND leaves the
    GPU clock cold, worth ~2.8x on this machine;
  * best-of-N, not a single sample;
  * match the prompt LENGTH, not the prompt text. Ollama tokenizes with its own
    GGUF vocab, so the prompt is grown/shrunk until prompt_eval_count lands
    within 2% of the target rather than assumed;
  * defeat the prefix cache. Ollama reuses the KV cache of a shared prompt
    prefix, so re-sending one prompt would make every run after the first
    report a near-zero prefill -- a spectacular and completely fake speedup.
    Each run therefore gets a unique token at the FRONT, which invalidates the
    whole prefix. prompt_eval_count is printed per run so a cache hit (a count
    far below the target) is visible rather than silently averaged in.

`raw=True` bypasses the chat template so the prompt is exactly the tokens we
sent -- no template tokens, no thinking harness inflating the decode count.
"""
import json
import sys
import time
import urllib.request

HOST = "http://localhost:11434"
FILLER = ("The history of computing begins long before the electronic computer. "
          "Mechanical calculators, punched cards and the theory of computation all "
          "predate the first stored-program machines by decades. ")


def post(path, payload, timeout=900):
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def gen(model, prompt, n_predict, num_ctx):
    return post("/api/generate", {
        "model": model, "prompt": prompt, "raw": True, "stream": False,
        "keep_alive": "10m",
        "options": {"num_predict": n_predict, "num_ctx": num_ctx,
                    "temperature": 0, "stop": []},
    })


def make_prompt(reps, tag):
    """Turn-wrapped so the model actually answers at length.

    A bare block of filler makes it emit EOS within ~2 tokens, which yields a
    decode "rate" measured over 2 tokens -- i.e. fixed overhead, not throughput.
    Wrapping the same filler in the Gemma turn format with an explicit
    instruction gets a full-length generation, so decode is measured over the
    tokens actually requested.

    `tag` sits at the FRONT so it invalidates Ollama's prefix cache.
    """
    return (f"<|turn>user\nNote {tag}. Read the passage and then write a long, "
            f"detailed essay about the history of computing.\n"
            + FILLER * reps
            + "<turn|>\n<|turn>model\n")


def fit_prompt(model, target, num_ctx):
    """Grow the filler until Ollama's own tokenizer reports ~target tokens."""
    reps = max(1, target // 40)
    for i in range(8):
        n = gen(model, make_prompt(reps, f"fit{i}"), 1, num_ctx)["prompt_eval_count"]
        if abs(n - target) <= target * 0.02:
            return reps, n
        reps = max(1, round(reps * target / n))
    return reps, n


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e2b"
    n_pre = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    n_dec = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    num_ctx = 1 << max(11, (n_pre + n_dec - 1).bit_length())

    print(f"model {model}   target {n_pre} prefill / {n_dec} decode   num_ctx {num_ctx}")
    reps, n_tok = fit_prompt(model, n_pre, num_ctx)
    print(f"prompt fitted to {n_tok} tokens")

    print("warming (model load + GPU clock)…")
    t0 = time.time()
    w = gen(model, make_prompt(reps, "warm"), 32, num_ctx)
    print(f"  load {w.get('load_duration', 0) / 1e9:.1f}s, first call {time.time() - t0:.1f}s")

    best_p, best_d, runs = 0.0, 0.0, []
    for i in range(3):
        r = gen(model, make_prompt(reps, f"run{i}"), n_dec, num_ctx)
        p = r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9)
        d = r["eval_count"] / (r["eval_duration"] / 1e9)
        runs.append((r["prompt_eval_count"], p, r["eval_count"], d))
        print(f"  run {i}: prefill {r['prompt_eval_count']:5d} tok @ {p:7.1f} tok/s   "
              f"decode {r['eval_count']:4d} tok @ {d:6.1f} tok/s")
        best_p, best_d = max(best_p, p), max(best_d, d)

    print(f"\nBEST  prefill {best_p:.1f} tok/s   decode {best_d:.1f} tok/s")
    if any(c < n_dec for _, _, c, _ in runs):
        print("NOTE: a run generated fewer than the requested tokens (hit a stop "
              "token); its decode rate is still measured over the tokens produced.")
    if any(pc < n_pre * 0.9 for pc, _, _, _ in runs):
        print("WARNING: a run's prompt_eval_count came in well under the target — "
              "the prefix cache was hit, so that prefill figure is not a real "
              "measurement. Vary the prompt further.")


if __name__ == "__main__":
    main()
