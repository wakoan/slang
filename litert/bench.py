"""Benchmark Google's LiteRT-LM runtime on Gemma 4 E2B, for comparison against
this repo's own backends (see gemma4_150/ and the top-level README perf matrix).

Model is the litert-community .litertlm (instruction-tuned, multimodal). Needs
`litert_lm` installed and the model cached under ~/.litert-lm/. Prints prefill +
decode tok/s for each available backend; see RESULTS.md for a recorded run.
"""
import litert_lm

MODEL = (
    "/Users/wako/.litert-lm/cache/huggingface/litert-community/"
    "gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm"
)


def run(name, backend, prefill=1024, decode=256):
    try:
        b = litert_lm.Benchmark(
            model_path=MODEL, backend=backend,
            prefill_tokens=prefill, decode_tokens=decode,
        )
        info = b.run()
        print(f"[{name}]  prefill {info.last_prefill_tokens_per_second:7.1f} tok/s   "
              f"decode {info.last_decode_tokens_per_second:6.1f} tok/s   "
              f"TTFT {info.time_to_first_token_in_second:.2f}s   "
              f"init {info.init_time_in_second:.1f}s")
    except Exception as e:
        print(f"[{name}]  FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    run("GPU", litert_lm.Backend.GPU())
    run("CPU", litert_lm.Backend.CPU())
