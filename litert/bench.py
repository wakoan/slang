import litert_lm

# Run a controlled performance benchmark
benchmark = litert_lm.Benchmark(
    model_path="/Users/wako/.litert-lm/cache/huggingface/litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm",
    backend=litert_lm.Backend.CPU(), # Or Backend.GPU()
    prefill_tokens=1024,
    decode_tokens=256
)

print(benchmark.run())

