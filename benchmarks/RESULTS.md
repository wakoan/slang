# Third-party runtimes vs this repo — Gemma 4 E2B, M4 Pro

All numbers measured on the same machine (M4 Pro, 24 GB), same shape:
**1024-token prefill, 256-token decode**. Reproduce with:

    python benchmarks/ollama_bench.py gemma4:e2b 1024 256    # Ollama
    python litert/bench.py                                   # LiteRT-LM
    python -m gemma4_150.prefill_research.test_prefill_batched 1024   # ours (prefill)

## Headline (2026-07-25)

| Runtime | Prefill | Decode @ 1024 ctx |
|---|---:|---:|
| **ours** (`gemma4_150`, native Metal) | **1208 tok/s** | **~148 tok/s** |
| Ollama 0.30.6 (`gemma4:e2b`) | 1153 tok/s | 87.2 tok/s |
| LiteRT-LM (GPU/WebGPU) | 1132 tok/s | 95.7 tok/s |
| LiteRT-LM (CPU/XNNPACK) | 443 tok/s | 40.2 tok/s |

Ahead on both axes, but by very different margins: **1.05× on prefill** against
Ollama, **1.70× on decode**. The prefill margin is the honest headline — see the
weight-size caveat below for why the decode one flatters us.

## Is it the same model?

Yes on architecture, no on weights. Verified via `ollama show`:

| | ours | Ollama | LiteRT |
|---|---|---|---|
| embedding length | 1536 | **1536** | — |
| params | 4.6B (text) | 5.1B (+vision/audio) | multimodal |
| weights | 2.11 GB int2/4/8 QAT | **7.2 GB Q4_K_M** | 2.59 GB |

Same Gemma 4 E2B base — the embedding length of 1536 pins it, and 8B/12B variants
were checked and rejected (embedding 2560/3840). The parameter delta is the
vision+audio towers, which text inference does not touch.

**The 3.4× weight-size difference is the caveat that matters.** Decode is
bandwidth-bound, so most of the 1.70× decode margin is quantization, not kernel
quality. In fact the interesting reading runs the other way: we carry **3.4×
fewer weight bytes but decode only 1.70× faster**, so Ollama extracts more
throughput per byte moved than we do. That is consistent with this repo's own
finding that effective bandwidth sits far below the hardware peak, and it is a
lead worth pulling on rather than a result to celebrate.

Prefill is the sounder comparison: it is compute-bound at large M, so it is far
less sensitive to the weight format, and it is where the batched-GEMM work
(`gemma4_150/prefill_gpu.py`) actually shows up.

## Measurement notes

Both hazards below produce large, confident, wrong numbers, and both were hit
here before being fixed:

1. **Ollama's prefix cache.** Re-sending one prompt makes every run after the
   first report a near-zero `prompt_eval_count` and an absurd prefill rate. Each
   run gets a unique token at the FRONT of the prompt; `prompt_eval_count` is
   printed per run so a cache hit is visible rather than silently averaged in.
2. **Early EOS destroying the decode sample.** A bare block of repetitive filler
   makes the model stop after ~2 tokens, and a decode rate over 2 tokens is
   fixed overhead, not throughput — it read 164 tok/s that way, nearly double the
   real 87. The prompt is now turn-wrapped with an explicit instruction, and the
   generated count is asserted against the request.

Plus the standing rules from `gemma4_150/prefill_research/README.md`: warm the
GPU clock first (2.8× swing cold), best-of-N, and never time a per-token
baseline as one command buffer per token.

## Open

Our decode at 1024 context measures **146.7–148.8 tok/s** across three runs
today, but `litert/RESULTS.md` records **125.5** for the same configuration.
Unreconciled. The higher figure is the more physically plausible one — 28 of 35
layers are sliding-window capped at 512 keys, so 512→1024 context should cost
little, and the measured 149.8→148.5 matches that — but until the discrepancy is
explained, treat ~148 as provisional.
