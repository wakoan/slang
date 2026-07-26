# Third-party runtimes vs this repo — Gemma 4 E2B, M4 Pro

All numbers measured on the same machine (M4 Pro, 24 GB), same shape:
**1024-token prefill, 256-token decode**. Reproduce with:

    python benchmarks/ollama_bench.py gemma4:e2b 1024 256    # Ollama
    python litert/bench.py                                   # LiteRT-LM
    python -m gemma4_150.prefill_research.test_prefill_batched 1024   # ours (prefill)

## Headline (2026-07-25)

| Runtime | Prefill | Decode @ 1024 ctx |
|---|---:|---:|
| **ours** (`gemma4_150`, native Metal) | **1208 tok/s** | **148.1 tok/s** |
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

## Portable prefill: what the non-Metal backends can reach

Prefill's 1208 tok/s uses `MPSMatrixMultiplication`, an Apple library with no
WGSL equivalent — so it cannot be "written once". `gemma4_150/kernels_dsl.py`
holds a DSL-authored tiled GEMM that can, and `PrefillGPU.GEMM` selects between
them (`"mps"` / `"dsl"`). One kernel source, two implementations.

Measured on the same 1024-token prefill, Metal, so the comparison isolates the
GEMM rather than the backend:

| GEMM | prefill | vs MPS |
|---|---:|---:|
| MPS (Apple-only) | 1216 tok/s | 1.00× |
| DSL `gemm_tiled` (portable, **all** GEMMs) | 564 tok/s | 0.46× |
| per-token (what wgpu/browser do today) | 158 tok/s | 0.13× |

The two produce **bit-identical KV caches**, so this is purely a speed tradeoff.

Isolated GEMM throughput on the dominant prefill shape (12288×1536 f16):

| M | DSL | MPS |
|---:|---:|---:|
| 64 | 1.48 | 2.41 |
| 256 | 2.50 | 5.22 |
| 1024 | **2.61** | **5.64** |

So the portable path is ~46% of MPS and ~3.6× what the non-Metal backends have
today. The `"dsl"` setting needs no fallback: one kernel covers both
orientations (weights are transposed-right, attention's P@V is not) and carries
explicit row strides for the padded score matrix, so nothing silently reverts to
MPS. `gemm_tiled` is a first tiled implementation (32×32×16 tiles, 2×2 per
thread); the gap to MPS is tuning headroom, not a hard limit — though the
earlier `simdgroup_matrix` prototype also landed at 2.59, which suggests ~2.6 is
where straightforward Metal-level tiling sits on this machine.

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

## Resolved: the old 125.5 was a timer bug, not a slow GPU

`litert/RESULTS.md` recorded 125.5 tok/s decode at 1024 context; measurement gave
~148. The gap was a bug in `generate()`, and it is now fixed in every backend.

Decode dispatches a whole chunk of 32 forwards before reading any token back, so
an early EOS discards up to 31 tokens **whose work is still inside the timer**.
The rate then divided the surviving tokens by the full elapsed time — a truncated
numerator over a full denominator. At a fixed 1024-token context, the same code
on the same machine reported:

| prompt | tokens generated | before fix | after fix |
|---|---:|---:|---:|
| "continue at length" | 240 | 150.3 | 149.4 |
| "summarize in one sentence" | 32 | 76.8 | 149.0 |
| "answer yes or no" | 5 | 23.9 | 148.3 |

So the reported figure depended on *what the model chose to say*, not on decode
speed. 125.5 was a prompt that stopped somewhere around 200 tokens.

Two things worth keeping from this:

* **`chat_turn` always divided by the executed step count; `generate` was the odd
  one out** — the correct form was already in the codebase, which is why the bug
  survived review.
* It was hit **three separate times in one session** before being recognised: a
  short prompt reading 59 tok/s instead of 166, the Ollama harness reading 164
  tok/s over 2 generated tokens, and this. A decode rate should always be
  reported alongside the number of tokens it was measured over; if that count is
  small or unstated, the rate is not a measurement.

Falsified along the way: memory/GPU contention. Holding Ollama's 7.1 GB model
resident at 100% GPU changed our decode by under 1% (167.4 / 153.2 / 149.2 vs
167.4 / 153.1 / 148.1 clean), so the co-resident LiteRT process was not the
cause.
