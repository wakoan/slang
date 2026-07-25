# LiteRT-LM vs this repo's backends — Gemma 4 E2B, M4 Pro

Google's official on-device runtime (LiteRT-LM) running Gemma 4 E2B, benchmarked
against this repo's own gemma4_150 backends on the same machine. Run with
`python litert/bench.py` (1024-token prefill, 256-token decode).

## LiteRT-LM (2026-07-25)

| Backend | Prefill | Decode | TTFT | Init |
|---|---:|---:|---:|---:|
| **GPU** (`GPU WebGPU`) | 1132 tok/s | **95.7 tok/s** | 0.92 s | 5.2 s |
| **CPU** (XNNPACK) | 443 tok/s | **40.2 tok/s** | 2.33 s | 0.28 s |

**Key finding:** LiteRT's "GPU" backend reports itself as `GPU WebGPU` in its logs —
it runs through **Dawn/WebGPU, not native Metal**, and does **not** use the
`simdgroup_matrix` units for decode. That's why its decode (95.7) sits in the same
tier as this repo's wgpu-py (97) and browser/Dawn (156) backends rather than the
native-Metal tier.

## Head-to-head decode, matched context

Decode legitimately slows as KV context grows, so the comparison must fix context.
This repo's Python native-Metal runner (`gemma4_150/metal_runner.py`), decode tok/s:

| Context | native-Metal (Python) | LiteRT GPU |
|---:|---:|---:|
| 64 | 160.7 | — |
| 512 | 149.4 | — |
| **1024** | **125.5** | **95.7** |

At the matched 1024-token context, native Metal is **~1.3× faster on decode**
(and Swift/Rust run a few tok/s above Python). At short chat context (~tens of
tokens) native Metal reaches ~170.

## The honest caveats

1. **Prefill: LiteRT wins decisively — 1132 tok/s vs this repo's one-token-per-pass.**
   LiteRT does *batched* prefill (the regime where matmul batching / matrix units pay
   off); these runners process the prompt one token per forward. This is the strongest
   argument for the "batched prefill" roadmap item and the reason LiteRT reaches
   first-token faster on long prompts.
2. **Different quantization.** LiteRT's `.litertlm` is 2.4 GB, instruction-tuned +
   multimodal; this repo runs the QAT `g4_150` (2.1 GB, text). Not an identical weight
   format, so decode-bandwidth is not perfectly matched.

## Bottom line

- **Decode** (short or matched 1024 context): this repo's native-Metal backends beat
  LiteRT's GPU backend (~1.3–1.8×), because LiteRT decodes through WebGPU/Dawn.
- **Prefill / time-to-first-token:** LiteRT wins big via batched prefill — the one
  thing these decode-focused runners don't do yet.
