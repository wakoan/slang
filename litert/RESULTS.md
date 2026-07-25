# LiteRT-LM vs this repo's backends — Gemma 4 E2B, M4 Pro

> **See also `benchmarks/RESULTS.md`** for the combined third-party table
> (ours vs Ollama vs LiteRT) measured at a matched 1024/256 shape, and for the
> unreconciled decode discrepancy against the 125.5 figure recorded below.

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
| 64 | 167.4 | — |
| 512 | 153.1 | — |
| **1024** | **148.1** | **95.7** |

At the matched 1024-token context, native Metal is **~1.55× faster on decode**
(and Swift/Rust run a few tok/s above Python). At short chat context (~tens of
tokens) native Metal reaches ~170.

**Corrected 2026-07-25 — this table previously read 160.7 / 149.4 / 125.5.** The
1024 entry was wrong, and the cause was a bug in `generate()`, not the GPU:
decode dispatches a whole chunk of forwards before reading any back, so an early
EOS discarded up to 31 tokens whose work was still inside the timer, and the rate
divided the *surviving* tokens by the *full* elapsed time. At a fixed 1024-token
context that made the same code report 150 tok/s for a long answer, 77 for a
one-sentence one and 24 for a yes/no. Every chunked backend had it; all now
divide by the forwards actually executed (`chat_turn` always did). The 64 and 512
entries barely moved because those prompts happened to run to the token cap.

## Prefill

| Prefill path | tok/s (256 / 1024) |
|---|---|
| Was: one **synchronous** forward per token | 122 / — |
| **Now: pipelined** (chunked, no per-token sync) | **162 / 158** |
| LiteRT GPU (**batched**) | 1132 |

Pipelining prefill (commit a chunk of forwards without waiting, token ids read from
a GPU buffer at an offset so the CPU never races the GPU) gives **~1.35×** and is
bit-identical to the synchronous path. Shipped on all three native-Metal backends.

**The remaining ~7× gap is true batching, not sync.** LiteRT processes S tokens per
pass, so each weight row is read **once for S tokens** instead of re-reading
~0.82 GB/token. Closing it needs batched (M=S) GEMM kernels: `embed_00`,
`plegather_01`, `rmssrq_69` and `attn_101` already carry a seq/rows dimension, but
every matmul the reference bundle captured (`qkv_70`, `oproj_73`, `gateup_74/95`,
`down_75/96`, `proj_68`, `plegate_76`, `pleproj_77`, `logits_33`) is a single-row
GEMV — the capture was decode-only. That is ~7 new kernels plus causal-mask
orchestration: a separate project, and the regime where matrix units finally pay off
(batch fills the N dimension a GEMV wastes).

## Batched-prefill prototype (measured warm, 2026-07-25)

Batched (M=S) variants of the two dominant matmuls — 2-bit gate/up (55.3% of prefill
FLOPs) and 2-bit down (27.7%) — built to measure the real win before porting all ~7.
Each thread holds a micro-tile of `N_ROWS` output rows × `SEQ` tokens, so **both**
weight and activation loads are amortized. Reproduce with
`python -m gemma4_150.prefill_research.bench_batched [S]`.

| Kernel | blocking | S=8 | S=16 |
|---|---|---:|---:|
| `gateup_95_b` | `N_ROWS=2, SEQ=4` | **1.21×** | **1.23×** |
| `down_96_b` | `N_ROWS=16, SEQ=4` | **1.45×** | **1.80×** |

**An earlier revision of this section reported 2.49× for gate/up. That was wrong**,
from two compounding measurement bugs, both now controlled for:

1. **Unfair baseline** — the S per-token dispatches were timed as S separate command
   buffers (commit+wait each) rather than S dispatches in one, which is how they
   actually run inside a forward pass. Inflated the baseline **1.80×**.
2. **Cold GPU clock** — the same `S × down_96` baseline reads **1.013 ms cold vs
   0.366 ms warm**, a 2.8× swing that lands on whichever kernel runs first. All
   numbers above follow a 2 s warm-up, min-of-3, with post-sweep drift < 1.5%.

**Consequence for the roadmap:** applying this across every matmul is worth
**~1.26–1.32× overall, i.e. prefill ~200–210 tok/s** — not the 350–400 previously
extrapolated. It is capped by gate/up, 55% of the FLOPs at only 1.21×. **That still
leaves a 5.4× gap to LiteRT, so token-batching is closed as a route to parity.**
The only route whose arithmetic reaches 1132 is dequantize-to-f16 once per prefill +
an MPS-class tiled GEMM at large M (LiteRT runs at 4.53 TFLOP/s ≈ 92% of this
machine's measured 4.90 GEMM ceiling; we are at 0.63). See
`gemma4_150/prefill_research/README.md`.

## The honest caveats

1. **Prefill: LiteRT still wins — 1132 vs 158 tok/s** (see above). Batched prefill
   remains the open roadmap item; pipelining closed only the sync portion.
2. **Different quantization.** LiteRT's `.litertlm` is 2.4 GB, instruction-tuned +
   multimodal; this repo runs the QAT `g4_150` (2.1 GB, text). Not an identical weight
   format, so decode-bandwidth is not perfectly matched.

## Bottom line

- **Decode** (short or matched 1024 context): this repo's native-Metal backends beat
  LiteRT's GPU backend (~1.3–1.8×), because LiteRT decodes through WebGPU/Dawn.
- **Prefill / time-to-first-token:** LiteRT wins big via batched prefill — the one
  thing these decode-focused runners don't do yet.
