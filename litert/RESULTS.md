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

## Batched-prefill prototype (measured, 2026-07-25)

`kernels_msl/gateup_95_b.metal` is a **batched (M=S) variant of the dominant matmul**
(2-bit gate/up, 12288×1536), built to measure the real win before porting all ~7
matmuls. It reads each weight word **once** and dots it against S activation vectors.
Output is **bit-exact** vs the per-token path at every S.

| S (tokens/pass) | S× GEMV | batched | speedup |
|---:|---:|---:|---:|
| 4 | 1.01 ms | 0.84 ms | 1.21× |
| 8 | 1.81 ms | 1.36 ms | 1.33× |
| **16** | 2.40 ms | **0.96 ms** | **2.49×** |
| 32 | 2.77 ms | 3.32 ms | 0.83× ← regression |

**Findings that contradict the naive roofline:**

1. **The GEMV path was never weight-bandwidth-bound at this size** — it runs at only
   ~42 GB/s, 15% of the 273 GB/s peak. So "read weights S× less" does not buy S×;
   the kernel is issue/occupancy-bound, not byte-bound.
2. **S=32 regresses** (0.83×): 2×32 = 64 float accumulators per thread spills
   registers. S=16 is the sweet spot for this simple structure.
3. The baseline is also flattered — S independent GEMV dispatches in one command
   buffer overlap on the GPU, so they scale sublinearly (1.01→2.77 ms for 4→32).
   Real prefill runs them as *sequential dependent forwards*, so in-context batching
   should beat this microbenchmark's 2.5×.

**Extrapolation:** applying S=16 batching to all ~7 matmuls plausibly takes prefill
from 158 to roughly **350–400 tok/s** — a solid 2–2.5×, but still ~3× short of
LiteRT's 1132. Closing the rest needs a **properly tiled GEMM** (threadgroup-memory
A/B tiles, register micro-tiles) rather than this "S accumulators in registers"
shape — and that is the regime where Apple's `simdgroup_matrix` units finally pay
off (falsified for *decode*, where a GEMV wastes 7 of 8 tile columns; a token batch
fills them).

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
