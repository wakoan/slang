# Prefill research — matching LiteRT's 1132 tok/s

Measured investigation (M4 Pro, 2026-07-25) into closing the prefill gap. Decode is
already ~1.3–1.8× faster than LiteRT; prefill was ~7× slower. These are prototypes
and measurements, **not yet wired into the runners**.

## The corrected roofline (this fixes an earlier wrong analysis)

I first computed LiteRT's prefill as **10.4 TFLOP/s** by assuming 2×4.6B FLOP/token.
That was wrong: Gemma 4 **E2B = "Effective 2B"** — the per-layer embeddings are
*lookups*, not matmuls, so per-token compute is ≈2×2B.

| | FLOP/token basis | implied TFLOP/s |
|---|---|---:|
| LiteRT prefill, 1132 tok/s | 2×2.0B | **4.53** |
| Ours, 158 tok/s | 2×2.0B | **0.63** |

**Measured GEMM ceilings on this machine** (12288×1536 f16, the gate/up shape):

| Implementation | M=64 | M=256 | M=1024 |
|---|---:|---:|---:|
| Apple **MPS** (hand-tuned) | 2.08 | 3.50 | **4.90** |
| My `simdgroup_matrix` kernel | **2.59** | — | — |

**So LiteRT runs at ~92% of the machine's practical GEMM ceiling (4.53 of 4.90).**
That settles two things: the 1132 target *is* achievable here, and the only route to
it is a real GEMM at large M — nothing exotic.

## simdgroup_matrix results (pure f16, no dequant)

Falsified for *decode* previously (a GEMV wastes 7 of 8 tile columns). In the GEMM
regime it works and is numerically correct (rel err ~0.014 vs f32 numpy).

| Variant | TFLOP/s | Note |
|---|---:|---|
| naive 8×8 tile/simdgroup | 1.11 | re-reads B per M-tile → memory-bound |
| + M-tiled (B read once) — `gemm_sgmat.metal` w/ SG=1 | 1.69 | now load-issue-bound |
| + **16 simdgroups/threadgroup** — `gemm_sgmat.metal` | **2.59** | occupancy was the lever |
| 2-D register tiling (MT×NT) — `gemm_regtile_FALSIFIED` | 0.28–0.64 | **worse**: grid collapses to 192–768 wgs of 32 threads; 64 accumulators spill |
| A staged in threadgroup memory — `gemm_tgstage_FALSIFIED` | 1.48 | **worse**: barriers cost more than the saved loads; A (196 KB) was already L2-cached |

Notably my kernel **beats MPS at M=64** (2.59 vs 2.08) but MPS wins decisively at
large M — and large M is exactly the prefill regime.

## The route to 1132 tok/s

Weights are 2-bit; MPS needs f16. The design that follows from the measurements:

1. **Dequantize each weight matrix to f16 scratch once per prefill**, not per token.
   Cost is trivially amortized: ~100 MB of f16 per layer ≈ 0.5 ms, ×35 layers ≈
   17 ms, against ~900 ms of prefill compute for S=1024.
2. **MPS (or an MPS-class GEMM) for all S tokens at once**, at M≥256 where it reaches
   3.5–4.9 TFLOP/s.
3. Batched norms/activations (`rmssrq_69`, `embed_00`, `plegather_01` already carry a
   rows/seq dimension) + causal-mask attention over the query batch.

Estimated landing zone: **~1000–1200 tok/s prefill**, i.e. parity with LiteRT.

## RETRACTION: the earlier batching numbers were measurement artifacts

An earlier revision of this file reported **2.49×** for `gateup_95_b` and **1.00×**
for `down_96_b`, and built a "~350–400 tok/s cheap path" on them. **Both numbers were
wrong**, from two independent methodology bugs. Re-measure with
`python -m gemma4_150.prefill_research.bench_batched [S]`, which now controls for both.

1. **Unfair baseline.** The S per-token decode dispatches were timed as S separate
   command buffers (commit + wait each) instead of S dispatches in one command
   buffer. Inside a real forward pass they share a command buffer and overlap.
   Measured inflation of the baseline: **1.80×**, i.e. pure fiction.
2. **Cold GPU clock.** Apple silicon idles the GPU at a low clock and needs
   order-seconds of sustained load to ramp. The *same* `S × down_96` baseline reads
   **1.013 ms cold and 0.366 ms warm — a 2.8× swing**, landing entirely on whichever
   kernel runs first. Every number below is taken after a 2 s warm-up, min-of-3, with
   the baseline re-measured after each sweep to confirm drift < 1.5%.

The "shape rule" that revision inferred (`n_in` small → batching wins) was also
wrong — an artifact of the same bad data. The real story is below.

## What actually governs these kernels: MAC per issued load

These are split-k GEMVs and they are **load-issue bound**, not byte bound. A thread
holding a micro-tile of `N_ROWS` output rows × `SEQ` tokens issues `N_ROWS` weight
loads + `4·SEQ` activation loads per weight-word to do `16·N_ROWS·SEQ` MACs:

    MAC / issued-load  =  16·N_ROWS·SEQ / (N_ROWS + 4·SEQ)

This explains the original failure exactly. `down_96` (decode) already blocks **4
output rows**; the first `down_96_b` dropped that to **1 row** while adding 8 tokens,
taking the ratio *down* from 8.0 to 3.9. It traded one amortization for another
instead of stacking them. **Both dimensions have to be blocked at once.**

Warm, fair measurements of the fixed kernels (bit-exact — see the tie note below):

| Kernel | blocking | S=8 | S=16 |
|---|---|---:|---:|
| `kernels_msl/gateup_95_b.metal` | `N_ROWS=2, SEQ=4` | **1.21×** | **1.23×** |
| `kernels_msl/down_96_b.metal` | `N_ROWS=16, SEQ=4` | **1.45×** | **1.80×** |

The ratio is necessary but **not sufficient** — it ignores register pressure. For
gate/up, `N_ROWS=8,SEQ=4` has the best ratio (32.0) and the *worst* time (0.43×):
`n_in=1536` is only 96 words = 3 per lane, too short a k-loop to amortize the
registers. `down`'s 768-word k-loop does pay for `N_ROWS=16`. `SEQ=8` spills
everywhere (0.27–0.41×). Sweep, don't extrapolate.

## Verdict: batching is NOT a route to parity

Exact FLOP shares (from the manifest): gate/up **55.3%**, down **27.7%**, qkv 7.8%,
o_proj 7.0%, PLE 1.5%, proj_68 0.7% — 1.88 G MAC/token.

| Scenario | overall | prefill |
|---|---:|---:|
| gate/up + down batched (measured) | 1.26× | ~199 tok/s |
| *plus* every other matmul also hitting 1.3× (optimistic) | 1.32× | ~209 tok/s |
| LiteRT | | **1132 tok/s** |

**The whole batching path is worth ~1.3×, and still leaves a 5.4× gap.** It is capped
by gate/up, which is 55% of the FLOPs and only yields 1.21×. This route is closed;
the dequant-to-f16 + GEMM route above is the only one whose arithmetic reaches 1132.

## Note on "bit-exact"

The batched kernels match the decode kernels except for occasional single-element
mismatches (1 in 24576 at S=16). These are **exact ties on the SRQ rounding grid**:
float64 reconstruction of one such element gives `50.500007`, where a last-ulp
difference in float accumulation order flips `round()` to the other side. The
underlying dot products agree exactly (verified in exact integer arithmetic).
