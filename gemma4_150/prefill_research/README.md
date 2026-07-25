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

## Cheaper alternative already proven

`../kernels_msl/gateup_95_b.metal` — simple token-batched GEMV (S accumulators in
registers), **bit-exact**, measured **2.49× at S=16** (regresses at S=32: register
spill). Rolling that across the ~7 matmuls lands prefill near **350–400 tok/s** for
far less work, without any dequant-staging redesign.
