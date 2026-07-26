# Archived: the token-batching and simdgroup-matrix prefill routes

Kept as the evidence behind two retractions, not as working code. Nothing in the
shipping path imports any of it, and the paths inside these files are stale.

## Token batching (`bench_batched.py`, `down_96_b.metal`, `gateup_95_b.metal`)

Register-blocked GEMVs that process S tokens per pass. **Measured, falsified,
closed.** Warm and fair, gate/up reaches only 1.21x and down 1.45–1.80x; since
gate/up is 55.3% of prefill FLOPs the whole route caps at ~1.3x overall, roughly
200–210 tok/s against LiteRT's 1132. Root cause: a GEMV holds S accumulators in
registers and spills at SEQ=8, so it dies long before the M>=256 that a real
GEMM needs. Superseded by dequantize-to-f16 + GEMM, which shipped at 1208 tok/s.

`bench_batched.py` also carries the `--unfair` flag that reproduces the original
bad measurement, deliberately, so the retraction stays checkable. An earlier
revision of this work reported 2.49x from two compounding bugs: timing the
baseline as S separate command buffers (1.80x inflation) and a cold GPU clock
(2.8x swing). Those two rules — one command buffer, warm the clock — are why the
benchmark methodology section exists at all.

## simdgroup_matrix prototypes (`gemm_sgmat.metal`, the two `*_FALSIFIED`)

`gemm_sgmat` reached 2.59 TFLOP/s and beat MPS at M=64, but lost decisively at
the large M that prefill actually runs at. The two FALSIFIED variants are worse
still: 2-D register tiling collapses the grid and spills 64 accumulators;
staging A in threadgroup memory costs more in barriers than it saves in loads.
The portable GEMM that shipped is `gemma4_150/kernels_dsl.py::gemm_tiled`, which
lands at the same ~2.6 TFLOP/s from a single Python source.

Full numbers: `gemma4_150/prefill_research/README.md` and `litert/RESULTS.md`.
