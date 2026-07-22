# gemma4_150 — standalone port targeting the webml space's ~150 tok/s decode

Clean-room reimplementation of the webml-community/gemma-4-webgpu-kernels decode
path (no DSL, no dependency on the project's `gemma4/` runner). Goal: match the
space's **~150 tok/s** on this M4 Pro, using its actual kernels (captured in
`../reference/webml_gemma4_kernels/`) + the SRQ int8-activation pipeline.

## Why 150 (recap — proven this session)

The gap over our 90 tok/s DSL demo is NOT a hardware feature (subgroup-matrix is
prefill-only). It's three things, all now understood and de-risked:

1. **SRQ int8 activations** — activations quantized to int8, stored as f16, with
   per-linear `input_activation_scale` / `output_activation_scale` (scalars) that
   ARE in the checkpoint (we skipped them; `loader.py` now loads them). Halves
   activation bandwidth and enables the fast `unpack4x8unorm` code dot. The SRQ
   math is validated numerically vs the checkpoint (see `validate_srq` history).
2. **Virtual-subgroup fused GEMVs** — 32-lane subgroup = independent GEMV unit,
   N_ROWS output rows each, subgroupAdd (no barriers). Only wins WITH f16/presrq
   activations (our naked subgroupAdd lost).
3. **Fusion** — q/k/v = 1 dispatch (70_srq); o-proj + post-attn residual-norm-add
   + pre-FFN norm = 1 (73_sg_sum); down + post-FFN norm-add = 1 (75_srq); gate/up
   geglu emitting down's int8 code = 1 (74/16/30_sg_sum); flash decode attention
   with same-dispatch atomic merge (101_srq). ~16 dispatches/layer -> ~6.

## Done (foundation, de-risked)

- `loader.py` — reads the QAT checkpoint standalone: packed sub-byte weights,
  per-row `weight_scale`, scalar `input/output_activation_scale`, per-layer bit
  widths (attn 4b; MLP 4b L0-14 / 2b L15-34; PLE 8b). Verified against real L4.
- SRQ math validated vs numpy: `out = srq(wscale * Σ(code-ZP)*srq(a,inScale), outScale)`.
- All 150-tok/s kernels captured + recipe documented in `../reference/`.

## Remaining stages (each: build -> validate numerically -> headless bench)

1. **SRQ activation quantizer** (`DecodeRmsSrq`): rms-norm + srq to int8-as-f16
   codes + per-row `sum_a`. This feeds the presrq matmuls.
2. **Weight repack** to each kernel's layout: row-major `o*WPR+wd` (25/74/75),
   block-major `blk*N+col` (33 dense/logits). Pack u32 low-bits-first.
3. **Port the ~8 decode kernels** as raw WGSL with correct baked constants per
   shape (K/N/BITS/WPR/N_ROWS/GRID). Kernels are in `../reference/`; params
   structs are in each file's header. Feature-detect subgroups+f16.
4. **Numpy oracle** (`reference.py`): full decode WITH SRQ (matches the kernels
   bit-for-bit) — the correctness gate for every kernel + the whole pipeline.
5. **Runner/app.js**: orchestrate the fused dispatch sequence per layer (PLE,
   KV-share, dual head dims, flash-atomic attention), GPU-resident chunked decode.
   Reuse `../gendemo4/bench_headless.mjs` harness (autonomous tok/s).
6. **Export + server**: standalone weights.bin (their layouts) + scales + a server.

## Status

Foundation + full de-risk complete; the recipe is exact and the data loads. The
remaining stages are a substantial multi-session build (an engine port), not a
tweak — but every unknown is resolved, so it can be executed straight through
with the reference kernels + the autonomous headless harness validating tok/s.
