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

## Exact per-layer decode dispatch chain (mapped from kernel headers)

Data flows SRQ-quantized between fused kernels (each emits its output already
srq'd for the next). Per decoder layer, ~6-7 dispatches:

1. **69_sg_sum** `DecodeRmsSrq`: weighted RMSNorm(x) → SRQ int8 (as f16) codes
   `a` + per-row `sum_a`. (input_layernorm; produces the qkv input)
2. **70_srq** qkv: one dispatch, reads presrq `a`+`sum_a`, weights `[q|k|v]_bits`,
   `scales=[qScale|kScale|vScale]`, `params=[q/k/v OutScale]` → out_q/k/v (f32).
3. **101_srq** flash decode attention: fuses q-RMSNorm+split-half RoPE, chunked
   (NCHUNK wgs), same-dispatch last-arriver atomic merge, emits SRQ'd attn out.
   (needs cosTbl/sinTbl, k/v caches as vec4<f32>, per-head; OUT_Q srq at merge.)
4. **73_sg_sum** o-proj + post-attn residual-add + pre-FFN norm: reads srq'd attn
   `a`, o `bits_buf`+`scale`, `w12=[w1|w2]`; updates `hidden`; emits gate/up presrq
   input `y2` (f16) + `sum2`. Atomic `pp` tail. params=[outScale, inScale2].
5. **74/16/30_sg_sum** gate/up geglu (presrq): reads `hidden`(f16 y2)+`sum_a`,
   `gate_bits`/`up_bits`+scales, `gelu_lut`; emits down input int8 code (f16).
   virtual-subgroup, N_ROWS=4. params=[gateOutScale, upOutScale, outQuantScale].
6. **75_srq** down + post-FFN residual-add: reads gate/up codes (f16), down
   `bits_buf`+`scale`, norm `nw`; updates `hidden`. Atomic `pp` last-arriver merge.

PLE block + embed (00/01_main gather) are separate. Weight layouts: row-major
`o*WPR+wd` (70/73/74/75), block-major `blk*N+col` (33 dense/logits). Codes are
LSB-first; ZP=2^(bits-1). All decode kernels: `enable subgroups` (+`f16` where
f16 buffers) and the 32-lane subgroupShuffleXor fallback for wide-subgroup GPUs.

## Validated kernels (bit-exact vs numpy on REAL weights)

Every stage runs the reference kernel unmodified in headless Chrome (CDP) on the
localhost:8000 secure origin and compares to a numpy oracle. `prep_stageN.py`
builds the buffers + oracle; `validate_stageN.mjs` runs + diffs.

| Stage | Kernel(s) | What | Result |
|---|---|---|---|
| 1 | 69 + 70 | RMS+SRQ, fused qkv | bit-exact |
| 2 | 74 | gate/up geglu | 0 mismatch / 6144 |
| 3 | 75 | down + atomic norm-add | maxAbsDiff 0 |
| 4 | 73 | o-proj + post-attn norm-add + pre-FFN norm | hidden/y2/sum2 all 0 |
| 5 | 101 | flash-decode attention (q-norm+RoPE+softmax+V, atomic merge) | maxAbsDiff 0 (90 & 1500 keys) |
| 6 | 33 | dense logits GEMV (2-bit block-major) | maxAbsDiff 0 (4096 real lm_head cols) |
| 7 | 00 | embed gather (2-bit) | maxAbsDiff 0 |
| 8 | 77 | PLE projection 256→1536 (int8) + norm-add + norm | hidden/y2/sum2 all 0 |
| 9 | 76 | PLE input gate (int8) + GELU-LUT × PLE | maxAbsDiff 0 |
| 10 | 01 | 4-bit PLE gather (8960, 35 groups) | maxAbsDiff 0 |

**All novel kernel patterns are proven** — SRQ int8 activations, virtual-subgroup
fused GEMVs, block-major dense GEMV, the two-chained-RMSNorm atomic last-arriver
tail (73/75/77), flash-decode attention with same-dispatch atomic merge (101),
int8 +128-biased unpack4x8unorm decode (76/77), and the packing/repacking
(checkpoint u8 rows == kernel u32 nibble layout; block-major = [.,BLK,4] then
transpose). Remaining kernels (68 dense proj hidden→8960, small norm/combine
ops) are low-risk variants of the above.

## Status

Foundation + full kernel de-risk complete. The recipe is exact, the data loads,
and 10 stages validate the reference kernels bit-exact on real weights. What
remains is the **runner integration** (app.js orchestration: weight export in
the kernels' layouts, KV cache, PLE input pipeline, the fused per-layer dispatch
chain, GPU-resident chunked decode, tokenization) plus the headless tok/s
harness — an execution build with no open unknowns.
