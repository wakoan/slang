# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`py_shader_lang_wgpu` is a Python-to-WGSL (WebGPU Shading Language) translator. The goal is to let developers write GPU compute shaders for ML models in Python syntax, then emit WGSL text — similar to Triton or numba-cuda but targeting WebGPU. The translator itself is pure Python; it does not execute shaders.

## Setup

```bash
source venv/bin/activate
python -m gemma3.download   # fetches model weights/tokenizer (~570MB, idempotent)
```

Dependencies already installed in `venv`: `wgpu`, `numpy`, `tokenizers`, `metalgpu`, `pytest`. Weights come from the ungated `unsloth/gemma-3-270m-it` mirror (the official Google repo is license-gated).

## Testing

```bash
python -m pytest tests/                            # full suite
python -m pytest tests/test_translator.py -k for   # single file / keyword filter
```

## Running

Run the matrix multiplication WebGPU example (requires a GPU-capable environment):

```bash
python matmul_wgpu.py
```

Use the translator package directly:

```python
from py_shader_lang_wgpu import translate
wgsl = translate(my_func)
```

## Architecture

- **`py_shader_lang_wgpu/`** — the DSL: `translator.py` (AST → WGSL, plus backend hooks), `msl.py` (AST → MSL), `types.py` (annotation types). `translate(func)` / `@kernel` are the entry points.
- **`matmul_wgpu.py`** — Standalone hand-written WGSL compute shader for MxK × KxN matrix multiplication, run via `wgpu`. Serves as a reference for correct WGSL output and the wgpu API pattern (buffer layout, bind groups, compute pipeline dispatch).
- **`design.md`** — Goals and open design questions. Key constraints: output is WGSL text only (no execution layer), and the architecture should be extendable to Metal/OpenCL backends.
- **`gemma3/`** — Gemma 3 270M LLM inference with every GPU shader written in the DSL (`kernels.py`), plus a torch-free bf16 safetensors loader, a numpy reference decoder for verification, a wgpu runner with KV cache, and a generation CLI: `python -m gemma3.generate "prompt"` (flags: `--max-tokens`, `--temperature`, `--profile`). Weights live in `models/gemma-3-270m-it/` (not in git). GQA kernels assume `num_key_value_heads == 1` (true for Gemma 3 270M and Gemma 4 E2B).
- **`gemma4/`** — Gemma 4 E2B (4.6B text params, 9.8GB bf16 checkpoint in `models/gemma-4-E2B/`) on wgpu: lazy `SafetensorsIndex` loader (never materializes full-model host copies; the 4.7GB PLE table stays mmap'd with per-token row gathers), streaming numpy reference (`@pytest.mark.slow`, ~seconds/token — test oracle only), and `Gemma4GPU` runner reusing gemma3 kernels plus E2B-specific ones (p-RoPE, no-scale attention, scale-free v_norm, softcap, PLE ops). CLI: `python -m gemma4.generate "prompt"` (plain completion — the checkpoint has no chat template). Key model facts vs Gemma 3: Gemma4RMSNorm scales by w directly, NOT (1+w) — the runner uploads (w-1) to reuse gemma3 norm kernels; KV sharing (layers 15-34 bind layer 13/14's caches); dual head dims 256/512; double-wide MLP on layers 15-34. Greedy decode is GPU-resident (step_setup_g4 + ple_gather_f16 from a two-half-buffer f16 PLE table + on-GPU argmax, +4.7GB GPU); sampling/profile/f32 use the CPU-driven per-step path. wgpu pitfall: GB-scale `create_buffer_with_data` uploads silently zero out unless flushed with `queue.submit([])` every ~256MB (see `upload()` in `gemma4/runner.py`).

## Performance notes (gemma4 runner, M4 Pro)

- 38 tok/s greedy f16 (resident), GPU-bound and near the f16 bandwidth floor: matvecs are ~75% of decode time at ~235 GB/s (mv_gateup), ~86% of the ~273 GB/s peak; ceiling ≈ 58 tok/s. History: 17 f32 → 22.6 f16 → 32.6 GPU argmax (killed the 23ms/step logits readback) → 33.0 resident → 34.0 fused post-attn add-norm+pre-FFN norm → 38.0 vec4 matvec.
- The matvec is `matvec_wg_packed_v4` (workgroup-per-row, 64 threads, vec4<u32>+2×vec4<f32> loads = 8 f16/iter). All E2B matvec n_in are divisible by 8. Norm fusions live in `rmsnorm_add_norm_wg` (post-attn add-norm + pre-FFN norm) and `rmsnorm_add_scale_wg` (PLE post-norm + scaled residual add).
- Falsified for E2B (measure before re-trying): (1) subgroup `_sg` kernels — net loss on the wide rows (31.5 vs 33.0 tok/s); `use_subgroups` hard-off. (2) 128-thread matvec vs 64 — slower (36 vs 38 tok/s: extra reduction level costs more than any bandwidth gain; 64 threads already saturate). (3) norm+trivial-elementwise fusion (PLE add_scale) — speed-neutral (real work is the reduction, not the dispatch).
- Remaining leads (diminishing without quantization): matvecs at 86% of peak leave ~14% there; norm_input could fold into the previous layer's PLE tail (~1 tok/s, cross-layer coupling). The real lever below the f16 floor is weight quantization — the reference WebGPU bundle (webml-community/gemma-4-webgpu-kernels, Xenova) runs the QAT-mobile checkpoint at 4-bit attn / 2-bit MLP, cutting weight bandwidth ~4-8× (a separate checkpoint + quantized-matmul project).

### QAT quantized runner (gemma4/qat_*.py)

`python -m gemma4.qat_generate "prompt"` runs the full model from the **QAT-mobile checkpoint** `google/gemma-4-E2B-it-qat-mobile-transformers` (ungated, 2.46GB int2/4/8, in `models/gemma-4-E2B-qat/`). `qat_loader.py` dequantizes packed int2/4/8 weights (per-row f32 scale, low-bits-first, -8/-2 offset — matches transformers `integrations/gemma_quant.py`; format also in the `gemma4-qat-quant-format` memory). `qat_kernels.py` has the dequant matmuls `matvec_dq4`/`matvec_dq2` (read packed u32 → int dot → one scale multiply; the symmetric per-row scale factors out) plus `qat_embed_2bit` (2-bit embed gather, tied 2-bit logits) and `qat_ple_gather_4bit` (4-bit PLE table, ~1.17GB single buffer). 8-bit PLE gate/proj and the unquantized `per_layer_model_projection` dequant to f16 and use the base vec4 matvec. `Gemma4QATGPU` (`qat_runner.py`) forks the f16 runner (resident decode). `qat_reference.py` is the weight-only numpy oracle; `Gemma4Config` builds from the QAT header unchanged (packing preserves `n_out`).

- This is the **instruction-tuned** checkpoint: wrap prompts in its chat format `<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n` (turn tokens 105/106) — plain completion loops. Weight-only inference (SRQ activation scales skipped) has excellent quality; verified GPU==numpy-reference (argmax exact) and coherent answers.
- Perf: **56 tok/s resident** (up from 32). Ceiling is **~334 tok/s** (model reads only **0.82GB/token**; the 1.17GB PLE table is row-gathered, not read whole). At 56 tok/s effective bandwidth is ~46 GB/s (17% of the 273 GB/s peak), so headroom remains.
- **THE lever (measured, decisive): output-blocking the dequant matvecs** — one workgroup computes **N adjacent output rows**, reading each shared-input element *once* for all N weight rows (amortizes the input read, which is the real decode bottleneck). This is what actually moved end-to-end: gate+up+geglu fusion 36→43 (its win was output-blocking 2 rows × gate+up, NOT the geglu fusion), then blocking `down` (`matvec_dq2_blk2`) 43→45, tied logits (`matvec_dq2_blk8/16`, one 2-D-free 32768-wg dispatch replacing 8 chunks) 45→50, blocked fused gate+up (`mv_gateup_geglu_dq2_blk8`) 50→56, blocked int4 q/o/k/v (`matvec_dq4_blk2`). Block-factor tuning: **blk8 for wide rows** (n_out≥12288, e.g. gate/up), **blk2 for narrow rows** (n_out≈1536, e.g. down/o — blk4 there *lost* to occupancy), **blk16 for logits** (n_out=262144). Hard cap: **blk-N uses N (or 2N for gate+up) separate `WorkgroupArray`s and Metal allows only ~31 threadgroup buffers** — blk32 logits fails to compile; blk16 is the ceiling.
- **FALSIFIED end-to-end in the resident path (all ~0, do NOT retry; the earlier "dispatch/latency-bound, ~8.5µs×680" diagnosis was WRONG — it described the per-*step* profile path, not resident decode):** qk-norm+rope fusion, 128-thread matvec, 4 independent FMA accumulators, collapsing the 8 logits dispatches to 1 *without* blocking, blk4 on narrow (n_out≈1536) rows. The resident path is neither dispatch-bound nor FMA-latency-bound; it is input-read-issue-bound, which *only* output-blocking addresses. Also still falsified from before: subgroup matmul, shared-x tiling, `dot4I8Packed`. The `--profile` per-kernel timings OVERWEIGHT tiny single-workgroup dispatches (norms show ~28% there but fusing them is neutral) — trust only end-to-end tok/s.
- **~56 tok/s is the ceiling for wgpu-py, and the reason is now proven, not guessed.** The reference bundle's 150 tok/s comes from **WebGPU cooperative/subgroup-matrix ops** (`subgroupMatrixMultiplyAccumulate` / `subgroupMatrixLoad/Store`, i.e. Apple `simdgroup_matrix` hardware matrix units) behind the **`chromium-experimental-subgroup-matrix`** feature. Verified by downloading its `gemma-4-e2b.js`: 170+ `subgroupMatrix` refs, `enable chromium_experimental_subgroup_matrix;`, and it ships two GEMV variants — `DenseGemv` (plain reduction, ≈ our path) and `DenseGemvSgmat` (matrix-unit path). Its `DecodeGateUpNorm` is otherwise like ours (f16 activations, packed weights, fused gelu via LUT, no int8 dot — `dot4I8Packed` count = 0). **Our runtime (wgpu-native) exposes only `subgroup`/`subgroup-barrier`, NOT `chromium-experimental-subgroup-matrix`** (checked `adapter.features`, 27 of them, none matrix). So matrix-unit throughput is unreachable from wgpu-py; ~56 is essentially the bundle's own non-matrix `DenseGemv` fallback tier. The only real path to the matrix units on this Mac is **outside wgpu** — a Metal backend emitting `simdgroup_matrix` (needs DSL matrix types + a Metal runtime), or running in-browser where the Chromium feature exists.
- Attention costs ~2.7ms/token (15%, measured by a 3×-dispatch diff) but is NOT improvable by more parallelism: **flash-decoding** (split-KV `attn_flash_partial` over nh·S workgroups + online-softmax `attn_flash_combine`) was built and verified argmax-exact but is **neutral-to-negative at every split** (S=4/8/16 → 52.9/54.9/55.7 vs 56.3) — short decode context (~90 keys) thread-starves each split; attn's cost is inherent barrier/dispatch latency across 35 sequential layers. Also falsified: **256-thread single-workgroup rmsnorm** (norms look big in `--profile` but are cheap in resident). QAT wins on **memory** (2.46GB vs 9.8GB) regardless.

## Backends

The translator emits **WGSL** (default) and **MSL** (`translate(fn, target="msl")`); `@kernel` attaches both as `fn.wgsl` / `fn.msl`. The MSL emitter (`py_shader_lang_wgpu/msl.py`) maps buffers to `device T*`, builtins to Metal thread attributes, `barrier()` to `threadgroup_barrier`, subgroup ops to `simd_*`, and mangles MSL-reserved identifiers (`half` → `half_`). Gemma runs on both: `python -m gemma3.generate "..." --backend metal` uses the `metalgpu` package (`gemma3/runner_metal.py`) — note metalgpu only supports 1-D dispatch with threadgroup = min(n, 1024), so Metal kernels (`kernels_metal.py`) reduce at simdgroup scope only; per-step params are plain numpy writes into shared-memory `buffer.contents`. metalgpu buffers can't be offset-bound (no QKV concat trick) and its `Buffer.__del__` double-releases — call `.close()` (see runner_metal).

## Tensor debugger (tensorscope)

`/tensorscope` on the same server: step-by-step single-token inference in the browser with capture of all intermediate tensors (13 per layer + embed/final/logits, ~2.3MB/step). The debug forward pass interleaves `copyBufferToBuffer` snapshots into a capture arena between compute passes (copies can't happen inside a pass — the pass is split around each capture). Canvas heatmap: symmetric blue/white/red scale, non-finite values magenta, hover for values, per-head-normalized attention maps (the fused kernel leaves scores as unnormalized exp). Standalone JS (`tensorscope/tensorscope.js`) — duplicates gendemo's setup deliberately; refactor into a shared module if a third page appears.

## Browser inference (gendemo)

`python -m gemma3.gendemo_server` (port 8000) serves a WebGPU page that runs the full model in the browser: `gendemo/app.js` mirrors runner.py (CPU-param prefill + GPU-resident chunked decode), `/kernels.json` is generated live from the DSL, and `gemma3/export_gendemo.py` packs weights into `models/.../gendemo/weights.bin` + manifest (GPU layout, f16-packed u32). Browser kernels are the portable set only — packed-u32 via core `unpack2x16float`, barrier-tree reductions, no `shader-f16`/subgroups features. Tokenize/detokenize stay server-side. `tests/test_gendemo_server.py::TestBrowserArtifactsEndToEnd` drives the exported artifacts through wgpu-py with app.js's exact dispatch sequence.

## DSL features beyond the basics

- Helper functions: any plain annotated function visible from a kernel (module global or enclosing scope) is auto-resolved and its definition emitted transitively, callee-first, on both backends — no decorator needed. `@device_fn` remains as an optional eager-validation marker. Resolution is lexical (per kernel's namespace), builtins take precedence, and closure *values* are NOT captured (only functions). Unknown call names raise `TranslationError` at translate time — extend `_KNOWN_BUILTINS` if a legitimate WGSL builtin is missing
- `WorkgroupArray[f32, N]` params → `var<workgroup>` shared memory; `barrier()` → `workgroupBarrier()`
- `f16` storage buffers emit `enable f16;` (needs the `shader-f16` device feature)
- `subgroupAdd`/`subgroupMax` calls and `Builtin.subgroup_*` emit `enable subgroups;` — the runner strips this directive if naga rejects it (current naga supports the ops but not the directive; feature is named `subgroup`, singular, in wgpu-py)
- WGSL reserved words (e.g. `shared`) are NOT caught by the translator — shader compilation fails; pick different Python names

## Performance notes (gemma3 runner, M4 Pro)

- Greedy decode runs GPU-resident: `step_setup` computes per-step params on-GPU, `argmax_stage1/2` feed the next token back without CPU round trips; CPU checks EOS once per 16-token chunk. Sampling (`temperature>0`) and `profile=True` use the per-step path.
- Weights are f16 (packed u32 loads via `unpack2x16float`); norm weights and activations stay f32. dtype="f32" available for exact verification vs the numpy reference.
- Profiler theories falsified so far: dispatch-count reduction alone (no effect), plain f16 loads (scalar f16 halves bandwidth). Verify with `--profile` before optimizing.

## Key Design Constraints

- The translator targets WGSL first but must be backend-agnostic in structure (Metal, OpenCL planned).
- `translate()` must be usable as a decorator or called directly on a function object.
- The `matmul_wgpu.py` WGSL struct layout (`size: vec2<u32>` followed by `data: array<f32>`) is the reference format for storage buffers.
