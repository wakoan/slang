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
python examples/matmul_wgpu.py
```

Use the translator package directly:

```python
from py_shader_lang_wgpu import translate
wgsl = translate(my_func)
```

## Architecture

- **`py_shader_lang_wgpu/`** — the DSL: `translator.py` (AST → WGSL, plus backend hooks), `msl.py` (AST → MSL), `types.py` (annotation types). `translate(func)` / `@kernel` are the entry points.
- **`examples/matmul_wgpu.py`** — Standalone hand-written WGSL compute shader for MxK × KxN matrix multiplication, run via `wgpu`. Serves as a reference for correct WGSL output and the wgpu API pattern (buffer layout, bind groups, compute pipeline dispatch).
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

### Batched prefill (gemma4_150/prefill_gpu.py)

`PrefillGPU` runs the whole prompt **layers-outer / tokens-inner**, so each weight is read once per *prompt* instead of once per token and every matmul becomes a GEMM at M=S. Weights are dequantized to f16 scratch once per layer (`prefill_research/dq_f16.metal`, one kernel for all three widths) and multiplied with **MPS** (`MPSMatrixMultiplication`, ~5.6 TFLOP/s f16 at M=1024 vs the per-token GEMV's 1.6). `metal_runner.G4.prefill` delegates here for prompts ≥16 tokens; `G4_BATCHED_PREFILL=0` forces the per-token path (which is how the tests get a reference — comparing the path against itself once produced a triumphant 0.0 divergence).

- **1208 tok/s at 1024 tokens** (per-token path: 158; Google's LiteRT-LM: 1132). Decode is unaffected at ~166.
- **Attention is two GEMMs too** (`qprep_b` → QK^T → `smax_b` → PV → `attnout_b`) and needs no per-head batching: kvHeads==1, so q's `[S][h][d]` layout is already `[S*h][d]` and every row multiplies the same K/V cache. Worth 276 → ~76 ms. The mask is never materialized — it is the key-loop bound, as in the fused `attn_prefill`. f16 scores are fine (99.3% of the output is bit-identical to the fused kernel, 0.02% moves more than one SRQ grid step); **f32 scores are not the fix they look like** — they cost ~40 ms.
- Do NOT hardcode a bit width: layers 0-14 pack gate/up/down at **4** bits, layers 15-34 at **2**. `_dq` derives it from the buffer length because getting this wrong is silent (it cost a debugging session).
- Only `down` passes `alpha != 1.0` to `_gemm` — its activations are integer codes; everything else takes real-valued activations.
- Batch sizes are bucketed (`BUCKET=16`) because MPS builds a pipeline per unique `(M,N,K,alpha,transpose)`; padding repeats the last token and lands at positions no real query can attend to.
- Measurement rules that this work paid for twice: warm the GPU clock ~2 s first (2.8× swing cold), time a per-token baseline in ONE command buffer (1.8× inflation otherwise), min-of-N. `ATTN_REPEAT` measures attention's true share by differencing; the per-kernel profiler overweights small dispatches.
- Next lever: band the score GEMMs by query block — the dense form computes and discards the causal upper triangle plus everything outside the sliding window (~2× wasted FLOPs).

### Portable batched prefill (gemma4_150/prefill_wgpu.py)

Same structure and the same DSL kernels, but every matmul is `kernels_dsl.gemm_tiled` instead of MPS, so it runs anywhere the DSL does. `G4Runner.prefill()` uses it for prompts ≥16 tokens; `G4_BATCHED_PREFILL=0` opts out on both Python runners.

- **145 tok/s at 1024 tokens vs 17.7 per-token (8.2×)**, against 1245 for banded MPS and 564 for this same kernel driven from Metal — the portable GEMM is **4× slower under wgpu-native than under Metal** (2.61 → 0.63 TFLOP/s, M=1024/N=12288/K=1536). f16 workgroup tiles are FALSIFIED as the cause (an f32-tile variant measures identically); naga's bounds-check clamps on the 64 workgroup reads per k-tile are the untested suspect. Re-measure in the browser before calling it portable-GEMM cost — Dawn already beats wgpu-native ~20% on decode with the same kernels.
- **Not bit-identical to per-token, and must not be gated as if it were.** Dequant+f16 GEMM replaces the exact int-dot. Measured drift vs each backend's own per-token path (relative L2 on the KV caches, n=16→256): wgpu 0.155→0.237, Metal 0.159→0.236 — equivalent, so the Metal envelope is the bar (`tests/test_prefill_wgpu.py`). Use **relative L2, not max-abs-relative**: max-abs is dominated by elements sitting on the SRQ rounding grid and reads 0.87 on a path that generates character-identical text.
- Two silent bugs this cost: (1) `_dq` needs a BYTE OFFSET into the concatenated `qkv_scales` (k at `q_dim`, v at `q_dim+head_dim`) — without it k/v dequantize with q's scales, reading as 1.1 relative; (2) `setup_caches()` must clear `_bgcache`, whose per-layer entries embed `kc`/`vc` — otherwise the second call onward leaves the forward path writing into the previous generation's caches (~1e10, and it would bite a chat REPL resetting context).
- Batched prefill must **drain before returning** (as the Metal path's `commit_wait()` does): it is one submit of ~500 dispatches, so without it the caller starts its decode timer with seconds of prefill still queued and decode "measures" 12 tok/s instead of 87.

## Backends

The translator emits **WGSL** (default) and **MSL** (`translate(fn, target="msl")`); `@kernel` attaches both as `fn.wgsl` / `fn.msl`. The MSL emitter (`py_shader_lang_wgpu/msl.py`) maps buffers to `device T*`, builtins to Metal thread attributes, `barrier()` to `threadgroup_barrier`, subgroup ops to `simd_*`, and mangles MSL-reserved identifiers (`half` → `half_`). Gemma runs on both: `python -m gemma3.generate "..." --backend metal` uses the `metalgpu` package (`gemma3/runner_metal.py`) — note metalgpu only supports 1-D dispatch with threadgroup = min(n, 1024), so Metal kernels (`kernels_metal.py`) reduce at simdgroup scope only; per-step params are plain numpy writes into shared-memory `buffer.contents`. metalgpu buffers can't be offset-bound (no QKV concat trick) and its `Buffer.__del__` double-releases — call `.close()` (see runner_metal).

## Tensor debugger (tensorscope)

`/tensorscope` on the same server: step-by-step single-token inference in the browser with capture of all intermediate tensors (13 per layer + embed/final/logits, ~2.3MB/step). The debug forward pass interleaves `copyBufferToBuffer` snapshots into a capture arena between compute passes (copies can't happen inside a pass — the pass is split around each capture). Canvas heatmap: symmetric blue/white/red scale, non-finite values magenta, hover for values, per-head-normalized attention maps (the fused kernel leaves scores as unnormalized exp). Standalone JS (`tensorscope/tensorscope.js`) — duplicates gendemo's setup deliberately; refactor into a shared module if a third page appears.

## Browser inference (gendemo)

`python -m gemma3.gendemo_server` (port 8000) serves a WebGPU page that runs the full model in the browser: `gendemo/app.js` mirrors runner.py (CPU-param prefill + GPU-resident chunked decode), `/kernels.json` is generated live from the DSL, and `gemma3/export_gendemo.py` packs weights into `models/.../gendemo/weights.bin` + manifest (GPU layout, f16-packed u32). Browser kernels are the portable set only — packed-u32 via core `unpack2x16float`, barrier-tree reductions, no `shader-f16`/subgroups features. Tokenize/detokenize stay server-side. `tests/test_gendemo_server.py::TestBrowserArtifactsEndToEnd` drives the exported artifacts through wgpu-py with app.js's exact dispatch sequence.

**Gemma 4 E2B QAT in the browser** (`python -m gemma4.qat_gendemo_server`, port 8000): the full 4.6B-param QAT model, every shader from the DSL. `gemma4/export_qat_gendemo.py` → 2.04GB `weights.bin` + manifest (per-linear kind/n_out/n_in + per-layer spec incl. KV-share/dual-head-dim); `gendemo4/app.js` mirrors `qat_runner.py` (PLE ctx/gather/combine, per-layer attention with KV-share aliasing, output-blocked dq matmuls, resident chunked decode, on-GPU argmax feedback). All 29 QAT kernels are portable (integer/packed-u32, no f16/subgroup features — verified). Needs Chrome with a ≥1.17GB max buffer (the PLE table). **Runs at 67 tok/s — ~20% FASTER than the wgpu-py runner's 56** (same DSL kernels; Dawn has lower per-dispatch overhead than wgpu-native), so the browser is the fastest non-`subgroupMatrix` backend. Verified: export bytes == qat_loader byte-for-byte; app.js's 569-op dispatch plan == traced `qat_runner._encode`. NB Dawn is stricter than naga (browser flushes out emitter bugs naga tolerates — e.g. the shift/arith paren fix in `translator._binop`).

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
- The `examples/matmul_wgpu.py` WGSL struct layout (`size: vec2<u32>` followed by `data: array<f32>`) is the reference format for storage buffers.


## Links

- https://huggingface.co/spaces/webml-community/gemma-4-webgpu-kernels
- https://google-ai-edge.github.io/LiteRT-LM/web_demos/chat/index.html