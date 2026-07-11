# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`py_shader_lang_wgpu` is a Python-to-WGSL (WebGPU Shading Language) translator. The goal is to let developers write GPU compute shaders for ML models in Python syntax, then emit WGSL text — similar to Triton or numba-cuda but targeting WebGPU. The translator itself is pure Python; it does not execute shaders.

## Setup

```bash
source venv/bin/activate
```

Dependencies already installed in `venv`: `wgpu`, `numpy`, `rendercanvas`.

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
