# Gemma 3 270M Swift inference

Standalone Swift/Metal inference for Gemma 3 270M — no Python at runtime. The
goal is to find the machine's real limit: full control over threadgroup sizes,
offset buffer binding, and single-command-buffer dispatch, none of which the
metalgpu backend allowed.

## Layout

- `kernels/*.metal` — MSL generated from the DSL (one file per kernel;
  regenerate with the snippet below). Function name = file basename; the
  threadgroup width is parsed from the generator's dispatch comment.
- `Sources/gemma/MetalRunner.swift` — per-file kernel compilation (runtime
  MSL, no Xcode toolchain needed), shared-storage buffers, batched command
  encoding with per-buffer offsets.
- `Sources/gemma/GemmaInference.swift` — `GemmaConfig` (config.json).
- `Sources/gemma/SafetensorsLoader.swift` — safetensors → f32 (bf16/f16/f32).
- `Sources/gemma/Tokenizer.swift` — stub; simple tokenizer planned.
- `Sources/gemma/main.swift` — currently a smoke test: compiles all kernels,
  verifies `matvec_simd_packed` against a CPU reference.

## Build & run

```bash
cd swift-inference
swift build
cd .. && swift-inference/.build/debug/gemma   # run from repo root (finds kernels/ + models/)
```

## Regenerating kernels

```python
# from repo root, venv active
from gemma3.kernels import KERNELS
from gemma3.kernels_metal import METAL_KERNELS
for name, k in {**KERNELS, **METAL_KERNELS}.items():
    open(f"swift-inference/kernels/{name}.metal", "w").write(k.msl)
```

## Status / next steps

- [x] Metal runner: per-file compile, batched dispatch, offset binding
- [x] All 32 DSL kernels compile as MSL at runtime
- [x] matvec verified vs CPU reference
- [ ] Weight loading into f16-packed GPU buffers (current loader is slow copy-based)
- [ ] Full decode step (use the `_wg`/`_sg` kernels — controllable threadgroups
      mean we are NOT limited to the simd-only `kernels_metal.py` set)
- [ ] Simple tokenizer from tokenizer.json
- [ ] tok/s benchmark vs wgpu (175) and metalgpu (120); f16 bandwidth floor ≈ 370
