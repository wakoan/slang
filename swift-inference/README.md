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

## Status

- [x] Metal runner: per-file compile, batched dispatch, offset binding
- [x] All 32 DSL kernels compile as MSL at runtime
- [x] Weights: mmap → bf16→f16 straight into MTLBuffers (~0.5 GB, warm ~0.07s)
- [x] Full decode, wgpu-parity kernel sequence (`_wg`/`_sg` set); greedy output
      matches the Python runner token-for-token
- [x] GPU-resident decode (step_setup + argmax on GPU, chunked EOS checks)

Usage: `gemma --ids 2,105,... [--max-tokens N] [--chunk N] [--step-mode] [--ignore-eos]`
(token ids from the Python tokenizer for now).

## Performance (M4 Pro, 256-token greedy decode, same prompt)

| backend | tok/s |
|---|---|
| Swift/Metal resident, chunk 64 | **202–210** |
| Swift/Metal CPU-driven steps | 172 |
| wgpu GPU-resident (same day, same prompt) | ~149 |
| metalgpu (simd-only kernels) | ~120 |

Wins over wgpu: coalesced `matvec_wg_packed_sg` for the logits matvec
(the per-thread `matvec_packed` costs ~7%), chunk 64, no Python overhead.
f16 bandwidth floor ≈ 370 tok/s — next: per-kernel GPU profiling
(MTLCounterSampleBuffer), attention scaling at long kv_len.

## Next steps

- [ ] Simple tokenizer from tokenizer.json (currently ids in/out via CLI)
- [ ] Per-kernel GPU timing to find the gap to the bandwidth floor
- [ ] Sampling (temperature) path
