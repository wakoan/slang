# slang — a Python→GPU-shader DSL, and LLM inference built on it

Two things live here:

1. **`py_shader_lang_wgpu/`** — the DSL. Write GPU compute shaders in a subset of
   Python, emit **WGSL** (WebGPU) and **MSL** (Metal) text. Like Triton/numba-cuda
   but targeting WebGPU/Metal. Pure translator — it does not execute shaders.
2. **LLM inference** that uses the DSL to run Gemma models on the GPU, across many
   backends — the proving ground and the interesting engineering.

## Repository map

```
py_shader_lang_wgpu/     the DSL: translator.py (→WGSL), msl.py (→MSL), types.py
examples/                matmul_wgpu.py (hand-written WGSL reference), DSL examples
docs/                    design.md and notes
reference/               vendored HF webml gemma-4 kernels (third-party, read-only)
tests/                   pytest suite

gemma3/                  Gemma 3 270M — DSL kernels, wgpu + Metal runners, browser demo
gemma3/... swift-gemma3/ Gemma 3 native Swift/Metal runner (was swift-inference)
gemma4/                  Gemma 4 E2B — bf16 + QAT wgpu runners
gemma4_150/              Gemma 4 E2B QAT, the "150 recipe" — the fast path (below)
gendemo/ gendemo4/       browser (WebGPU) demos for gemma3 / gemma4
tensorscope/             in-browser tensor debugger
models/                  weights (git-ignored)
```

## gemma4_150 — the fast Gemma 4 E2B QAT path

A clean-room port of the HF `webml-community/gemma-4-webgpu-kernels` recipe, with
**one shared set of kernels** driven by six backends. Everything reads the same
`models/gemma-4-E2B-qat/g4_150/{manifest.json,weights.bin}` and the same MSL/WGSL
shaders.

```
gemma4_150/
  loader.py reference.py export.py         core (weights, numpy oracle, exporter)
  runner.py generate.py server.py          Python runners (wgpu + browser server)
  metal_runner.py                          Python native-Metal runner (PyObjC)
  kernels_msl/                             shared MSL shaders (used by all 3 Metal backends)
  backends/
    rust_wgpu/                             Rust + wgpu
    rust_metal/                            Rust + native Metal
    swift_metal/                           Swift + native Metal (compiles kernels_msl/)
    browser/                               WebGPU page (app.js mirrors runner.py)
  archive/                                 one-time bring-up scaffolding (historical)
```

### Backends & measured speed (M4 Pro, greedy decode, warmed)

| Backend | Path to GPU | tok/s | How to run |
|---|---|---:|---|
| Swift + native Metal | Metal | **~170** | `cd gemma4_150/backends/swift_metal && swift run -c release g4 chat` |
| Rust + native Metal | Metal | **~170** | `cd gemma4_150/backends/rust_metal && cargo run --release -- chat` |
| **Python + native Metal** (PyObjC) | Metal | **~170** | `python -m gemma4_150.metal_runner chat` |
| Browser | Dawn → Metal | ~156 | `python -m gemma4_150.server` then open `localhost:8000` |
| Python + wgpu | wgpu-native → Metal | ~97 | `python -m gemma4_150.runner "prompt"` |
| Rust + wgpu | wgpu-native → Metal | ~90 | `cd gemma4_150/backends/rust_wgpu && cargo run --release -- "prompt"` |

The three native-Metal backends (Swift/Rust/**Python**) all hit the same ~170
tok/s ceiling — **the host language never mattered.** The ~2× gap to the wgpu
backends is wgpu's per-dispatch binding/validation overhead, not the language
(PyObjC records a token's ~570 dispatches in ~1.8 ms, 3× the GPU rate). The wall
above ~170 on this Mac is the `simdgroup_matrix` hardware units, reachable only
from a kernel emitting `simdgroupMatrix` ops (not through wgpu at all).

The native-Metal runners share a **ring-of-32 per-step uniform buffers** so their
chunked GPU-resident decode can commit many forwards without the CPU racing the
GPU on position params — the fix that keeps long outputs coherent.

## Setup

```bash
source venv/bin/activate
python -m gemma4.download            # weights (git-ignored)
pip install pyobjc-framework-Metal   # for the Python native-Metal runner
```

Rust backends need `~/.cargo/bin` on PATH; Swift backends need a recent Xcode
toolchain.

## Testing

```bash
python -m pytest tests/
```

See `CLAUDE.md` for deep architecture notes and performance history.
