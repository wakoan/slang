# DSL kernel port — status

Porting `kernels_msl/*.metal` to a single Python source in `kernels_dsl.py`,
which emits both MSL and WGSL. Two gates, both required before a kernel is used:

1. **Parity** — `tests/test_kernels_dsl_parity.py`, bit-exact vs the
   hand-written kernel on synthetic inputs.
2. **End-to-end** — `verify_dsl_kernels.py` (decode: tokens must be identical
   and throughput must hold) and `verify_dsl_prefill.py` (prefill: KV caches
   must be BIT-IDENTICAL, stock vs DSL, on both attention paths).

For anything whose output feeds a quantizer, also `replay_layer.py`, which runs
both kernels at every dispatch of a real multi-token decode. Gate 1 is not
sufficient on its own — see the lesson below.

## Status: 30 of 32

**Ported and enabled (30).** The entire decode path and the entire batched
prefill path are DSL-authored: embed, PLE gather/gate/proj, qkv, attention
(decode flash + prefill, both fused and GEMM forms), o-proj, MLP at both bit
widths, every norm, logits, argmax, and all the batched prefill tails.

**Not ported (2), deliberately.** `down_96_b` and `gateup_95_b` are used only by
`prefill_research/bench_batched.py`. They are register-blocked token-batching
matmuls from a route that was measured, falsified and closed (batching alone
caps at ~1.3x; the dequant+GEMM route replaced it). They are research artifacts
retained as evidence for that retraction, not production kernels, and porting
them would be work with no consumer. If they are ever deleted, the falsification
record in `prefill_research/README.md` should go with them.

## Backend adoption

Porting the kernels and switching a backend to USE them are separate jobs. Where
each backend loads its kernels from today:

| backend | source | verified |
|---|---|---|
| `metal_runner.py` (PyObjC) | **kernels_dsl** | tokens identical, KV bit-identical |
| `runner.py` (wgpu) | **kernels_dsl** | tokens identical vs captured WGSL |
| browser (`server.py` + `app.js`) | **kernels_dsl** | headless Chrome: "**Paris**" at ~160 tok/s |
| Swift, Rust | `kernels_msl/*.metal` | not started |

Each gate compares the DEFAULT against the other source, so the direction flips
when a backend switches over. Getting that wrong turns a gate into a no-op that
passes unconditionally.

### FIXED en route: a real uniformity bug in five kernels

Dawn rejected `oproj_73`, `down_75`, `down_96`, `pleproj_77` and `attn_101` with
*"'workgroupBarrier' must only be called from uniform control flow"*. The
last-arriver tail branches on a workgroup variable, and reading it directly
leaves the branch non-uniform, so every barrier inside it is illegal WGSL.

The reference does `if (workgroupUniformLoad(&lastFlag) != 1u) { return; }` --
that builtin carries the barrier AND makes the value provably uniform. The DSL
has had `workgroupUniformLoad` since the atomics work; the port simply never
used it. **naga accepts the plain read, so this was invisible in wgpu-py and
only surfaced under Dawn** -- the same "Dawn is stricter than naga" pattern that
caught a translator bug earlier in this project. All five now use it, and the
Metal parity/gates are unchanged.

### Browser: RESOLVED

`python -m gemma4_150.server` now serves the DSL bundle by default, and headless
Chrome produces "The capital of France is **Paris**." at ~160 tok/s sustained
(captured bundle: 156.9). G4_KERNEL_SOURCE=ref for the captured kernels.

Three faults, found in order, each hidden by the previous one:

1. **drive.mjs swallowed console errors**, so a failing shader looked like a
   successful run with empty output. Nothing else was diagnosable until this was
   fixed; `checkshaders.mjs` now compiles every served kernel under Dawn and
   prints the real messages.
2. **Non-uniform barriers** -- see above. A genuine WGSL bug in five kernels
   that naga tolerates and Dawn does not.
3. **Unpatched shape constants.** app.js patched the names the CAPTURED kernels
   use; the DSL kernels declare a different set (101_srq also has HD4,
   J_GROUPS, PP_COUNTER_BASE; oproj calls it WPR, not WORDS_PER_ROW). The
   unpatched ones kept their 512-wide values, so every 256-wide layer ran with
   512-wide constants -- computing at full speed and returning nonsense. app.js
   now patches the union of both sets; a name that is absent simply does not
   match, so one call serves either bundle.

The signature is worth remembering: **correct throughput, wrong output** points
at specialization, not at the kernels. Empty output pointed at compilation.

## The lesson this port paid for

**Bit-exactness on synthetic inputs is necessary but NOT sufficient.** Kernels
whose output feeds a quantizer need real-data verification: the failure mode is
a rounding flip, and random inputs essentially never land on a `round()`
boundary while real activations — already on a quantization grid — do constantly.

Concretely:

* **Port the STRUCTURE, not the meaning.** A per-thread array, a set of scalars
  and a device round-trip are semantically equal and numerically distinct: Metal
  compiles with fast math, so multiply-add contraction follows the expression
  tree. `oproj_73` was bit-exact on 10 seeds and on a single real dispatch, and
  still changed the generated tokens from step 161 — traced by `replay_layer` to
  ONE element of `y2` off by one quantization step at token 98, layer 30. The
  fix was `PrivateArray`, so the kernel could be written the way the reference
  is instead of approximated.
* **A control on random data can inherit the blind spot it is meant to rule
  out.** That substitution was dismissed on 12 random seeds before being
  confirmed as the cause on captured real inputs.
* **Check the binding width.** `attn_101`'s and `smax_b`'s params structs are
  eight words bound as one buffer; `Uniform[vec4[u32]]` is four, so declaring two
  of them claims two binding slots and the host fills only one. Both became
  read-only storage bindings.
* **A parity test that can pass on two empty buffers is not a parity test.** The
  qkv grid was miscomputed as `Q_WGS + KV_WGS`; both sides came out empty and
  `array_equal` passed. Every parity test now also asserts the kernel wrote
  something.
* `unpack4x8unorm` (divides by 255) and `unpack4xU8` (raw bytes) are not
  interchangeable — the epilogue's `fma(...,255,...)` is what pairs with the
  former.
