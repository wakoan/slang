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
