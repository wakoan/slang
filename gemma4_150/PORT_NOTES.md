# DSL kernel port — status and open issues

Porting `kernels_msl/*.metal` (and the 111 captured `.wgsl`) to a single Python
source in `kernels_dsl.py`. Two gates, both required before a kernel is used:

1. `tests/test_kernels_dsl_parity.py` — bit-exact vs the hand-written kernel.
2. `python -m gemma4_150.verify_dsl_kernels` — swap into the live runner;
   generated tokens must be IDENTICAL and throughput must hold.

## OPEN: oproj_73 passes gate 1 and fails gate 2

`oproj_73` is ported and its parity test passes, but it is deliberately NOT in
`verify_dsl_kernels.PORTED` and the runner still uses the hand-written kernel.

**What is established:**

* Bit-exact vs the reference across 10 random seeds, both `q_dim` shapes
  (WPR 256 and 512) — `pp` (the GEMV output), `hidden`, `y2` and `sum2` all
  compare equal with `np.array_equal`.
* Bit-exact on the model's OWN buffers for a single dispatch, checked on a
  sliding layer (L0, q_dim 2048) and a full layer (L4, q_dim 4096).
* The counter self-resets to 0, so the next token's merge still fires.
* Stock decode is deterministic: 4 consecutive 192-token runs are identical.
* Swapping ONLY the two oproj variants changes greedy decode from token 161,
  reproducibly. So the divergence is real and caused by this kernel.

**CAUSE FOUND — Metal fast-math FMA contraction.** `gemma4_150/replay_layer.py`
compares stock vs DSL at every dispatch across real decode and reported the
exact case: **token 98, layer 30**, where ONE element of `y2` (index 1078)
differs by exactly one quantization step (-0.4966 vs -0.5190) while `pp` and
`hidden` are bit-identical. So the whole divergence is one `srq` rounding flip
in the pre-FFN norm, amplified from a last-ulp difference in `n2`.

The ulp difference comes from how the Metal compiler contracts multiply-adds.
`newLibraryWithSource_options_error_(src, None, None)` compiles with fast math,
which permits contraction, and the choice depends on the exact expression
structure. Demonstrated three ways on the captured failing input:

* reference `hloc[e]` (a per-thread ARRAY) vs the same kernel patched to
  re-read `hidden[j]` — reproduces the divergence exactly, index 1078 and all;
* rewriting the DSL version to hold the six residuals in SCALARS instead of
  re-reading — identical divergence, so it is not register-vs-memory;
* splitting `normed` into its own statement to match the reference
  line-for-line — made it WORSE, `hidden` itself then differing by 1.9e-6 at
  token 0.

So matching is not about semantics; it is about emitting an expression tree the
compiler contracts identically. The reference stores into `float hloc[6]`, and
an array element is treated differently from a scalar.

**Next step:** DSL support for private (function-scope) arrays —
`var x: array<f32,6>` in WGSL, `float x[6]` in MSL — so the kernel can be
written the way the reference is instead of approximated. That is the only
remaining gap; every other construct oproj_73 needs already works.

An alternative worth measuring first: compile both with fast math DISABLED
(`MTLCompileOptions.fastMathEnabled = False`). If contraction is off, the
structure stops mattering — but it would change the stock kernels' numerics
too, so it is a runner-wide decision, not a port detail.

**Ruled out:**

* ~~The `hloc` -> re-read substitution.~~ RETRACTED: this was ruled out on 12
  random seeds, and that control had the same blind spot as the parity test.
  On the captured failing input it reproduces the divergence exactly. Random
  data does not land on `round()` boundaries; real activations do constantly.
* Float addition associativity — a real bug found and fixed here (`a += X + Y`
  is `a + (X+Y)`, not `(a+X)+Y`; the same slip was present in `down_75` and
  `plegate_76` and is fixed in all three). Fixing it did not change the
  divergence, so it was latent rather than causal.
* Dispatch geometry (192 workgroups, 256 threads) and the threadgroup size the
  runner parses from the emitted comment.

**Next step when resumed:** capture each layer's real `attn`/`hidden` inputs
during a decode step and replay both kernels per layer until the first layer
that differs, rather than testing one arbitrary input. A per-layer replay
harness is the missing tool — synthetic parity is demonstrably not sufficient
for this kernel.

## Lesson worth keeping

Bit-exactness on synthetic inputs is necessary but NOT sufficient. This is the
first kernel where the two gates disagree, and the end-to-end gate is the one
that was right. Kernels whose output feeds a quantizer need real-data
verification, because the failure mode is a rounding flip that random inputs
almost never trigger.
