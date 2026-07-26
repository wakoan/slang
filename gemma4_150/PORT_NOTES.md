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

**What that means:** the difference is input-dependent and does not show up in
either synthetic data or a single dispatch on one real input. The most likely
mechanism is a last-ulp difference that flips an `srq` rounding: activations are
already on a quantization grid, so real inputs sit exactly on `round()`
boundaries far more often than random data does, and one flip anywhere in
35 layers x 161 tokens is enough to change the sampled token.

**Ruled out:**

* The `hloc` -> re-read substitution (the DSL has no local arrays, so the second
  norm pass re-reads `hidden[j]` instead of a per-thread cache). Patching the
  HAND-WRITTEN kernel with only that change is bit-identical over 12 seeds.
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
