# DSL kernel port — status and open issues

Porting `kernels_msl/*.metal` (and the 111 captured `.wgsl`) to a single Python
source in `kernels_dsl.py`. Two gates, both required before a kernel is used:

1. `tests/test_kernels_dsl_parity.py` — bit-exact vs the hand-written kernel.
2. `python -m gemma4_150.verify_dsl_kernels` — swap into the live runner;
   generated tokens must be IDENTICAL and throughput must hold.

## RESOLVED: oproj_73 (was: passes gate 1, fails gate 2)

Fixed by adding `PrivateArray` to the DSL. Now enabled in the runner; the
per-layer sweep reports **0 divergences in 7000 dispatches over 200 tokens**.

The story is worth keeping because the failure mode generalises.

**Symptom.** oproj_73 was bit-exact against the reference on 10 random seeds,
on both q_dim shapes, and on a single dispatch using the model's own buffers for
a sliding AND a full layer — yet swapping it changed greedy decode from token
161 onward, reproducibly, while stock decode is deterministic over 4 runs.

**Diagnosis.** `gemma4_150/replay_layer.py` runs both kernels at every dispatch
of a real multi-token decode and reports the first difference. It found: token
98, layer 30, ONE element of `y2` (index 1078) off by exactly one quantization
step, with `pp` and `hidden` bit-identical. So the whole behavioural change was
a single `srq` rounding flip amplified from a last-ulp difference.

**Cause.** Metal compiles with fast math by default, so multiply-add contraction
follows the expression tree. The reference caches six per-thread residuals in
`float hloc[6]`; the DSL had no local arrays, so the port re-read `hidden[j]`
instead. That is semantically identical and numerically is not. Confirmed by
patching ONLY that line in the hand-written kernel — it reproduces the
divergence exactly, index 1078 and all.

Two substitutes were tried and both failed the same way: six scalars instead of
an array (identical divergence, so it is not register-vs-memory), and splitting
`normed` into its own statement to match the reference line-for-line (made it
WORSE — `hidden` itself then differed at token 0).

**Fix.** `PrivateArray[f32, 6]` — `var x: array<f32,6>` in WGSL, `float x[6]` in
MSL — so the kernel is written the way the reference is rather than approximated.

**Retraction.** An earlier revision of this file listed the hloc -> re-read
substitution as ruled out, on the strength of 12 random seeds. That control had
exactly the same blind spot as the parity test. It was the cause all along.

## Lesson worth keeping

Bit-exactness on synthetic inputs is necessary but NOT sufficient. Kernels whose
output feeds a quantizer need real-data verification: the failure mode is a
rounding flip, and random inputs essentially never land on a `round()` boundary
while real activations — already on a quantization grid — do constantly.

Concretely, for the rest of the port:

* Do not "simplify" a reference kernel's storage. A per-thread array, a scalar
  and a device round-trip are semantically equal and numerically distinct under
  fast math. Port the structure, not the meaning.
* `replay_layer.py` is the tool that settles these. Run it before enabling any
  kernel whose output is quantized.
* A control experiment on random data can inherit the exact blind spot it is
  meant to rule out. Confirm on captured real inputs.
