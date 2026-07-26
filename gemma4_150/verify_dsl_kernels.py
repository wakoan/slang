"""Gate: the runner's DSL kernels must match the hand-written ones exactly.

    python -m gemma4_150.verify_dsl_kernels

NOTE THE DIRECTION. The runner now compiles from `kernels_dsl.py` by DEFAULT,
so this loads the hand-written `kernels_msl/*.metal` as the REFERENCE and
compares the default against it. Before the switch it ran the other way round;
leaving it that way would have made the gate compare DSL against DSL and pass
unconditionally -- the same self-comparison trap that once reported a triumphant
0.0 divergence for batched prefill.

Two observables, both required:
  * generated tokens IDENTICAL -- greedy decode is deterministic, so any
    difference is a real defect, not drift;
  * decode throughput held, so the port is not buying correctness with speed.

Not in tests/ because it loads the full 2 GB model.
"""
import sys

from tokenizers import Tokenizer

from gemma4_150.metal_runner import G4, TOKJSON

PROMPT = "Write a 200-word essay about the sea."


def recompile(g, use_dsl):
    """Rebuild every kernel from the chosen source."""
    g.kernels.clear()
    g.USE_DSL = use_dsl
    g._compile_all()
    g._pf = None            # batched prefill re-derives its variants on next use


def main():
    print("loading model...")
    g = G4()
    tok = Tokenizer.from_file(TOKJSON)
    ids = [g.bos] + tok.encode(
        f"<|turn>user\n{PROMPT}<turn|>\n<|turn>model\n", add_special_tokens=False).ids

    def run():
        return g.generate(ids, 192)

    recompile(g, True)                     # the runner default
    run()                                  # warm clock + pipelines
    dsl_out, dsl_tps = run()

    recompile(g, False)                    # hand-written reference
    run()
    ref_out, ref_tps = run()

    recompile(g, True)                     # leave the runner as we found it

    same = ref_out == dsl_out
    slow = dsl_tps < ref_tps * 0.95
    print(f"\nkernel source comparison ({len(ref_out)} tokens)")
    print(f"  kernels_msl (reference) : {ref_tps:6.1f} tok/s")
    print(f"  kernels_dsl (default)   : {dsl_tps:6.1f} tok/s")
    print(f"  tokens identical : {same}")
    print(f"  throughput held  : {not slow} ({dsl_tps / ref_tps:.3f}x)")
    if not same:
        n = next((i for i, (a, b) in enumerate(zip(ref_out, dsl_out)) if a != b), None)
        print(f"  FIRST DIVERGENCE at token {n}")
    print(f"\n{'PASS' if same and not slow else 'FAIL'}")
    return 0 if (same and not slow) else 1


if __name__ == "__main__":
    sys.exit(main())
