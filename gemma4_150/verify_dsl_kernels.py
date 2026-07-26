"""End-to-end gate for the DSL kernel port.

    python -m gemma4_150.verify_dsl_kernels

The unit tests (tests/test_kernels_dsl_parity.py) prove each DSL kernel is
bit-exact against the hand-written one in isolation. This proves the swap is
safe in the assembled model, which is a different claim: a kernel can be
bit-exact on synthetic inputs and still be wrong in place — bound at the wrong
index, dispatched with the wrong grid, or reading a buffer the runner lays out
differently.

Two observables, both of which must hold:
  * generated tokens IDENTICAL to the stock kernels (not "similar" — greedy
    decode is deterministic, so any difference is a real defect);
  * decode throughput unchanged, so the port is not quietly buying correctness
    with speed.

This does not live in tests/ because it loads the full 2 GB model.
"""
import sys
import time

from tokenizers import Tokenizer

from py_shader_lang_wgpu import translate
from gemma4_150 import kernels_dsl
from gemma4_150.metal_runner import G4, TOKJSON

# Kernels ported to gemma4_150/kernels_dsl.py, by the key the runner compiles
# them under. Add a name here once its parity test passes.
PORTED = ["rmssrq_69", "combine", "srqh_b", "srq_b", "geglu_b", "down_75",
          "embed_00", "plegather_01", "argmax1_34", "argmax2_35",
          "rmsadd_b", "rmssrqh_b", "combine_b", "proj_68", "plegate_76"]

# Shape-parameterized kernels: one DSL source, one variant per layer geometry.
# (dsl name, runner key, consts, threads) — the runner registers these under
# keys like "kvnorm_256" / "oproj_2048", built by string-patching the .metal.
SPECIALIZED = (
    [("kvnorm", f"kvnorm_{hd}", {"HD": hd, "HALF": hd // 2}, hd) for hd in (256, 512)]
    # NB oproj_73 is NOT here. It is bit-exact against the reference in the
    # parity tests (10 seeds, both q_dim shapes, and a single dispatch on the
    # model's own buffers for a sliding AND a full layer) — yet swapping it
    # alone changes greedy decode from token 161 onward, reproducibly, while
    # stock-vs-stock is deterministic over 4 runs. Something input-dependent
    # differs that synthetic and single-dispatch checks do not reach; until it
    # is found the kernel stays out of the runner. See PORT_NOTES.md.
)

PROMPT = "Write a 200-word essay about the sea."


def main():
    print("loading model + compiling kernels…")
    g = G4()
    tok = Tokenizer.from_file(TOKJSON)
    ids = [g.bos] + tok.encode(
        f"<|turn>user\n{PROMPT}<turn|>\n<|turn>model\n", add_special_tokens=False).ids

    def run():
        return g.generate(ids, 192)

    run()                                  # warm the GPU clock and the pipelines
    base_out, base_tps = run()

    for name in PORTED:
        fn = getattr(kernels_dsl, name)
        g._compile_one(translate(fn, target="msl"), name, name)
    for name, key, consts, threads in SPECIALIZED:
        if key in g.kernels:
            g._compile_one(translate(getattr(kernels_dsl, name),
                                     workgroup_size=(threads, 1, 1),
                                     target="msl", consts=consts), name, key)
    g._pf = None                           # rebuild batched prefill against them

    run()
    dsl_out, dsl_tps = run()

    same = base_out == dsl_out
    slow = dsl_tps < base_tps * 0.95
    print(f"\nswapped {len(PORTED)} kernels + {len(SPECIALIZED)} specialized variants")
    print(f"  {', '.join(PORTED)}")
    print(f"  specialized: {', '.join(k for _, k, _, _ in SPECIALIZED)}")
    print(f"  stock : {len(base_out):3d} tokens, {base_tps:6.1f} tok/s")
    print(f"  DSL   : {len(dsl_out):3d} tokens, {dsl_tps:6.1f} tok/s")
    print(f"  tokens identical : {same}")
    print(f"  throughput held  : {not slow} ({dsl_tps / base_tps:.3f}x)")
    if not same:
        n = next((i for i, (a, b) in enumerate(zip(base_out, dsl_out)) if a != b), None)
        print(f"  FIRST DIVERGENCE at token {n}")
    print(f"\n{'PASS' if same and not slow else 'FAIL'}")
    return 0 if (same and not slow) else 1


if __name__ == "__main__":
    sys.exit(main())
