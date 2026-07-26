"""Gate: the runner's DSL prefill kernels must match the hand-written ones.

    python -m gemma4_150.verify_dsl_prefill [n_tokens]

Same inverted direction as verify_dsl_kernels: the runner compiles from
kernels_dsl.py by default, so kernels_msl is the REFERENCE here.

The observable is the KV cache, because that is prefill's entire contract, and
the bar is BIT-IDENTICAL rather than a tolerance -- any difference is a port
defect, not the known f16-GEMM divergence from the per-token path. Both
attention paths are exercised: the GEMM one (qprep_b / smax_b / attnout_b) and
the fused attn_prefill fallback, since they use disjoint kernel sets.
"""
import sys

import numpy as np
from tokenizers import Tokenizer

from py_shader_lang_wgpu import translate
from gemma4_150 import kernels_dsl
from gemma4_150.metal_runner import G4, TOKJSON
from gemma4_150.prefill_gpu import PrefillGPU

def recompile(g, use_dsl):
    g.kernels.clear()
    g.USE_DSL = use_dsl
    g._compile_all()


def snapshot(g, n):
    out = {}
    for kind, d in (("k", g.kc), ("v", g.vc)):
        for l, buf in d.items():
            w = n * g.layers[l]["head_dim"] * 4
            out[(kind, l)] = np.frombuffer(buf.contents().as_buffer(w), np.float32).copy()
    return out


def zero(g):
    for d in (g.kc, g.vc):
        for buf in d.values():
            buf.contents().as_buffer(buf.length())[:] = b"\x00" * buf.length()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    print("loading model…")
    g = G4()
    tok = Tokenizer.from_file(TOKJSON)
    text = ("The history of computing begins long before the electronic computer. "
            "Mechanical calculators, punched cards and the theory of computation "
            "all predate the first stored-program machines by decades. ") * 40
    ids = ([g.bos] + tok.encode(text, add_special_tokens=False).ids)[:n]
    n = len(ids)
    print(f"prompt: {n} tokens")

    ok = True
    for gemm_attn in (True, False):
        label = "GEMM attention" if gemm_attn else "fused attn_prefill"
        snaps = {}
        for use_dsl in (True, False):
            recompile(g, use_dsl)
            pf = PrefillGPU(g)             # rebuilds its variants from this source
            pf.ATTN_GEMM = gemm_attn
            zero(g); pf.prefill(ids, 0)    # warm / build MPS pipelines
            zero(g); pf.prefill(ids, 0)
            snaps[use_dsl] = snapshot(g, n)
        ref, got = snaps[False], snaps[True]
        worst = max(int(np.count_nonzero(ref[key] != got[key])) for key in ref)
        same = worst == 0
        ok &= same
        print(f"  {label:20}  KV bit-identical: {same}"
              + ("" if same else f"   ({worst} elements differ)"))
        if not same:
            bad = [k for k in ref if not np.array_equal(ref[k], got[k])]
            print(f"    first differing cache: {sorted(bad)[0]}")
    recompile(g, True)                     # leave the runner as we found it

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
