"""End-to-end gate for the DSL-authored PREFILL kernels.

    python -m gemma4_150.verify_dsl_prefill [n_tokens]

verify_dsl_kernels covers the decode path; this covers the batched prefill one.
The observable is the KV cache, because that is prefill's entire contract — and
the comparison is stock-vs-DSL rather than batched-vs-per-token, so the bar is
BIT-IDENTICAL rather than a tolerance. Any difference is a port defect, not the
known f16-GEMM divergence from the per-token path.

Both attention paths are exercised: the GEMM one (qprep_b / smax_b / attnout_b)
and the fused attn_prefill fallback, since they use disjoint kernel sets.
"""
import sys

import numpy as np
from tokenizers import Tokenizer

from py_shader_lang_wgpu import translate
from gemma4_150 import kernels_dsl
from gemma4_150.metal_runner import G4, TOKJSON
from gemma4_150.prefill_gpu import PrefillGPU

# Prefill kernels ported to kernels_dsl.py, by the key the runner compiles them
# under. srqh_b/srq_b/geglu_b/rmsadd_b/rmssrqh_b/combine_b are shared with the
# decode gate and already covered there, but re-swapping them here is free.
PORTED = ["srqh_b", "srq_b", "geglu_b", "rmsadd_b", "rmssrqh_b", "combine_b",
          "attnout_b", "plegate_b", "smax_b"]

# Shape-parameterized: (dsl name, key, consts, threads)
def specialized(g):
    out = []
    for hd in sorted({g.layers[l]["head_dim"] for l in range(g.nL)}):
        out.append(("kvnorm_b", f"kvnormb_{hd}", {"HD": hd, "HALF": hd // 2}, hd))
        out.append(("qprep_b", f"qprep_{hd}",
                    {"HEAD_DIM": hd, "HALF_DIM": hd // 2}, hd))
    for l in range(g.nL):
        hd = g.layers[l]["head_dim"]
        out.append(("attn_prefill", f"attnp_{hd}_{l}",
                    {"HEAD_DIM": hd, "HALF_DIM": hd // 2, "HD4": hd // 4,
                     "J_GROUPS": 256 // (hd // 4), "OUT_Q": g.sc(l, "o_in")}, 256))
    return out


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
    pf = PrefillGPU(g)
    print(f"prompt: {n} tokens")

    ok = True
    for gemm_attn in (True, False):
        pf.ATTN_GEMM = gemm_attn
        label = "GEMM attention" if gemm_attn else "fused attn_prefill"
        zero(g); pf.prefill(ids, 0)                 # warm / build pipelines
        zero(g); pf.prefill(ids, 0)
        ref = snapshot(g, n)

        for name in PORTED:
            if name in g.kernels:
                g._compile_one(translate(getattr(kernels_dsl, name), target="msl"),
                               name, name)
        for name, key, consts, threads in specialized(g):
            if key in g.kernels:
                g._compile_one(translate(getattr(kernels_dsl, name),
                                         workgroup_size=(threads, 1, 1),
                                         target="msl", consts=consts), name, key)
        zero(g); pf.prefill(ids, 0)
        zero(g); pf.prefill(ids, 0)
        got = snapshot(g, n)

        worst = max(int(np.count_nonzero(ref[key] != got[key])) for key in ref)
        same = worst == 0
        ok &= same
        print(f"  {label:20}  KV bit-identical: {same}"
              + ("" if same else f"   ({worst} elements differ)"))
        if not same:
            bad = [k for k in ref if not np.array_equal(ref[k], got[k])]
            print(f"    first differing cache: {sorted(bad)[0]}")

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
