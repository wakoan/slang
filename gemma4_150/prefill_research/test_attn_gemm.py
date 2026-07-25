"""Verify GEMM attention against the fused attn_prefill kernel it replaces.

    python -m gemma4_150.prefill_research.test_attn_gemm [n_tokens]

attn_prefill was itself verified against the decode kernel attn_101
(test_attn_prefill.py), so this compares against a checked reference rather than
against numpy again.

Two observables, because one alone would be misleading:

  1. Layer 0's attention output directly (truncating the layer loop leaves it in
     the scratch buffer). This is the tight check — no downstream amplification,
     so the numbers mean what they say.
  2. The KV caches after a full prefill run both ways. Layer 0's cache is
     attention-independent and must come out BIT-IDENTICAL; layer 1's is the
     first consumer of layer-0 attention, so a wrong mask, a wrong row/head
     mapping or a missing normalisation shows up there at full strength.

Deeper layers accumulate and are reported, not asserted: f16 score arithmetic
legitimately differs from the fused kernel's f32 online softmax, and a 35-layer
residual stream amplifies any perturbation (the same effect already documented
for the f16-dequant matmuls in test_prefill_batched.py).
"""
import sys, time

import numpy as np

from gemma4_150.metal_runner import G4, TOKJSON
from gemma4_150.prefill_gpu import PrefillGPU
from gemma4_150.prefill_research.test_prefill_batched import snapshot, zero


def attn_only(pf, ids, n, gemm):
    """Layer 0's attention output. Stubbing out layers 1+ leaves attnh holding
    layer 0's result at commit time; everything before the layer loop (embed,
    PLE context, input norm) still runs, so the input is the real thing."""
    orig = PrefillGPU._layer
    PrefillGPU._layer = lambda self, b, l, S, sp: None if l else orig(self, b, l, S, sp)
    try:
        pf.ATTN_GEMM = gemm
        zero(pf.g)
        pf.prefill(ids, 0)
    finally:
        PrefillGPU._layer = orig
    qd = pf.g.layers[0]["q_dim"]
    return np.frombuffer(pf._buf["attnh"].contents().as_buffer(n * qd * 2),
                         dtype=np.float16).astype(np.float64)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    print("loading model + compiling kernels…")
    g = G4()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKJSON)
    text = ("The history of computing begins long before the electronic computer. "
            "Mechanical calculators, punched cards and the theory of computation all "
            "predate the first stored-program machines by decades. ") * 40
    ids = ([g.bos] + tok.encode(text, add_special_tokens=False).ids)[:n]
    n = len(ids)
    print(f"prompt: {n} tokens")

    pf = PrefillGPU(g)

    # ---- 1. the attention output itself
    a, b = attn_only(pf, ids, n, False), attn_only(pf, ids, n, True)
    d = np.abs(a - b)
    oin = g.sc(0, "o_in")
    rel = d.max() / np.abs(a).max()
    one_step = np.count_nonzero(np.isclose(d, oin, rtol=1e-2)) / d.size
    over = np.count_nonzero(d > oin * 1.01) / d.size
    print(f"\nlayer-0 attention output, fused vs GEMM (SRQ grid step {oin:.6f}):")
    print(f"   identical {np.count_nonzero(d == 0) / d.size:6.2%}"
          f"   one grid step {one_step:6.2%}   more {over:6.2%}")
    print(f"   max abs diff {d.max():.6f} of max|ref| {np.abs(a).max():.4f}  (rel {rel:.3e})")

    # ---- 2. KV caches after the full model
    out = {}
    for gemm in (False, True):
        pf.ATTN_GEMM = gemm
        zero(g)
        pf.prefill(ids, 0)          # warm: compiles MPS pipelines, warms the clock
        zero(g)
        t0 = time.time()
        pf.prefill(ids, 0)
        out[gemm] = (snapshot(g, n), time.time() - t0)

    (ref, t_ref), (got, t_gemm) = out[False], out[True]
    print(f"\nfused attn_prefill : {t_ref * 1e3:7.1f} ms  ({n / t_ref:6.1f} tok/s)")
    print(f"GEMM attention     : {t_gemm * 1e3:7.1f} ms  ({n / t_gemm:6.1f} tok/s)"
          f"   speedup {t_ref / t_gemm:.2f}x")

    errs = {}
    for k in ref:
        a, b = ref[k], got[k]
        errs[k] = np.abs(a - b).max() / max(np.abs(a).max(), 1e-6)
    print("\nKV cache rel error, fused vs GEMM attention:")
    for l in sorted({l for _, l in errs}):
        print(f"   layer {l:2d}: k {errs[('k', l)]:.3e}   v {errs[('v', l)]:.3e}")

    l0 = max(errs[("k", 0)], errs[("v", 0)])
    l1 = max(errs[("k", 1)], errs[("v", 1)])
    # The attention output must agree to within the SRQ grid the o-projection
    # quantizes it onto anyway: anything under one step is arithmetic noise the
    # model cannot see, so what matters is how few elements move MORE than that.
    ok = over < 1e-3 and l0 == 0.0 and l1 < 5e-2
    print(f"\nattention output beyond one SRQ step: {over:.2%}  -> {'ok' if over < 1e-3 else 'BUG'}")
    print(f"layer 0 (attention-independent, must be exact): {l0:.3e}"
          f"  -> {'ok' if l0 == 0.0 else 'BUG'}")
    print(f"layer 1 (first consumer of layer-0 attention): {l1:.3e}"
          f"  -> {'ok' if l1 < 5e-2 else 'BUG'}")
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
