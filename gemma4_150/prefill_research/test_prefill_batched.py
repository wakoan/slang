"""Verify batched prefill against the per-token prefill it replaces.

    python -m gemma4_150.prefill_research.test_prefill_batched [n_tokens]

Prefill's only job is to leave the KV caches correct, so the caches ARE the
contract: run the same prompt through both paths from a zeroed cache and compare
every owning layer's K and V.

MPS emits f16 and the dequantized weights lose the exact int-dot arithmetic, so
this is a tolerance comparison, not bit-identity — deliberately, and the
tolerance is reported rather than merely asserted.
"""
import sys, time

import numpy as np

from gemma4_150.metal_runner import G4
from gemma4_150.prefill_gpu import PrefillGPU


def snapshot(g, n):
    out = {}
    for l, buf in g.kc.items():
        out[("k", l)] = np.frombuffer(buf.contents().as_buffer(n * g.layers[l]["head_dim"] * 4),
                                      dtype=np.float32).copy()
    for l, buf in g.vc.items():
        out[("v", l)] = np.frombuffer(buf.contents().as_buffer(n * g.layers[l]["head_dim"] * 4),
                                      dtype=np.float32).copy()
    return out


def zero(g):
    for d in (g.kc, g.vc):
        for buf in d.values():
            buf.contents().as_buffer(buf.length())[:] = b"\x00" * buf.length()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    print("loading model + compiling kernels…")
    g = G4()
    from tokenizers import Tokenizer
    from gemma4_150.metal_runner import TOKJSON
    tok = Tokenizer.from_file(TOKJSON)
    text = ("The history of computing begins long before the electronic computer. "
            "Mechanical calculators, punched cards and the theory of computation all "
            "predate the first stored-program machines by decades. ") * 6
    ids = ([g.bos] + tok.encode(text, add_special_tokens=False).ids)[:n]
    n = len(ids)
    print(f"prompt: {n} tokens")

    zero(g)
    t0 = time.time()
    g.prefill(ids, 0)
    t_ref = time.time() - t0
    ref = snapshot(g, n)

    pf = PrefillGPU(g)
    zero(g)
    t0 = time.time()
    pos = pf.prefill(ids, 0)
    t_cold = time.time() - t0
    got = snapshot(g, n)
    # MPS builds a pipeline per unique (M,N,K,alpha), so the first prefill at a
    # given S pays a one-off compile. Time a second run for the steady state.
    zero(g)
    t0 = time.time()
    pf.prefill(ids, 0)
    t_bat = time.time() - t0
    print(f"(batched cold-start incl. MPS pipeline build: {t_cold * 1e3:.1f} ms)")

    assert pos == n, f"position {pos} != {n}"
    worst, worst_key = 0.0, None
    for k in ref:
        a, b = ref[k], got[k]
        denom = max(np.abs(a).max(), 1e-6)
        rel = np.abs(a - b).max() / denom
        if rel > worst:
            worst, worst_key = rel, k
    print(f"\nper-token prefill : {t_ref * 1e3:7.1f} ms  ({n / t_ref:6.1f} tok/s)")
    print(f"batched prefill   : {t_bat * 1e3:7.1f} ms  ({n / t_bat:6.1f} tok/s)"
          f"   speedup {t_ref / t_bat:.2f}x")
    print(f"\nKV cache rel error vs the per-token path, by layer:")
    errs = {}
    for (kind, l) in ref:
        a, b2 = ref[(kind, l)], got[(kind, l)]
        errs[(kind, l)] = np.abs(a - b2).max() / max(np.abs(a).max(), 1e-6)
    for l in sorted({l for _, l in errs}):
        print(f"   layer {l:2d}: k {errs[('k', l)]:.3e}   v {errs[('v', l)]:.3e}")

    # Layer 0 is the plumbing check: its inputs are identical in both paths, so
    # anything beyond f16 rounding there means a real wiring/semantics bug (this
    # is exactly how the 4-bit-vs-2-bit MLP mixup was caught).
    l0 = max(errs[("k", 0)], errs[("v", 0)])
    ok = l0 < 3e-2 and pos == n
    print(f"\nlayer-0 divergence (plumbing check): {l0:.3e}  -> {'ok' if l0 < 3e-2 else 'BUG'}")
    print(f"deepest-layer divergence:            {worst:.3e} at {worst_key}")
    print("""
NOTE: the deep-layer divergence is expected and is NOT bit-rot. The decode
kernels compute an exact integer dot product and apply the f32 row scale ONCE at
the end; dequantizing to f16 instead applies that scale per weight, so every
matmul carries ~1e-3 relative error. A 35-layer residual stream amplifies such
perturbations, hence the growth with depth. Generation stays coherent (the
trajectory differs, the quality does not) -- but this path is NOT numerically
equivalent to the per-token one, and should not be presented as such. Whether an
f32 GEMM removes it, and at what throughput cost, is untested.""")
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
