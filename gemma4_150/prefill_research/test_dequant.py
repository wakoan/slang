"""Verify dq_f16 (2/4/8-bit QAT -> f16) against a float64 reconstruction.

    python -m gemma4_150.prefill_research.test_dequant

Uses real checkpoint weights of each width. The oracle is independent of the
GPU: it re-derives the packing from the bit layout documented in dq_f16.metal
and applies w = scale*(q - ZP) in float64, then rounds to f16. Getting the
lane order wrong is silent — the dot products just pair the wrong operands — so
this is checked EXACTLY, not to a tolerance.
"""
import os, struct, sys

import numpy as np

from gemma4_150.metal_runner import G4, _SHARED, Batch

HERE = os.path.dirname(os.path.abspath(__file__))
L = 20


def check(g, bits_name, scale_name, n_in, n_out, nbits):
    dev = g.dev
    bits, scale = g.w[bits_name], g.w[scale_name]
    vpw = 32 // nbits
    per_byte = 8 // nbits
    zp = 1 << (nbits - 1)
    mask = (1 << nbits) - 1
    wpr = n_in // vpw
    nwords = n_out * wpr

    key = f"dq{nbits}"
    if key not in g.kernels:
        src = open(os.path.join(HERE, "dq_f16.metal")).read().replace("BITS = 2u", f"BITS = {nbits}u")
        g._compile_one(src, "dq_f16", key)

    out = dev.newBufferWithLength_options_(n_out * n_in * 2, _SHARED)
    par = dev.newBufferWithBytes_length_options_(struct.pack("<4I", n_in, n_out, 0, 0), 16, _SHARED)
    b = Batch(g)
    b.dg(key, [(bits, 0), (scale, 0), (out, 0), (par, 0)], (nwords + 255) // 256)
    b.commit_wait()
    got = np.frombuffer(out.contents().as_buffer(n_out * n_in * 2), dtype=np.float16).reshape(n_out, n_in)

    raw = np.frombuffer(bits.contents().as_buffer(nwords * 4), dtype=np.uint32).reshape(n_out, wpr)
    sc = np.frombuffer(scale.contents().as_buffer(n_out * 4), dtype=np.float32)
    sh = np.array([8 * (i // per_byte) + nbits * (i % per_byte) for i in range(vpw)], dtype=np.uint64)
    rows = [0, 1, min(777, n_out - 1), n_out - 1]
    bad = 0
    for o in rows:
        q = ((raw[o].astype(np.uint64)[:, None] >> sh[None, :]) & mask).astype(np.float64).reshape(-1)
        ref = (sc[o].astype(np.float64) * (q - zp)).astype(np.float16)
        bad += int(np.count_nonzero(ref != got[o]))
    print(f"  {nbits}-bit  {bits_name:24s} [{n_out:6d} x {n_in:5d}]  ZP={zp:<4d} "
          f"{'EXACT' if not bad else f'{bad} MISMATCHES'}")
    return bad == 0


def main():
    print("loading model + compiling kernels…")
    g = G4()
    li = g.layers[L]
    H, qd, ple_d = g.H, li["q_dim"], g.ple_d
    print("\ndq_f16 vs float64 reconstruction (rows 0, 1, 777, last):")
    ok = all([
        check(g, f"L{L}.gate_bits", f"L{L}.gate_scale", H, li["intermediate"], 2),
        check(g, f"L{L}.q_bits", f"L{L}.q_scale", H, qd, 4),
        check(g, f"L{L}.plegate_codes", f"L{L}.plegate_rowscale", H, ple_d, 8),
        check(g, f"L{L}.pleproj_codes", f"L{L}.pleproj_rowscale", ple_d, H, 8),
    ])
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
