"""Verify attn_prefill (batched causal attention) against attn_101 (decode).

    python -m gemma4_150.prefill_research.test_attn_prefill

attn_101 IS the oracle: running it once per query position, with the same KV
cache and the same q, must reproduce what attn_prefill produces for the whole
batch in one dispatch. Covers both head-dim families (sliding 256 / full 512)
and drives S past the 256-key tile boundary so the online-softmax rescale path
is actually exercised.
"""
import os, struct, sys

import numpy as np

from gemma4_150.metal_runner import G4, KDIR, _SHARED, Batch

S = 300          # > one 256-key tile, so the running-max correction is exercised
NCHUNK = 32


def check(g, L, S=S):
    dev = g.dev
    li = g.layers[L]
    hd, nH = li["head_dim"], g.nH
    half, qdim = hd // 2, nH * hd
    window = g.window if li["sliding"] else 0
    oin = g.sc(L, "o_in")

    rng = np.random.default_rng(5)
    qv = rng.standard_normal(S * qdim).astype(np.float32) * 0.5
    kv = rng.standard_normal(S * hd).astype(np.float32) * 0.5
    vv = rng.standard_normal(S * hd).astype(np.float32) * 0.5
    qb = dev.newBufferWithBytes_length_options_(qv.tobytes(), qv.nbytes, _SHARED)
    kb = dev.newBufferWithBytes_length_options_(kv.tobytes(), kv.nbytes, _SHARED)
    vb = dev.newBufferWithBytes_length_options_(vv.tobytes(), vv.nbytes, _SHARED)
    qn = g.w[f"L{L}.q_norm"]
    cb, sb = g.pool[f"rcosT_{hd}"], g.pool[f"rsinT_{hd}"]
    outr = dev.newBufferWithLength_options_(qdim * 4, _SHARED)
    outb = dev.newBufferWithLength_options_(S * qdim * 4, _SHARED)
    pp = dev.newBufferWithLength_options_((8 * NCHUNK * (hd + 2) + 8) * 4, _SHARED)
    pp.contents().as_buffer(pp.length())[:] = b"\x00" * pp.length()

    # oracle: attn_101, one dispatch per query position
    ref = np.empty((S, qdim), dtype=np.float32)
    for pos in range(S):
        par = dev.newBufferWithBytes_length_options_(
            struct.pack("<8I", 1, pos + 1, pos, nH, 1, window, 0, 0), 32, _SHARED)
        roff = pos * half * 4
        b = Batch(g)
        b.dg2d(f"attn_{hd}_{L}", [(qb, pos * qdim * 4), (qn, 0), (cb, roff), (sb, roff),
                                  (kb, 0), (vb, 0), (pp, 0), (outr, 0), (par, 0)], nH, NCHUNK)
        b.commit_wait()
        ref[pos] = np.frombuffer(outr.contents().as_buffer(qdim * 4), dtype=np.float32)

    key = f"attnp_{hd}_{L}"
    src = open(os.path.join(KDIR, "attn_prefill.metal")).read()
    src = src.replace("HEAD_DIM=512u, HALF_DIM=256u", f"HEAD_DIM={hd}u, HALF_DIM={half}u")
    src = src.replace("OUT_Q=0.014886821620166302f", f"OUT_Q={oin!r}f")
    g._compile_one(src, "attn_prefill", key)
    par = dev.newBufferWithBytes_length_options_(
        struct.pack("<8I", 0, nH, 1, window, S, 0, 0, 0), 32, _SHARED)
    b = Batch(g)
    b.dg2d(key, [(qb, 0), (qn, 0), (cb, 0), (sb, 0), (kb, 0), (vb, 0), (outb, 0), (par, 0)],
           nH, S)
    b.commit_wait()
    got = np.frombuffer(outb.contents().as_buffer(S * qdim * 4), dtype=np.float32).reshape(S, qdim)

    # The outputs are SRQ-quantized to a grid of step `oin`, so a value landing on
    # a .5 boundary can round either way on a last-ulp difference in accumulation
    # order. Such a mismatch is exactly one grid step and is benign; anything else
    # is a real divergence.
    d = np.abs(got - ref)
    ties = int(np.count_nonzero(np.isclose(d, oin, rtol=1e-3)))
    real = int(np.count_nonzero(d > oin * 1.001))
    kind = "sliding" if li["sliding"] else "full"
    status = "EXACT" if not d.any() else f"{ties} tie(s), {real} real diff(s)"
    print(f"  L{L:<3d} {kind:8s} head_dim={hd:<4d} window={window:<5d} S={S}: {status}"
          + (f"  max|d|={d.max():.3e}" if real else ""))
    return real == 0


def main():
    print("loading model + compiling kernels…")
    g = G4()
    layers = [20, next(i for i, l in enumerate(g.layers) if not l["sliding"])]
    print(f"\nattn_prefill vs attn_101 (decode) as oracle:")
    ok = all([check(g, L) for L in layers])
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
