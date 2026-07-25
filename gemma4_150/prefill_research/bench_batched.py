"""Reproducible benchmark for the batched (prefill) matmul kernels.

    python -m gemma4_150.prefill_research.bench_batched [S]

Compares each batched kernel against S dispatches of the decode kernel it
replaces, and sweeps the (N_ROWS, SEQ) register-blocking factors.

THE BASELINE IS THE WHOLE BALLGAME. All S decode dispatches go into ONE command
buffer, so they overlap on the GPU exactly as they do inside a real forward
pass. Timing them as S separate command buffers (commit+wait each) inflates the
baseline ~1.8x and manufactures speedups that do not exist — that is precisely
how this file's predecessor "measured" 2.49x for a kernel that is really 0.55x.
`--unfair` reproduces that mistake so it stays falsifiable.

Correctness is checked against the decode kernel's own output. Expect a handful
of single-element mismatches: they are exact ties on the SRQ rounding grid
(verified in float64 — one landed on 50.500007), where a last-ulp difference in
float accumulation order flips round() to the other side. The dot products
themselves agree exactly.
"""
import os, struct, sys, time, random

import numpy as np

from gemma4_150.metal_runner import G4, KDIR, _SHARED, Batch

L = 20                                  # sliding layer, intermediate=12288 (2-bit)
H, INTER, OUT_F = 1536, 12288, 1536
ITERS = 60


def _time(fn, iters=ITERS, reps=3):
    """Min-of-reps. Min, not mean: we want the clean-run time, and the noise here
    is one-sided (scheduling, contention)."""
    fn()                                # warm
    best = float("inf")
    for _ in range(reps):
        t0 = time.time()
        for _ in range(iters):
            fn()
        best = min(best, (time.time() - t0) / iters * 1e3)
    return best


def warmup(g, seconds=2.0):
    """Ramp the GPU clock before ANY timing.

    Non-negotiable on Apple silicon: the GPU idles at a low clock and takes
    order-seconds of sustained load to reach full frequency. Measured cold, the
    S x down_96 baseline reads 1.013 ms; warm it reads 0.366 ms — a 2.8x swing
    that lands entirely on whichever kernel happens to run first, and silently
    inflates its "speedup". Every number in this file is taken warm.
    """
    dev = g.dev
    W = lambda n: g.w[f"L{L}.{n}"]
    hid = dev.newBufferWithLength_options_(H * 2, _SHARED)
    suma = dev.newBufferWithLength_options_(4, _SHARED)
    out = dev.newBufferWithLength_options_(INTER * 2, _SHARED)
    par = dev.newBufferWithBytes_length_options_(struct.pack("<fffI", 1.0, 1.0, 1.0, 0), 16, _SHARED)
    a = [(hid, 0), (W("gate_bits"), 0), (W("gate_scale"), 0), (W("up_bits"), 0),
         (W("up_scale"), 0), (suma, 0), (out, 0), (W("gelu_gate"), 0), (par, 0)]
    t0 = time.time()
    while time.time() - t0 < seconds:
        b = Batch(g)
        for _ in range(32):
            b.dg("gateup_95", a, 3072)
        b.commit_wait()


def bench_down(g, S, unfair=False):
    dev = g.dev
    bits, scale, nw = (g.w[f"L{L}.down_bits"], g.w[f"L{L}.down_scale"], g.w[f"L{L}.down_nw"])
    inS, outS = g.sc(L, "down_in"), g.sc(L, "down_out")
    random.seed(7)
    av = np.array([float(random.randint(-128, 127)) for _ in range(S * INTER)], dtype=np.float16)
    a = dev.newBufferWithBytes_length_options_(av.tobytes(), S * INTER * 2, _SHARED)
    hidden = dev.newBufferWithLength_options_(OUT_F * 4, _SHARED)
    pp = dev.newBufferWithLength_options_((OUT_F + 1) * 4, _SHARED)
    pp.contents().as_buffer((OUT_F + 1) * 4)[:] = b"\x00" * ((OUT_F + 1) * 4)
    yb = dev.newBufferWithLength_options_(S * OUT_F * 4, _SHARED)
    par = dev.newBufferWithBytes_length_options_(struct.pack("<ffII", inS, outS, 0, 0), 16, _SHARED)
    args = lambda s: [(a, s * INTER * 2), (bits, 0), (pp, 0), (scale, 0), (hidden, 0), (nw, 0), (par, 0)]

    ref = []
    for s in range(S):
        b = Batch(g); b.dg("down_96", args(s), 384); b.commit_wait()
        ref.append(np.frombuffer(pp.contents().as_buffer(OUT_F * 4), dtype=np.float32).copy())
    ref = np.stack(ref)

    def base():
        b = Batch(g)
        for s in range(S):
            b.dg("down_96", args(s), 384)
        b.commit_wait()

    def base_unfair():
        for s in range(S):
            b = Batch(g); b.dg("down_96", args(s), 384); b.commit_wait()

    ms0 = _time(base_unfair if unfair else base)
    print(f"\ndown  (2-bit, {INTER}->{OUT_F}, 28% of prefill FLOPs)   S={S}")
    print(f"  baseline: S x down_96 = {ms0:.3f} ms" + ("   [UNFAIR: per-token command buffers]" if unfair else ""))
    src0 = open(os.path.join(KDIR, "down_96_b.metal")).read()
    print(f"  {'N_ROWS':>7} {'SEQ':>4} {'ratio':>6} {'ms':>8} {'speedup':>8}  mismatches")
    for nr, seq in [(1, 8), (4, 4), (8, 4), (16, 4), (4, 8), (8, 8)]:
        if S % seq or OUT_F % nr:
            continue
        key = f"db_{nr}_{seq}"
        try:
            g._compile_one(src0.replace("N_ROWS=8u, SEQ=4u", f"N_ROWS={nr}u, SEQ={seq}u"), "down_96_b", key)
        except RuntimeError as e:
            print(f"  {nr:>7} {seq:>4}  COMPILE FAIL: {str(e)[:50]}"); continue
        ba = [(a, 0), (bits, 0), (scale, 0), (yb, 0), (par, 0)]
        run = lambda: (lambda b: (b.dg2d(key, ba, OUT_F // nr, S // seq), b.commit_wait()))(Batch(g))
        run()
        got = np.frombuffer(yb.contents().as_buffer(S * OUT_F * 4), dtype=np.float32).reshape(S, OUT_F)
        ms = _time(run)
        r = 16 * nr * seq / (nr + 4 * seq)
        print(f"  {nr:>7} {seq:>4} {r:>6.1f} {ms:>8.3f} {ms0 / ms:>7.2f}x  {int((got != ref).sum())}")
    ms1 = _time(base_unfair if unfair else base)
    print(f"  baseline re-measured after sweep: {ms1:.3f} ms  (drift {ms1 / ms0 - 1:+.1%})")


def bench_gateup(g, S, unfair=False):
    dev = g.dev
    W = lambda n: g.w[f"L{L}.{n}"]
    par = dev.newBufferWithBytes_length_options_(
        struct.pack("<fffI", g.sc(L, "gate_out"), g.sc(L, "up_out"), g.sc(L, "down_in"), 0), 16, _SHARED)
    random.seed(11)
    hv = np.array([float(random.randint(-128, 127)) for _ in range(S * H)], dtype=np.float16)
    hid = dev.newBufferWithBytes_length_options_(hv.tobytes(), S * H * 2, _SHARED)
    sa = np.array([hv[s * H:(s + 1) * H].astype(np.float32).sum() for s in range(S)], dtype=np.float32)
    suma = dev.newBufferWithBytes_length_options_(sa.tobytes(), S * 4, _SHARED)
    out1 = dev.newBufferWithLength_options_(INTER * 2, _SHARED)
    outB = dev.newBufferWithLength_options_(S * INTER * 2, _SHARED)
    args = lambda s: [(hid, s * H * 2), (W("gate_bits"), 0), (W("gate_scale"), 0), (W("up_bits"), 0),
                      (W("up_scale"), 0), (suma, s * 4), (out1, 0), (W("gelu_gate"), 0), (par, 0)]

    ref = []
    for s in range(S):
        b = Batch(g); b.dg("gateup_95", args(s), 3072); b.commit_wait()
        ref.append(np.frombuffer(out1.contents().as_buffer(INTER * 2), dtype=np.float16).copy())
    ref = np.stack(ref)

    def base():
        b = Batch(g)
        for s in range(S):
            b.dg("gateup_95", args(s), 3072)
        b.commit_wait()

    def base_unfair():
        for s in range(S):
            b = Batch(g); b.dg("gateup_95", args(s), 3072); b.commit_wait()

    ms0 = _time(base_unfair if unfair else base)
    print(f"\ngate/up (2-bit, {H}->2x{INTER}, 56% of prefill FLOPs)   S={S}")
    print(f"  baseline: S x gateup_95 = {ms0:.3f} ms" + ("   [UNFAIR: per-token command buffers]" if unfair else ""))
    src0 = open(os.path.join(KDIR, "gateup_95_b.metal")).read()
    print(f"  {'N_ROWS':>7} {'SEQ':>4} {'ratio':>6} {'ms':>8} {'speedup':>8}  mismatches")
    for nr, seq in [(1, 8), (2, 4), (4, 2), (4, 4), (8, 4), (2, 8), (4, 8)]:
        if S % seq or INTER % (2 * nr):
            continue
        key = f"gu_{nr}_{seq}"
        try:
            g._compile_one(src0.replace("N_ROWS=2u, SEQ=4u", f"N_ROWS={nr}u, SEQ={seq}u"), "gateup_95_b", key)
        except RuntimeError as e:
            print(f"  {nr:>7} {seq:>4}  COMPILE FAIL: {str(e)[:50]}"); continue
        ba = [(hid, 0), (W("gate_bits"), 0), (W("gate_scale"), 0), (W("up_bits"), 0), (W("up_scale"), 0),
              (suma, 0), (outB, 0), (W("gelu_gate"), 0), (par, 0)]
        gx, gy = INTER // (2 * nr), S // seq
        run = lambda: (lambda b: (b.dg2d(key, ba, gx, gy), b.commit_wait()))(Batch(g))
        run()
        got = np.frombuffer(outB.contents().as_buffer(S * INTER * 2), dtype=np.float16).reshape(S, INTER)
        ms = _time(run)
        r = 32 * nr * seq / (2 * nr + 4 * seq)
        print(f"  {nr:>7} {seq:>4} {r:>6.1f} {ms:>8.3f} {ms0 / ms:>7.2f}x  {int((got != ref).sum())}")
    ms1 = _time(base_unfair if unfair else base)
    print(f"  baseline re-measured after sweep: {ms1:.3f} ms  (drift {ms1 / ms0 - 1:+.1%})")


def main():
    argv = [a for a in sys.argv[1:] if a != "--unfair"]
    unfair = "--unfair" in sys.argv
    S = int(argv[0]) if argv else 8
    print("loading model + compiling kernels…")
    g = G4()
    print("warming GPU clock…")
    warmup(g)
    bench_gateup(g, S, unfair)
    bench_down(g, S, unfair)


if __name__ == "__main__":
    main()
