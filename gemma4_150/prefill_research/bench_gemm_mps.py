"""Does dequantize-to-f16 + MPS GEMM actually reach LiteRT-class prefill throughput?

    python -m gemma4_150.prefill_research.bench_gemm_mps

Microbenchmark on the dominant prefill shape (2-bit gate/up, n_in=1536,
n_out=12288, 55.3% of prefill FLOPs) before committing to the full
layers-outer/tokens-inner restructure. Measures, at several batch sizes M:

  * one-off dequant cost (amortized once per layer per prompt),
  * MPS f16 GEMM throughput,
  * the same work through the existing per-token GEMV path, for scale.

Correctness is checked against a float64 numpy reconstruction of the QAT weights,
which also validates the dequant kernel's bit layout.

Same methodology rules as bench_batched.py: warm the GPU clock first, min-of-N,
and never time a baseline as one-command-buffer-per-token.
"""
import os, struct, sys, time, random

import numpy as np
import Metal
import MetalPerformanceShaders as mps

from gemma4_150.metal_runner import G4, KDIR, _SHARED, Batch
from gemma4_150.prefill_research.bench_batched import warmup, _time

L = 20
N_IN, N_OUT = 1536, 12288
HERE = os.path.dirname(os.path.abspath(__file__))
F16 = mps.MPSDataTypeFloat16


def _matrix(buf, rows, cols, dtype=F16, esz=2):
    d = mps.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        rows, cols, cols * esz, dtype)
    return mps.MPSMatrix.alloc().initWithBuffer_descriptor_(buf, d)


def main():
    print("loading model + compiling kernels…")
    g = G4()
    dev = g.dev
    g._compile_one(open(os.path.join(HERE, "dq_f16.metal")).read(), "dq_f16", "dq_f16")

    bits, scale = g.w[f"L{L}.gate_bits"], g.w[f"L{L}.gate_scale"]
    wf16 = dev.newBufferWithLength_options_(N_OUT * N_IN * 2, _SHARED)
    pdq = dev.newBufferWithBytes_length_options_(struct.pack("<4I", N_IN, N_OUT, 0, 0), 16, _SHARED)
    nwords = N_OUT * (N_IN // 16)

    def dequant():
        b = Batch(g)
        b.dg("dq_f16", [(bits, 0), (scale, 0), (wf16, 0), (pdq, 0)], nwords // 256)
        b.commit_wait()

    print("warming GPU clock…")
    warmup(g)

    # ---- correctness: dequant kernel vs float64 reconstruction of the QAT weights
    dequant()
    got = np.frombuffer(wf16.contents().as_buffer(N_OUT * N_IN * 2), dtype=np.float16)
    got = got.reshape(N_OUT, N_IN)
    raw = np.frombuffer(bits.contents().as_buffer(nwords * 4), dtype=np.uint32).reshape(N_OUT, -1)
    sc = np.frombuffer(scale.contents().as_buffer(N_OUT * 4), dtype=np.float32)
    sh = np.array([8 * (i // 4) + 2 * (i % 4) for i in range(16)], dtype=np.uint64)
    rows = [0, 1, 777, N_OUT - 1]
    ok = True
    for o in rows:
        q = ((raw[o].astype(np.uint64)[:, None] >> sh[None, :]) & 3).astype(np.float64).reshape(-1)
        ref = (sc[o] * (q - 2.0)).astype(np.float16)
        ok &= np.array_equal(ref, got[o])
    print(f"\ndequant kernel vs float64 QAT reconstruction: {'EXACT' if ok else 'MISMATCH'} "
          f"(rows {rows})")

    ms_dq = _time(dequant, iters=20)
    mb = N_OUT * N_IN * 2 / 1e6
    print(f"dequant gate [{N_OUT}x{N_IN}] -> f16: {ms_dq:.3f} ms  ({mb:.0f} MB written, "
          f"{mb / ms_dq:.0f} GB/s)")

    # ---- MPS GEMM sweep
    print(f"\nMPS f16 GEMM  C[M,{N_OUT}] = A[M,{N_IN}] x W[{N_OUT},{N_IN}]^T")
    print(f"  {'M':>6} {'ms':>9} {'TFLOP/s':>9}   {'max rel err':>11}")
    Bm = _matrix(wf16, N_OUT, N_IN)
    for M in (16, 64, 256, 1024):
        random.seed(3)
        av = np.array([random.randint(-128, 127) for _ in range(M * N_IN)], dtype=np.float16)
        abuf = dev.newBufferWithBytes_length_options_(av.tobytes(), M * N_IN * 2, _SHARED)
        cbuf = dev.newBufferWithLength_options_(M * N_OUT * 2, _SHARED)
        Am, Cm = _matrix(abuf, M, N_IN), _matrix(cbuf, M, N_OUT)
        mm = mps.MPSMatrixMultiplication.alloc()
        mm = mm.initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
            dev, False, True, M, N_OUT, N_IN, 1.0, 0.0)

        def run():
            cb = g.q.commandBuffer()
            mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(cb, Am, Bm, Cm)
            cb.commit(); cb.waitUntilCompleted()

        run()
        C = np.frombuffer(cbuf.contents().as_buffer(M * N_OUT * 2), dtype=np.float16).reshape(M, N_OUT)
        # reference on a few output columns, in float64
        A64 = av.astype(np.float64).reshape(M, N_IN)
        cols = [0, 1234, N_OUT - 1]
        ref = np.stack([A64 @ got[c].astype(np.float64) for c in cols], axis=1)
        rel = np.abs(ref - C[:, cols].astype(np.float64)) / np.maximum(np.abs(ref), 1e-6)
        ms = _time(run, iters=30)
        tf = 2 * M * N_IN * N_OUT / (ms * 1e-3) / 1e12
        print(f"  {M:>6} {ms:>9.3f} {tf:>9.2f}   {rel.max():>11.2e}")

    # ---- for scale: the same tokens through the existing per-token GEMV
    W = lambda n: g.w[f"L{L}.{n}"]
    par = dev.newBufferWithBytes_length_options_(
        struct.pack("<fffI", g.sc(L, "gate_out"), g.sc(L, "up_out"), g.sc(L, "down_in"), 0), 16, _SHARED)
    hid = dev.newBufferWithLength_options_(N_IN * 2, _SHARED)
    suma = dev.newBufferWithLength_options_(4, _SHARED)
    out1 = dev.newBufferWithLength_options_(N_OUT * 2, _SHARED)
    args = [(hid, 0), (W("gate_bits"), 0), (W("gate_scale"), 0), (W("up_bits"), 0),
            (W("up_scale"), 0), (suma, 0), (out1, 0), (W("gelu_gate"), 0), (par, 0)]
    for M in (64, 256):
        def run_gemv():
            b = Batch(g)
            for _ in range(M):
                b.dg("gateup_95", args, 3072)
            b.commit_wait()
        ms = _time(run_gemv, iters=5)
        # gateup_95 does gate AND up, so it is 2x the FLOPs of the single GEMM above
        tf = 2 * 2 * M * N_IN * N_OUT / (ms * 1e-3) / 1e12
        print(f"  [per-token GEMV, M={M}: {ms:.2f} ms for gate+up = {tf:.2f} TFLOP/s]")


if __name__ == "__main__":
    main()
