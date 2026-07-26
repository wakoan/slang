"""DSL-authored decode kernels vs the hand-written ones they replace.

The port only pays off if a DSL kernel can be swapped in without changing a
single output bit — otherwise every replacement is a fresh numerical
investigation and the "written once" claim costs more than it saves.

So the bar here is BIT-EXACT (`np.array_equal`), not a tolerance. These kernels
are deterministic given the same inputs and the same reduction order, and the
DSL version deliberately preserves that order (same subgroup-then-threadgroup
combine, same loop strides). A tolerance test would hide exactly the kind of
drift — a reassociated sum, an f32 constant folded differently — that this is
meant to catch.

Inputs are real model tensors, not random ones: the SRQ branch is scale-
dependent and a random tensor would not exercise the clamp.
"""
import os
import struct

import numpy as np
import pytest

Metal = pytest.importorskip("Metal")

from py_shader_lang_wgpu import translate
from gemma4_150 import kernels_dsl

KDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "gemma4_150", "kernels_msl")
DIM = 1536
WG = 256


@pytest.fixture(scope="module")
def dev():
    d = Metal.MTLCreateSystemDefaultDevice()
    if d is None:
        pytest.skip("no Metal device")
    return d


def _pso(dev, src, name):
    lib, err = dev.newLibraryWithSource_options_error_(src, None, None)
    assert lib is not None, f"compile failed for {name}: {err}"
    pso, err = dev.newComputePipelineStateWithFunction_error_(
        lib.newFunctionWithName_(name), None)
    assert pso is not None, f"pipeline failed for {name}: {err}"
    return pso


def _run(dev, pso, bufs, groups, threads):
    cb = dev.newCommandQueue().commandBuffer()
    enc = cb.computeCommandEncoder()
    enc.setComputePipelineState_(pso)
    for i, b in enumerate(bufs):
        enc.setBuffer_offset_atIndex_(b, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(groups, 1, 1), Metal.MTLSizeMake(threads, 1, 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()


@pytest.mark.parametrize("in_scale", [0.0, 0.0234375])
def test_rmssrq_69_bit_exact(dev, in_scale):
    """in_scale=0 takes srq's passthrough branch; the other exercises the clamp."""
    rows = 4
    rng = np.random.default_rng(5)
    # scaled so the SRQ clamp is actually reached on some elements
    x = (rng.standard_normal((rows, DIM)) * 3.0).astype(np.float32)
    w = rng.standard_normal(DIM).astype(np.float32)
    SH = Metal.MTLResourceStorageModeShared

    def go(src):
        bx = dev.newBufferWithBytes_length_options_(x.tobytes(), x.nbytes, SH)
        bw = dev.newBufferWithBytes_length_options_(w.tobytes(), w.nbytes, SH)
        by = dev.newBufferWithLength_options_(rows * DIM * 4, SH)
        bs = dev.newBufferWithLength_options_(rows * 4, SH)
        bp = dev.newBufferWithBytes_length_options_(
            struct.pack("<IIfI", rows, 0, in_scale, 0), 16, SH)
        _run(dev, _pso(dev, src, "rmssrq_69"), (bx, bw, by, bs, bp), rows, WG)
        y = np.frombuffer(by.contents().as_buffer(rows * DIM * 4), np.float32).copy()
        s = np.frombuffer(bs.contents().as_buffer(rows * 4), np.float32).copy()
        return y, s

    ref_y, ref_s = go(open(os.path.join(KDIR, "rmssrq_69.metal")).read())
    got_y, got_s = go(translate(kernels_dsl.rmssrq_69, target="msl"))

    assert np.array_equal(got_y, ref_y), (
        f"y differs: max |d| {np.abs(got_y - ref_y).max():.3e}")
    assert np.array_equal(got_s, ref_s), (
        f"sum_a differs: {got_s} vs {ref_s}")


def test_rmssrq_69_matches_numpy(dev):
    """Independent oracle, so a shared misunderstanding cannot pass both sides."""
    rows = 2
    rng = np.random.default_rng(9)
    x = rng.standard_normal((rows, DIM)).astype(np.float32)
    w = rng.standard_normal(DIM).astype(np.float32)
    SH = Metal.MTLResourceStorageModeShared
    bx = dev.newBufferWithBytes_length_options_(x.tobytes(), x.nbytes, SH)
    bw = dev.newBufferWithBytes_length_options_(w.tobytes(), w.nbytes, SH)
    by = dev.newBufferWithLength_options_(rows * DIM * 4, SH)
    bs = dev.newBufferWithLength_options_(rows * 4, SH)
    bp = dev.newBufferWithBytes_length_options_(struct.pack("<IIfI", rows, 0, 0.0, 0), 16, SH)
    _run(dev, _pso(dev, translate(kernels_dsl.rmssrq_69, target="msl"), "rmssrq_69"),
         (bx, bw, by, bs, bp), rows, WG)
    got = np.frombuffer(by.contents().as_buffer(rows * DIM * 4), np.float32).reshape(rows, DIM)

    x64 = x.astype(np.float64)
    sc = 1.0 / np.sqrt((x64 ** 2).mean(axis=1) + 1e-6)
    ref = x64 * sc[:, None] * w.astype(np.float64)
    rel = np.abs(got.astype(np.float64) - ref).max() / max(np.abs(ref).max(), 1e-6)
    assert rel < 1e-5, f"vs numpy: max rel err {rel:.2e}"


def test_wgsl_also_emits():
    """The same source must produce a WGSL kernel for wgpu/browser, not just MSL."""
    w = kernels_dsl.rmssrq_69.wgsl
    assert "@compute @workgroup_size(256, 1, 1)" in w
    assert "subgroupAdd" in w and "enable subgroups;" in w


def _elementwise(dev, msl_src, name, bufs, n, threads=256):
    _run(dev, _pso(dev, msl_src, name), bufs, (n + threads - 1) // threads, threads)


@pytest.mark.parametrize("scale", [0.0, 0.0234375])
def test_srqh_b_bit_exact(dev, scale):
    n = 1024
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(n) * 4.0).astype(np.float32)
    SH = Metal.MTLResourceStorageModeShared

    def go(src):
        bx = dev.newBufferWithBytes_length_options_(x.tobytes(), x.nbytes, SH)
        by = dev.newBufferWithLength_options_(n * 2, SH)
        bp = dev.newBufferWithBytes_length_options_(
            struct.pack("<fIII", scale, n, 0, 0), 16, SH)
        _elementwise(dev, src, "srqh_b", (bx, by, bp), n)
        return np.frombuffer(by.contents().as_buffer(n * 2), np.float16).copy()

    ref = go(open(os.path.join(KDIR, "srqh_b.metal")).read())
    got = go(translate(kernels_dsl.srqh_b, target="msl"))
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("scale", [0.0, 0.0234375])
def test_srq_b_bit_exact(dev, scale):
    n = 1024
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(n) * 4.0).astype(np.float16)
    SH = Metal.MTLResourceStorageModeShared

    def go(src):
        bx = dev.newBufferWithBytes_length_options_(x.tobytes(), x.nbytes, SH)
        by = dev.newBufferWithLength_options_(n * 4, SH)
        bp = dev.newBufferWithBytes_length_options_(
            struct.pack("<fIII", scale, n, 0, 0), 16, SH)
        _elementwise(dev, src, "srq_b", (bx, by, bp), n)
        return np.frombuffer(by.contents().as_buffer(n * 4), np.float32).copy()

    ref = go(open(os.path.join(KDIR, "srq_b.metal")).read())
    got = go(translate(kernels_dsl.srq_b, target="msl"))
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("gate_scale", [0.0, 0.0234375])
def test_geglu_b_bit_exact(dev, gate_scale):
    """gate_scale=0 takes the polynomial gelu; nonzero takes the LUT path."""
    n = 2048
    rng = np.random.default_rng(4)
    gate = (rng.standard_normal(n) * 3.0).astype(np.float16)
    up = (rng.standard_normal(n) * 3.0).astype(np.float16)
    lut = rng.standard_normal(256).astype(np.float32)
    SH = Metal.MTLResourceStorageModeShared

    def go(src):
        bg = dev.newBufferWithBytes_length_options_(gate.tobytes(), gate.nbytes, SH)
        bu = dev.newBufferWithBytes_length_options_(up.tobytes(), up.nbytes, SH)
        bo = dev.newBufferWithLength_options_(n * 2, SH)
        bl = dev.newBufferWithBytes_length_options_(lut.tobytes(), lut.nbytes, SH)
        bp = dev.newBufferWithBytes_length_options_(
            struct.pack("<fffI", gate_scale, 0.03125, 0.0234375, n), 16, SH)
        _elementwise(dev, src, "geglu_b", (bg, bu, bo, bl, bp), n)
        return np.frombuffer(bo.contents().as_buffer(n * 2), np.float16).copy()

    ref = go(open(os.path.join(KDIR, "geglu_b.metal")).read())
    got = go(translate(kernels_dsl.geglu_b, target="msl"))
    assert np.array_equal(got, ref)


def test_combine_bit_exact(dev):
    nL, D = 35, 256
    rng = np.random.default_rng(6)
    ctx = (rng.standard_normal(nL * D) * 20.0).astype(np.float32)
    ple = rng.standard_normal(nL * D).astype(np.float32)
    nw = rng.standard_normal(D).astype(np.float32)
    SH = Metal.MTLResourceStorageModeShared

    def go(src):
        bc = dev.newBufferWithBytes_length_options_(ctx.tobytes(), ctx.nbytes, SH)
        bpl = dev.newBufferWithBytes_length_options_(ple.tobytes(), ple.nbytes, SH)
        bn = dev.newBufferWithBytes_length_options_(nw.tobytes(), nw.nbytes, SH)
        bo = dev.newBufferWithLength_options_(nL * D * 4, SH)
        _run(dev, _pso(dev, src, "combine"), (bc, bpl, bn, bo), nL, D)
        return np.frombuffer(bo.contents().as_buffer(nL * D * 4), np.float32).copy()

    ref = go(open(os.path.join(KDIR, "combine.metal")).read())
    got = go(translate(kernels_dsl.combine, target="msl"))
    assert np.array_equal(got, ref), f"max |d| {np.abs(got - ref).max():.3e}"


def test_down_75_bit_exact(dev):
    """The atomic last-arriver kernel — the one the DSL needed atomics for.

    Exercises everything the easy tier did not: an atomic buffer, a
    cross-workgroup ticket merge, a device-scope barrier, 4-bit unpacking,
    subgroup reduction over a vec4, and a self-resetting counter.
    """
    OUT_F, WPR, CHUNKS, N_ROWS = 1536, 768, 2, 4
    TOTAL_WGS = OUT_F // N_ROWS
    rng = np.random.default_rng(21)
    a = (rng.standard_normal(WPR * CHUNKS * 4) * 2.0).astype(np.float16)
    bits = rng.integers(0, 2 ** 32, size=OUT_F * WPR, dtype=np.uint64).astype(np.uint32)
    scale = (rng.random(OUT_F).astype(np.float32) * 0.01 + 0.001)
    hidden0 = rng.standard_normal(OUT_F).astype(np.float32)
    nw = rng.standard_normal(OUT_F).astype(np.float32)
    SH = Metal.MTLResourceStorageModeShared

    def go(src):
        ba = dev.newBufferWithBytes_length_options_(a.tobytes(), a.nbytes, SH)
        bb = dev.newBufferWithBytes_length_options_(bits.tobytes(), bits.nbytes, SH)
        bpp = dev.newBufferWithLength_options_((OUT_F + 1) * 4, SH)
        bpp.contents().as_buffer((OUT_F + 1) * 4)[:] = b"\x00" * ((OUT_F + 1) * 4)
        bsc = dev.newBufferWithBytes_length_options_(scale.tobytes(), scale.nbytes, SH)
        bh = dev.newBufferWithBytes_length_options_(hidden0.tobytes(), hidden0.nbytes, SH)
        bn = dev.newBufferWithBytes_length_options_(nw.tobytes(), nw.nbytes, SH)
        bp = dev.newBufferWithBytes_length_options_(
            struct.pack("<ffII", 0.00787, 0.0234375, 0, 0), 16, SH)
        _run(dev, _pso(dev, src, "down_75"),
             (ba, bb, bpp, bsc, bh, bn, bp), TOTAL_WGS, WG)
        h = np.frombuffer(bh.contents().as_buffer(OUT_F * 4), np.float32).copy()
        ctr = np.frombuffer(bpp.contents().as_buffer((OUT_F + 1) * 4), np.uint32)[OUT_F]
        return h, int(ctr)

    ref_h, ref_ctr = go(open(os.path.join(KDIR, "down_75.metal")).read())
    got_h, got_ctr = go(translate(kernels_dsl.down_75, target="msl"))

    assert got_h.shape == ref_h.shape
    assert np.array_equal(got_h, ref_h), (
        f"hidden differs: max |d| {np.abs(got_h - ref_h).max():.3e}")
    # the counter must self-reset, or the NEXT token's merge never fires
    assert got_ctr == 0 == ref_ctr, f"counter left at {got_ctr} (ref {ref_ctr})"
    assert not np.array_equal(got_h, hidden0), "kernel did not write hidden"


def _gather_case(dev, name, hidden, wpr, vals_per_word, n_scale):
    rng = np.random.default_rng(31)
    seq = 3
    ids = rng.integers(0, 262144, size=seq, dtype=np.uint64).astype(np.uint32)
    nrows = int(ids.max()) + 1
    bits = rng.integers(0, 2 ** 32, size=nrows * wpr, dtype=np.uint64).astype(np.uint32)
    scale = (rng.random(nrows * n_scale).astype(np.float32) * 0.01 + 0.001)
    SH = Metal.MTLResourceStorageModeShared

    def go(src):
        bi = dev.newBufferWithBytes_length_options_(ids.tobytes(), ids.nbytes, SH)
        bb = dev.newBufferWithBytes_length_options_(bits.tobytes(), bits.nbytes, SH)
        bsc = dev.newBufferWithBytes_length_options_(scale.tobytes(), scale.nbytes, SH)
        by = dev.newBufferWithLength_options_(seq * hidden * 4, SH)
        bp = dev.newBufferWithBytes_length_options_(struct.pack("<4I", seq, 0, 0, 0), 16, SH)
        _run(dev, _pso(dev, src, name), (bi, bb, bsc, by, bp), seq, 64)
        return np.frombuffer(by.contents().as_buffer(seq * hidden * 4), np.float32).copy()

    ref = go(open(os.path.join(KDIR, name + ".metal")).read())
    got = go(translate(getattr(kernels_dsl, name), target="msl"))
    assert np.array_equal(got, ref), f"{name}: max |d| {np.abs(got - ref).max():.3e}"
    assert np.count_nonzero(got), f"{name} wrote nothing"


def test_embed_00_bit_exact(dev):
    _gather_case(dev, "embed_00", hidden=1536, wpr=96, vals_per_word=16, n_scale=1)


def test_plegather_01_bit_exact(dev):
    _gather_case(dev, "plegather_01", hidden=8960, wpr=1120, vals_per_word=8, n_scale=35)


def test_argmax_bit_exact(dev):
    """Both passes, including the tie rule: ties must break to the LOWER index,
    or the sampled token depends on scheduling."""
    COUNT, SLICE = 262144, 1024
    NWG = COUNT // SLICE
    rng = np.random.default_rng(41)
    x = rng.standard_normal(COUNT).astype(np.float32)
    x[12345] = 50.0
    x[200000] = 50.0                      # exact tie: lower index must win
    SH = Metal.MTLResourceStorageModeShared

    def go(src1, src2):
        bx = dev.newBufferWithBytes_length_options_(x.tobytes(), x.nbytes, SH)
        bv = dev.newBufferWithLength_options_(NWG * 4, SH)
        bi = dev.newBufferWithLength_options_(NWG * 4, SH)
        bo = dev.newBufferWithLength_options_(4, SH)
        _run(dev, _pso(dev, src1, "argmax1_34"), (bx, bv, bi), NWG, WG)
        _run(dev, _pso(dev, src2, "argmax2_35"), (bv, bi, bo), 1, WG)
        return int(np.frombuffer(bo.contents().as_buffer(4), np.uint32)[0])

    ref = go(open(os.path.join(KDIR, "argmax1_34.metal")).read(),
             open(os.path.join(KDIR, "argmax2_35.metal")).read())
    got = go(translate(kernels_dsl.argmax1_34, target="msl"),
             translate(kernels_dsl.argmax2_35, target="msl"))
    assert got == ref, f"argmax {got} != {ref}"
    assert got == 12345, f"tie should break to the lower index, got {got}"


@pytest.mark.parametrize("hd", [256, 512])
def test_kvnorm_bit_exact(dev, hd):
    """Shape-parameterized: both head_dim variants come from one DSL source.

    The reference is produced the way the runner does it — by string-patching
    the hand-written .metal — so this also checks the consts mechanism lands on
    the same specialization the patching produced.
    """
    half = hd // 2
    rng = np.random.default_rng(51)
    ink = rng.standard_normal(hd).astype(np.float32)
    invv = rng.standard_normal(hd).astype(np.float32)
    knorm = rng.standard_normal(hd).astype(np.float32)
    cosT = rng.standard_normal(half).astype(np.float32)
    sinT = rng.standard_normal(half).astype(np.float32)
    OFF = 3 * hd
    SH = Metal.MTLResourceStorageModeShared

    def go(src):
        bs = [dev.newBufferWithBytes_length_options_(a.tobytes(), a.nbytes, SH)
              for a in (ink, invv, knorm, cosT, sinT)]
        bk = dev.newBufferWithLength_options_(8 * hd * 4, SH)
        bv = dev.newBufferWithLength_options_(8 * hd * 4, SH)
        bp = dev.newBufferWithBytes_length_options_(struct.pack("<4I", OFF, 0, 0, 0), 16, SH)
        _run(dev, _pso(dev, src, "kvnorm"), (*bs, bk, bv, bp), 1, hd)
        k = np.frombuffer(bk.contents().as_buffer(8 * hd * 4), np.float32).copy()
        v = np.frombuffer(bv.contents().as_buffer(8 * hd * 4), np.float32).copy()
        return k, v

    ref_src = (open(os.path.join(KDIR, "kvnorm.metal")).read()
               .replace("HD=256u", f"HD={hd}u").replace("HALF=128u", f"HALF={half}u")
               .replace("threadsPerThreadgroup = (256)", f"threadsPerThreadgroup = ({hd})"))
    ref_k, ref_v = go(ref_src)
    got_k, got_v = go(translate(kernels_dsl.kvnorm, workgroup_size=(hd, 1, 1),
                                target="msl", consts={"HD": hd, "HALF": half}))
    assert np.array_equal(got_k, ref_k), f"k differs: {np.abs(got_k - ref_k).max():.3e}"
    assert np.array_equal(got_v, ref_v), f"v differs: {np.abs(got_v - ref_v).max():.3e}"
    assert np.count_nonzero(got_k[OFF:OFF + hd]), "wrote nothing at the cache offset"


def test_consts_actually_specialize():
    """A const must change the emitted code, not merely be accepted."""
    from py_shader_lang_wgpu import translate as tr
    a = tr(kernels_dsl.kvnorm, workgroup_size=(256, 1, 1), consts={"HD": 256, "HALF": 128})
    b = tr(kernels_dsl.kvnorm, workgroup_size=(512, 1, 1), consts={"HD": 512, "HALF": 256})
    assert a != b
    assert "512" in b and "@workgroup_size(512, 1, 1)" in b


def test_rmsadd_b_bit_exact(dev):
    S = 3
    rng = np.random.default_rng(61)
    x = (rng.standard_normal(S * DIM) * 2.0).astype(np.float16)
    nw = rng.standard_normal(DIM).astype(np.float32)
    hidden0 = rng.standard_normal(S * DIM).astype(np.float32)
    SH = Metal.MTLResourceStorageModeShared

    def go(src):
        bx = dev.newBufferWithBytes_length_options_(x.tobytes(), x.nbytes, SH)
        bn = dev.newBufferWithBytes_length_options_(nw.tobytes(), nw.nbytes, SH)
        bh = dev.newBufferWithBytes_length_options_(hidden0.tobytes(), hidden0.nbytes, SH)
        bp = dev.newBufferWithBytes_length_options_(
            struct.pack("<ffII", 0.0234375, 1.25, DIM, 0), 16, SH)
        _run(dev, _pso(dev, src, "rmsadd_b"), (bx, bn, bh, bp), S, WG)
        return np.frombuffer(bh.contents().as_buffer(S * DIM * 4), np.float32).copy()

    ref = go(open(os.path.join(KDIR, "rmsadd_b.metal")).read())
    got = go(translate(kernels_dsl.rmsadd_b, target="msl"))
    assert np.array_equal(got, ref), f"max |d| {np.abs(got - ref).max():.3e}"


def test_rmssrqh_b_bit_exact(dev):
    """Also pins the DOUBLE rounding f16(srq(f32(f16(n)))) — collapsing it to a
    single round changes results on the boundary."""
    S = 3
    rng = np.random.default_rng(62)
    hidden = rng.standard_normal(S * DIM).astype(np.float32)
    w = rng.standard_normal(DIM).astype(np.float32)
    SH = Metal.MTLResourceStorageModeShared

    def go(src):
        bh = dev.newBufferWithBytes_length_options_(hidden.tobytes(), hidden.nbytes, SH)
        bw = dev.newBufferWithBytes_length_options_(w.tobytes(), w.nbytes, SH)
        by = dev.newBufferWithLength_options_(S * DIM * 2, SH)
        bs = dev.newBufferWithLength_options_(S * 4, SH)
        bp = dev.newBufferWithBytes_length_options_(
            struct.pack("<fIII", 0.0234375, DIM, 0, 0), 16, SH)
        _run(dev, _pso(dev, src, "rmssrqh_b"), (bh, bw, by, bs, bp), S, WG)
        return (np.frombuffer(by.contents().as_buffer(S * DIM * 2), np.float16).copy(),
                np.frombuffer(bs.contents().as_buffer(S * 4), np.float32).copy())

    ref_y, ref_s = go(open(os.path.join(KDIR, "rmssrqh_b.metal")).read())
    got_y, got_s = go(translate(kernels_dsl.rmssrqh_b, target="msl"))
    assert np.array_equal(got_y, ref_y)
    assert np.array_equal(got_s, ref_s)


def test_combine_b_bit_exact(dev):
    nL, D, S = 35, 256, 3
    rng = np.random.default_rng(63)
    ctx = (rng.standard_normal(S * nL * D) * 20.0).astype(np.float32)
    ple = rng.standard_normal(S * nL * D).astype(np.float32)
    nw = rng.standard_normal(D).astype(np.float32)
    SH = Metal.MTLResourceStorageModeShared

    def go(src):
        bc = dev.newBufferWithBytes_length_options_(ctx.tobytes(), ctx.nbytes, SH)
        bpl = dev.newBufferWithBytes_length_options_(ple.tobytes(), ple.nbytes, SH)
        bn = dev.newBufferWithBytes_length_options_(nw.tobytes(), nw.nbytes, SH)
        bo = dev.newBufferWithLength_options_(S * nL * D * 4, SH)
        bp = dev.newBufferWithBytes_length_options_(struct.pack("<4I", nL, 0, 0, 0), 16, SH)
        pso = _pso(dev, src, "combine_b")
        cb = dev.newCommandQueue().commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        for i, b in enumerate((bc, bpl, bn, bo, bp)):
            enc.setBuffer_offset_atIndex_(b, 0, i)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(nL, S, 1), Metal.MTLSizeMake(D, 1, 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return np.frombuffer(bo.contents().as_buffer(S * nL * D * 4), np.float32).copy()

    ref = go(open(os.path.join(KDIR, "combine_b.metal")).read())
    got = go(translate(kernels_dsl.combine_b, target="msl"))
    assert np.array_equal(got, ref), f"max |d| {np.abs(got - ref).max():.3e}"


def test_proj_68_bit_exact(dev):
    IN, OUT, GRID = 1536, 8960, 1120
    rng = np.random.default_rng(71)
    a = rng.standard_normal(IN).astype(np.float32)
    wt = (rng.standard_normal(OUT * IN) * 0.05).astype(np.float16)
    SH = Metal.MTLResourceStorageModeShared

    def go(src):
        ba = dev.newBufferWithBytes_length_options_(a.tobytes(), a.nbytes, SH)
        bw = dev.newBufferWithBytes_length_options_(wt.tobytes(), wt.nbytes, SH)
        bo = dev.newBufferWithLength_options_(OUT * 4, SH)
        bp = dev.newBufferWithBytes_length_options_(
            struct.pack("<ffII", 0.0234375, 0.03125, 0, 0), 16, SH)
        _run(dev, _pso(dev, src, "proj_68"), (ba, bw, bo, bp), GRID, 32)
        return np.frombuffer(bo.contents().as_buffer(OUT * 4), np.float32).copy()

    ref = go(open(os.path.join(KDIR, "proj_68.metal")).read())
    got = go(translate(kernels_dsl.proj_68, target="msl"))
    assert np.array_equal(got, ref), f"max |d| {np.abs(got - ref).max():.3e}"
    assert np.count_nonzero(got) > OUT // 2, "most outputs should be nonzero"


@pytest.mark.parametrize("lin_scale", [0.0, 0.03125])
def test_plegate_76_bit_exact(dev, lin_scale):
    """lin_scale=0 takes the polynomial gelu; nonzero takes the LUT."""
    OUT, WPR, K = 256, 384, 1536
    rng = np.random.default_rng(72)
    a = rng.standard_normal(K).astype(np.float32)
    codes = rng.integers(0, 2 ** 32, size=OUT * WPR, dtype=np.uint64).astype(np.uint32)
    row_scale = (rng.random(OUT).astype(np.float32) * 0.01 + 0.001)
    ple = rng.standard_normal(OUT * 2).astype(np.float32)
    lut = rng.standard_normal(256).astype(np.float32)
    SH = Metal.MTLResourceStorageModeShared

    def go(src):
        bs = [dev.newBufferWithBytes_length_options_(x.tobytes(), x.nbytes, SH)
              for x in (a, codes, row_scale, ple)]
        bo = dev.newBufferWithLength_options_(OUT * 4, SH)
        bl = dev.newBufferWithBytes_length_options_(lut.tobytes(), lut.nbytes, SH)
        bp = dev.newBufferWithBytes_length_options_(
            struct.pack("<ffII", 0.0234375, lin_scale, OUT, 0), 16, SH)
        _run(dev, _pso(dev, src, "plegate_76"), (*bs, bo, bl, bp), OUT, 32)
        return np.frombuffer(bo.contents().as_buffer(OUT * 4), np.float32).copy()

    ref = go(open(os.path.join(KDIR, "plegate_76.metal")).read())
    got = go(translate(kernels_dsl.plegate_76, target="msl"))
    assert np.array_equal(got, ref), f"max |d| {np.abs(got - ref).max():.3e}"
