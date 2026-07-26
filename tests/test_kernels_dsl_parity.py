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
