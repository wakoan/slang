"""GPU tests for the QAT dequant matmul kernels vs numpy. No weights needed."""

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")

from test_gemma3_kernels_gpu import device, run_kernel, u32  # noqa: F401

from gemma4.qat_kernels import (
    matvec_dq2,
    matvec_dq4,
    qat_embed_2bit,
    qat_ple_gather_4bit,
)

rng = np.random.default_rng(7)


def f32a(*v):
    return np.array(v, dtype=np.float32)


def _pack4(ints: np.ndarray) -> np.ndarray:
    """[n_out, n_in] int in [-8,7] -> u32 [n_out, n_in/8] (low nibble first)."""
    u = (ints.astype(np.int16) + 8).astype(np.uint8)
    b = (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)  # [n_out, n_in/2]
    return np.frombuffer(np.ascontiguousarray(b).tobytes(),
                         dtype=np.uint32).reshape(ints.shape[0], ints.shape[1] // 8)


def _pack2(ints: np.ndarray) -> np.ndarray:
    """[n_out, n_in] int in [-2,1] -> u32 [n_out, n_in/16] (low bits first)."""
    u = (ints.astype(np.int16) + 2).astype(np.uint8)  # [0,3]
    b = (u[:, 0::4] | (u[:, 1::4] << 2) | (u[:, 2::4] << 4) | (u[:, 3::4] << 6)).astype(np.uint8)
    return np.frombuffer(np.ascontiguousarray(b).tobytes(),
                         dtype=np.uint32).reshape(ints.shape[0], ints.shape[1] // 16)


class TestMatvecDq4:
    @pytest.mark.parametrize("n_out,n_in", [(512, 1536), (2048, 768 * 2), (256, 256)])
    def test_matches_numpy(self, device, n_out, n_in):
        ints = rng.integers(-8, 8, (n_out, n_in)).astype(np.int8)
        scale = (rng.random(n_out).astype(np.float32) * 0.1 + 0.01)
        x = rng.standard_normal(n_in, dtype=np.float32)
        y = np.zeros(n_out, dtype=np.float32)
        res = run_kernel(
            device, matvec_dq4,
            [(_pack4(ints), False), (x, False), (scale, False), (y, True),
             (u32(n_out, n_in), False)],
            (n_out, 1, 1),
        )
        expected = (ints.astype(np.float32) * scale[:, None]) @ x
        np.testing.assert_allclose(res[3], expected, rtol=1e-4, atol=1e-3)


class TestMatvecDq2:
    @pytest.mark.parametrize("n_out,n_in", [(512, 1536), (1536, 12288), (256, 256)])
    def test_matches_numpy(self, device, n_out, n_in):
        ints = rng.integers(-2, 2, (n_out, n_in)).astype(np.int8)
        scale = (rng.random(n_out).astype(np.float32) * 0.1 + 0.01)
        x = rng.standard_normal(n_in, dtype=np.float32)
        y = np.zeros(n_out, dtype=np.float32)
        res = run_kernel(
            device, matvec_dq2,
            [(_pack2(ints), False), (x, False), (scale, False), (y, True),
             (u32(n_out, n_in), False)],
            (n_out, 1, 1),
        )
        expected = (ints.astype(np.float32) * scale[:, None]) @ x
        np.testing.assert_allclose(res[3], expected, rtol=1e-4, atol=1e-3)


class TestQatEmbed2bit:
    def test_matches_numpy(self, device):
        vocab, hidden = 40, 1536
        ints = rng.integers(-2, 2, (vocab, hidden)).astype(np.int8)
        scale = (rng.random(vocab).astype(np.float32) * 0.1 + 0.01)
        embed_scale = np.float32(np.sqrt(hidden))
        table = _pack2(ints)  # [vocab, hidden/16]
        tok = 17
        out = np.zeros(hidden, dtype=np.float32)
        res = run_kernel(
            device, qat_embed_2bit,
            [(u32(tok), False), (table, False), (scale, False), (out, True),
             (f32a(embed_scale), False), (u32(hidden), False)],
            ((hidden + 63) // 64, 1, 1),
        )
        expected = ints[tok].astype(np.float32) * scale[tok] * embed_scale
        np.testing.assert_allclose(res[3], expected, rtol=1e-5, atol=1e-4)


class TestQatPleGather4bit:
    def test_matches_numpy(self, device):
        vocab, n_layers, ple_h = 30, 35, 256
        n = n_layers * ple_h  # 8960
        ints = rng.integers(-8, 8, (vocab, n)).astype(np.int8)
        scale = (rng.random((vocab, n_layers)).astype(np.float32) * 0.1 + 0.01)
        ple_scale = np.float32(np.sqrt(ple_h))
        table = _pack4(ints)  # [vocab, n/8]
        tok = 11
        out = np.zeros(n, dtype=np.float32)
        res = run_kernel(
            device, qat_ple_gather_4bit,
            [(u32(tok), False), (table, False), (scale, False), (out, True),
             (f32a(ple_scale), False), (u32(n, ple_h, n_layers), False)],
            ((n + 63) // 64, 1, 1),
        )
        per_layer = np.repeat(scale[tok], ple_h)  # [n]
        expected = ints[tok].astype(np.float32) * per_layer * ple_scale
        np.testing.assert_allclose(res[3], expected, rtol=1e-5, atol=1e-4)
