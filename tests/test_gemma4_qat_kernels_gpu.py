"""GPU tests for the QAT dequant matmul kernels vs numpy. No weights needed."""

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")

from test_gemma3_kernels_gpu import device, run_kernel, u32  # noqa: F401

from gemma4.qat_kernels import matvec_dq2, matvec_dq4

rng = np.random.default_rng(7)


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
