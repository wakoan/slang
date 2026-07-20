"""QAT (int2/4/8) GPU kernels for the Gemma 4 E2B mobile checkpoint.

The speed win: dequant matmuls read packed sub-byte weights (0.25-0.5
byte/weight vs 2 for f16), so the bandwidth-bound decode gets ~4-8x less
weight traffic. The per-output-row scale is symmetric, so it factors out
of the dot product — the hot loop is an integer dot, one scale multiply
at the end. int4/int2 unpack matches transformers gemma_quant.py
(low-bits-first, -8 / -2 offset).

Everything non-matmul (norms, rope, attention, geglu, argmax, PLE
elementwise) is reused from gemma4.kernels / gemma3.kernels; 8-bit and
unquantized modules dequant to f16 on load and use the base f16 matvec.
"""

from __future__ import annotations

from py_shader_lang_wgpu import kernel
from py_shader_lang_wgpu.types import (
    Builtin,
    StorageBuffer,
    WorkgroupArray,
    f32,
    i32,
    u32,
)


@kernel(workgroup_size=(64,))
def matvec_dq4(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    w_packed: StorageBuffer[u32, "read"],     # [n_out, n_in/8] (8 int4/u32)
    x_in: StorageBuffer[f32, "read"],         # [n_in]
    scale: StorageBuffer[f32, "read"],        # [n_out] per-row weight scale
    y_out: StorageBuffer[f32, "read_write"],  # [n_out]
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in] (n_in % 8 == 0)
    partial: WorkgroupArray[f32, 64],
):
    # int4: low nibble first, value = (nibble) - 8. Scale factored out.
    r: u32 = wid.x
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    n8: u32 = n_in / 8

    acc: f32 = 0.0
    if r < n_out:
        for j in range(li, n8, 64):
            p: u32 = w_packed[r * n8 + j]
            base: u32 = 8 * j
            b0: u32 = p & 255
            acc += f32(i32(b0 & 15) - 8) * x_in[base]
            acc += f32(i32(b0 >> 4) - 8) * x_in[base + 1]
            b1: u32 = (p >> 8) & 255
            acc += f32(i32(b1 & 15) - 8) * x_in[base + 2]
            acc += f32(i32(b1 >> 4) - 8) * x_in[base + 3]
            b2: u32 = (p >> 16) & 255
            acc += f32(i32(b2 & 15) - 8) * x_in[base + 4]
            acc += f32(i32(b2 >> 4) - 8) * x_in[base + 5]
            b3: u32 = (p >> 24) & 255
            acc += f32(i32(b3 & 15) - 8) * x_in[base + 6]
            acc += f32(i32(b3 >> 4) - 8) * x_in[base + 7]
    partial[li] = acc
    barrier()
    s: u32 = 32
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2
    if li == 0 and r < n_out:
        y_out[r] = partial[0] * scale[r]


@kernel(workgroup_size=(64,))
def matvec_dq2(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    w_packed: StorageBuffer[u32, "read"],     # [n_out, n_in/16] (16 int2/u32)
    x_in: StorageBuffer[f32, "read"],         # [n_in]
    scale: StorageBuffer[f32, "read"],        # [n_out] per-row weight scale
    y_out: StorageBuffer[f32, "read_write"],  # [n_out]
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in] (n_in % 16 == 0)
    partial: WorkgroupArray[f32, 64],
):
    # int2: 16 values/u32, value m = ((p >> 2m) & 3) - 2. Scale factored out.
    r: u32 = wid.x
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    n16: u32 = n_in / 16

    acc: f32 = 0.0
    if r < n_out:
        for j in range(li, n16, 64):
            p: u32 = w_packed[r * n16 + j]
            base: u32 = 16 * j
            for m in range(16):
                v: i32 = i32((p >> (2 * m)) & 3) - 2
                acc += f32(v) * x_in[base + m]
    partial[li] = acc
    barrier()
    s: u32 = 32
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2
    if li == 0 and r < n_out:
        y_out[r] = partial[0] * scale[r]
