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
    vec4,
)

from gemma3.kernels import gelu  # auto-resolved DSL helper (geglu activation)

from . import kernels as K4


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
def qat_embed_2bit(
    gid: Builtin.global_invocation_id,
    token: StorageBuffer[u32, "read"],        # [1] token id
    table: StorageBuffer[u32, "read"],        # [vocab, hidden/16] packed int2
    scale: StorageBuffer[f32, "read"],        # [vocab] per-row scale
    x_out: StorageBuffer[f32, "read_write"],  # [hidden]
    fparams: StorageBuffer[f32, "read"],      # [embed_scale] = sqrt(hidden)
    dims: StorageBuffer[u32, "read"],         # [hidden]
):
    # 2-bit embedding gather: x_out[i] = (crumb - 2) * scale[token] * embed_scale
    i: u32 = gid.x
    hidden: u32 = dims[0]
    if i >= hidden:
        return
    t: u32 = token[0]
    row: u32 = hidden / 16
    word: u32 = table[t * row + i / 16]
    v: i32 = i32((word >> (2 * (i % 16))) & 3) - 2
    x_out[i] = f32(v) * scale[t] * fparams[0]


@kernel(workgroup_size=(64,))
def qat_ple_gather_4bit(
    gid: Builtin.global_invocation_id,
    token: StorageBuffer[u32, "read"],        # [1] token id
    table: StorageBuffer[u32, "read"],        # [vocab, n/8] packed int4
    scale: StorageBuffer[f32, "read"],        # [vocab, n_layers] per-layer scale
    out: StorageBuffer[f32, "read_write"],    # [n] = n_layers*ple_hidden
    fparams: StorageBuffer[f32, "read"],      # [ple_embed_scale] = sqrt(ple_hidden)
    dims: StorageBuffer[u32, "read"],         # [n, ple_hidden, n_layers]
):
    # 4-bit PLE table gather: out[i] = (nibble - 8) * scale[token, layer]
    #                                  * ple_embed_scale, layer = i / ple_hidden
    i: u32 = gid.x
    n: u32 = dims[0]
    ple_hidden: u32 = dims[1]
    n_layers: u32 = dims[2]
    if i >= n:
        return
    t: u32 = token[0]
    row: u32 = n / 8
    word: u32 = table[t * row + i / 8]
    v: i32 = i32((word >> (4 * (i % 8))) & 15) - 8
    layer: u32 = i / ple_hidden
    out[i] = f32(v) * scale[t * n_layers + layer] * fparams[0]


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
            b: u32 = 16 * j
            # unrolled constant shifts — a variable-shift loop is ~2.5x slower
            acc += f32(i32(p & 3) - 2) * x_in[b]
            acc += f32(i32((p >> 2) & 3) - 2) * x_in[b + 1]
            acc += f32(i32((p >> 4) & 3) - 2) * x_in[b + 2]
            acc += f32(i32((p >> 6) & 3) - 2) * x_in[b + 3]
            acc += f32(i32((p >> 8) & 3) - 2) * x_in[b + 4]
            acc += f32(i32((p >> 10) & 3) - 2) * x_in[b + 5]
            acc += f32(i32((p >> 12) & 3) - 2) * x_in[b + 6]
            acc += f32(i32((p >> 14) & 3) - 2) * x_in[b + 7]
            acc += f32(i32((p >> 16) & 3) - 2) * x_in[b + 8]
            acc += f32(i32((p >> 18) & 3) - 2) * x_in[b + 9]
            acc += f32(i32((p >> 20) & 3) - 2) * x_in[b + 10]
            acc += f32(i32((p >> 22) & 3) - 2) * x_in[b + 11]
            acc += f32(i32((p >> 24) & 3) - 2) * x_in[b + 12]
            acc += f32(i32((p >> 26) & 3) - 2) * x_in[b + 13]
            acc += f32(i32((p >> 28) & 3) - 2) * x_in[b + 14]
            acc += f32(i32((p >> 30) & 3) - 2) * x_in[b + 15]
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
def matvec_dq2_blk2(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    w_packed: StorageBuffer[u32, "read"],     # [n_out, n_in/16] (16 int2/u32)
    x_in: StorageBuffer[f32, "read"],         # [n_in]
    scale: StorageBuffer[f32, "read"],        # [n_out] per-row weight scale
    y_out: StorageBuffer[f32, "read_write"],  # [n_out]
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in] (n_out even)
    p0: WorkgroupArray[f32, 64],
    p1: WorkgroupArray[f32, 64],
):
    # Output-blocked matvec: one workgroup computes two adjacent rows, reading
    # each x element once for both weight rows. Tests whether gate+up's win came
    # from amortizing input reads via output blocking (vs from fusing geglu).
    r0: u32 = 2 * wid.x
    r1: u32 = 2 * wid.x + 1
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    n16: u32 = n_in / 16
    a0: f32 = 0.0
    a1: f32 = 0.0
    for j in range(li, n16, 64):
        p: u32 = w_packed[r0 * n16 + j]
        q: u32 = w_packed[r1 * n16 + j]
        b: u32 = 16 * j
        a0 += f32(i32(p & 3) - 2) * x_in[b]
        a1 += f32(i32(q & 3) - 2) * x_in[b]
        a0 += f32(i32((p >> 2) & 3) - 2) * x_in[b + 1]
        a1 += f32(i32((q >> 2) & 3) - 2) * x_in[b + 1]
        a0 += f32(i32((p >> 4) & 3) - 2) * x_in[b + 2]
        a1 += f32(i32((q >> 4) & 3) - 2) * x_in[b + 2]
        a0 += f32(i32((p >> 6) & 3) - 2) * x_in[b + 3]
        a1 += f32(i32((q >> 6) & 3) - 2) * x_in[b + 3]
        a0 += f32(i32((p >> 8) & 3) - 2) * x_in[b + 4]
        a1 += f32(i32((q >> 8) & 3) - 2) * x_in[b + 4]
        a0 += f32(i32((p >> 10) & 3) - 2) * x_in[b + 5]
        a1 += f32(i32((q >> 10) & 3) - 2) * x_in[b + 5]
        a0 += f32(i32((p >> 12) & 3) - 2) * x_in[b + 6]
        a1 += f32(i32((q >> 12) & 3) - 2) * x_in[b + 6]
        a0 += f32(i32((p >> 14) & 3) - 2) * x_in[b + 7]
        a1 += f32(i32((q >> 14) & 3) - 2) * x_in[b + 7]
        a0 += f32(i32((p >> 16) & 3) - 2) * x_in[b + 8]
        a1 += f32(i32((q >> 16) & 3) - 2) * x_in[b + 8]
        a0 += f32(i32((p >> 18) & 3) - 2) * x_in[b + 9]
        a1 += f32(i32((q >> 18) & 3) - 2) * x_in[b + 9]
        a0 += f32(i32((p >> 20) & 3) - 2) * x_in[b + 10]
        a1 += f32(i32((q >> 20) & 3) - 2) * x_in[b + 10]
        a0 += f32(i32((p >> 22) & 3) - 2) * x_in[b + 11]
        a1 += f32(i32((q >> 22) & 3) - 2) * x_in[b + 11]
        a0 += f32(i32((p >> 24) & 3) - 2) * x_in[b + 12]
        a1 += f32(i32((q >> 24) & 3) - 2) * x_in[b + 12]
        a0 += f32(i32((p >> 26) & 3) - 2) * x_in[b + 13]
        a1 += f32(i32((q >> 26) & 3) - 2) * x_in[b + 13]
        a0 += f32(i32((p >> 28) & 3) - 2) * x_in[b + 14]
        a1 += f32(i32((q >> 28) & 3) - 2) * x_in[b + 14]
        a0 += f32(i32((p >> 30) & 3) - 2) * x_in[b + 15]
        a1 += f32(i32((q >> 30) & 3) - 2) * x_in[b + 15]
    p0[li] = a0
    p1[li] = a1
    barrier()
    s: u32 = 32
    while s > 0:
        if li < s:
            p0[li] = p0[li] + p0[li + s]
            p1[li] = p1[li] + p1[li + s]
        barrier()
        s = s / 2
    if li == 0:
        y_out[r0] = p0[0] * scale[r0]
        y_out[r1] = p1[0] * scale[r1]


@kernel(workgroup_size=(64,))
def matvec_dq2_blk8(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    w_packed: StorageBuffer[u32, "read"],     # [n_out, n_in/16]
    x_in: StorageBuffer[f32, "read"],         # [n_in]
    scale: StorageBuffer[f32, "read"],        # [n_out]
    y_out: StorageBuffer[f32, "read_write"],  # [n_out]
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in] (n_out % 8 == 0)
    q0: WorkgroupArray[f32, 64],
    q1: WorkgroupArray[f32, 64],
    q2: WorkgroupArray[f32, 64],
    q3: WorkgroupArray[f32, 64],
    q4: WorkgroupArray[f32, 64],
    q5: WorkgroupArray[f32, 64],
    q6: WorkgroupArray[f32, 64],
    q7: WorkgroupArray[f32, 64],
):
    # Output-blocked 8 rows/workgroup: each x element is read once and
    # reused across 8 weight rows — amortizes the shared-input read (the
    # decode bottleneck). Used for the 262144-row tied logits matvec.
    r0: u32 = 8 * wid.x + 0
    r1: u32 = 8 * wid.x + 1
    r2: u32 = 8 * wid.x + 2
    r3: u32 = 8 * wid.x + 3
    r4: u32 = 8 * wid.x + 4
    r5: u32 = 8 * wid.x + 5
    r6: u32 = 8 * wid.x + 6
    r7: u32 = 8 * wid.x + 7
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    n16: u32 = n_in / 16
    a0: f32 = 0.0
    a1: f32 = 0.0
    a2: f32 = 0.0
    a3: f32 = 0.0
    a4: f32 = 0.0
    a5: f32 = 0.0
    a6: f32 = 0.0
    a7: f32 = 0.0
    for j in range(li, n16, 64):
        p0: u32 = w_packed[r0 * n16 + j]
        p1: u32 = w_packed[r1 * n16 + j]
        p2: u32 = w_packed[r2 * n16 + j]
        p3: u32 = w_packed[r3 * n16 + j]
        p4: u32 = w_packed[r4 * n16 + j]
        p5: u32 = w_packed[r5 * n16 + j]
        p6: u32 = w_packed[r6 * n16 + j]
        p7: u32 = w_packed[r7 * n16 + j]
        b: u32 = 16 * j
        a0 += f32(i32(p0 & 3) - 2) * x_in[b]
        a1 += f32(i32(p1 & 3) - 2) * x_in[b]
        a2 += f32(i32(p2 & 3) - 2) * x_in[b]
        a3 += f32(i32(p3 & 3) - 2) * x_in[b]
        a4 += f32(i32(p4 & 3) - 2) * x_in[b]
        a5 += f32(i32(p5 & 3) - 2) * x_in[b]
        a6 += f32(i32(p6 & 3) - 2) * x_in[b]
        a7 += f32(i32(p7 & 3) - 2) * x_in[b]
        a0 += f32(i32((p0 >> 2) & 3) - 2) * x_in[b + 1]
        a1 += f32(i32((p1 >> 2) & 3) - 2) * x_in[b + 1]
        a2 += f32(i32((p2 >> 2) & 3) - 2) * x_in[b + 1]
        a3 += f32(i32((p3 >> 2) & 3) - 2) * x_in[b + 1]
        a4 += f32(i32((p4 >> 2) & 3) - 2) * x_in[b + 1]
        a5 += f32(i32((p5 >> 2) & 3) - 2) * x_in[b + 1]
        a6 += f32(i32((p6 >> 2) & 3) - 2) * x_in[b + 1]
        a7 += f32(i32((p7 >> 2) & 3) - 2) * x_in[b + 1]
        a0 += f32(i32((p0 >> 4) & 3) - 2) * x_in[b + 2]
        a1 += f32(i32((p1 >> 4) & 3) - 2) * x_in[b + 2]
        a2 += f32(i32((p2 >> 4) & 3) - 2) * x_in[b + 2]
        a3 += f32(i32((p3 >> 4) & 3) - 2) * x_in[b + 2]
        a4 += f32(i32((p4 >> 4) & 3) - 2) * x_in[b + 2]
        a5 += f32(i32((p5 >> 4) & 3) - 2) * x_in[b + 2]
        a6 += f32(i32((p6 >> 4) & 3) - 2) * x_in[b + 2]
        a7 += f32(i32((p7 >> 4) & 3) - 2) * x_in[b + 2]
        a0 += f32(i32((p0 >> 6) & 3) - 2) * x_in[b + 3]
        a1 += f32(i32((p1 >> 6) & 3) - 2) * x_in[b + 3]
        a2 += f32(i32((p2 >> 6) & 3) - 2) * x_in[b + 3]
        a3 += f32(i32((p3 >> 6) & 3) - 2) * x_in[b + 3]
        a4 += f32(i32((p4 >> 6) & 3) - 2) * x_in[b + 3]
        a5 += f32(i32((p5 >> 6) & 3) - 2) * x_in[b + 3]
        a6 += f32(i32((p6 >> 6) & 3) - 2) * x_in[b + 3]
        a7 += f32(i32((p7 >> 6) & 3) - 2) * x_in[b + 3]
        a0 += f32(i32((p0 >> 8) & 3) - 2) * x_in[b + 4]
        a1 += f32(i32((p1 >> 8) & 3) - 2) * x_in[b + 4]
        a2 += f32(i32((p2 >> 8) & 3) - 2) * x_in[b + 4]
        a3 += f32(i32((p3 >> 8) & 3) - 2) * x_in[b + 4]
        a4 += f32(i32((p4 >> 8) & 3) - 2) * x_in[b + 4]
        a5 += f32(i32((p5 >> 8) & 3) - 2) * x_in[b + 4]
        a6 += f32(i32((p6 >> 8) & 3) - 2) * x_in[b + 4]
        a7 += f32(i32((p7 >> 8) & 3) - 2) * x_in[b + 4]
        a0 += f32(i32((p0 >> 10) & 3) - 2) * x_in[b + 5]
        a1 += f32(i32((p1 >> 10) & 3) - 2) * x_in[b + 5]
        a2 += f32(i32((p2 >> 10) & 3) - 2) * x_in[b + 5]
        a3 += f32(i32((p3 >> 10) & 3) - 2) * x_in[b + 5]
        a4 += f32(i32((p4 >> 10) & 3) - 2) * x_in[b + 5]
        a5 += f32(i32((p5 >> 10) & 3) - 2) * x_in[b + 5]
        a6 += f32(i32((p6 >> 10) & 3) - 2) * x_in[b + 5]
        a7 += f32(i32((p7 >> 10) & 3) - 2) * x_in[b + 5]
        a0 += f32(i32((p0 >> 12) & 3) - 2) * x_in[b + 6]
        a1 += f32(i32((p1 >> 12) & 3) - 2) * x_in[b + 6]
        a2 += f32(i32((p2 >> 12) & 3) - 2) * x_in[b + 6]
        a3 += f32(i32((p3 >> 12) & 3) - 2) * x_in[b + 6]
        a4 += f32(i32((p4 >> 12) & 3) - 2) * x_in[b + 6]
        a5 += f32(i32((p5 >> 12) & 3) - 2) * x_in[b + 6]
        a6 += f32(i32((p6 >> 12) & 3) - 2) * x_in[b + 6]
        a7 += f32(i32((p7 >> 12) & 3) - 2) * x_in[b + 6]
        a0 += f32(i32((p0 >> 14) & 3) - 2) * x_in[b + 7]
        a1 += f32(i32((p1 >> 14) & 3) - 2) * x_in[b + 7]
        a2 += f32(i32((p2 >> 14) & 3) - 2) * x_in[b + 7]
        a3 += f32(i32((p3 >> 14) & 3) - 2) * x_in[b + 7]
        a4 += f32(i32((p4 >> 14) & 3) - 2) * x_in[b + 7]
        a5 += f32(i32((p5 >> 14) & 3) - 2) * x_in[b + 7]
        a6 += f32(i32((p6 >> 14) & 3) - 2) * x_in[b + 7]
        a7 += f32(i32((p7 >> 14) & 3) - 2) * x_in[b + 7]
        a0 += f32(i32((p0 >> 16) & 3) - 2) * x_in[b + 8]
        a1 += f32(i32((p1 >> 16) & 3) - 2) * x_in[b + 8]
        a2 += f32(i32((p2 >> 16) & 3) - 2) * x_in[b + 8]
        a3 += f32(i32((p3 >> 16) & 3) - 2) * x_in[b + 8]
        a4 += f32(i32((p4 >> 16) & 3) - 2) * x_in[b + 8]
        a5 += f32(i32((p5 >> 16) & 3) - 2) * x_in[b + 8]
        a6 += f32(i32((p6 >> 16) & 3) - 2) * x_in[b + 8]
        a7 += f32(i32((p7 >> 16) & 3) - 2) * x_in[b + 8]
        a0 += f32(i32((p0 >> 18) & 3) - 2) * x_in[b + 9]
        a1 += f32(i32((p1 >> 18) & 3) - 2) * x_in[b + 9]
        a2 += f32(i32((p2 >> 18) & 3) - 2) * x_in[b + 9]
        a3 += f32(i32((p3 >> 18) & 3) - 2) * x_in[b + 9]
        a4 += f32(i32((p4 >> 18) & 3) - 2) * x_in[b + 9]
        a5 += f32(i32((p5 >> 18) & 3) - 2) * x_in[b + 9]
        a6 += f32(i32((p6 >> 18) & 3) - 2) * x_in[b + 9]
        a7 += f32(i32((p7 >> 18) & 3) - 2) * x_in[b + 9]
        a0 += f32(i32((p0 >> 20) & 3) - 2) * x_in[b + 10]
        a1 += f32(i32((p1 >> 20) & 3) - 2) * x_in[b + 10]
        a2 += f32(i32((p2 >> 20) & 3) - 2) * x_in[b + 10]
        a3 += f32(i32((p3 >> 20) & 3) - 2) * x_in[b + 10]
        a4 += f32(i32((p4 >> 20) & 3) - 2) * x_in[b + 10]
        a5 += f32(i32((p5 >> 20) & 3) - 2) * x_in[b + 10]
        a6 += f32(i32((p6 >> 20) & 3) - 2) * x_in[b + 10]
        a7 += f32(i32((p7 >> 20) & 3) - 2) * x_in[b + 10]
        a0 += f32(i32((p0 >> 22) & 3) - 2) * x_in[b + 11]
        a1 += f32(i32((p1 >> 22) & 3) - 2) * x_in[b + 11]
        a2 += f32(i32((p2 >> 22) & 3) - 2) * x_in[b + 11]
        a3 += f32(i32((p3 >> 22) & 3) - 2) * x_in[b + 11]
        a4 += f32(i32((p4 >> 22) & 3) - 2) * x_in[b + 11]
        a5 += f32(i32((p5 >> 22) & 3) - 2) * x_in[b + 11]
        a6 += f32(i32((p6 >> 22) & 3) - 2) * x_in[b + 11]
        a7 += f32(i32((p7 >> 22) & 3) - 2) * x_in[b + 11]
        a0 += f32(i32((p0 >> 24) & 3) - 2) * x_in[b + 12]
        a1 += f32(i32((p1 >> 24) & 3) - 2) * x_in[b + 12]
        a2 += f32(i32((p2 >> 24) & 3) - 2) * x_in[b + 12]
        a3 += f32(i32((p3 >> 24) & 3) - 2) * x_in[b + 12]
        a4 += f32(i32((p4 >> 24) & 3) - 2) * x_in[b + 12]
        a5 += f32(i32((p5 >> 24) & 3) - 2) * x_in[b + 12]
        a6 += f32(i32((p6 >> 24) & 3) - 2) * x_in[b + 12]
        a7 += f32(i32((p7 >> 24) & 3) - 2) * x_in[b + 12]
        a0 += f32(i32((p0 >> 26) & 3) - 2) * x_in[b + 13]
        a1 += f32(i32((p1 >> 26) & 3) - 2) * x_in[b + 13]
        a2 += f32(i32((p2 >> 26) & 3) - 2) * x_in[b + 13]
        a3 += f32(i32((p3 >> 26) & 3) - 2) * x_in[b + 13]
        a4 += f32(i32((p4 >> 26) & 3) - 2) * x_in[b + 13]
        a5 += f32(i32((p5 >> 26) & 3) - 2) * x_in[b + 13]
        a6 += f32(i32((p6 >> 26) & 3) - 2) * x_in[b + 13]
        a7 += f32(i32((p7 >> 26) & 3) - 2) * x_in[b + 13]
        a0 += f32(i32((p0 >> 28) & 3) - 2) * x_in[b + 14]
        a1 += f32(i32((p1 >> 28) & 3) - 2) * x_in[b + 14]
        a2 += f32(i32((p2 >> 28) & 3) - 2) * x_in[b + 14]
        a3 += f32(i32((p3 >> 28) & 3) - 2) * x_in[b + 14]
        a4 += f32(i32((p4 >> 28) & 3) - 2) * x_in[b + 14]
        a5 += f32(i32((p5 >> 28) & 3) - 2) * x_in[b + 14]
        a6 += f32(i32((p6 >> 28) & 3) - 2) * x_in[b + 14]
        a7 += f32(i32((p7 >> 28) & 3) - 2) * x_in[b + 14]
        a0 += f32(i32((p0 >> 30) & 3) - 2) * x_in[b + 15]
        a1 += f32(i32((p1 >> 30) & 3) - 2) * x_in[b + 15]
        a2 += f32(i32((p2 >> 30) & 3) - 2) * x_in[b + 15]
        a3 += f32(i32((p3 >> 30) & 3) - 2) * x_in[b + 15]
        a4 += f32(i32((p4 >> 30) & 3) - 2) * x_in[b + 15]
        a5 += f32(i32((p5 >> 30) & 3) - 2) * x_in[b + 15]
        a6 += f32(i32((p6 >> 30) & 3) - 2) * x_in[b + 15]
        a7 += f32(i32((p7 >> 30) & 3) - 2) * x_in[b + 15]
    q0[li] = a0
    q1[li] = a1
    q2[li] = a2
    q3[li] = a3
    q4[li] = a4
    q5[li] = a5
    q6[li] = a6
    q7[li] = a7
    barrier()
    s: u32 = 32
    while s > 0:
        if li < s:
            q0[li] = q0[li] + q0[li + s]
            q1[li] = q1[li] + q1[li + s]
            q2[li] = q2[li] + q2[li + s]
            q3[li] = q3[li] + q3[li + s]
            q4[li] = q4[li] + q4[li + s]
            q5[li] = q5[li] + q5[li + s]
            q6[li] = q6[li] + q6[li + s]
            q7[li] = q7[li] + q7[li + s]
        barrier()
        s = s / 2
    if li == 0:
        y_out[r0] = q0[0] * scale[r0]
        y_out[r1] = q1[0] * scale[r1]
        y_out[r2] = q2[0] * scale[r2]
        y_out[r3] = q3[0] * scale[r3]
        y_out[r4] = q4[0] * scale[r4]
        y_out[r5] = q5[0] * scale[r5]
        y_out[r6] = q6[0] * scale[r6]
        y_out[r7] = q7[0] * scale[r7]


@kernel(workgroup_size=(64,))
def mv_gateup_geglu_dq2(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    gate_w: StorageBuffer[u32, "read"],       # [n_out, n_in/16] int2 gate
    up_w: StorageBuffer[u32, "read"],         # [n_out, n_in/16] int2 up
    x_in: StorageBuffer[f32, "read"],         # [n_in]
    gscale: StorageBuffer[f32, "read"],       # [n_out] gate per-row scale
    uscale: StorageBuffer[f32, "read"],       # [n_out] up per-row scale
    y_out: StorageBuffer[f32, "read_write"],  # [n_out] = geglu(gate, up)
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in]
    pg: WorkgroupArray[f32, 64],
    pu: WorkgroupArray[f32, 64],
):
    # Fused gate+up matmul + geglu: one workgroup per output row computes both
    # projections of the same row, then y = gelu(gate*gs) * (up*us) — collapses
    # three dispatches (gate, up, geglu) into one and skips the gate/up VRAM
    # round-trip. Scales factor out of the int dot (symmetric per-row).
    r: u32 = wid.x
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    n16: u32 = n_in / 16
    ag: f32 = 0.0
    au: f32 = 0.0
    if r < n_out:
        for j in range(li, n16, 64):
            p: u32 = gate_w[r * n16 + j]
            q: u32 = up_w[r * n16 + j]
            b: u32 = 16 * j
            ag += f32(i32(p & 3) - 2) * x_in[b]
            au += f32(i32(q & 3) - 2) * x_in[b]
            ag += f32(i32((p >> 2) & 3) - 2) * x_in[b + 1]
            au += f32(i32((q >> 2) & 3) - 2) * x_in[b + 1]
            ag += f32(i32((p >> 4) & 3) - 2) * x_in[b + 2]
            au += f32(i32((q >> 4) & 3) - 2) * x_in[b + 2]
            ag += f32(i32((p >> 6) & 3) - 2) * x_in[b + 3]
            au += f32(i32((q >> 6) & 3) - 2) * x_in[b + 3]
            ag += f32(i32((p >> 8) & 3) - 2) * x_in[b + 4]
            au += f32(i32((q >> 8) & 3) - 2) * x_in[b + 4]
            ag += f32(i32((p >> 10) & 3) - 2) * x_in[b + 5]
            au += f32(i32((q >> 10) & 3) - 2) * x_in[b + 5]
            ag += f32(i32((p >> 12) & 3) - 2) * x_in[b + 6]
            au += f32(i32((q >> 12) & 3) - 2) * x_in[b + 6]
            ag += f32(i32((p >> 14) & 3) - 2) * x_in[b + 7]
            au += f32(i32((q >> 14) & 3) - 2) * x_in[b + 7]
            ag += f32(i32((p >> 16) & 3) - 2) * x_in[b + 8]
            au += f32(i32((q >> 16) & 3) - 2) * x_in[b + 8]
            ag += f32(i32((p >> 18) & 3) - 2) * x_in[b + 9]
            au += f32(i32((q >> 18) & 3) - 2) * x_in[b + 9]
            ag += f32(i32((p >> 20) & 3) - 2) * x_in[b + 10]
            au += f32(i32((q >> 20) & 3) - 2) * x_in[b + 10]
            ag += f32(i32((p >> 22) & 3) - 2) * x_in[b + 11]
            au += f32(i32((q >> 22) & 3) - 2) * x_in[b + 11]
            ag += f32(i32((p >> 24) & 3) - 2) * x_in[b + 12]
            au += f32(i32((q >> 24) & 3) - 2) * x_in[b + 12]
            ag += f32(i32((p >> 26) & 3) - 2) * x_in[b + 13]
            au += f32(i32((q >> 26) & 3) - 2) * x_in[b + 13]
            ag += f32(i32((p >> 28) & 3) - 2) * x_in[b + 14]
            au += f32(i32((q >> 28) & 3) - 2) * x_in[b + 14]
            ag += f32(i32((p >> 30) & 3) - 2) * x_in[b + 15]
            au += f32(i32((q >> 30) & 3) - 2) * x_in[b + 15]
    pg[li] = ag
    pu[li] = au
    barrier()
    s: u32 = 32
    while s > 0:
        if li < s:
            pg[li] = pg[li] + pg[li + s]
            pu[li] = pu[li] + pu[li + s]
        barrier()
        s = s / 2
    if li == 0 and r < n_out:
        y_out[r] = gelu(pg[0] * gscale[r]) * (pu[0] * uscale[r])


@kernel(workgroup_size=(64,))
def mv_gateup_geglu_dq4(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    gate_w: StorageBuffer[u32, "read"],       # [n_out, n_in/8] int4 gate
    up_w: StorageBuffer[u32, "read"],         # [n_out, n_in/8] int4 up
    x_in: StorageBuffer[f32, "read"],         # [n_in]
    gscale: StorageBuffer[f32, "read"],       # [n_out] gate per-row scale
    uscale: StorageBuffer[f32, "read"],       # [n_out] up per-row scale
    y_out: StorageBuffer[f32, "read_write"],  # [n_out] = geglu(gate, up)
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in]
    pg: WorkgroupArray[f32, 64],
    pu: WorkgroupArray[f32, 64],
):
    # int4 fused gate+up+geglu; see mv_gateup_geglu_dq2.
    r: u32 = wid.x
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    n8: u32 = n_in / 8
    ag: f32 = 0.0
    au: f32 = 0.0
    if r < n_out:
        for j in range(li, n8, 64):
            p: u32 = gate_w[r * n8 + j]
            q: u32 = up_w[r * n8 + j]
            base: u32 = 8 * j
            pb0: u32 = p & 255
            qb0: u32 = q & 255
            ag += f32(i32(pb0 & 15) - 8) * x_in[base]
            au += f32(i32(qb0 & 15) - 8) * x_in[base]
            ag += f32(i32(pb0 >> 4) - 8) * x_in[base + 1]
            au += f32(i32(qb0 >> 4) - 8) * x_in[base + 1]
            pb1: u32 = (p >> 8) & 255
            qb1: u32 = (q >> 8) & 255
            ag += f32(i32(pb1 & 15) - 8) * x_in[base + 2]
            au += f32(i32(qb1 & 15) - 8) * x_in[base + 2]
            ag += f32(i32(pb1 >> 4) - 8) * x_in[base + 3]
            au += f32(i32(qb1 >> 4) - 8) * x_in[base + 3]
            pb2: u32 = (p >> 16) & 255
            qb2: u32 = (q >> 16) & 255
            ag += f32(i32(pb2 & 15) - 8) * x_in[base + 4]
            au += f32(i32(qb2 & 15) - 8) * x_in[base + 4]
            ag += f32(i32(pb2 >> 4) - 8) * x_in[base + 5]
            au += f32(i32(qb2 >> 4) - 8) * x_in[base + 5]
            pb3: u32 = (p >> 24) & 255
            qb3: u32 = (q >> 24) & 255
            ag += f32(i32(pb3 & 15) - 8) * x_in[base + 6]
            au += f32(i32(qb3 & 15) - 8) * x_in[base + 6]
            ag += f32(i32(pb3 >> 4) - 8) * x_in[base + 7]
            au += f32(i32(qb3 >> 4) - 8) * x_in[base + 7]
    pg[li] = ag
    pu[li] = au
    barrier()
    s: u32 = 32
    while s > 0:
        if li < s:
            pg[li] = pg[li] + pg[li + s]
            pu[li] = pu[li] + pu[li + s]
        barrier()
        s = s / 2
    if li == 0 and r < n_out:
        y_out[r] = gelu(pg[0] * gscale[r]) * (pu[0] * uscale[r])


@kernel(workgroup_size=(64,))
def mv_geglu_f16(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    w_mat: StorageBuffer[vec4[u32], "read"],  # [n_out, n_in/8] f16 gate weight
    x_in: StorageBuffer[vec4[f32], "read"],   # [n_in/4]
    up_in: StorageBuffer[f32, "read"],        # [n_out] geglu up operand (buffer)
    y_out: StorageBuffer[f32, "read_write"],  # [n_out] = gelu(gate) * up_in
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in]
    partial: WorkgroupArray[f32, 64],
):
    # f16 vec4 matvec (like matvec_wg_packed_v4) with a fused geglu tail whose
    # "up" operand is a plain buffer (the PLE per-layer input slice), collapsing
    # the PLE gate matmul + geglu into one dispatch.
    r: u32 = wid.x
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    n8: u32 = n_in / 8
    acc: f32 = 0.0
    if r < n_out:
        for j in range(li, n8, 64):
            wv = w_mat[r * n8 + j]
            p0 = unpack2x16float(wv.x)
            p1 = unpack2x16float(wv.y)
            p2 = unpack2x16float(wv.z)
            p3 = unpack2x16float(wv.w)
            xa = x_in[2 * j]
            xb = x_in[2 * j + 1]
            acc += p0.x * xa.x + p0.y * xa.y + p1.x * xa.z + p1.y * xa.w
            acc += p2.x * xb.x + p2.y * xb.y + p3.x * xb.z + p3.y * xb.w
    partial[li] = acc
    barrier()
    s: u32 = 32
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2
    if li == 0 and r < n_out:
        y_out[r] = gelu(partial[0]) * up_in[r]


@kernel(workgroup_size=(64,))
def mv_gateup_geglu_dq2_blk2(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    gate_w: StorageBuffer[u32, "read"],       # [n_out, n_in/16] int2 gate
    up_w: StorageBuffer[u32, "read"],         # [n_out, n_in/16] int2 up
    x_in: StorageBuffer[f32, "read"],         # [n_in]
    gscale: StorageBuffer[f32, "read"],       # [n_out]
    uscale: StorageBuffer[f32, "read"],       # [n_out]
    y_out: StorageBuffer[f32, "read_write"],  # [n_out] = geglu(gate, up)
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in] (n_out % 2 == 0)
    pg0: WorkgroupArray[f32, 64],
    pu0: WorkgroupArray[f32, 64],
    pg1: WorkgroupArray[f32, 64],
    pu1: WorkgroupArray[f32, 64],
):
    # Output-blocked fused gate+up+geglu: 2 rows/workgroup, x read once
    # per element for all 4 weight rows (gate+up x 2).
    r0: u32 = 2 * wid.x + 0
    r1: u32 = 2 * wid.x + 1
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    n16: u32 = n_in / 16
    g0: f32 = 0.0
    u0: f32 = 0.0
    g1: f32 = 0.0
    u1: f32 = 0.0
    for j in range(li, n16, 64):
        gw0: u32 = gate_w[r0 * n16 + j]
        uw0: u32 = up_w[r0 * n16 + j]
        gw1: u32 = gate_w[r1 * n16 + j]
        uw1: u32 = up_w[r1 * n16 + j]
        b: u32 = 16 * j
        g0 += f32(i32(gw0 & 3) - 2) * x_in[b]
        u0 += f32(i32(uw0 & 3) - 2) * x_in[b]
        g1 += f32(i32(gw1 & 3) - 2) * x_in[b]
        u1 += f32(i32(uw1 & 3) - 2) * x_in[b]
        g0 += f32(i32((gw0 >> 2) & 3) - 2) * x_in[b + 1]
        u0 += f32(i32((uw0 >> 2) & 3) - 2) * x_in[b + 1]
        g1 += f32(i32((gw1 >> 2) & 3) - 2) * x_in[b + 1]
        u1 += f32(i32((uw1 >> 2) & 3) - 2) * x_in[b + 1]
        g0 += f32(i32((gw0 >> 4) & 3) - 2) * x_in[b + 2]
        u0 += f32(i32((uw0 >> 4) & 3) - 2) * x_in[b + 2]
        g1 += f32(i32((gw1 >> 4) & 3) - 2) * x_in[b + 2]
        u1 += f32(i32((uw1 >> 4) & 3) - 2) * x_in[b + 2]
        g0 += f32(i32((gw0 >> 6) & 3) - 2) * x_in[b + 3]
        u0 += f32(i32((uw0 >> 6) & 3) - 2) * x_in[b + 3]
        g1 += f32(i32((gw1 >> 6) & 3) - 2) * x_in[b + 3]
        u1 += f32(i32((uw1 >> 6) & 3) - 2) * x_in[b + 3]
        g0 += f32(i32((gw0 >> 8) & 3) - 2) * x_in[b + 4]
        u0 += f32(i32((uw0 >> 8) & 3) - 2) * x_in[b + 4]
        g1 += f32(i32((gw1 >> 8) & 3) - 2) * x_in[b + 4]
        u1 += f32(i32((uw1 >> 8) & 3) - 2) * x_in[b + 4]
        g0 += f32(i32((gw0 >> 10) & 3) - 2) * x_in[b + 5]
        u0 += f32(i32((uw0 >> 10) & 3) - 2) * x_in[b + 5]
        g1 += f32(i32((gw1 >> 10) & 3) - 2) * x_in[b + 5]
        u1 += f32(i32((uw1 >> 10) & 3) - 2) * x_in[b + 5]
        g0 += f32(i32((gw0 >> 12) & 3) - 2) * x_in[b + 6]
        u0 += f32(i32((uw0 >> 12) & 3) - 2) * x_in[b + 6]
        g1 += f32(i32((gw1 >> 12) & 3) - 2) * x_in[b + 6]
        u1 += f32(i32((uw1 >> 12) & 3) - 2) * x_in[b + 6]
        g0 += f32(i32((gw0 >> 14) & 3) - 2) * x_in[b + 7]
        u0 += f32(i32((uw0 >> 14) & 3) - 2) * x_in[b + 7]
        g1 += f32(i32((gw1 >> 14) & 3) - 2) * x_in[b + 7]
        u1 += f32(i32((uw1 >> 14) & 3) - 2) * x_in[b + 7]
        g0 += f32(i32((gw0 >> 16) & 3) - 2) * x_in[b + 8]
        u0 += f32(i32((uw0 >> 16) & 3) - 2) * x_in[b + 8]
        g1 += f32(i32((gw1 >> 16) & 3) - 2) * x_in[b + 8]
        u1 += f32(i32((uw1 >> 16) & 3) - 2) * x_in[b + 8]
        g0 += f32(i32((gw0 >> 18) & 3) - 2) * x_in[b + 9]
        u0 += f32(i32((uw0 >> 18) & 3) - 2) * x_in[b + 9]
        g1 += f32(i32((gw1 >> 18) & 3) - 2) * x_in[b + 9]
        u1 += f32(i32((uw1 >> 18) & 3) - 2) * x_in[b + 9]
        g0 += f32(i32((gw0 >> 20) & 3) - 2) * x_in[b + 10]
        u0 += f32(i32((uw0 >> 20) & 3) - 2) * x_in[b + 10]
        g1 += f32(i32((gw1 >> 20) & 3) - 2) * x_in[b + 10]
        u1 += f32(i32((uw1 >> 20) & 3) - 2) * x_in[b + 10]
        g0 += f32(i32((gw0 >> 22) & 3) - 2) * x_in[b + 11]
        u0 += f32(i32((uw0 >> 22) & 3) - 2) * x_in[b + 11]
        g1 += f32(i32((gw1 >> 22) & 3) - 2) * x_in[b + 11]
        u1 += f32(i32((uw1 >> 22) & 3) - 2) * x_in[b + 11]
        g0 += f32(i32((gw0 >> 24) & 3) - 2) * x_in[b + 12]
        u0 += f32(i32((uw0 >> 24) & 3) - 2) * x_in[b + 12]
        g1 += f32(i32((gw1 >> 24) & 3) - 2) * x_in[b + 12]
        u1 += f32(i32((uw1 >> 24) & 3) - 2) * x_in[b + 12]
        g0 += f32(i32((gw0 >> 26) & 3) - 2) * x_in[b + 13]
        u0 += f32(i32((uw0 >> 26) & 3) - 2) * x_in[b + 13]
        g1 += f32(i32((gw1 >> 26) & 3) - 2) * x_in[b + 13]
        u1 += f32(i32((uw1 >> 26) & 3) - 2) * x_in[b + 13]
        g0 += f32(i32((gw0 >> 28) & 3) - 2) * x_in[b + 14]
        u0 += f32(i32((uw0 >> 28) & 3) - 2) * x_in[b + 14]
        g1 += f32(i32((gw1 >> 28) & 3) - 2) * x_in[b + 14]
        u1 += f32(i32((uw1 >> 28) & 3) - 2) * x_in[b + 14]
        g0 += f32(i32((gw0 >> 30) & 3) - 2) * x_in[b + 15]
        u0 += f32(i32((uw0 >> 30) & 3) - 2) * x_in[b + 15]
        g1 += f32(i32((gw1 >> 30) & 3) - 2) * x_in[b + 15]
        u1 += f32(i32((uw1 >> 30) & 3) - 2) * x_in[b + 15]
    pg0[li] = g0
    pu0[li] = u0
    pg1[li] = g1
    pu1[li] = u1
    barrier()
    s: u32 = 32
    while s > 0:
        if li < s:
            pg0[li] = pg0[li] + pg0[li + s]
            pu0[li] = pu0[li] + pu0[li + s]
            pg1[li] = pg1[li] + pg1[li + s]
            pu1[li] = pu1[li] + pu1[li + s]
        barrier()
        s = s / 2
    if li == 0:
        y_out[r0] = gelu(pg0[0] * gscale[r0]) * (pu0[0] * uscale[r0])
        y_out[r1] = gelu(pg1[0] * gscale[r1]) * (pu1[0] * uscale[r1])


# Registry for the QAT runner: reusable base gemma4 kernels (norms, rope,
# attention, geglu, argmax, resident-decode helpers, f16 matvec for 8-bit /
# unquantized modules) plus the QAT-specific dequant matmuls and gathers.
_REUSE = (
    "rmsnorm_wg", "rmsnorm_ns_wg", "rmsnorm_add_wg", "rmsnorm_add_norm_wg",
    "rmsnorm_add_scale_wg", "rope_pl", "attention_fused_g4", "geglu",
    "kv_append", "combine_scaled", "softcap", "argmax_stage1", "argmax_stage2",
    "step_setup_g4", "matvec_wg_packed_v4",
)
KERNELS = {name: K4.KERNELS[name] for name in _REUSE}
KERNELS.update({
    "matvec_dq4": matvec_dq4,
    "matvec_dq2": matvec_dq2,
    "matvec_dq2_blk2": matvec_dq2_blk2,
    "matvec_dq2_blk8": matvec_dq2_blk8,
    "mv_gateup_geglu_dq2": mv_gateup_geglu_dq2,
    "mv_gateup_geglu_dq2_blk2": mv_gateup_geglu_dq2_blk2,
    "mv_gateup_geglu_dq4": mv_gateup_geglu_dq4,
    "mv_geglu_f16": mv_geglu_f16,
    "qat_embed_2bit": qat_embed_2bit,
    "qat_ple_gather_4bit": qat_ple_gather_4bit,
})
