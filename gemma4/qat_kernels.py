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


@kernel(workgroup_size=(64,))
def matvec_dq4_blk2(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    w_packed: StorageBuffer[u32, "read"],     # [n_out, n_in/8] (8 int4/u32)
    x_in: StorageBuffer[f32, "read"],         # [n_in]
    scale: StorageBuffer[f32, "read"],        # [n_out]
    y_out: StorageBuffer[f32, "read_write"],  # [n_out]
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in] (n_out % 2 == 0)
    q0: WorkgroupArray[f32, 64],
    q1: WorkgroupArray[f32, 64],
):
    # Output-blocked int4 matvec: 2 rows/workgroup, x read once per element.
    r0: u32 = 2 * wid.x + 0
    r1: u32 = 2 * wid.x + 1
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    n8: u32 = n_in / 8
    a0: f32 = 0.0
    a1: f32 = 0.0
    for j in range(li, n8, 64):
        p0: u32 = w_packed[r0 * n8 + j]
        p1: u32 = w_packed[r1 * n8 + j]
        b: u32 = 8 * j
        a0 += f32(i32(p0 & 15) - 8) * x_in[b]
        a1 += f32(i32(p1 & 15) - 8) * x_in[b]
        a0 += f32(i32((p0 >> 4) & 15) - 8) * x_in[b + 1]
        a1 += f32(i32((p1 >> 4) & 15) - 8) * x_in[b + 1]
        a0 += f32(i32((p0 >> 8) & 15) - 8) * x_in[b + 2]
        a1 += f32(i32((p1 >> 8) & 15) - 8) * x_in[b + 2]
        a0 += f32(i32((p0 >> 12) & 15) - 8) * x_in[b + 3]
        a1 += f32(i32((p1 >> 12) & 15) - 8) * x_in[b + 3]
        a0 += f32(i32((p0 >> 16) & 15) - 8) * x_in[b + 4]
        a1 += f32(i32((p1 >> 16) & 15) - 8) * x_in[b + 4]
        a0 += f32(i32((p0 >> 20) & 15) - 8) * x_in[b + 5]
        a1 += f32(i32((p1 >> 20) & 15) - 8) * x_in[b + 5]
        a0 += f32(i32((p0 >> 24) & 15) - 8) * x_in[b + 6]
        a1 += f32(i32((p1 >> 24) & 15) - 8) * x_in[b + 6]
        a0 += f32(i32((p0 >> 28) & 15) - 8) * x_in[b + 7]
        a1 += f32(i32((p1 >> 28) & 15) - 8) * x_in[b + 7]
    q0[li] = a0
    q1[li] = a1
    barrier()
    s: u32 = 32
    while s > 0:
        if li < s:
            q0[li] = q0[li] + q0[li + s]
            q1[li] = q1[li] + q1[li + s]
        barrier()
        s = s / 2
    if li == 0:
        y_out[r0] = q0[0] * scale[r0]
        y_out[r1] = q1[0] * scale[r1]


@kernel(workgroup_size=(64,))
def mv_gateup_geglu_dq2_blk4(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    gate_w: StorageBuffer[u32, "read"],       # [n_out, n_in/16] int2 gate
    up_w: StorageBuffer[u32, "read"],         # [n_out, n_in/16] int2 up
    x_in: StorageBuffer[f32, "read"],         # [n_in]
    gscale: StorageBuffer[f32, "read"],       # [n_out]
    uscale: StorageBuffer[f32, "read"],       # [n_out]
    y_out: StorageBuffer[f32, "read_write"],  # [n_out] = geglu(gate, up)
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in] (n_out % 4 == 0)
    pg0: WorkgroupArray[f32, 64],
    pu0: WorkgroupArray[f32, 64],
    pg1: WorkgroupArray[f32, 64],
    pu1: WorkgroupArray[f32, 64],
    pg2: WorkgroupArray[f32, 64],
    pu2: WorkgroupArray[f32, 64],
    pg3: WorkgroupArray[f32, 64],
    pu3: WorkgroupArray[f32, 64],
):
    # Output-blocked fused gate+up+geglu: 4 rows/workgroup, x read once
    # per element for all 8 weight rows (gate+up x 4).
    r0: u32 = 4 * wid.x + 0
    r1: u32 = 4 * wid.x + 1
    r2: u32 = 4 * wid.x + 2
    r3: u32 = 4 * wid.x + 3
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    n16: u32 = n_in / 16
    g0: f32 = 0.0
    u0: f32 = 0.0
    g1: f32 = 0.0
    u1: f32 = 0.0
    g2: f32 = 0.0
    u2: f32 = 0.0
    g3: f32 = 0.0
    u3: f32 = 0.0
    for j in range(li, n16, 64):
        gw0: u32 = gate_w[r0 * n16 + j]
        uw0: u32 = up_w[r0 * n16 + j]
        gw1: u32 = gate_w[r1 * n16 + j]
        uw1: u32 = up_w[r1 * n16 + j]
        gw2: u32 = gate_w[r2 * n16 + j]
        uw2: u32 = up_w[r2 * n16 + j]
        gw3: u32 = gate_w[r3 * n16 + j]
        uw3: u32 = up_w[r3 * n16 + j]
        b: u32 = 16 * j
        g0 += f32(i32(gw0 & 3) - 2) * x_in[b]
        u0 += f32(i32(uw0 & 3) - 2) * x_in[b]
        g1 += f32(i32(gw1 & 3) - 2) * x_in[b]
        u1 += f32(i32(uw1 & 3) - 2) * x_in[b]
        g2 += f32(i32(gw2 & 3) - 2) * x_in[b]
        u2 += f32(i32(uw2 & 3) - 2) * x_in[b]
        g3 += f32(i32(gw3 & 3) - 2) * x_in[b]
        u3 += f32(i32(uw3 & 3) - 2) * x_in[b]
        g0 += f32(i32((gw0 >> 2) & 3) - 2) * x_in[b + 1]
        u0 += f32(i32((uw0 >> 2) & 3) - 2) * x_in[b + 1]
        g1 += f32(i32((gw1 >> 2) & 3) - 2) * x_in[b + 1]
        u1 += f32(i32((uw1 >> 2) & 3) - 2) * x_in[b + 1]
        g2 += f32(i32((gw2 >> 2) & 3) - 2) * x_in[b + 1]
        u2 += f32(i32((uw2 >> 2) & 3) - 2) * x_in[b + 1]
        g3 += f32(i32((gw3 >> 2) & 3) - 2) * x_in[b + 1]
        u3 += f32(i32((uw3 >> 2) & 3) - 2) * x_in[b + 1]
        g0 += f32(i32((gw0 >> 4) & 3) - 2) * x_in[b + 2]
        u0 += f32(i32((uw0 >> 4) & 3) - 2) * x_in[b + 2]
        g1 += f32(i32((gw1 >> 4) & 3) - 2) * x_in[b + 2]
        u1 += f32(i32((uw1 >> 4) & 3) - 2) * x_in[b + 2]
        g2 += f32(i32((gw2 >> 4) & 3) - 2) * x_in[b + 2]
        u2 += f32(i32((uw2 >> 4) & 3) - 2) * x_in[b + 2]
        g3 += f32(i32((gw3 >> 4) & 3) - 2) * x_in[b + 2]
        u3 += f32(i32((uw3 >> 4) & 3) - 2) * x_in[b + 2]
        g0 += f32(i32((gw0 >> 6) & 3) - 2) * x_in[b + 3]
        u0 += f32(i32((uw0 >> 6) & 3) - 2) * x_in[b + 3]
        g1 += f32(i32((gw1 >> 6) & 3) - 2) * x_in[b + 3]
        u1 += f32(i32((uw1 >> 6) & 3) - 2) * x_in[b + 3]
        g2 += f32(i32((gw2 >> 6) & 3) - 2) * x_in[b + 3]
        u2 += f32(i32((uw2 >> 6) & 3) - 2) * x_in[b + 3]
        g3 += f32(i32((gw3 >> 6) & 3) - 2) * x_in[b + 3]
        u3 += f32(i32((uw3 >> 6) & 3) - 2) * x_in[b + 3]
        g0 += f32(i32((gw0 >> 8) & 3) - 2) * x_in[b + 4]
        u0 += f32(i32((uw0 >> 8) & 3) - 2) * x_in[b + 4]
        g1 += f32(i32((gw1 >> 8) & 3) - 2) * x_in[b + 4]
        u1 += f32(i32((uw1 >> 8) & 3) - 2) * x_in[b + 4]
        g2 += f32(i32((gw2 >> 8) & 3) - 2) * x_in[b + 4]
        u2 += f32(i32((uw2 >> 8) & 3) - 2) * x_in[b + 4]
        g3 += f32(i32((gw3 >> 8) & 3) - 2) * x_in[b + 4]
        u3 += f32(i32((uw3 >> 8) & 3) - 2) * x_in[b + 4]
        g0 += f32(i32((gw0 >> 10) & 3) - 2) * x_in[b + 5]
        u0 += f32(i32((uw0 >> 10) & 3) - 2) * x_in[b + 5]
        g1 += f32(i32((gw1 >> 10) & 3) - 2) * x_in[b + 5]
        u1 += f32(i32((uw1 >> 10) & 3) - 2) * x_in[b + 5]
        g2 += f32(i32((gw2 >> 10) & 3) - 2) * x_in[b + 5]
        u2 += f32(i32((uw2 >> 10) & 3) - 2) * x_in[b + 5]
        g3 += f32(i32((gw3 >> 10) & 3) - 2) * x_in[b + 5]
        u3 += f32(i32((uw3 >> 10) & 3) - 2) * x_in[b + 5]
        g0 += f32(i32((gw0 >> 12) & 3) - 2) * x_in[b + 6]
        u0 += f32(i32((uw0 >> 12) & 3) - 2) * x_in[b + 6]
        g1 += f32(i32((gw1 >> 12) & 3) - 2) * x_in[b + 6]
        u1 += f32(i32((uw1 >> 12) & 3) - 2) * x_in[b + 6]
        g2 += f32(i32((gw2 >> 12) & 3) - 2) * x_in[b + 6]
        u2 += f32(i32((uw2 >> 12) & 3) - 2) * x_in[b + 6]
        g3 += f32(i32((gw3 >> 12) & 3) - 2) * x_in[b + 6]
        u3 += f32(i32((uw3 >> 12) & 3) - 2) * x_in[b + 6]
        g0 += f32(i32((gw0 >> 14) & 3) - 2) * x_in[b + 7]
        u0 += f32(i32((uw0 >> 14) & 3) - 2) * x_in[b + 7]
        g1 += f32(i32((gw1 >> 14) & 3) - 2) * x_in[b + 7]
        u1 += f32(i32((uw1 >> 14) & 3) - 2) * x_in[b + 7]
        g2 += f32(i32((gw2 >> 14) & 3) - 2) * x_in[b + 7]
        u2 += f32(i32((uw2 >> 14) & 3) - 2) * x_in[b + 7]
        g3 += f32(i32((gw3 >> 14) & 3) - 2) * x_in[b + 7]
        u3 += f32(i32((uw3 >> 14) & 3) - 2) * x_in[b + 7]
        g0 += f32(i32((gw0 >> 16) & 3) - 2) * x_in[b + 8]
        u0 += f32(i32((uw0 >> 16) & 3) - 2) * x_in[b + 8]
        g1 += f32(i32((gw1 >> 16) & 3) - 2) * x_in[b + 8]
        u1 += f32(i32((uw1 >> 16) & 3) - 2) * x_in[b + 8]
        g2 += f32(i32((gw2 >> 16) & 3) - 2) * x_in[b + 8]
        u2 += f32(i32((uw2 >> 16) & 3) - 2) * x_in[b + 8]
        g3 += f32(i32((gw3 >> 16) & 3) - 2) * x_in[b + 8]
        u3 += f32(i32((uw3 >> 16) & 3) - 2) * x_in[b + 8]
        g0 += f32(i32((gw0 >> 18) & 3) - 2) * x_in[b + 9]
        u0 += f32(i32((uw0 >> 18) & 3) - 2) * x_in[b + 9]
        g1 += f32(i32((gw1 >> 18) & 3) - 2) * x_in[b + 9]
        u1 += f32(i32((uw1 >> 18) & 3) - 2) * x_in[b + 9]
        g2 += f32(i32((gw2 >> 18) & 3) - 2) * x_in[b + 9]
        u2 += f32(i32((uw2 >> 18) & 3) - 2) * x_in[b + 9]
        g3 += f32(i32((gw3 >> 18) & 3) - 2) * x_in[b + 9]
        u3 += f32(i32((uw3 >> 18) & 3) - 2) * x_in[b + 9]
        g0 += f32(i32((gw0 >> 20) & 3) - 2) * x_in[b + 10]
        u0 += f32(i32((uw0 >> 20) & 3) - 2) * x_in[b + 10]
        g1 += f32(i32((gw1 >> 20) & 3) - 2) * x_in[b + 10]
        u1 += f32(i32((uw1 >> 20) & 3) - 2) * x_in[b + 10]
        g2 += f32(i32((gw2 >> 20) & 3) - 2) * x_in[b + 10]
        u2 += f32(i32((uw2 >> 20) & 3) - 2) * x_in[b + 10]
        g3 += f32(i32((gw3 >> 20) & 3) - 2) * x_in[b + 10]
        u3 += f32(i32((uw3 >> 20) & 3) - 2) * x_in[b + 10]
        g0 += f32(i32((gw0 >> 22) & 3) - 2) * x_in[b + 11]
        u0 += f32(i32((uw0 >> 22) & 3) - 2) * x_in[b + 11]
        g1 += f32(i32((gw1 >> 22) & 3) - 2) * x_in[b + 11]
        u1 += f32(i32((uw1 >> 22) & 3) - 2) * x_in[b + 11]
        g2 += f32(i32((gw2 >> 22) & 3) - 2) * x_in[b + 11]
        u2 += f32(i32((uw2 >> 22) & 3) - 2) * x_in[b + 11]
        g3 += f32(i32((gw3 >> 22) & 3) - 2) * x_in[b + 11]
        u3 += f32(i32((uw3 >> 22) & 3) - 2) * x_in[b + 11]
        g0 += f32(i32((gw0 >> 24) & 3) - 2) * x_in[b + 12]
        u0 += f32(i32((uw0 >> 24) & 3) - 2) * x_in[b + 12]
        g1 += f32(i32((gw1 >> 24) & 3) - 2) * x_in[b + 12]
        u1 += f32(i32((uw1 >> 24) & 3) - 2) * x_in[b + 12]
        g2 += f32(i32((gw2 >> 24) & 3) - 2) * x_in[b + 12]
        u2 += f32(i32((uw2 >> 24) & 3) - 2) * x_in[b + 12]
        g3 += f32(i32((gw3 >> 24) & 3) - 2) * x_in[b + 12]
        u3 += f32(i32((uw3 >> 24) & 3) - 2) * x_in[b + 12]
        g0 += f32(i32((gw0 >> 26) & 3) - 2) * x_in[b + 13]
        u0 += f32(i32((uw0 >> 26) & 3) - 2) * x_in[b + 13]
        g1 += f32(i32((gw1 >> 26) & 3) - 2) * x_in[b + 13]
        u1 += f32(i32((uw1 >> 26) & 3) - 2) * x_in[b + 13]
        g2 += f32(i32((gw2 >> 26) & 3) - 2) * x_in[b + 13]
        u2 += f32(i32((uw2 >> 26) & 3) - 2) * x_in[b + 13]
        g3 += f32(i32((gw3 >> 26) & 3) - 2) * x_in[b + 13]
        u3 += f32(i32((uw3 >> 26) & 3) - 2) * x_in[b + 13]
        g0 += f32(i32((gw0 >> 28) & 3) - 2) * x_in[b + 14]
        u0 += f32(i32((uw0 >> 28) & 3) - 2) * x_in[b + 14]
        g1 += f32(i32((gw1 >> 28) & 3) - 2) * x_in[b + 14]
        u1 += f32(i32((uw1 >> 28) & 3) - 2) * x_in[b + 14]
        g2 += f32(i32((gw2 >> 28) & 3) - 2) * x_in[b + 14]
        u2 += f32(i32((uw2 >> 28) & 3) - 2) * x_in[b + 14]
        g3 += f32(i32((gw3 >> 28) & 3) - 2) * x_in[b + 14]
        u3 += f32(i32((uw3 >> 28) & 3) - 2) * x_in[b + 14]
        g0 += f32(i32((gw0 >> 30) & 3) - 2) * x_in[b + 15]
        u0 += f32(i32((uw0 >> 30) & 3) - 2) * x_in[b + 15]
        g1 += f32(i32((gw1 >> 30) & 3) - 2) * x_in[b + 15]
        u1 += f32(i32((uw1 >> 30) & 3) - 2) * x_in[b + 15]
        g2 += f32(i32((gw2 >> 30) & 3) - 2) * x_in[b + 15]
        u2 += f32(i32((uw2 >> 30) & 3) - 2) * x_in[b + 15]
        g3 += f32(i32((gw3 >> 30) & 3) - 2) * x_in[b + 15]
        u3 += f32(i32((uw3 >> 30) & 3) - 2) * x_in[b + 15]
    pg0[li] = g0
    pu0[li] = u0
    pg1[li] = g1
    pu1[li] = u1
    pg2[li] = g2
    pu2[li] = u2
    pg3[li] = g3
    pu3[li] = u3
    barrier()
    s: u32 = 32
    while s > 0:
        if li < s:
            pg0[li] = pg0[li] + pg0[li + s]
            pu0[li] = pu0[li] + pu0[li + s]
            pg1[li] = pg1[li] + pg1[li + s]
            pu1[li] = pu1[li] + pu1[li + s]
            pg2[li] = pg2[li] + pg2[li + s]
            pu2[li] = pu2[li] + pu2[li + s]
            pg3[li] = pg3[li] + pg3[li + s]
            pu3[li] = pu3[li] + pu3[li + s]
        barrier()
        s = s / 2
    if li == 0:
        y_out[r0] = gelu(pg0[0] * gscale[r0]) * (pu0[0] * uscale[r0])
        y_out[r1] = gelu(pg1[0] * gscale[r1]) * (pu1[0] * uscale[r1])
        y_out[r2] = gelu(pg2[0] * gscale[r2]) * (pu2[0] * uscale[r2])
        y_out[r3] = gelu(pg3[0] * gscale[r3]) * (pu3[0] * uscale[r3])


@kernel(workgroup_size=(64,))
def matvec_dq2_blk16(
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
    q8: WorkgroupArray[f32, 64],
    q9: WorkgroupArray[f32, 64],
    q10: WorkgroupArray[f32, 64],
    q11: WorkgroupArray[f32, 64],
    q12: WorkgroupArray[f32, 64],
    q13: WorkgroupArray[f32, 64],
    q14: WorkgroupArray[f32, 64],
    q15: WorkgroupArray[f32, 64],
):
    # Output-blocked 16 rows/workgroup: each x element is read once and
    # reused across 16 weight rows — amortizes the shared-input read (the
    # decode bottleneck). Used for the 262144-row tied logits matvec.
    r0: u32 = 16 * wid.x + 0
    r1: u32 = 16 * wid.x + 1
    r2: u32 = 16 * wid.x + 2
    r3: u32 = 16 * wid.x + 3
    r4: u32 = 16 * wid.x + 4
    r5: u32 = 16 * wid.x + 5
    r6: u32 = 16 * wid.x + 6
    r7: u32 = 16 * wid.x + 7
    r8: u32 = 16 * wid.x + 8
    r9: u32 = 16 * wid.x + 9
    r10: u32 = 16 * wid.x + 10
    r11: u32 = 16 * wid.x + 11
    r12: u32 = 16 * wid.x + 12
    r13: u32 = 16 * wid.x + 13
    r14: u32 = 16 * wid.x + 14
    r15: u32 = 16 * wid.x + 15
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
    a8: f32 = 0.0
    a9: f32 = 0.0
    a10: f32 = 0.0
    a11: f32 = 0.0
    a12: f32 = 0.0
    a13: f32 = 0.0
    a14: f32 = 0.0
    a15: f32 = 0.0
    for j in range(li, n16, 64):
        p0: u32 = w_packed[r0 * n16 + j]
        p1: u32 = w_packed[r1 * n16 + j]
        p2: u32 = w_packed[r2 * n16 + j]
        p3: u32 = w_packed[r3 * n16 + j]
        p4: u32 = w_packed[r4 * n16 + j]
        p5: u32 = w_packed[r5 * n16 + j]
        p6: u32 = w_packed[r6 * n16 + j]
        p7: u32 = w_packed[r7 * n16 + j]
        p8: u32 = w_packed[r8 * n16 + j]
        p9: u32 = w_packed[r9 * n16 + j]
        p10: u32 = w_packed[r10 * n16 + j]
        p11: u32 = w_packed[r11 * n16 + j]
        p12: u32 = w_packed[r12 * n16 + j]
        p13: u32 = w_packed[r13 * n16 + j]
        p14: u32 = w_packed[r14 * n16 + j]
        p15: u32 = w_packed[r15 * n16 + j]
        b: u32 = 16 * j
        a0 += f32(i32(p0 & 3) - 2) * x_in[b]
        a1 += f32(i32(p1 & 3) - 2) * x_in[b]
        a2 += f32(i32(p2 & 3) - 2) * x_in[b]
        a3 += f32(i32(p3 & 3) - 2) * x_in[b]
        a4 += f32(i32(p4 & 3) - 2) * x_in[b]
        a5 += f32(i32(p5 & 3) - 2) * x_in[b]
        a6 += f32(i32(p6 & 3) - 2) * x_in[b]
        a7 += f32(i32(p7 & 3) - 2) * x_in[b]
        a8 += f32(i32(p8 & 3) - 2) * x_in[b]
        a9 += f32(i32(p9 & 3) - 2) * x_in[b]
        a10 += f32(i32(p10 & 3) - 2) * x_in[b]
        a11 += f32(i32(p11 & 3) - 2) * x_in[b]
        a12 += f32(i32(p12 & 3) - 2) * x_in[b]
        a13 += f32(i32(p13 & 3) - 2) * x_in[b]
        a14 += f32(i32(p14 & 3) - 2) * x_in[b]
        a15 += f32(i32(p15 & 3) - 2) * x_in[b]
        a0 += f32(i32((p0 >> 2) & 3) - 2) * x_in[b + 1]
        a1 += f32(i32((p1 >> 2) & 3) - 2) * x_in[b + 1]
        a2 += f32(i32((p2 >> 2) & 3) - 2) * x_in[b + 1]
        a3 += f32(i32((p3 >> 2) & 3) - 2) * x_in[b + 1]
        a4 += f32(i32((p4 >> 2) & 3) - 2) * x_in[b + 1]
        a5 += f32(i32((p5 >> 2) & 3) - 2) * x_in[b + 1]
        a6 += f32(i32((p6 >> 2) & 3) - 2) * x_in[b + 1]
        a7 += f32(i32((p7 >> 2) & 3) - 2) * x_in[b + 1]
        a8 += f32(i32((p8 >> 2) & 3) - 2) * x_in[b + 1]
        a9 += f32(i32((p9 >> 2) & 3) - 2) * x_in[b + 1]
        a10 += f32(i32((p10 >> 2) & 3) - 2) * x_in[b + 1]
        a11 += f32(i32((p11 >> 2) & 3) - 2) * x_in[b + 1]
        a12 += f32(i32((p12 >> 2) & 3) - 2) * x_in[b + 1]
        a13 += f32(i32((p13 >> 2) & 3) - 2) * x_in[b + 1]
        a14 += f32(i32((p14 >> 2) & 3) - 2) * x_in[b + 1]
        a15 += f32(i32((p15 >> 2) & 3) - 2) * x_in[b + 1]
        a0 += f32(i32((p0 >> 4) & 3) - 2) * x_in[b + 2]
        a1 += f32(i32((p1 >> 4) & 3) - 2) * x_in[b + 2]
        a2 += f32(i32((p2 >> 4) & 3) - 2) * x_in[b + 2]
        a3 += f32(i32((p3 >> 4) & 3) - 2) * x_in[b + 2]
        a4 += f32(i32((p4 >> 4) & 3) - 2) * x_in[b + 2]
        a5 += f32(i32((p5 >> 4) & 3) - 2) * x_in[b + 2]
        a6 += f32(i32((p6 >> 4) & 3) - 2) * x_in[b + 2]
        a7 += f32(i32((p7 >> 4) & 3) - 2) * x_in[b + 2]
        a8 += f32(i32((p8 >> 4) & 3) - 2) * x_in[b + 2]
        a9 += f32(i32((p9 >> 4) & 3) - 2) * x_in[b + 2]
        a10 += f32(i32((p10 >> 4) & 3) - 2) * x_in[b + 2]
        a11 += f32(i32((p11 >> 4) & 3) - 2) * x_in[b + 2]
        a12 += f32(i32((p12 >> 4) & 3) - 2) * x_in[b + 2]
        a13 += f32(i32((p13 >> 4) & 3) - 2) * x_in[b + 2]
        a14 += f32(i32((p14 >> 4) & 3) - 2) * x_in[b + 2]
        a15 += f32(i32((p15 >> 4) & 3) - 2) * x_in[b + 2]
        a0 += f32(i32((p0 >> 6) & 3) - 2) * x_in[b + 3]
        a1 += f32(i32((p1 >> 6) & 3) - 2) * x_in[b + 3]
        a2 += f32(i32((p2 >> 6) & 3) - 2) * x_in[b + 3]
        a3 += f32(i32((p3 >> 6) & 3) - 2) * x_in[b + 3]
        a4 += f32(i32((p4 >> 6) & 3) - 2) * x_in[b + 3]
        a5 += f32(i32((p5 >> 6) & 3) - 2) * x_in[b + 3]
        a6 += f32(i32((p6 >> 6) & 3) - 2) * x_in[b + 3]
        a7 += f32(i32((p7 >> 6) & 3) - 2) * x_in[b + 3]
        a8 += f32(i32((p8 >> 6) & 3) - 2) * x_in[b + 3]
        a9 += f32(i32((p9 >> 6) & 3) - 2) * x_in[b + 3]
        a10 += f32(i32((p10 >> 6) & 3) - 2) * x_in[b + 3]
        a11 += f32(i32((p11 >> 6) & 3) - 2) * x_in[b + 3]
        a12 += f32(i32((p12 >> 6) & 3) - 2) * x_in[b + 3]
        a13 += f32(i32((p13 >> 6) & 3) - 2) * x_in[b + 3]
        a14 += f32(i32((p14 >> 6) & 3) - 2) * x_in[b + 3]
        a15 += f32(i32((p15 >> 6) & 3) - 2) * x_in[b + 3]
        a0 += f32(i32((p0 >> 8) & 3) - 2) * x_in[b + 4]
        a1 += f32(i32((p1 >> 8) & 3) - 2) * x_in[b + 4]
        a2 += f32(i32((p2 >> 8) & 3) - 2) * x_in[b + 4]
        a3 += f32(i32((p3 >> 8) & 3) - 2) * x_in[b + 4]
        a4 += f32(i32((p4 >> 8) & 3) - 2) * x_in[b + 4]
        a5 += f32(i32((p5 >> 8) & 3) - 2) * x_in[b + 4]
        a6 += f32(i32((p6 >> 8) & 3) - 2) * x_in[b + 4]
        a7 += f32(i32((p7 >> 8) & 3) - 2) * x_in[b + 4]
        a8 += f32(i32((p8 >> 8) & 3) - 2) * x_in[b + 4]
        a9 += f32(i32((p9 >> 8) & 3) - 2) * x_in[b + 4]
        a10 += f32(i32((p10 >> 8) & 3) - 2) * x_in[b + 4]
        a11 += f32(i32((p11 >> 8) & 3) - 2) * x_in[b + 4]
        a12 += f32(i32((p12 >> 8) & 3) - 2) * x_in[b + 4]
        a13 += f32(i32((p13 >> 8) & 3) - 2) * x_in[b + 4]
        a14 += f32(i32((p14 >> 8) & 3) - 2) * x_in[b + 4]
        a15 += f32(i32((p15 >> 8) & 3) - 2) * x_in[b + 4]
        a0 += f32(i32((p0 >> 10) & 3) - 2) * x_in[b + 5]
        a1 += f32(i32((p1 >> 10) & 3) - 2) * x_in[b + 5]
        a2 += f32(i32((p2 >> 10) & 3) - 2) * x_in[b + 5]
        a3 += f32(i32((p3 >> 10) & 3) - 2) * x_in[b + 5]
        a4 += f32(i32((p4 >> 10) & 3) - 2) * x_in[b + 5]
        a5 += f32(i32((p5 >> 10) & 3) - 2) * x_in[b + 5]
        a6 += f32(i32((p6 >> 10) & 3) - 2) * x_in[b + 5]
        a7 += f32(i32((p7 >> 10) & 3) - 2) * x_in[b + 5]
        a8 += f32(i32((p8 >> 10) & 3) - 2) * x_in[b + 5]
        a9 += f32(i32((p9 >> 10) & 3) - 2) * x_in[b + 5]
        a10 += f32(i32((p10 >> 10) & 3) - 2) * x_in[b + 5]
        a11 += f32(i32((p11 >> 10) & 3) - 2) * x_in[b + 5]
        a12 += f32(i32((p12 >> 10) & 3) - 2) * x_in[b + 5]
        a13 += f32(i32((p13 >> 10) & 3) - 2) * x_in[b + 5]
        a14 += f32(i32((p14 >> 10) & 3) - 2) * x_in[b + 5]
        a15 += f32(i32((p15 >> 10) & 3) - 2) * x_in[b + 5]
        a0 += f32(i32((p0 >> 12) & 3) - 2) * x_in[b + 6]
        a1 += f32(i32((p1 >> 12) & 3) - 2) * x_in[b + 6]
        a2 += f32(i32((p2 >> 12) & 3) - 2) * x_in[b + 6]
        a3 += f32(i32((p3 >> 12) & 3) - 2) * x_in[b + 6]
        a4 += f32(i32((p4 >> 12) & 3) - 2) * x_in[b + 6]
        a5 += f32(i32((p5 >> 12) & 3) - 2) * x_in[b + 6]
        a6 += f32(i32((p6 >> 12) & 3) - 2) * x_in[b + 6]
        a7 += f32(i32((p7 >> 12) & 3) - 2) * x_in[b + 6]
        a8 += f32(i32((p8 >> 12) & 3) - 2) * x_in[b + 6]
        a9 += f32(i32((p9 >> 12) & 3) - 2) * x_in[b + 6]
        a10 += f32(i32((p10 >> 12) & 3) - 2) * x_in[b + 6]
        a11 += f32(i32((p11 >> 12) & 3) - 2) * x_in[b + 6]
        a12 += f32(i32((p12 >> 12) & 3) - 2) * x_in[b + 6]
        a13 += f32(i32((p13 >> 12) & 3) - 2) * x_in[b + 6]
        a14 += f32(i32((p14 >> 12) & 3) - 2) * x_in[b + 6]
        a15 += f32(i32((p15 >> 12) & 3) - 2) * x_in[b + 6]
        a0 += f32(i32((p0 >> 14) & 3) - 2) * x_in[b + 7]
        a1 += f32(i32((p1 >> 14) & 3) - 2) * x_in[b + 7]
        a2 += f32(i32((p2 >> 14) & 3) - 2) * x_in[b + 7]
        a3 += f32(i32((p3 >> 14) & 3) - 2) * x_in[b + 7]
        a4 += f32(i32((p4 >> 14) & 3) - 2) * x_in[b + 7]
        a5 += f32(i32((p5 >> 14) & 3) - 2) * x_in[b + 7]
        a6 += f32(i32((p6 >> 14) & 3) - 2) * x_in[b + 7]
        a7 += f32(i32((p7 >> 14) & 3) - 2) * x_in[b + 7]
        a8 += f32(i32((p8 >> 14) & 3) - 2) * x_in[b + 7]
        a9 += f32(i32((p9 >> 14) & 3) - 2) * x_in[b + 7]
        a10 += f32(i32((p10 >> 14) & 3) - 2) * x_in[b + 7]
        a11 += f32(i32((p11 >> 14) & 3) - 2) * x_in[b + 7]
        a12 += f32(i32((p12 >> 14) & 3) - 2) * x_in[b + 7]
        a13 += f32(i32((p13 >> 14) & 3) - 2) * x_in[b + 7]
        a14 += f32(i32((p14 >> 14) & 3) - 2) * x_in[b + 7]
        a15 += f32(i32((p15 >> 14) & 3) - 2) * x_in[b + 7]
        a0 += f32(i32((p0 >> 16) & 3) - 2) * x_in[b + 8]
        a1 += f32(i32((p1 >> 16) & 3) - 2) * x_in[b + 8]
        a2 += f32(i32((p2 >> 16) & 3) - 2) * x_in[b + 8]
        a3 += f32(i32((p3 >> 16) & 3) - 2) * x_in[b + 8]
        a4 += f32(i32((p4 >> 16) & 3) - 2) * x_in[b + 8]
        a5 += f32(i32((p5 >> 16) & 3) - 2) * x_in[b + 8]
        a6 += f32(i32((p6 >> 16) & 3) - 2) * x_in[b + 8]
        a7 += f32(i32((p7 >> 16) & 3) - 2) * x_in[b + 8]
        a8 += f32(i32((p8 >> 16) & 3) - 2) * x_in[b + 8]
        a9 += f32(i32((p9 >> 16) & 3) - 2) * x_in[b + 8]
        a10 += f32(i32((p10 >> 16) & 3) - 2) * x_in[b + 8]
        a11 += f32(i32((p11 >> 16) & 3) - 2) * x_in[b + 8]
        a12 += f32(i32((p12 >> 16) & 3) - 2) * x_in[b + 8]
        a13 += f32(i32((p13 >> 16) & 3) - 2) * x_in[b + 8]
        a14 += f32(i32((p14 >> 16) & 3) - 2) * x_in[b + 8]
        a15 += f32(i32((p15 >> 16) & 3) - 2) * x_in[b + 8]
        a0 += f32(i32((p0 >> 18) & 3) - 2) * x_in[b + 9]
        a1 += f32(i32((p1 >> 18) & 3) - 2) * x_in[b + 9]
        a2 += f32(i32((p2 >> 18) & 3) - 2) * x_in[b + 9]
        a3 += f32(i32((p3 >> 18) & 3) - 2) * x_in[b + 9]
        a4 += f32(i32((p4 >> 18) & 3) - 2) * x_in[b + 9]
        a5 += f32(i32((p5 >> 18) & 3) - 2) * x_in[b + 9]
        a6 += f32(i32((p6 >> 18) & 3) - 2) * x_in[b + 9]
        a7 += f32(i32((p7 >> 18) & 3) - 2) * x_in[b + 9]
        a8 += f32(i32((p8 >> 18) & 3) - 2) * x_in[b + 9]
        a9 += f32(i32((p9 >> 18) & 3) - 2) * x_in[b + 9]
        a10 += f32(i32((p10 >> 18) & 3) - 2) * x_in[b + 9]
        a11 += f32(i32((p11 >> 18) & 3) - 2) * x_in[b + 9]
        a12 += f32(i32((p12 >> 18) & 3) - 2) * x_in[b + 9]
        a13 += f32(i32((p13 >> 18) & 3) - 2) * x_in[b + 9]
        a14 += f32(i32((p14 >> 18) & 3) - 2) * x_in[b + 9]
        a15 += f32(i32((p15 >> 18) & 3) - 2) * x_in[b + 9]
        a0 += f32(i32((p0 >> 20) & 3) - 2) * x_in[b + 10]
        a1 += f32(i32((p1 >> 20) & 3) - 2) * x_in[b + 10]
        a2 += f32(i32((p2 >> 20) & 3) - 2) * x_in[b + 10]
        a3 += f32(i32((p3 >> 20) & 3) - 2) * x_in[b + 10]
        a4 += f32(i32((p4 >> 20) & 3) - 2) * x_in[b + 10]
        a5 += f32(i32((p5 >> 20) & 3) - 2) * x_in[b + 10]
        a6 += f32(i32((p6 >> 20) & 3) - 2) * x_in[b + 10]
        a7 += f32(i32((p7 >> 20) & 3) - 2) * x_in[b + 10]
        a8 += f32(i32((p8 >> 20) & 3) - 2) * x_in[b + 10]
        a9 += f32(i32((p9 >> 20) & 3) - 2) * x_in[b + 10]
        a10 += f32(i32((p10 >> 20) & 3) - 2) * x_in[b + 10]
        a11 += f32(i32((p11 >> 20) & 3) - 2) * x_in[b + 10]
        a12 += f32(i32((p12 >> 20) & 3) - 2) * x_in[b + 10]
        a13 += f32(i32((p13 >> 20) & 3) - 2) * x_in[b + 10]
        a14 += f32(i32((p14 >> 20) & 3) - 2) * x_in[b + 10]
        a15 += f32(i32((p15 >> 20) & 3) - 2) * x_in[b + 10]
        a0 += f32(i32((p0 >> 22) & 3) - 2) * x_in[b + 11]
        a1 += f32(i32((p1 >> 22) & 3) - 2) * x_in[b + 11]
        a2 += f32(i32((p2 >> 22) & 3) - 2) * x_in[b + 11]
        a3 += f32(i32((p3 >> 22) & 3) - 2) * x_in[b + 11]
        a4 += f32(i32((p4 >> 22) & 3) - 2) * x_in[b + 11]
        a5 += f32(i32((p5 >> 22) & 3) - 2) * x_in[b + 11]
        a6 += f32(i32((p6 >> 22) & 3) - 2) * x_in[b + 11]
        a7 += f32(i32((p7 >> 22) & 3) - 2) * x_in[b + 11]
        a8 += f32(i32((p8 >> 22) & 3) - 2) * x_in[b + 11]
        a9 += f32(i32((p9 >> 22) & 3) - 2) * x_in[b + 11]
        a10 += f32(i32((p10 >> 22) & 3) - 2) * x_in[b + 11]
        a11 += f32(i32((p11 >> 22) & 3) - 2) * x_in[b + 11]
        a12 += f32(i32((p12 >> 22) & 3) - 2) * x_in[b + 11]
        a13 += f32(i32((p13 >> 22) & 3) - 2) * x_in[b + 11]
        a14 += f32(i32((p14 >> 22) & 3) - 2) * x_in[b + 11]
        a15 += f32(i32((p15 >> 22) & 3) - 2) * x_in[b + 11]
        a0 += f32(i32((p0 >> 24) & 3) - 2) * x_in[b + 12]
        a1 += f32(i32((p1 >> 24) & 3) - 2) * x_in[b + 12]
        a2 += f32(i32((p2 >> 24) & 3) - 2) * x_in[b + 12]
        a3 += f32(i32((p3 >> 24) & 3) - 2) * x_in[b + 12]
        a4 += f32(i32((p4 >> 24) & 3) - 2) * x_in[b + 12]
        a5 += f32(i32((p5 >> 24) & 3) - 2) * x_in[b + 12]
        a6 += f32(i32((p6 >> 24) & 3) - 2) * x_in[b + 12]
        a7 += f32(i32((p7 >> 24) & 3) - 2) * x_in[b + 12]
        a8 += f32(i32((p8 >> 24) & 3) - 2) * x_in[b + 12]
        a9 += f32(i32((p9 >> 24) & 3) - 2) * x_in[b + 12]
        a10 += f32(i32((p10 >> 24) & 3) - 2) * x_in[b + 12]
        a11 += f32(i32((p11 >> 24) & 3) - 2) * x_in[b + 12]
        a12 += f32(i32((p12 >> 24) & 3) - 2) * x_in[b + 12]
        a13 += f32(i32((p13 >> 24) & 3) - 2) * x_in[b + 12]
        a14 += f32(i32((p14 >> 24) & 3) - 2) * x_in[b + 12]
        a15 += f32(i32((p15 >> 24) & 3) - 2) * x_in[b + 12]
        a0 += f32(i32((p0 >> 26) & 3) - 2) * x_in[b + 13]
        a1 += f32(i32((p1 >> 26) & 3) - 2) * x_in[b + 13]
        a2 += f32(i32((p2 >> 26) & 3) - 2) * x_in[b + 13]
        a3 += f32(i32((p3 >> 26) & 3) - 2) * x_in[b + 13]
        a4 += f32(i32((p4 >> 26) & 3) - 2) * x_in[b + 13]
        a5 += f32(i32((p5 >> 26) & 3) - 2) * x_in[b + 13]
        a6 += f32(i32((p6 >> 26) & 3) - 2) * x_in[b + 13]
        a7 += f32(i32((p7 >> 26) & 3) - 2) * x_in[b + 13]
        a8 += f32(i32((p8 >> 26) & 3) - 2) * x_in[b + 13]
        a9 += f32(i32((p9 >> 26) & 3) - 2) * x_in[b + 13]
        a10 += f32(i32((p10 >> 26) & 3) - 2) * x_in[b + 13]
        a11 += f32(i32((p11 >> 26) & 3) - 2) * x_in[b + 13]
        a12 += f32(i32((p12 >> 26) & 3) - 2) * x_in[b + 13]
        a13 += f32(i32((p13 >> 26) & 3) - 2) * x_in[b + 13]
        a14 += f32(i32((p14 >> 26) & 3) - 2) * x_in[b + 13]
        a15 += f32(i32((p15 >> 26) & 3) - 2) * x_in[b + 13]
        a0 += f32(i32((p0 >> 28) & 3) - 2) * x_in[b + 14]
        a1 += f32(i32((p1 >> 28) & 3) - 2) * x_in[b + 14]
        a2 += f32(i32((p2 >> 28) & 3) - 2) * x_in[b + 14]
        a3 += f32(i32((p3 >> 28) & 3) - 2) * x_in[b + 14]
        a4 += f32(i32((p4 >> 28) & 3) - 2) * x_in[b + 14]
        a5 += f32(i32((p5 >> 28) & 3) - 2) * x_in[b + 14]
        a6 += f32(i32((p6 >> 28) & 3) - 2) * x_in[b + 14]
        a7 += f32(i32((p7 >> 28) & 3) - 2) * x_in[b + 14]
        a8 += f32(i32((p8 >> 28) & 3) - 2) * x_in[b + 14]
        a9 += f32(i32((p9 >> 28) & 3) - 2) * x_in[b + 14]
        a10 += f32(i32((p10 >> 28) & 3) - 2) * x_in[b + 14]
        a11 += f32(i32((p11 >> 28) & 3) - 2) * x_in[b + 14]
        a12 += f32(i32((p12 >> 28) & 3) - 2) * x_in[b + 14]
        a13 += f32(i32((p13 >> 28) & 3) - 2) * x_in[b + 14]
        a14 += f32(i32((p14 >> 28) & 3) - 2) * x_in[b + 14]
        a15 += f32(i32((p15 >> 28) & 3) - 2) * x_in[b + 14]
        a0 += f32(i32((p0 >> 30) & 3) - 2) * x_in[b + 15]
        a1 += f32(i32((p1 >> 30) & 3) - 2) * x_in[b + 15]
        a2 += f32(i32((p2 >> 30) & 3) - 2) * x_in[b + 15]
        a3 += f32(i32((p3 >> 30) & 3) - 2) * x_in[b + 15]
        a4 += f32(i32((p4 >> 30) & 3) - 2) * x_in[b + 15]
        a5 += f32(i32((p5 >> 30) & 3) - 2) * x_in[b + 15]
        a6 += f32(i32((p6 >> 30) & 3) - 2) * x_in[b + 15]
        a7 += f32(i32((p7 >> 30) & 3) - 2) * x_in[b + 15]
        a8 += f32(i32((p8 >> 30) & 3) - 2) * x_in[b + 15]
        a9 += f32(i32((p9 >> 30) & 3) - 2) * x_in[b + 15]
        a10 += f32(i32((p10 >> 30) & 3) - 2) * x_in[b + 15]
        a11 += f32(i32((p11 >> 30) & 3) - 2) * x_in[b + 15]
        a12 += f32(i32((p12 >> 30) & 3) - 2) * x_in[b + 15]
        a13 += f32(i32((p13 >> 30) & 3) - 2) * x_in[b + 15]
        a14 += f32(i32((p14 >> 30) & 3) - 2) * x_in[b + 15]
        a15 += f32(i32((p15 >> 30) & 3) - 2) * x_in[b + 15]
    q0[li] = a0
    q1[li] = a1
    q2[li] = a2
    q3[li] = a3
    q4[li] = a4
    q5[li] = a5
    q6[li] = a6
    q7[li] = a7
    q8[li] = a8
    q9[li] = a9
    q10[li] = a10
    q11[li] = a11
    q12[li] = a12
    q13[li] = a13
    q14[li] = a14
    q15[li] = a15
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
            q8[li] = q8[li] + q8[li + s]
            q9[li] = q9[li] + q9[li + s]
            q10[li] = q10[li] + q10[li + s]
            q11[li] = q11[li] + q11[li + s]
            q12[li] = q12[li] + q12[li + s]
            q13[li] = q13[li] + q13[li + s]
            q14[li] = q14[li] + q14[li + s]
            q15[li] = q15[li] + q15[li + s]
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
        y_out[r8] = q8[0] * scale[r8]
        y_out[r9] = q9[0] * scale[r9]
        y_out[r10] = q10[0] * scale[r10]
        y_out[r11] = q11[0] * scale[r11]
        y_out[r12] = q12[0] * scale[r12]
        y_out[r13] = q13[0] * scale[r13]
        y_out[r14] = q14[0] * scale[r14]
        y_out[r15] = q15[0] * scale[r15]


@kernel(workgroup_size=(64,))
def mv_gateup_geglu_dq2_blk8(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    gate_w: StorageBuffer[u32, "read"],       # [n_out, n_in/16] int2 gate
    up_w: StorageBuffer[u32, "read"],         # [n_out, n_in/16] int2 up
    x_in: StorageBuffer[f32, "read"],         # [n_in]
    gscale: StorageBuffer[f32, "read"],       # [n_out]
    uscale: StorageBuffer[f32, "read"],       # [n_out]
    y_out: StorageBuffer[f32, "read_write"],  # [n_out] = geglu(gate, up)
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in] (n_out % 8 == 0)
    pg0: WorkgroupArray[f32, 64],
    pu0: WorkgroupArray[f32, 64],
    pg1: WorkgroupArray[f32, 64],
    pu1: WorkgroupArray[f32, 64],
    pg2: WorkgroupArray[f32, 64],
    pu2: WorkgroupArray[f32, 64],
    pg3: WorkgroupArray[f32, 64],
    pu3: WorkgroupArray[f32, 64],
    pg4: WorkgroupArray[f32, 64],
    pu4: WorkgroupArray[f32, 64],
    pg5: WorkgroupArray[f32, 64],
    pu5: WorkgroupArray[f32, 64],
    pg6: WorkgroupArray[f32, 64],
    pu6: WorkgroupArray[f32, 64],
    pg7: WorkgroupArray[f32, 64],
    pu7: WorkgroupArray[f32, 64],
):
    # Output-blocked fused gate+up+geglu: 8 rows/workgroup, x read once
    # per element for all 16 weight rows (gate+up x 8).
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
    g0: f32 = 0.0
    u0: f32 = 0.0
    g1: f32 = 0.0
    u1: f32 = 0.0
    g2: f32 = 0.0
    u2: f32 = 0.0
    g3: f32 = 0.0
    u3: f32 = 0.0
    g4: f32 = 0.0
    u4: f32 = 0.0
    g5: f32 = 0.0
    u5: f32 = 0.0
    g6: f32 = 0.0
    u6: f32 = 0.0
    g7: f32 = 0.0
    u7: f32 = 0.0
    for j in range(li, n16, 64):
        gw0: u32 = gate_w[r0 * n16 + j]
        uw0: u32 = up_w[r0 * n16 + j]
        gw1: u32 = gate_w[r1 * n16 + j]
        uw1: u32 = up_w[r1 * n16 + j]
        gw2: u32 = gate_w[r2 * n16 + j]
        uw2: u32 = up_w[r2 * n16 + j]
        gw3: u32 = gate_w[r3 * n16 + j]
        uw3: u32 = up_w[r3 * n16 + j]
        gw4: u32 = gate_w[r4 * n16 + j]
        uw4: u32 = up_w[r4 * n16 + j]
        gw5: u32 = gate_w[r5 * n16 + j]
        uw5: u32 = up_w[r5 * n16 + j]
        gw6: u32 = gate_w[r6 * n16 + j]
        uw6: u32 = up_w[r6 * n16 + j]
        gw7: u32 = gate_w[r7 * n16 + j]
        uw7: u32 = up_w[r7 * n16 + j]
        b: u32 = 16 * j
        g0 += f32(i32(gw0 & 3) - 2) * x_in[b]
        u0 += f32(i32(uw0 & 3) - 2) * x_in[b]
        g1 += f32(i32(gw1 & 3) - 2) * x_in[b]
        u1 += f32(i32(uw1 & 3) - 2) * x_in[b]
        g2 += f32(i32(gw2 & 3) - 2) * x_in[b]
        u2 += f32(i32(uw2 & 3) - 2) * x_in[b]
        g3 += f32(i32(gw3 & 3) - 2) * x_in[b]
        u3 += f32(i32(uw3 & 3) - 2) * x_in[b]
        g4 += f32(i32(gw4 & 3) - 2) * x_in[b]
        u4 += f32(i32(uw4 & 3) - 2) * x_in[b]
        g5 += f32(i32(gw5 & 3) - 2) * x_in[b]
        u5 += f32(i32(uw5 & 3) - 2) * x_in[b]
        g6 += f32(i32(gw6 & 3) - 2) * x_in[b]
        u6 += f32(i32(uw6 & 3) - 2) * x_in[b]
        g7 += f32(i32(gw7 & 3) - 2) * x_in[b]
        u7 += f32(i32(uw7 & 3) - 2) * x_in[b]
        g0 += f32(i32((gw0 >> 2) & 3) - 2) * x_in[b + 1]
        u0 += f32(i32((uw0 >> 2) & 3) - 2) * x_in[b + 1]
        g1 += f32(i32((gw1 >> 2) & 3) - 2) * x_in[b + 1]
        u1 += f32(i32((uw1 >> 2) & 3) - 2) * x_in[b + 1]
        g2 += f32(i32((gw2 >> 2) & 3) - 2) * x_in[b + 1]
        u2 += f32(i32((uw2 >> 2) & 3) - 2) * x_in[b + 1]
        g3 += f32(i32((gw3 >> 2) & 3) - 2) * x_in[b + 1]
        u3 += f32(i32((uw3 >> 2) & 3) - 2) * x_in[b + 1]
        g4 += f32(i32((gw4 >> 2) & 3) - 2) * x_in[b + 1]
        u4 += f32(i32((uw4 >> 2) & 3) - 2) * x_in[b + 1]
        g5 += f32(i32((gw5 >> 2) & 3) - 2) * x_in[b + 1]
        u5 += f32(i32((uw5 >> 2) & 3) - 2) * x_in[b + 1]
        g6 += f32(i32((gw6 >> 2) & 3) - 2) * x_in[b + 1]
        u6 += f32(i32((uw6 >> 2) & 3) - 2) * x_in[b + 1]
        g7 += f32(i32((gw7 >> 2) & 3) - 2) * x_in[b + 1]
        u7 += f32(i32((uw7 >> 2) & 3) - 2) * x_in[b + 1]
        g0 += f32(i32((gw0 >> 4) & 3) - 2) * x_in[b + 2]
        u0 += f32(i32((uw0 >> 4) & 3) - 2) * x_in[b + 2]
        g1 += f32(i32((gw1 >> 4) & 3) - 2) * x_in[b + 2]
        u1 += f32(i32((uw1 >> 4) & 3) - 2) * x_in[b + 2]
        g2 += f32(i32((gw2 >> 4) & 3) - 2) * x_in[b + 2]
        u2 += f32(i32((uw2 >> 4) & 3) - 2) * x_in[b + 2]
        g3 += f32(i32((gw3 >> 4) & 3) - 2) * x_in[b + 2]
        u3 += f32(i32((uw3 >> 4) & 3) - 2) * x_in[b + 2]
        g4 += f32(i32((gw4 >> 4) & 3) - 2) * x_in[b + 2]
        u4 += f32(i32((uw4 >> 4) & 3) - 2) * x_in[b + 2]
        g5 += f32(i32((gw5 >> 4) & 3) - 2) * x_in[b + 2]
        u5 += f32(i32((uw5 >> 4) & 3) - 2) * x_in[b + 2]
        g6 += f32(i32((gw6 >> 4) & 3) - 2) * x_in[b + 2]
        u6 += f32(i32((uw6 >> 4) & 3) - 2) * x_in[b + 2]
        g7 += f32(i32((gw7 >> 4) & 3) - 2) * x_in[b + 2]
        u7 += f32(i32((uw7 >> 4) & 3) - 2) * x_in[b + 2]
        g0 += f32(i32((gw0 >> 6) & 3) - 2) * x_in[b + 3]
        u0 += f32(i32((uw0 >> 6) & 3) - 2) * x_in[b + 3]
        g1 += f32(i32((gw1 >> 6) & 3) - 2) * x_in[b + 3]
        u1 += f32(i32((uw1 >> 6) & 3) - 2) * x_in[b + 3]
        g2 += f32(i32((gw2 >> 6) & 3) - 2) * x_in[b + 3]
        u2 += f32(i32((uw2 >> 6) & 3) - 2) * x_in[b + 3]
        g3 += f32(i32((gw3 >> 6) & 3) - 2) * x_in[b + 3]
        u3 += f32(i32((uw3 >> 6) & 3) - 2) * x_in[b + 3]
        g4 += f32(i32((gw4 >> 6) & 3) - 2) * x_in[b + 3]
        u4 += f32(i32((uw4 >> 6) & 3) - 2) * x_in[b + 3]
        g5 += f32(i32((gw5 >> 6) & 3) - 2) * x_in[b + 3]
        u5 += f32(i32((uw5 >> 6) & 3) - 2) * x_in[b + 3]
        g6 += f32(i32((gw6 >> 6) & 3) - 2) * x_in[b + 3]
        u6 += f32(i32((uw6 >> 6) & 3) - 2) * x_in[b + 3]
        g7 += f32(i32((gw7 >> 6) & 3) - 2) * x_in[b + 3]
        u7 += f32(i32((uw7 >> 6) & 3) - 2) * x_in[b + 3]
        g0 += f32(i32((gw0 >> 8) & 3) - 2) * x_in[b + 4]
        u0 += f32(i32((uw0 >> 8) & 3) - 2) * x_in[b + 4]
        g1 += f32(i32((gw1 >> 8) & 3) - 2) * x_in[b + 4]
        u1 += f32(i32((uw1 >> 8) & 3) - 2) * x_in[b + 4]
        g2 += f32(i32((gw2 >> 8) & 3) - 2) * x_in[b + 4]
        u2 += f32(i32((uw2 >> 8) & 3) - 2) * x_in[b + 4]
        g3 += f32(i32((gw3 >> 8) & 3) - 2) * x_in[b + 4]
        u3 += f32(i32((uw3 >> 8) & 3) - 2) * x_in[b + 4]
        g4 += f32(i32((gw4 >> 8) & 3) - 2) * x_in[b + 4]
        u4 += f32(i32((uw4 >> 8) & 3) - 2) * x_in[b + 4]
        g5 += f32(i32((gw5 >> 8) & 3) - 2) * x_in[b + 4]
        u5 += f32(i32((uw5 >> 8) & 3) - 2) * x_in[b + 4]
        g6 += f32(i32((gw6 >> 8) & 3) - 2) * x_in[b + 4]
        u6 += f32(i32((uw6 >> 8) & 3) - 2) * x_in[b + 4]
        g7 += f32(i32((gw7 >> 8) & 3) - 2) * x_in[b + 4]
        u7 += f32(i32((uw7 >> 8) & 3) - 2) * x_in[b + 4]
        g0 += f32(i32((gw0 >> 10) & 3) - 2) * x_in[b + 5]
        u0 += f32(i32((uw0 >> 10) & 3) - 2) * x_in[b + 5]
        g1 += f32(i32((gw1 >> 10) & 3) - 2) * x_in[b + 5]
        u1 += f32(i32((uw1 >> 10) & 3) - 2) * x_in[b + 5]
        g2 += f32(i32((gw2 >> 10) & 3) - 2) * x_in[b + 5]
        u2 += f32(i32((uw2 >> 10) & 3) - 2) * x_in[b + 5]
        g3 += f32(i32((gw3 >> 10) & 3) - 2) * x_in[b + 5]
        u3 += f32(i32((uw3 >> 10) & 3) - 2) * x_in[b + 5]
        g4 += f32(i32((gw4 >> 10) & 3) - 2) * x_in[b + 5]
        u4 += f32(i32((uw4 >> 10) & 3) - 2) * x_in[b + 5]
        g5 += f32(i32((gw5 >> 10) & 3) - 2) * x_in[b + 5]
        u5 += f32(i32((uw5 >> 10) & 3) - 2) * x_in[b + 5]
        g6 += f32(i32((gw6 >> 10) & 3) - 2) * x_in[b + 5]
        u6 += f32(i32((uw6 >> 10) & 3) - 2) * x_in[b + 5]
        g7 += f32(i32((gw7 >> 10) & 3) - 2) * x_in[b + 5]
        u7 += f32(i32((uw7 >> 10) & 3) - 2) * x_in[b + 5]
        g0 += f32(i32((gw0 >> 12) & 3) - 2) * x_in[b + 6]
        u0 += f32(i32((uw0 >> 12) & 3) - 2) * x_in[b + 6]
        g1 += f32(i32((gw1 >> 12) & 3) - 2) * x_in[b + 6]
        u1 += f32(i32((uw1 >> 12) & 3) - 2) * x_in[b + 6]
        g2 += f32(i32((gw2 >> 12) & 3) - 2) * x_in[b + 6]
        u2 += f32(i32((uw2 >> 12) & 3) - 2) * x_in[b + 6]
        g3 += f32(i32((gw3 >> 12) & 3) - 2) * x_in[b + 6]
        u3 += f32(i32((uw3 >> 12) & 3) - 2) * x_in[b + 6]
        g4 += f32(i32((gw4 >> 12) & 3) - 2) * x_in[b + 6]
        u4 += f32(i32((uw4 >> 12) & 3) - 2) * x_in[b + 6]
        g5 += f32(i32((gw5 >> 12) & 3) - 2) * x_in[b + 6]
        u5 += f32(i32((uw5 >> 12) & 3) - 2) * x_in[b + 6]
        g6 += f32(i32((gw6 >> 12) & 3) - 2) * x_in[b + 6]
        u6 += f32(i32((uw6 >> 12) & 3) - 2) * x_in[b + 6]
        g7 += f32(i32((gw7 >> 12) & 3) - 2) * x_in[b + 6]
        u7 += f32(i32((uw7 >> 12) & 3) - 2) * x_in[b + 6]
        g0 += f32(i32((gw0 >> 14) & 3) - 2) * x_in[b + 7]
        u0 += f32(i32((uw0 >> 14) & 3) - 2) * x_in[b + 7]
        g1 += f32(i32((gw1 >> 14) & 3) - 2) * x_in[b + 7]
        u1 += f32(i32((uw1 >> 14) & 3) - 2) * x_in[b + 7]
        g2 += f32(i32((gw2 >> 14) & 3) - 2) * x_in[b + 7]
        u2 += f32(i32((uw2 >> 14) & 3) - 2) * x_in[b + 7]
        g3 += f32(i32((gw3 >> 14) & 3) - 2) * x_in[b + 7]
        u3 += f32(i32((uw3 >> 14) & 3) - 2) * x_in[b + 7]
        g4 += f32(i32((gw4 >> 14) & 3) - 2) * x_in[b + 7]
        u4 += f32(i32((uw4 >> 14) & 3) - 2) * x_in[b + 7]
        g5 += f32(i32((gw5 >> 14) & 3) - 2) * x_in[b + 7]
        u5 += f32(i32((uw5 >> 14) & 3) - 2) * x_in[b + 7]
        g6 += f32(i32((gw6 >> 14) & 3) - 2) * x_in[b + 7]
        u6 += f32(i32((uw6 >> 14) & 3) - 2) * x_in[b + 7]
        g7 += f32(i32((gw7 >> 14) & 3) - 2) * x_in[b + 7]
        u7 += f32(i32((uw7 >> 14) & 3) - 2) * x_in[b + 7]
        g0 += f32(i32((gw0 >> 16) & 3) - 2) * x_in[b + 8]
        u0 += f32(i32((uw0 >> 16) & 3) - 2) * x_in[b + 8]
        g1 += f32(i32((gw1 >> 16) & 3) - 2) * x_in[b + 8]
        u1 += f32(i32((uw1 >> 16) & 3) - 2) * x_in[b + 8]
        g2 += f32(i32((gw2 >> 16) & 3) - 2) * x_in[b + 8]
        u2 += f32(i32((uw2 >> 16) & 3) - 2) * x_in[b + 8]
        g3 += f32(i32((gw3 >> 16) & 3) - 2) * x_in[b + 8]
        u3 += f32(i32((uw3 >> 16) & 3) - 2) * x_in[b + 8]
        g4 += f32(i32((gw4 >> 16) & 3) - 2) * x_in[b + 8]
        u4 += f32(i32((uw4 >> 16) & 3) - 2) * x_in[b + 8]
        g5 += f32(i32((gw5 >> 16) & 3) - 2) * x_in[b + 8]
        u5 += f32(i32((uw5 >> 16) & 3) - 2) * x_in[b + 8]
        g6 += f32(i32((gw6 >> 16) & 3) - 2) * x_in[b + 8]
        u6 += f32(i32((uw6 >> 16) & 3) - 2) * x_in[b + 8]
        g7 += f32(i32((gw7 >> 16) & 3) - 2) * x_in[b + 8]
        u7 += f32(i32((uw7 >> 16) & 3) - 2) * x_in[b + 8]
        g0 += f32(i32((gw0 >> 18) & 3) - 2) * x_in[b + 9]
        u0 += f32(i32((uw0 >> 18) & 3) - 2) * x_in[b + 9]
        g1 += f32(i32((gw1 >> 18) & 3) - 2) * x_in[b + 9]
        u1 += f32(i32((uw1 >> 18) & 3) - 2) * x_in[b + 9]
        g2 += f32(i32((gw2 >> 18) & 3) - 2) * x_in[b + 9]
        u2 += f32(i32((uw2 >> 18) & 3) - 2) * x_in[b + 9]
        g3 += f32(i32((gw3 >> 18) & 3) - 2) * x_in[b + 9]
        u3 += f32(i32((uw3 >> 18) & 3) - 2) * x_in[b + 9]
        g4 += f32(i32((gw4 >> 18) & 3) - 2) * x_in[b + 9]
        u4 += f32(i32((uw4 >> 18) & 3) - 2) * x_in[b + 9]
        g5 += f32(i32((gw5 >> 18) & 3) - 2) * x_in[b + 9]
        u5 += f32(i32((uw5 >> 18) & 3) - 2) * x_in[b + 9]
        g6 += f32(i32((gw6 >> 18) & 3) - 2) * x_in[b + 9]
        u6 += f32(i32((uw6 >> 18) & 3) - 2) * x_in[b + 9]
        g7 += f32(i32((gw7 >> 18) & 3) - 2) * x_in[b + 9]
        u7 += f32(i32((uw7 >> 18) & 3) - 2) * x_in[b + 9]
        g0 += f32(i32((gw0 >> 20) & 3) - 2) * x_in[b + 10]
        u0 += f32(i32((uw0 >> 20) & 3) - 2) * x_in[b + 10]
        g1 += f32(i32((gw1 >> 20) & 3) - 2) * x_in[b + 10]
        u1 += f32(i32((uw1 >> 20) & 3) - 2) * x_in[b + 10]
        g2 += f32(i32((gw2 >> 20) & 3) - 2) * x_in[b + 10]
        u2 += f32(i32((uw2 >> 20) & 3) - 2) * x_in[b + 10]
        g3 += f32(i32((gw3 >> 20) & 3) - 2) * x_in[b + 10]
        u3 += f32(i32((uw3 >> 20) & 3) - 2) * x_in[b + 10]
        g4 += f32(i32((gw4 >> 20) & 3) - 2) * x_in[b + 10]
        u4 += f32(i32((uw4 >> 20) & 3) - 2) * x_in[b + 10]
        g5 += f32(i32((gw5 >> 20) & 3) - 2) * x_in[b + 10]
        u5 += f32(i32((uw5 >> 20) & 3) - 2) * x_in[b + 10]
        g6 += f32(i32((gw6 >> 20) & 3) - 2) * x_in[b + 10]
        u6 += f32(i32((uw6 >> 20) & 3) - 2) * x_in[b + 10]
        g7 += f32(i32((gw7 >> 20) & 3) - 2) * x_in[b + 10]
        u7 += f32(i32((uw7 >> 20) & 3) - 2) * x_in[b + 10]
        g0 += f32(i32((gw0 >> 22) & 3) - 2) * x_in[b + 11]
        u0 += f32(i32((uw0 >> 22) & 3) - 2) * x_in[b + 11]
        g1 += f32(i32((gw1 >> 22) & 3) - 2) * x_in[b + 11]
        u1 += f32(i32((uw1 >> 22) & 3) - 2) * x_in[b + 11]
        g2 += f32(i32((gw2 >> 22) & 3) - 2) * x_in[b + 11]
        u2 += f32(i32((uw2 >> 22) & 3) - 2) * x_in[b + 11]
        g3 += f32(i32((gw3 >> 22) & 3) - 2) * x_in[b + 11]
        u3 += f32(i32((uw3 >> 22) & 3) - 2) * x_in[b + 11]
        g4 += f32(i32((gw4 >> 22) & 3) - 2) * x_in[b + 11]
        u4 += f32(i32((uw4 >> 22) & 3) - 2) * x_in[b + 11]
        g5 += f32(i32((gw5 >> 22) & 3) - 2) * x_in[b + 11]
        u5 += f32(i32((uw5 >> 22) & 3) - 2) * x_in[b + 11]
        g6 += f32(i32((gw6 >> 22) & 3) - 2) * x_in[b + 11]
        u6 += f32(i32((uw6 >> 22) & 3) - 2) * x_in[b + 11]
        g7 += f32(i32((gw7 >> 22) & 3) - 2) * x_in[b + 11]
        u7 += f32(i32((uw7 >> 22) & 3) - 2) * x_in[b + 11]
        g0 += f32(i32((gw0 >> 24) & 3) - 2) * x_in[b + 12]
        u0 += f32(i32((uw0 >> 24) & 3) - 2) * x_in[b + 12]
        g1 += f32(i32((gw1 >> 24) & 3) - 2) * x_in[b + 12]
        u1 += f32(i32((uw1 >> 24) & 3) - 2) * x_in[b + 12]
        g2 += f32(i32((gw2 >> 24) & 3) - 2) * x_in[b + 12]
        u2 += f32(i32((uw2 >> 24) & 3) - 2) * x_in[b + 12]
        g3 += f32(i32((gw3 >> 24) & 3) - 2) * x_in[b + 12]
        u3 += f32(i32((uw3 >> 24) & 3) - 2) * x_in[b + 12]
        g4 += f32(i32((gw4 >> 24) & 3) - 2) * x_in[b + 12]
        u4 += f32(i32((uw4 >> 24) & 3) - 2) * x_in[b + 12]
        g5 += f32(i32((gw5 >> 24) & 3) - 2) * x_in[b + 12]
        u5 += f32(i32((uw5 >> 24) & 3) - 2) * x_in[b + 12]
        g6 += f32(i32((gw6 >> 24) & 3) - 2) * x_in[b + 12]
        u6 += f32(i32((uw6 >> 24) & 3) - 2) * x_in[b + 12]
        g7 += f32(i32((gw7 >> 24) & 3) - 2) * x_in[b + 12]
        u7 += f32(i32((uw7 >> 24) & 3) - 2) * x_in[b + 12]
        g0 += f32(i32((gw0 >> 26) & 3) - 2) * x_in[b + 13]
        u0 += f32(i32((uw0 >> 26) & 3) - 2) * x_in[b + 13]
        g1 += f32(i32((gw1 >> 26) & 3) - 2) * x_in[b + 13]
        u1 += f32(i32((uw1 >> 26) & 3) - 2) * x_in[b + 13]
        g2 += f32(i32((gw2 >> 26) & 3) - 2) * x_in[b + 13]
        u2 += f32(i32((uw2 >> 26) & 3) - 2) * x_in[b + 13]
        g3 += f32(i32((gw3 >> 26) & 3) - 2) * x_in[b + 13]
        u3 += f32(i32((uw3 >> 26) & 3) - 2) * x_in[b + 13]
        g4 += f32(i32((gw4 >> 26) & 3) - 2) * x_in[b + 13]
        u4 += f32(i32((uw4 >> 26) & 3) - 2) * x_in[b + 13]
        g5 += f32(i32((gw5 >> 26) & 3) - 2) * x_in[b + 13]
        u5 += f32(i32((uw5 >> 26) & 3) - 2) * x_in[b + 13]
        g6 += f32(i32((gw6 >> 26) & 3) - 2) * x_in[b + 13]
        u6 += f32(i32((uw6 >> 26) & 3) - 2) * x_in[b + 13]
        g7 += f32(i32((gw7 >> 26) & 3) - 2) * x_in[b + 13]
        u7 += f32(i32((uw7 >> 26) & 3) - 2) * x_in[b + 13]
        g0 += f32(i32((gw0 >> 28) & 3) - 2) * x_in[b + 14]
        u0 += f32(i32((uw0 >> 28) & 3) - 2) * x_in[b + 14]
        g1 += f32(i32((gw1 >> 28) & 3) - 2) * x_in[b + 14]
        u1 += f32(i32((uw1 >> 28) & 3) - 2) * x_in[b + 14]
        g2 += f32(i32((gw2 >> 28) & 3) - 2) * x_in[b + 14]
        u2 += f32(i32((uw2 >> 28) & 3) - 2) * x_in[b + 14]
        g3 += f32(i32((gw3 >> 28) & 3) - 2) * x_in[b + 14]
        u3 += f32(i32((uw3 >> 28) & 3) - 2) * x_in[b + 14]
        g4 += f32(i32((gw4 >> 28) & 3) - 2) * x_in[b + 14]
        u4 += f32(i32((uw4 >> 28) & 3) - 2) * x_in[b + 14]
        g5 += f32(i32((gw5 >> 28) & 3) - 2) * x_in[b + 14]
        u5 += f32(i32((uw5 >> 28) & 3) - 2) * x_in[b + 14]
        g6 += f32(i32((gw6 >> 28) & 3) - 2) * x_in[b + 14]
        u6 += f32(i32((uw6 >> 28) & 3) - 2) * x_in[b + 14]
        g7 += f32(i32((gw7 >> 28) & 3) - 2) * x_in[b + 14]
        u7 += f32(i32((uw7 >> 28) & 3) - 2) * x_in[b + 14]
        g0 += f32(i32((gw0 >> 30) & 3) - 2) * x_in[b + 15]
        u0 += f32(i32((uw0 >> 30) & 3) - 2) * x_in[b + 15]
        g1 += f32(i32((gw1 >> 30) & 3) - 2) * x_in[b + 15]
        u1 += f32(i32((uw1 >> 30) & 3) - 2) * x_in[b + 15]
        g2 += f32(i32((gw2 >> 30) & 3) - 2) * x_in[b + 15]
        u2 += f32(i32((uw2 >> 30) & 3) - 2) * x_in[b + 15]
        g3 += f32(i32((gw3 >> 30) & 3) - 2) * x_in[b + 15]
        u3 += f32(i32((uw3 >> 30) & 3) - 2) * x_in[b + 15]
        g4 += f32(i32((gw4 >> 30) & 3) - 2) * x_in[b + 15]
        u4 += f32(i32((uw4 >> 30) & 3) - 2) * x_in[b + 15]
        g5 += f32(i32((gw5 >> 30) & 3) - 2) * x_in[b + 15]
        u5 += f32(i32((uw5 >> 30) & 3) - 2) * x_in[b + 15]
        g6 += f32(i32((gw6 >> 30) & 3) - 2) * x_in[b + 15]
        u6 += f32(i32((uw6 >> 30) & 3) - 2) * x_in[b + 15]
        g7 += f32(i32((gw7 >> 30) & 3) - 2) * x_in[b + 15]
        u7 += f32(i32((uw7 >> 30) & 3) - 2) * x_in[b + 15]
    pg0[li] = g0
    pu0[li] = u0
    pg1[li] = g1
    pu1[li] = u1
    pg2[li] = g2
    pu2[li] = u2
    pg3[li] = g3
    pu3[li] = u3
    pg4[li] = g4
    pu4[li] = u4
    pg5[li] = g5
    pu5[li] = u5
    pg6[li] = g6
    pu6[li] = u6
    pg7[li] = g7
    pu7[li] = u7
    barrier()
    s: u32 = 32
    while s > 0:
        if li < s:
            pg0[li] = pg0[li] + pg0[li + s]
            pu0[li] = pu0[li] + pu0[li + s]
            pg1[li] = pg1[li] + pg1[li + s]
            pu1[li] = pu1[li] + pu1[li + s]
            pg2[li] = pg2[li] + pg2[li + s]
            pu2[li] = pu2[li] + pu2[li + s]
            pg3[li] = pg3[li] + pg3[li + s]
            pu3[li] = pu3[li] + pu3[li + s]
            pg4[li] = pg4[li] + pg4[li + s]
            pu4[li] = pu4[li] + pu4[li + s]
            pg5[li] = pg5[li] + pg5[li + s]
            pu5[li] = pu5[li] + pu5[li + s]
            pg6[li] = pg6[li] + pg6[li + s]
            pu6[li] = pu6[li] + pu6[li + s]
            pg7[li] = pg7[li] + pg7[li + s]
            pu7[li] = pu7[li] + pu7[li + s]
        barrier()
        s = s / 2
    if li == 0:
        y_out[r0] = gelu(pg0[0] * gscale[r0]) * (pu0[0] * uscale[r0])
        y_out[r1] = gelu(pg1[0] * gscale[r1]) * (pu1[0] * uscale[r1])
        y_out[r2] = gelu(pg2[0] * gscale[r2]) * (pu2[0] * uscale[r2])
        y_out[r3] = gelu(pg3[0] * gscale[r3]) * (pu3[0] * uscale[r3])
        y_out[r4] = gelu(pg4[0] * gscale[r4]) * (pu4[0] * uscale[r4])
        y_out[r5] = gelu(pg5[0] * gscale[r5]) * (pu5[0] * uscale[r5])
        y_out[r6] = gelu(pg6[0] * gscale[r6]) * (pu6[0] * uscale[r6])
        y_out[r7] = gelu(pg7[0] * gscale[r7]) * (pu7[0] * uscale[r7])


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
    "matvec_dq4_blk2": matvec_dq4_blk2,
    "matvec_dq2": matvec_dq2,
    "matvec_dq2_blk2": matvec_dq2_blk2,
    "matvec_dq2_blk8": matvec_dq2_blk8,
    "matvec_dq2_blk16": matvec_dq2_blk16,
    "mv_gateup_geglu_dq2": mv_gateup_geglu_dq2,
    "mv_gateup_geglu_dq2_blk2": mv_gateup_geglu_dq2_blk2,
    "mv_gateup_geglu_dq2_blk4": mv_gateup_geglu_dq2_blk4,
    "mv_gateup_geglu_dq2_blk8": mv_gateup_geglu_dq2_blk8,
    "mv_gateup_geglu_dq4": mv_gateup_geglu_dq4,
    "mv_geglu_f16": mv_geglu_f16,
    "qat_embed_2bit": qat_embed_2bit,
    "qat_ple_gather_4bit": qat_ple_gather_4bit,
})


# ---------------------------------------------------------------------------
# Browser-only experiment: int8 activations + dot4I8Packed (hardware int dot).
# Kept OUT of KERNELS (qat_runner's naga may reject the `requires` directive;
# these need int8-quantized activations the runner doesn't produce). Served to
# the browser via qat_gendemo_server.BROWSER_EXTRA. Dawn/Apple emits the real
# hardware packed-int-dot instruction (wgpu-native only emulates it).
# ---------------------------------------------------------------------------

@kernel(workgroup_size=(256,))
def quant_i8(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    x_in: StorageBuffer[f32, "read"],            # [n]
    xq_out: StorageBuffer[u32, "read_write"],    # [n/4] four int8 per u32
    scale_out: StorageBuffer[f32, "read_write"], # [1] per-vector scale
    dims: StorageBuffer[u32, "read"],            # [n] (n % 4 == 0)
    smax: WorkgroupArray[f32, 256],
):
    # Dynamic per-vector symmetric int8 quantization: scale = max|x|/127,
    # xq = round(x/scale) clamped to [-127,127], packed 4 per u32 (byte order).
    li: u32 = lid.x
    n: u32 = dims[0]
    m: f32 = 0.0
    for j in range(li, n, 256):
        m = max(m, abs(x_in[j]))
    smax[li] = m
    barrier()
    s: u32 = 128
    while s > 0:
        if li < s:
            smax[li] = max(smax[li], smax[li + s])
        barrier()
        s = s / 2
    scale: f32 = smax[0] / 127.0
    inv: f32 = 0.0
    if scale > 0.0:
        inv = 1.0 / scale
    if li == 0:
        scale_out[0] = scale
    n4: u32 = n / 4
    for j in range(li, n4, 256):
        b: u32 = 4 * j
        v0: i32 = i32(clamp(round(x_in[b] * inv), -127.0, 127.0))
        v1: i32 = i32(clamp(round(x_in[b + 1] * inv), -127.0, 127.0))
        v2: i32 = i32(clamp(round(x_in[b + 2] * inv), -127.0, 127.0))
        v3: i32 = i32(clamp(round(x_in[b + 3] * inv), -127.0, 127.0))
        xq_out[j] = (u32(v0) & 255) | ((u32(v1) & 255) << 8) | \
                    ((u32(v2) & 255) << 16) | ((u32(v3) & 255) << 24)


@kernel(workgroup_size=(64,))
def matvec_dq2_i8(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    w_packed: StorageBuffer[u32, "read"],     # [n_out, n_in/16] int2
    xq: StorageBuffer[u32, "read"],           # [n_in/4] int8 packed activations
    wscale: StorageBuffer[f32, "read"],       # [n_out] per-row weight scale
    actscale: StorageBuffer[f32, "read"],     # [1] activation scale
    y_out: StorageBuffer[f32, "read_write"],  # [n_out]
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in]
    partial: WorkgroupArray[i32, 64],
):
    # int2 weights x int8 activations via dot4I8Packed: expand 4 int2 crumbs to
    # a packed-int8 u32, hardware-dot with 4 packed int8 activations, sum in i32,
    # then one f32 scale (weight per-row * activation) at the end.
    r: u32 = wid.x
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    n16: u32 = n_in / 16
    acc: i32 = 0
    if r < n_out:
        for j in range(li, n16, 64):
            p: u32 = w_packed[r * n16 + j]
            w0: u32 = (u32(i32(p & 3) - 2) & 255) | \
                      ((u32(i32((p >> 2) & 3) - 2) & 255) << 8) | \
                      ((u32(i32((p >> 4) & 3) - 2) & 255) << 16) | \
                      ((u32(i32((p >> 6) & 3) - 2) & 255) << 24)
            acc += dot4I8Packed(w0, xq[4 * j])
            w1: u32 = (u32(i32((p >> 8) & 3) - 2) & 255) | \
                      ((u32(i32((p >> 10) & 3) - 2) & 255) << 8) | \
                      ((u32(i32((p >> 12) & 3) - 2) & 255) << 16) | \
                      ((u32(i32((p >> 14) & 3) - 2) & 255) << 24)
            acc += dot4I8Packed(w1, xq[4 * j + 1])
            w2: u32 = (u32(i32((p >> 16) & 3) - 2) & 255) | \
                      ((u32(i32((p >> 18) & 3) - 2) & 255) << 8) | \
                      ((u32(i32((p >> 20) & 3) - 2) & 255) << 16) | \
                      ((u32(i32((p >> 22) & 3) - 2) & 255) << 24)
            acc += dot4I8Packed(w2, xq[4 * j + 2])
            w3: u32 = (u32(i32((p >> 24) & 3) - 2) & 255) | \
                      ((u32(i32((p >> 26) & 3) - 2) & 255) << 8) | \
                      ((u32(i32((p >> 28) & 3) - 2) & 255) << 16) | \
                      ((u32(i32((p >> 30) & 3) - 2) & 255) << 24)
            acc += dot4I8Packed(w3, xq[4 * j + 3])
    partial[li] = acc
    barrier()
    s: u32 = 32
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2
    if li == 0 and r < n_out:
        y_out[r] = f32(partial[0]) * wscale[r] * actscale[0]



# --- 256-thread norm variants (browser: better single-workgroup occupancy) ---

@kernel(workgroup_size=(256,))
def rmsnorm_wg_t256(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    x_in: StorageBuffer[f32, "read"],         # [n_rows, row_len]
    w: StorageBuffer[f32, "read"],            # [row_len]
    x_out: StorageBuffer[f32, "read_write"],  # [n_rows, row_len]
    dims: StorageBuffer[u32, "read"],         # [n_rows, row_len]
    partial: WorkgroupArray[f32, 256],
):
    # One workgroup per row: phase 1 reduces sum-of-squares in shared
    # memory, phase 2 has all 64 threads write the normalised row.
    row: u32 = wid.x
    li: u32 = lid.x
    n_rows: u32 = dims[0]
    row_len: u32 = dims[1]

    acc: f32 = 0.0
    if row < n_rows:
        for j in range(li, row_len, 256):
            v: f32 = x_in[row * row_len + j]
            acc += v * v
    partial[li] = acc
    barrier()

    s: u32 = 128
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2

    inv: f32 = 1.0 / sqrt(partial[0] / f32(row_len) + 1e-6)
    if row < n_rows:
        for j in range(li, row_len, 256):
            x_out[row * row_len + j] = x_in[row * row_len + j] * inv * (1.0 + w[j])

@kernel(workgroup_size=(256,))
def rmsnorm_add_wg_t256(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    x_in: StorageBuffer[f32, "read"],         # [n_rows, row_len] block output
    w: StorageBuffer[f32, "read"],            # [row_len] norm weight
    x_io: StorageBuffer[f32, "read_write"],   # [n_rows, row_len] residual accumulator
    dims: StorageBuffer[u32, "read"],         # [n_rows, row_len]
    partial: WorkgroupArray[f32, 256],
):
    # Fused post-norm + residual add: x_io += rmsnorm(x_in) * (1 + w)
    row: u32 = wid.x
    li: u32 = lid.x
    n_rows: u32 = dims[0]
    row_len: u32 = dims[1]

    acc: f32 = 0.0
    if row < n_rows:
        for j in range(li, row_len, 256):
            v: f32 = x_in[row * row_len + j]
            acc += v * v
    partial[li] = acc
    barrier()

    s: u32 = 128
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2

    inv: f32 = 1.0 / sqrt(partial[0] / f32(row_len) + 1e-6)
    if row < n_rows:
        for j in range(li, row_len, 256):
            idx: u32 = row * row_len + j
            x_io[idx] = x_io[idx] + x_in[idx] * inv * (1.0 + w[j])

@kernel(workgroup_size=(256,))
def rmsnorm_add_norm_wg_t256(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    src: StorageBuffer[f32, "read"],          # [n_rows, row_len] branch output
    w1: StorageBuffer[f32, "read"],           # [row_len] add-norm weight (w-1)
    w2: StorageBuffer[f32, "read"],           # [row_len] second-norm weight (w-1)
    x_io: StorageBuffer[f32, "read_write"],   # [n_rows, row_len] residual accumulator
    x_out: StorageBuffer[f32, "read_write"],  # [n_rows, row_len] second-norm output
    dims: StorageBuffer[u32, "read"],         # [n_rows, row_len]
    partial: WorkgroupArray[f32, 256],
):
    # Fuses rmsnorm_add + rmsnorm (Gemma 4 post-attention residual add-norm
    # then pre-FFN norm) in one workgroup, saving a dispatch and a re-read
    # of the residual from VRAM:
    #   x_io  += rmsnorm(src) * (1 + w1)
    #   x_out  = rmsnorm(x_io) * (1 + w2)
    row: u32 = wid.x
    li: u32 = lid.x
    n_rows: u32 = dims[0]
    row_len: u32 = dims[1]

    acc: f32 = 0.0
    if row < n_rows:
        for j in range(li, row_len, 256):
            v: f32 = src[row * row_len + j]
            acc += v * v
    partial[li] = acc
    barrier()
    s: u32 = 128
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2
    inv1: f32 = 1.0 / sqrt(partial[0] / f32(row_len) + 1e-6)
    barrier()  # all lanes read partial[0] before it is overwritten below

    acc2: f32 = 0.0
    if row < n_rows:
        for j in range(li, row_len, 256):
            idx: u32 = row * row_len + j
            v2: f32 = x_io[idx] + src[idx] * inv1 * (1.0 + w1[j])
            x_io[idx] = v2
            acc2 += v2 * v2
    partial[li] = acc2
    barrier()
    s = 128
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2
    inv2: f32 = 1.0 / sqrt(partial[0] / f32(row_len) + 1e-6)
    if row < n_rows:
        for j in range(li, row_len, 256):
            idx2: u32 = row * row_len + j
            x_out[idx2] = x_io[idx2] * inv2 * (1.0 + w2[j])

@kernel(workgroup_size=(256,))
def rmsnorm_add_scale_wg_t256(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    src: StorageBuffer[f32, "read"],          # [n_rows, row_len] PLE projection
    w: StorageBuffer[f32, "read"],            # [row_len] norm weight (w-1)
    x_io: StorageBuffer[f32, "read_write"],   # [n_rows, row_len] residual accumulator
    fparams: StorageBuffer[f32, "read"],      # [scale] = learned layer_scalar
    dims: StorageBuffer[u32, "read"],         # [n_rows, row_len]
    partial: WorkgroupArray[f32, 256],
):
    # Fuses the PLE post-norm with the scaled residual add that ends the
    # per-layer-embedding block (one dispatch, no separate ple_norm buffer):
    #   x_io = (x_io + rmsnorm(src) * (1 + w)) * scale
    row: u32 = wid.x
    li: u32 = lid.x
    n_rows: u32 = dims[0]
    row_len: u32 = dims[1]

    acc: f32 = 0.0
    if row < n_rows:
        for j in range(li, row_len, 256):
            v: f32 = src[row * row_len + j]
            acc += v * v
    partial[li] = acc
    barrier()
    s: u32 = 128
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2
    inv: f32 = 1.0 / sqrt(partial[0] / f32(row_len) + 1e-6)
    scale: f32 = fparams[0]
    if row < n_rows:
        for j in range(li, row_len, 256):
            idx: u32 = row * row_len + j
            x_io[idx] = (x_io[idx] + src[idx] * inv * (1.0 + w[j])) * scale


BROWSER_EXTRA_KERNELS = {
    "rmsnorm_wg_t256": rmsnorm_wg_t256,
    "rmsnorm_add_wg_t256": rmsnorm_add_wg_t256,
    "rmsnorm_add_norm_wg_t256": rmsnorm_add_norm_wg_t256,
    "rmsnorm_add_scale_wg_t256": rmsnorm_add_scale_wg_t256,
    "quant_i8": quant_i8,
    "matvec_dq2_i8": matvec_dq2_i8,
}


# --- flash-decoding attention (browser: attention runs grid=nh=8, occupancy-
# starved; split KV across nh*S workgroups + online-softmax combine). Neutral in
# wgpu-py but the browser rewards occupancy. dims = step_setup-updated [nh,hd,
# kv_len,start,max_seq]; S passed separately so it stays step_setup-compatible.
@kernel(workgroup_size=(64,))
def attn_flash_partial(
    wid: Builtin.workgroup_id,                 # (head, split)
    lid: Builtin.local_invocation_id,
    q: StorageBuffer[f32, "read"],             # [nh, hd]
    k_cache: StorageBuffer[f32, "read"],       # [max_seq, hd]
    v_cache: StorageBuffer[f32, "read"],       # [max_seq, hd]
    part_m: StorageBuffer[f32, "read_write"],  # [nh, S]
    part_l: StorageBuffer[f32, "read_write"],  # [nh, S]
    part_o: StorageBuffer[f32, "read_write"],  # [nh, S, hd]
    dims: StorageBuffer[u32, "read"],          # [nh, hd, kv_len, start, max_seq]
    sbuf: StorageBuffer[u32, "read"],          # [S]
    smem: WorkgroupArray[f32, 64],
    escore: WorkgroupArray[f32, 512],
):
    h: u32 = wid.x
    s: u32 = wid.y
    li: u32 = lid.x
    head_dim: u32 = dims[1]
    kv_len: u32 = dims[2]
    start: u32 = dims[3]
    S: u32 = sbuf[0]
    idx: u32 = h * S + s
    total: u32 = kv_len - start
    per: u32 = (total + S - 1) / S
    t0: u32 = start + s * per
    t1: u32 = min(t0 + per, kv_len)
    if t0 >= t1:
        if li == 0:
            part_m[idx] = -1e30
            part_l[idx] = 0.0
        for d in range(li, head_dim, 64):
            part_o[idx * head_dim + d] = 0.0
        return
    m: f32 = -1e30
    for t in range(t0 + li, t1, 64):
        a: f32 = 0.0
        for j in range(head_dim):
            a += q[h * head_dim + j] * k_cache[t * head_dim + j]
        escore[t - t0] = a
        m = max(m, a)
    smem[li] = m
    barrier()
    r: u32 = 32
    while r > 0:
        if li < r:
            smem[li] = max(smem[li], smem[li + r])
        barrier()
        r = r / 2
    mloc: f32 = smem[0]
    barrier()
    lsum: f32 = 0.0
    for t in range(t0 + li, t1, 64):
        e: f32 = exp(escore[t - t0] - mloc)
        escore[t - t0] = e
        lsum += e
    smem[li] = lsum
    barrier()
    r = 32
    while r > 0:
        if li < r:
            smem[li] = smem[li] + smem[li + r]
        barrier()
        r = r / 2
    if li == 0:
        part_m[idx] = mloc
        part_l[idx] = smem[0]
    barrier()
    for d in range(li, head_dim, 64):
        acc: f32 = 0.0
        for t in range(t0, t1):
            acc += escore[t - t0] * v_cache[t * head_dim + d]
        part_o[idx * head_dim + d] = acc


@kernel(workgroup_size=(64,))
def attn_flash_combine(
    wid: Builtin.workgroup_id,                 # (head,)
    lid: Builtin.local_invocation_id,
    part_m: StorageBuffer[f32, "read"],        # [nh, S]
    part_l: StorageBuffer[f32, "read"],        # [nh, S]
    part_o: StorageBuffer[f32, "read"],        # [nh, S, hd]
    out_vec: StorageBuffer[f32, "read_write"], # [nh, hd]
    dims: StorageBuffer[u32, "read"],          # [nh, head_dim, S]
):
    h: u32 = wid.x
    li: u32 = lid.x
    head_dim: u32 = dims[1]
    S: u32 = dims[2]
    M: f32 = -1e30
    for s in range(S):
        M = max(M, part_m[h * S + s])
    denom: f32 = 0.0
    for s in range(S):
        denom += exp(part_m[h * S + s] - M) * part_l[h * S + s]
    for d in range(li, head_dim, 64):
        acc: f32 = 0.0
        for s in range(S):
            acc += exp(part_m[h * S + s] - M) * part_o[(h * S + s) * head_dim + d]
        out_vec[h * head_dim + d] = acc / denom


# 128/256-thread down variants (browser: long n_in=12288 reduction)
@kernel(workgroup_size=(128,))
def matvec_dq2_blk2_t128(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    w_packed: StorageBuffer[u32, "read"],     # [n_out, n_in/16] (16 int2/u32)
    x_in: StorageBuffer[f32, "read"],         # [n_in]
    scale: StorageBuffer[f32, "read"],        # [n_out] per-row weight scale
    y_out: StorageBuffer[f32, "read_write"],  # [n_out]
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in] (n_out even)
    p0: WorkgroupArray[f32, 128],
    p1: WorkgroupArray[f32, 128],
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
    for j in range(li, n16, 128):
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
    s: u32 = 64
    while s > 0:
        if li < s:
            p0[li] = p0[li] + p0[li + s]
            p1[li] = p1[li] + p1[li + s]
        barrier()
        s = s / 2
    if li == 0:
        y_out[r0] = p0[0] * scale[r0]
        y_out[r1] = p1[0] * scale[r1]

@kernel(workgroup_size=(256,))
def matvec_dq2_blk2_t256(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    w_packed: StorageBuffer[u32, "read"],     # [n_out, n_in/16] (16 int2/u32)
    x_in: StorageBuffer[f32, "read"],         # [n_in]
    scale: StorageBuffer[f32, "read"],        # [n_out] per-row weight scale
    y_out: StorageBuffer[f32, "read_write"],  # [n_out]
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in] (n_out even)
    p0: WorkgroupArray[f32, 256],
    p1: WorkgroupArray[f32, 256],
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
    for j in range(li, n16, 256):
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
    s: u32 = 128
    while s > 0:
        if li < s:
            p0[li] = p0[li] + p0[li + s]
            p1[li] = p1[li] + p1[li + s]
        barrier()
        s = s / 2
    if li == 0:
        y_out[r0] = p0[0] * scale[r0]
        y_out[r1] = p1[0] * scale[r1]

@kernel(workgroup_size=(32,))
def mv_gateup_geglu_dq2_sg4(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    gate_w: StorageBuffer[u32, "read"],
    up_w: StorageBuffer[u32, "read"],
    x_in: StorageBuffer[f32, "read"],
    gscale: StorageBuffer[f32, "read"],
    uscale: StorageBuffer[f32, "read"],
    y_out: StorageBuffer[f32, "read_write"],
    dims: StorageBuffer[u32, "read"],
):
    # 4 rows/wg, subgroupAdd reduction (no shared memory, no barriers).
    r0: u32 = 4 * wid.x + 0
    r1: u32 = 4 * wid.x + 1
    r2: u32 = 4 * wid.x + 2
    r3: u32 = 4 * wid.x + 3
    li: u32 = lid.x
    n_in: u32 = dims[1]
    n16: u32 = n_in / 16
    g0: f32 = 0.0
    u0: f32 = 0.0
    g1: f32 = 0.0
    u1: f32 = 0.0
    g2: f32 = 0.0
    u2: f32 = 0.0
    g3: f32 = 0.0
    u3: f32 = 0.0
    for j in range(li, n16, 32):
        gw0: u32 = gate_w[r0 * n16 + j]
        uw0: u32 = up_w[r0 * n16 + j]
        gw1: u32 = gate_w[r1 * n16 + j]
        uw1: u32 = up_w[r1 * n16 + j]
        gw2: u32 = gate_w[r2 * n16 + j]
        uw2: u32 = up_w[r2 * n16 + j]
        gw3: u32 = gate_w[r3 * n16 + j]
        uw3: u32 = up_w[r3 * n16 + j]
        b: u32 = 16 * j
        g0 += f32(i32(gw0 & 3) - 2) * x_in[b]
        u0 += f32(i32(uw0 & 3) - 2) * x_in[b]
        g1 += f32(i32(gw1 & 3) - 2) * x_in[b]
        u1 += f32(i32(uw1 & 3) - 2) * x_in[b]
        g2 += f32(i32(gw2 & 3) - 2) * x_in[b]
        u2 += f32(i32(uw2 & 3) - 2) * x_in[b]
        g3 += f32(i32(gw3 & 3) - 2) * x_in[b]
        u3 += f32(i32(uw3 & 3) - 2) * x_in[b]
        g0 += f32(i32((gw0 >> 2) & 3) - 2) * x_in[b + 1]
        u0 += f32(i32((uw0 >> 2) & 3) - 2) * x_in[b + 1]
        g1 += f32(i32((gw1 >> 2) & 3) - 2) * x_in[b + 1]
        u1 += f32(i32((uw1 >> 2) & 3) - 2) * x_in[b + 1]
        g2 += f32(i32((gw2 >> 2) & 3) - 2) * x_in[b + 1]
        u2 += f32(i32((uw2 >> 2) & 3) - 2) * x_in[b + 1]
        g3 += f32(i32((gw3 >> 2) & 3) - 2) * x_in[b + 1]
        u3 += f32(i32((uw3 >> 2) & 3) - 2) * x_in[b + 1]
        g0 += f32(i32((gw0 >> 4) & 3) - 2) * x_in[b + 2]
        u0 += f32(i32((uw0 >> 4) & 3) - 2) * x_in[b + 2]
        g1 += f32(i32((gw1 >> 4) & 3) - 2) * x_in[b + 2]
        u1 += f32(i32((uw1 >> 4) & 3) - 2) * x_in[b + 2]
        g2 += f32(i32((gw2 >> 4) & 3) - 2) * x_in[b + 2]
        u2 += f32(i32((uw2 >> 4) & 3) - 2) * x_in[b + 2]
        g3 += f32(i32((gw3 >> 4) & 3) - 2) * x_in[b + 2]
        u3 += f32(i32((uw3 >> 4) & 3) - 2) * x_in[b + 2]
        g0 += f32(i32((gw0 >> 6) & 3) - 2) * x_in[b + 3]
        u0 += f32(i32((uw0 >> 6) & 3) - 2) * x_in[b + 3]
        g1 += f32(i32((gw1 >> 6) & 3) - 2) * x_in[b + 3]
        u1 += f32(i32((uw1 >> 6) & 3) - 2) * x_in[b + 3]
        g2 += f32(i32((gw2 >> 6) & 3) - 2) * x_in[b + 3]
        u2 += f32(i32((uw2 >> 6) & 3) - 2) * x_in[b + 3]
        g3 += f32(i32((gw3 >> 6) & 3) - 2) * x_in[b + 3]
        u3 += f32(i32((uw3 >> 6) & 3) - 2) * x_in[b + 3]
        g0 += f32(i32((gw0 >> 8) & 3) - 2) * x_in[b + 4]
        u0 += f32(i32((uw0 >> 8) & 3) - 2) * x_in[b + 4]
        g1 += f32(i32((gw1 >> 8) & 3) - 2) * x_in[b + 4]
        u1 += f32(i32((uw1 >> 8) & 3) - 2) * x_in[b + 4]
        g2 += f32(i32((gw2 >> 8) & 3) - 2) * x_in[b + 4]
        u2 += f32(i32((uw2 >> 8) & 3) - 2) * x_in[b + 4]
        g3 += f32(i32((gw3 >> 8) & 3) - 2) * x_in[b + 4]
        u3 += f32(i32((uw3 >> 8) & 3) - 2) * x_in[b + 4]
        g0 += f32(i32((gw0 >> 10) & 3) - 2) * x_in[b + 5]
        u0 += f32(i32((uw0 >> 10) & 3) - 2) * x_in[b + 5]
        g1 += f32(i32((gw1 >> 10) & 3) - 2) * x_in[b + 5]
        u1 += f32(i32((uw1 >> 10) & 3) - 2) * x_in[b + 5]
        g2 += f32(i32((gw2 >> 10) & 3) - 2) * x_in[b + 5]
        u2 += f32(i32((uw2 >> 10) & 3) - 2) * x_in[b + 5]
        g3 += f32(i32((gw3 >> 10) & 3) - 2) * x_in[b + 5]
        u3 += f32(i32((uw3 >> 10) & 3) - 2) * x_in[b + 5]
        g0 += f32(i32((gw0 >> 12) & 3) - 2) * x_in[b + 6]
        u0 += f32(i32((uw0 >> 12) & 3) - 2) * x_in[b + 6]
        g1 += f32(i32((gw1 >> 12) & 3) - 2) * x_in[b + 6]
        u1 += f32(i32((uw1 >> 12) & 3) - 2) * x_in[b + 6]
        g2 += f32(i32((gw2 >> 12) & 3) - 2) * x_in[b + 6]
        u2 += f32(i32((uw2 >> 12) & 3) - 2) * x_in[b + 6]
        g3 += f32(i32((gw3 >> 12) & 3) - 2) * x_in[b + 6]
        u3 += f32(i32((uw3 >> 12) & 3) - 2) * x_in[b + 6]
        g0 += f32(i32((gw0 >> 14) & 3) - 2) * x_in[b + 7]
        u0 += f32(i32((uw0 >> 14) & 3) - 2) * x_in[b + 7]
        g1 += f32(i32((gw1 >> 14) & 3) - 2) * x_in[b + 7]
        u1 += f32(i32((uw1 >> 14) & 3) - 2) * x_in[b + 7]
        g2 += f32(i32((gw2 >> 14) & 3) - 2) * x_in[b + 7]
        u2 += f32(i32((uw2 >> 14) & 3) - 2) * x_in[b + 7]
        g3 += f32(i32((gw3 >> 14) & 3) - 2) * x_in[b + 7]
        u3 += f32(i32((uw3 >> 14) & 3) - 2) * x_in[b + 7]
        g0 += f32(i32((gw0 >> 16) & 3) - 2) * x_in[b + 8]
        u0 += f32(i32((uw0 >> 16) & 3) - 2) * x_in[b + 8]
        g1 += f32(i32((gw1 >> 16) & 3) - 2) * x_in[b + 8]
        u1 += f32(i32((uw1 >> 16) & 3) - 2) * x_in[b + 8]
        g2 += f32(i32((gw2 >> 16) & 3) - 2) * x_in[b + 8]
        u2 += f32(i32((uw2 >> 16) & 3) - 2) * x_in[b + 8]
        g3 += f32(i32((gw3 >> 16) & 3) - 2) * x_in[b + 8]
        u3 += f32(i32((uw3 >> 16) & 3) - 2) * x_in[b + 8]
        g0 += f32(i32((gw0 >> 18) & 3) - 2) * x_in[b + 9]
        u0 += f32(i32((uw0 >> 18) & 3) - 2) * x_in[b + 9]
        g1 += f32(i32((gw1 >> 18) & 3) - 2) * x_in[b + 9]
        u1 += f32(i32((uw1 >> 18) & 3) - 2) * x_in[b + 9]
        g2 += f32(i32((gw2 >> 18) & 3) - 2) * x_in[b + 9]
        u2 += f32(i32((uw2 >> 18) & 3) - 2) * x_in[b + 9]
        g3 += f32(i32((gw3 >> 18) & 3) - 2) * x_in[b + 9]
        u3 += f32(i32((uw3 >> 18) & 3) - 2) * x_in[b + 9]
        g0 += f32(i32((gw0 >> 20) & 3) - 2) * x_in[b + 10]
        u0 += f32(i32((uw0 >> 20) & 3) - 2) * x_in[b + 10]
        g1 += f32(i32((gw1 >> 20) & 3) - 2) * x_in[b + 10]
        u1 += f32(i32((uw1 >> 20) & 3) - 2) * x_in[b + 10]
        g2 += f32(i32((gw2 >> 20) & 3) - 2) * x_in[b + 10]
        u2 += f32(i32((uw2 >> 20) & 3) - 2) * x_in[b + 10]
        g3 += f32(i32((gw3 >> 20) & 3) - 2) * x_in[b + 10]
        u3 += f32(i32((uw3 >> 20) & 3) - 2) * x_in[b + 10]
        g0 += f32(i32((gw0 >> 22) & 3) - 2) * x_in[b + 11]
        u0 += f32(i32((uw0 >> 22) & 3) - 2) * x_in[b + 11]
        g1 += f32(i32((gw1 >> 22) & 3) - 2) * x_in[b + 11]
        u1 += f32(i32((uw1 >> 22) & 3) - 2) * x_in[b + 11]
        g2 += f32(i32((gw2 >> 22) & 3) - 2) * x_in[b + 11]
        u2 += f32(i32((uw2 >> 22) & 3) - 2) * x_in[b + 11]
        g3 += f32(i32((gw3 >> 22) & 3) - 2) * x_in[b + 11]
        u3 += f32(i32((uw3 >> 22) & 3) - 2) * x_in[b + 11]
        g0 += f32(i32((gw0 >> 24) & 3) - 2) * x_in[b + 12]
        u0 += f32(i32((uw0 >> 24) & 3) - 2) * x_in[b + 12]
        g1 += f32(i32((gw1 >> 24) & 3) - 2) * x_in[b + 12]
        u1 += f32(i32((uw1 >> 24) & 3) - 2) * x_in[b + 12]
        g2 += f32(i32((gw2 >> 24) & 3) - 2) * x_in[b + 12]
        u2 += f32(i32((uw2 >> 24) & 3) - 2) * x_in[b + 12]
        g3 += f32(i32((gw3 >> 24) & 3) - 2) * x_in[b + 12]
        u3 += f32(i32((uw3 >> 24) & 3) - 2) * x_in[b + 12]
        g0 += f32(i32((gw0 >> 26) & 3) - 2) * x_in[b + 13]
        u0 += f32(i32((uw0 >> 26) & 3) - 2) * x_in[b + 13]
        g1 += f32(i32((gw1 >> 26) & 3) - 2) * x_in[b + 13]
        u1 += f32(i32((uw1 >> 26) & 3) - 2) * x_in[b + 13]
        g2 += f32(i32((gw2 >> 26) & 3) - 2) * x_in[b + 13]
        u2 += f32(i32((uw2 >> 26) & 3) - 2) * x_in[b + 13]
        g3 += f32(i32((gw3 >> 26) & 3) - 2) * x_in[b + 13]
        u3 += f32(i32((uw3 >> 26) & 3) - 2) * x_in[b + 13]
        g0 += f32(i32((gw0 >> 28) & 3) - 2) * x_in[b + 14]
        u0 += f32(i32((uw0 >> 28) & 3) - 2) * x_in[b + 14]
        g1 += f32(i32((gw1 >> 28) & 3) - 2) * x_in[b + 14]
        u1 += f32(i32((uw1 >> 28) & 3) - 2) * x_in[b + 14]
        g2 += f32(i32((gw2 >> 28) & 3) - 2) * x_in[b + 14]
        u2 += f32(i32((uw2 >> 28) & 3) - 2) * x_in[b + 14]
        g3 += f32(i32((gw3 >> 28) & 3) - 2) * x_in[b + 14]
        u3 += f32(i32((uw3 >> 28) & 3) - 2) * x_in[b + 14]
        g0 += f32(i32((gw0 >> 30) & 3) - 2) * x_in[b + 15]
        u0 += f32(i32((uw0 >> 30) & 3) - 2) * x_in[b + 15]
        g1 += f32(i32((gw1 >> 30) & 3) - 2) * x_in[b + 15]
        u1 += f32(i32((uw1 >> 30) & 3) - 2) * x_in[b + 15]
        g2 += f32(i32((gw2 >> 30) & 3) - 2) * x_in[b + 15]
        u2 += f32(i32((uw2 >> 30) & 3) - 2) * x_in[b + 15]
        g3 += f32(i32((gw3 >> 30) & 3) - 2) * x_in[b + 15]
        u3 += f32(i32((uw3 >> 30) & 3) - 2) * x_in[b + 15]
    sg0: f32 = subgroupAdd(g0)
    su0: f32 = subgroupAdd(u0)
    sg1: f32 = subgroupAdd(g1)
    su1: f32 = subgroupAdd(u1)
    sg2: f32 = subgroupAdd(g2)
    su2: f32 = subgroupAdd(u2)
    sg3: f32 = subgroupAdd(g3)
    su3: f32 = subgroupAdd(u3)
    if li == 0:
        y_out[r0] = gelu(sg0 * gscale[r0]) * (su0 * uscale[r0])
        y_out[r1] = gelu(sg1 * gscale[r1]) * (su1 * uscale[r1])
        y_out[r2] = gelu(sg2 * gscale[r2]) * (su2 * uscale[r2])
        y_out[r3] = gelu(sg3 * gscale[r3]) * (su3 * uscale[r3])


@kernel(workgroup_size=(64,))
def mv_gateup_geglu_dq4_blk2(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    gate_w: StorageBuffer[u32, "read"],
    up_w: StorageBuffer[u32, "read"],
    x_in: StorageBuffer[f32, "read"],
    gscale: StorageBuffer[f32, "read"],
    uscale: StorageBuffer[f32, "read"],
    y_out: StorageBuffer[f32, "read_write"],
    dims: StorageBuffer[u32, "read"],
    pg0: WorkgroupArray[f32, 64],
    pu0: WorkgroupArray[f32, 64],
    pg1: WorkgroupArray[f32, 64],
    pu1: WorkgroupArray[f32, 64],
):
    # int4 gate+up+geglu, 2 rows/wg (amortize x read); nibble unpack.
    r0: u32 = 2 * wid.x + 0
    r1: u32 = 2 * wid.x + 1
    li: u32 = lid.x
    n_in: u32 = dims[1]
    n8: u32 = n_in / 8
    g0: f32 = 0.0
    u0: f32 = 0.0
    g1: f32 = 0.0
    u1: f32 = 0.0
    for j in range(li, n8, 64):
        gw0: u32 = gate_w[r0 * n8 + j]
        uw0: u32 = up_w[r0 * n8 + j]
        gw1: u32 = gate_w[r1 * n8 + j]
        uw1: u32 = up_w[r1 * n8 + j]
        base: u32 = 8 * j
        g0 += f32(i32(gw0 & 15) - 8) * x_in[base]
        u0 += f32(i32(uw0 & 15) - 8) * x_in[base]
        g1 += f32(i32(gw1 & 15) - 8) * x_in[base]
        u1 += f32(i32(uw1 & 15) - 8) * x_in[base]
        g0 += f32(i32((gw0 >> 4) & 15) - 8) * x_in[base + 1]
        u0 += f32(i32((uw0 >> 4) & 15) - 8) * x_in[base + 1]
        g1 += f32(i32((gw1 >> 4) & 15) - 8) * x_in[base + 1]
        u1 += f32(i32((uw1 >> 4) & 15) - 8) * x_in[base + 1]
        g0 += f32(i32((gw0 >> 8) & 15) - 8) * x_in[base + 2]
        u0 += f32(i32((uw0 >> 8) & 15) - 8) * x_in[base + 2]
        g1 += f32(i32((gw1 >> 8) & 15) - 8) * x_in[base + 2]
        u1 += f32(i32((uw1 >> 8) & 15) - 8) * x_in[base + 2]
        g0 += f32(i32((gw0 >> 12) & 15) - 8) * x_in[base + 3]
        u0 += f32(i32((uw0 >> 12) & 15) - 8) * x_in[base + 3]
        g1 += f32(i32((gw1 >> 12) & 15) - 8) * x_in[base + 3]
        u1 += f32(i32((uw1 >> 12) & 15) - 8) * x_in[base + 3]
        g0 += f32(i32((gw0 >> 16) & 15) - 8) * x_in[base + 4]
        u0 += f32(i32((uw0 >> 16) & 15) - 8) * x_in[base + 4]
        g1 += f32(i32((gw1 >> 16) & 15) - 8) * x_in[base + 4]
        u1 += f32(i32((uw1 >> 16) & 15) - 8) * x_in[base + 4]
        g0 += f32(i32((gw0 >> 20) & 15) - 8) * x_in[base + 5]
        u0 += f32(i32((uw0 >> 20) & 15) - 8) * x_in[base + 5]
        g1 += f32(i32((gw1 >> 20) & 15) - 8) * x_in[base + 5]
        u1 += f32(i32((uw1 >> 20) & 15) - 8) * x_in[base + 5]
        g0 += f32(i32((gw0 >> 24) & 15) - 8) * x_in[base + 6]
        u0 += f32(i32((uw0 >> 24) & 15) - 8) * x_in[base + 6]
        g1 += f32(i32((gw1 >> 24) & 15) - 8) * x_in[base + 6]
        u1 += f32(i32((uw1 >> 24) & 15) - 8) * x_in[base + 6]
        g0 += f32(i32((gw0 >> 28) & 15) - 8) * x_in[base + 7]
        u0 += f32(i32((uw0 >> 28) & 15) - 8) * x_in[base + 7]
        g1 += f32(i32((gw1 >> 28) & 15) - 8) * x_in[base + 7]
        u1 += f32(i32((uw1 >> 28) & 15) - 8) * x_in[base + 7]
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



@kernel(workgroup_size=(64,))
def matvec_dq2_blk2_f16(
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
        a0 += f32(f16(i32(p & 3) - 2) * f16(x_in[b]))
        a1 += f32(f16(i32(q & 3) - 2) * f16(x_in[b]))
        a0 += f32(f16(i32((p >> 2) & 3) - 2) * f16(x_in[b + 1]))
        a1 += f32(f16(i32((q >> 2) & 3) - 2) * f16(x_in[b + 1]))
        a0 += f32(f16(i32((p >> 4) & 3) - 2) * f16(x_in[b + 2]))
        a1 += f32(f16(i32((q >> 4) & 3) - 2) * f16(x_in[b + 2]))
        a0 += f32(f16(i32((p >> 6) & 3) - 2) * f16(x_in[b + 3]))
        a1 += f32(f16(i32((q >> 6) & 3) - 2) * f16(x_in[b + 3]))
        a0 += f32(f16(i32((p >> 8) & 3) - 2) * f16(x_in[b + 4]))
        a1 += f32(f16(i32((q >> 8) & 3) - 2) * f16(x_in[b + 4]))
        a0 += f32(f16(i32((p >> 10) & 3) - 2) * f16(x_in[b + 5]))
        a1 += f32(f16(i32((q >> 10) & 3) - 2) * f16(x_in[b + 5]))
        a0 += f32(f16(i32((p >> 12) & 3) - 2) * f16(x_in[b + 6]))
        a1 += f32(f16(i32((q >> 12) & 3) - 2) * f16(x_in[b + 6]))
        a0 += f32(f16(i32((p >> 14) & 3) - 2) * f16(x_in[b + 7]))
        a1 += f32(f16(i32((q >> 14) & 3) - 2) * f16(x_in[b + 7]))
        a0 += f32(f16(i32((p >> 16) & 3) - 2) * f16(x_in[b + 8]))
        a1 += f32(f16(i32((q >> 16) & 3) - 2) * f16(x_in[b + 8]))
        a0 += f32(f16(i32((p >> 18) & 3) - 2) * f16(x_in[b + 9]))
        a1 += f32(f16(i32((q >> 18) & 3) - 2) * f16(x_in[b + 9]))
        a0 += f32(f16(i32((p >> 20) & 3) - 2) * f16(x_in[b + 10]))
        a1 += f32(f16(i32((q >> 20) & 3) - 2) * f16(x_in[b + 10]))
        a0 += f32(f16(i32((p >> 22) & 3) - 2) * f16(x_in[b + 11]))
        a1 += f32(f16(i32((q >> 22) & 3) - 2) * f16(x_in[b + 11]))
        a0 += f32(f16(i32((p >> 24) & 3) - 2) * f16(x_in[b + 12]))
        a1 += f32(f16(i32((q >> 24) & 3) - 2) * f16(x_in[b + 12]))
        a0 += f32(f16(i32((p >> 26) & 3) - 2) * f16(x_in[b + 13]))
        a1 += f32(f16(i32((q >> 26) & 3) - 2) * f16(x_in[b + 13]))
        a0 += f32(f16(i32((p >> 28) & 3) - 2) * f16(x_in[b + 14]))
        a1 += f32(f16(i32((q >> 28) & 3) - 2) * f16(x_in[b + 14]))
        a0 += f32(f16(i32((p >> 30) & 3) - 2) * f16(x_in[b + 15]))
        a1 += f32(f16(i32((q >> 30) & 3) - 2) * f16(x_in[b + 15]))
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
def matvec_dq2_vec4(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    w: StorageBuffer[vec4[u32], "read"],   # [n_out, n_in/64] (vec4<u32>=64 int2)
    x: StorageBuffer[vec4[f32], "read"],   # [n_in/4]
    scale: StorageBuffer[f32, "read"],     # [n_out]
    y_out: StorageBuffer[f32, "read_write"],
    dims: StorageBuffer[u32, "read"],      # [n_out, n_in] (n_in % 64 == 0)
    partial: WorkgroupArray[f32, 64],
):
    r: u32 = wid.x
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    n64: u32 = n_in / 64
    acc: f32 = 0.0
    if r < n_out:
        for j in range(li, n64, 64):
            wv = w[r * n64 + j]
            xb: u32 = 16 * j
            xa00 = x[xb + 0]
            acc += f32(i32(wv.x & 3) - 2) * xa00.x
            acc += f32(i32((wv.x >> 2) & 3) - 2) * xa00.y
            acc += f32(i32((wv.x >> 4) & 3) - 2) * xa00.z
            acc += f32(i32((wv.x >> 6) & 3) - 2) * xa00.w
            xa01 = x[xb + 1]
            acc += f32(i32((wv.x >> 8) & 3) - 2) * xa01.x
            acc += f32(i32((wv.x >> 10) & 3) - 2) * xa01.y
            acc += f32(i32((wv.x >> 12) & 3) - 2) * xa01.z
            acc += f32(i32((wv.x >> 14) & 3) - 2) * xa01.w
            xa02 = x[xb + 2]
            acc += f32(i32((wv.x >> 16) & 3) - 2) * xa02.x
            acc += f32(i32((wv.x >> 18) & 3) - 2) * xa02.y
            acc += f32(i32((wv.x >> 20) & 3) - 2) * xa02.z
            acc += f32(i32((wv.x >> 22) & 3) - 2) * xa02.w
            xa03 = x[xb + 3]
            acc += f32(i32((wv.x >> 24) & 3) - 2) * xa03.x
            acc += f32(i32((wv.x >> 26) & 3) - 2) * xa03.y
            acc += f32(i32((wv.x >> 28) & 3) - 2) * xa03.z
            acc += f32(i32((wv.x >> 30) & 3) - 2) * xa03.w
            xa10 = x[xb + 4]
            acc += f32(i32(wv.y & 3) - 2) * xa10.x
            acc += f32(i32((wv.y >> 2) & 3) - 2) * xa10.y
            acc += f32(i32((wv.y >> 4) & 3) - 2) * xa10.z
            acc += f32(i32((wv.y >> 6) & 3) - 2) * xa10.w
            xa11 = x[xb + 5]
            acc += f32(i32((wv.y >> 8) & 3) - 2) * xa11.x
            acc += f32(i32((wv.y >> 10) & 3) - 2) * xa11.y
            acc += f32(i32((wv.y >> 12) & 3) - 2) * xa11.z
            acc += f32(i32((wv.y >> 14) & 3) - 2) * xa11.w
            xa12 = x[xb + 6]
            acc += f32(i32((wv.y >> 16) & 3) - 2) * xa12.x
            acc += f32(i32((wv.y >> 18) & 3) - 2) * xa12.y
            acc += f32(i32((wv.y >> 20) & 3) - 2) * xa12.z
            acc += f32(i32((wv.y >> 22) & 3) - 2) * xa12.w
            xa13 = x[xb + 7]
            acc += f32(i32((wv.y >> 24) & 3) - 2) * xa13.x
            acc += f32(i32((wv.y >> 26) & 3) - 2) * xa13.y
            acc += f32(i32((wv.y >> 28) & 3) - 2) * xa13.z
            acc += f32(i32((wv.y >> 30) & 3) - 2) * xa13.w
            xa20 = x[xb + 8]
            acc += f32(i32(wv.z & 3) - 2) * xa20.x
            acc += f32(i32((wv.z >> 2) & 3) - 2) * xa20.y
            acc += f32(i32((wv.z >> 4) & 3) - 2) * xa20.z
            acc += f32(i32((wv.z >> 6) & 3) - 2) * xa20.w
            xa21 = x[xb + 9]
            acc += f32(i32((wv.z >> 8) & 3) - 2) * xa21.x
            acc += f32(i32((wv.z >> 10) & 3) - 2) * xa21.y
            acc += f32(i32((wv.z >> 12) & 3) - 2) * xa21.z
            acc += f32(i32((wv.z >> 14) & 3) - 2) * xa21.w
            xa22 = x[xb + 10]
            acc += f32(i32((wv.z >> 16) & 3) - 2) * xa22.x
            acc += f32(i32((wv.z >> 18) & 3) - 2) * xa22.y
            acc += f32(i32((wv.z >> 20) & 3) - 2) * xa22.z
            acc += f32(i32((wv.z >> 22) & 3) - 2) * xa22.w
            xa23 = x[xb + 11]
            acc += f32(i32((wv.z >> 24) & 3) - 2) * xa23.x
            acc += f32(i32((wv.z >> 26) & 3) - 2) * xa23.y
            acc += f32(i32((wv.z >> 28) & 3) - 2) * xa23.z
            acc += f32(i32((wv.z >> 30) & 3) - 2) * xa23.w
            xa30 = x[xb + 12]
            acc += f32(i32(wv.w & 3) - 2) * xa30.x
            acc += f32(i32((wv.w >> 2) & 3) - 2) * xa30.y
            acc += f32(i32((wv.w >> 4) & 3) - 2) * xa30.z
            acc += f32(i32((wv.w >> 6) & 3) - 2) * xa30.w
            xa31 = x[xb + 13]
            acc += f32(i32((wv.w >> 8) & 3) - 2) * xa31.x
            acc += f32(i32((wv.w >> 10) & 3) - 2) * xa31.y
            acc += f32(i32((wv.w >> 12) & 3) - 2) * xa31.z
            acc += f32(i32((wv.w >> 14) & 3) - 2) * xa31.w
            xa32 = x[xb + 14]
            acc += f32(i32((wv.w >> 16) & 3) - 2) * xa32.x
            acc += f32(i32((wv.w >> 18) & 3) - 2) * xa32.y
            acc += f32(i32((wv.w >> 20) & 3) - 2) * xa32.z
            acc += f32(i32((wv.w >> 22) & 3) - 2) * xa32.w
            xa33 = x[xb + 15]
            acc += f32(i32((wv.w >> 24) & 3) - 2) * xa33.x
            acc += f32(i32((wv.w >> 26) & 3) - 2) * xa33.y
            acc += f32(i32((wv.w >> 28) & 3) - 2) * xa33.z
            acc += f32(i32((wv.w >> 30) & 3) - 2) * xa33.w
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



BROWSER_EXTRA_KERNELS.update({
    "matvec_dq2_vec4": matvec_dq2_vec4,
    "matvec_dq2_blk2_f16": matvec_dq2_blk2_f16,
    "mv_gateup_geglu_dq4_blk2": mv_gateup_geglu_dq4_blk2,
    "mv_gateup_geglu_dq2_sg4": mv_gateup_geglu_dq2_sg4,
    "matvec_dq2_blk2_t128": matvec_dq2_blk2_t128,
    "matvec_dq2_blk2_t256": matvec_dq2_blk2_t256,
    "attn_flash_partial": attn_flash_partial,
    "attn_flash_combine": attn_flash_combine,
})
