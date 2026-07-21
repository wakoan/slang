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
    "mv_gateup_geglu_dq2": mv_gateup_geglu_dq2,
    "mv_gateup_geglu_dq4": mv_gateup_geglu_dq4,
    "mv_geglu_f16": mv_geglu_f16,
    "qat_embed_2bit": qat_embed_2bit,
    "qat_ple_gather_4bit": qat_ple_gather_4bit,
})
