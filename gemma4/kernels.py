"""Gemma 4 E2B GPU kernels in the py_shader_lang_wgpu DSL.

Only the ops that differ from Gemma 3 are defined here; everything else is
imported unchanged from gemma3.kernels into KERNELS. Norm-weight convention:
Gemma4RMSNorm scales by w directly, so the runner uploads (w - 1) and reuses
the Gemma 3 (1 + w) kernels for every scaled norm. The scale-free v_norm
needs its own kernel (rmsnorm_ns_wg).

Runtime shapes flow through small `dims`/`fparams` storage buffers, as in
gemma3.kernels — nothing shape-specific is baked into the WGSL.
"""

from __future__ import annotations

from py_shader_lang_wgpu import kernel
from py_shader_lang_wgpu.types import (
    Builtin,
    StorageBuffer,
    WorkgroupArray,
    f16,
    f32,
    u32,
)

from gemma3.kernels import (
    argmax_stage1,
    argmax_stage2,
    embed_scale,
    embed_scale_f16,
    geglu,
    kv_append,
    matvec_wg,
    matvec_wg_packed,
    matvec_wg_packed_sg,
    probe_sg,
    rmsnorm_add_wg,
    rmsnorm_add_wg_sg,
    rmsnorm_wg,
    rmsnorm_wg_sg,
)


@kernel(workgroup_size=(64,))
def rope_pl(
    gid: Builtin.global_invocation_id,
    x_io: StorageBuffer[f32, "read_write"],   # [n_heads, head_dim] q or k
    fparams: StorageBuffer[f32, "read"],      # [theta, position]
    dims: StorageBuffer[u32, "read"],         # [n_heads, head_dim, cutoff]
):
    # Gemma 4 p-RoPE: frequency pairs >= cutoff have inv_freq 0 (identity),
    # so they are simply left untouched. cutoff == head_dim/2 is a full
    # rotation (sliding layers); full-attention layers use cutoff 64 of 256.
    idx: u32 = gid.x
    n_heads: u32 = dims[0]
    head_dim: u32 = dims[1]
    cutoff: u32 = dims[2]
    half: u32 = head_dim / 2
    if idx >= n_heads * half:
        return
    h: u32 = idx / half
    i: u32 = idx % half
    if i >= cutoff:
        return

    theta: f32 = fparams[0]
    pos: f32 = fparams[1]
    # rotate_half convention (HF): pair (i, i + half)
    inv_freq: f32 = 1.0 / pow(theta, f32(2 * i) / f32(head_dim))
    angle: f32 = pos * inv_freq
    c: f32 = cos(angle)
    s: f32 = sin(angle)

    a: f32 = x_io[h * head_dim + i]
    b: f32 = x_io[h * head_dim + i + half]
    x_io[h * head_dim + i] = a * c - b * s
    x_io[h * head_dim + i + half] = b * c + a * s


@kernel(workgroup_size=(256,))
def attention_fused_g4(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    q: StorageBuffer[f32, "read"],             # [n_heads, head_dim] normed+roped
    k_cache: StorageBuffer[f32, "read"],       # [max_seq, head_dim]
    v_cache: StorageBuffer[f32, "read"],       # [max_seq, head_dim]
    scores: StorageBuffer[f32, "read_write"],  # [n_heads, max_seq] scratch
    out_vec: StorageBuffer[f32, "read_write"], # [n_heads, head_dim]
    dims: StorageBuffer[u32, "read"],          # [n_heads, head_dim, kv_len, start, max_seq]
    smem: WorkgroupArray[f32, 256],
):
    # gemma3 attention_fused minus the 1/sqrt(head_dim) score scale:
    # Gemma 4 uses scaling = 1.0 (q_norm does the work).
    h: u32 = wid.x
    li: u32 = lid.x
    head_dim: u32 = dims[1]
    kv_len: u32 = dims[2]
    start: u32 = dims[3]
    max_seq: u32 = dims[4]
    base: u32 = h * max_seq

    # 1) raw scores over [start, kv_len)
    for t in range(start + li, kv_len, 256):
        dot: f32 = 0.0
        for j in range(head_dim):
            dot += q[h * head_dim + j] * k_cache[t * head_dim + j]
        scores[base + t] = dot
    barrier()

    # 2) max-reduce for numerical stability
    m_local: f32 = -1e30
    for t in range(start + li, kv_len, 256):
        m_local = max(m_local, scores[base + t])
    smem[li] = m_local
    barrier()
    s: u32 = 128
    while s > 0:
        if li < s:
            smem[li] = max(smem[li], smem[li + s])
        barrier()
        s = s / 2
    m: f32 = smem[0]
    barrier()

    # 3) exp + sum-reduce
    sum_local: f32 = 0.0
    for t in range(start + li, kv_len, 256):
        e: f32 = exp(scores[base + t] - m)
        scores[base + t] = e
        sum_local += e
    smem[li] = sum_local
    barrier()
    s = 128
    while s > 0:
        if li < s:
            smem[li] = smem[li] + smem[li + s]
        barrier()
        s = s / 2
    denom: f32 = smem[0]

    # 4) weighted V sum, normalised
    for d in range(li, head_dim, 256):
        acc: f32 = 0.0
        for t in range(start, kv_len):
            acc += scores[base + t] * v_cache[t * head_dim + d]
        out_vec[h * head_dim + d] = acc / denom


@kernel(workgroup_size=(256,))
def attention_fused_g4_sg(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    lane: Builtin.subgroup_invocation_id,
    sg_size: Builtin.subgroup_size,
    q: StorageBuffer[f32, "read"],             # [n_heads, head_dim] normed+roped
    k_cache: StorageBuffer[f32, "read"],       # [max_seq, head_dim]
    v_cache: StorageBuffer[f32, "read"],       # [max_seq, head_dim]
    scores: StorageBuffer[f32, "read_write"],  # [n_heads, max_seq] scratch
    out_vec: StorageBuffer[f32, "read_write"], # [n_heads, head_dim]
    dims: StorageBuffer[u32, "read"],          # [n_heads, head_dim, kv_len, start, max_seq]
    smem: WorkgroupArray[f32, 8],              # one slot per subgroup (256/32)
):
    # gemma3 attention_fused_sg minus the 1/sqrt(head_dim) score scale.
    h: u32 = wid.x
    li: u32 = lid.x
    head_dim: u32 = dims[1]
    kv_len: u32 = dims[2]
    start: u32 = dims[3]
    max_seq: u32 = dims[4]
    base: u32 = h * max_seq
    sg_id: u32 = li / sg_size
    n_sg: u32 = 256 / sg_size

    for t in range(start + li, kv_len, 256):
        dot: f32 = 0.0
        for j in range(head_dim):
            dot += q[h * head_dim + j] * k_cache[t * head_dim + j]
        scores[base + t] = dot
    barrier()

    # max: subgroup reduce, then tiny cross-subgroup pass
    m_local: f32 = -1e30
    for t in range(start + li, kv_len, 256):
        m_local = max(m_local, scores[base + t])
    m_sg: f32 = subgroupMax(m_local)
    if lane == 0:
        smem[sg_id] = m_sg
    barrier()
    m: f32 = smem[0]
    for i in range(1, n_sg):
        m = max(m, smem[i])
    barrier()

    # exp + sum
    sum_local: f32 = 0.0
    for t in range(start + li, kv_len, 256):
        e: f32 = exp(scores[base + t] - m)
        scores[base + t] = e
        sum_local += e
    s_sg: f32 = subgroupAdd(sum_local)
    if lane == 0:
        smem[sg_id] = s_sg
    barrier()
    denom: f32 = smem[0]
    for i in range(1, n_sg):
        denom += smem[i]

    for d in range(li, head_dim, 256):
        acc: f32 = 0.0
        for t in range(start, kv_len):
            acc += scores[base + t] * v_cache[t * head_dim + d]
        out_vec[h * head_dim + d] = acc / denom


@kernel(workgroup_size=(64,))
def rmsnorm_ns_wg(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    x_in: StorageBuffer[f32, "read"],         # [n_rows, row_len]
    x_out: StorageBuffer[f32, "read_write"],  # [n_rows, row_len]
    dims: StorageBuffer[u32, "read"],         # [n_rows, row_len]
    partial: WorkgroupArray[f32, 64],
):
    # Parameter-free RMSNorm (Gemma 4 v_norm, with_scale=False): x / rms(x).
    row: u32 = wid.x
    li: u32 = lid.x
    n_rows: u32 = dims[0]
    row_len: u32 = dims[1]

    acc: f32 = 0.0
    if row < n_rows:
        for j in range(li, row_len, 64):
            v: f32 = x_in[row * row_len + j]
            acc += v * v
    partial[li] = acc
    barrier()

    s: u32 = 32
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2

    inv: f32 = 1.0 / sqrt(partial[0] / f32(row_len) + 1e-6)
    if row < n_rows:
        for j in range(li, row_len, 64):
            x_out[row * row_len + j] = x_in[row * row_len + j] * inv


@kernel(workgroup_size=(64,))
def rmsnorm_add_norm_wg(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    src: StorageBuffer[f32, "read"],          # [n_rows, row_len] branch output
    w1: StorageBuffer[f32, "read"],           # [row_len] add-norm weight (w-1)
    w2: StorageBuffer[f32, "read"],           # [row_len] second-norm weight (w-1)
    x_io: StorageBuffer[f32, "read_write"],   # [n_rows, row_len] residual accumulator
    x_out: StorageBuffer[f32, "read_write"],  # [n_rows, row_len] second-norm output
    dims: StorageBuffer[u32, "read"],         # [n_rows, row_len]
    partial: WorkgroupArray[f32, 64],
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
        for j in range(li, row_len, 64):
            v: f32 = src[row * row_len + j]
            acc += v * v
    partial[li] = acc
    barrier()
    s: u32 = 32
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2
    inv1: f32 = 1.0 / sqrt(partial[0] / f32(row_len) + 1e-6)
    barrier()  # all lanes read partial[0] before it is overwritten below

    acc2: f32 = 0.0
    if row < n_rows:
        for j in range(li, row_len, 64):
            idx: u32 = row * row_len + j
            v2: f32 = x_io[idx] + src[idx] * inv1 * (1.0 + w1[j])
            x_io[idx] = v2
            acc2 += v2 * v2
    partial[li] = acc2
    barrier()
    s = 32
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2
    inv2: f32 = 1.0 / sqrt(partial[0] / f32(row_len) + 1e-6)
    if row < n_rows:
        for j in range(li, row_len, 64):
            idx2: u32 = row * row_len + j
            x_out[idx2] = x_io[idx2] * inv2 * (1.0 + w2[j])


@kernel(workgroup_size=(64,))
def softcap(
    gid: Builtin.global_invocation_id,
    x_io: StorageBuffer[f32, "read_write"],   # [n] logits
    fparams: StorageBuffer[f32, "read"],      # [cap]
    dims: StorageBuffer[u32, "read"],         # [n]
):
    i: u32 = gid.x
    if i >= dims[0]:
        return
    cap: f32 = fparams[0]
    x_io[i] = cap * tanh(x_io[i] / cap)


@kernel(workgroup_size=(64,))
def add_scale(
    gid: Builtin.global_invocation_id,
    x_in: StorageBuffer[f32, "read"],         # [n] branch output
    x_io: StorageBuffer[f32, "read_write"],   # [n] residual accumulator
    fparams: StorageBuffer[f32, "read"],      # [scale]
    dims: StorageBuffer[u32, "read"],         # [n]
):
    # End of the PLE block: x = (residual + branch) * layer_scalar.
    i: u32 = gid.x
    if i >= dims[0]:
        return
    x_io[i] = (x_io[i] + x_in[i]) * fparams[0]


@kernel(workgroup_size=(64,))
def combine_scaled(
    gid: Builtin.global_invocation_id,
    a_in: StorageBuffer[f32, "read"],         # [n] context projection (normed)
    b_in: StorageBuffer[f32, "read"],         # [n] scaled PLE table rows
    x_out: StorageBuffer[f32, "read_write"],  # [n]
    fparams: StorageBuffer[f32, "read"],      # [scale] = 2^-0.5
    dims: StorageBuffer[u32, "read"],         # [n]
):
    i: u32 = gid.x
    if i >= dims[0]:
        return
    x_out[i] = (a_in[i] + b_in[i]) * fparams[0]


@kernel(workgroup_size=(1,))
def step_setup_g4(
    gid: Builtin.global_invocation_id,
    pos_buf: StorageBuffer[u32, "read_write"],  # [1] this step's position
    kv_s: StorageBuffer[u32, "read_write"],     # [vec_len, pos] sliding layers
    kv_f: StorageBuffer[u32, "read_write"],     # [vec_len, pos] full layers
    sc_slide: StorageBuffer[u32, "read_write"], # [nh, hd, kv_len, start, max_seq]
    sc_full: StorageBuffer[u32, "read_write"],
    rope_l: StorageBuffer[f32, "read_write"],   # [theta, pos]
    rope_g: StorageBuffer[f32, "read_write"],
    cfg_c: StorageBuffer[u32, "read"],          # [sliding_window]
):
    # Computes all position-derived step parameters on-GPU so a decode
    # chain never round-trips to the CPU. Increments pos for the next step.
    if gid.x > 0:
        return
    pos: u32 = pos_buf[0]
    kv_len: u32 = pos + 1
    window: u32 = cfg_c[0]

    kv_s[1] = pos
    kv_f[1] = pos
    sc_slide[2] = kv_len
    if kv_len > window:
        sc_slide[3] = kv_len - window
    else:
        sc_slide[3] = 0
    sc_full[2] = kv_len
    rope_l[1] = f32(pos)
    rope_g[1] = f32(pos)
    pos_buf[0] = pos + 1


@kernel(workgroup_size=(64,))
def ple_gather_f16(
    gid: Builtin.global_invocation_id,
    token: StorageBuffer[u32, "read"],          # [1] current token id
    table_lo: StorageBuffer[f16, "read"],       # rows [0, split)
    table_hi: StorageBuffer[f16, "read"],       # rows [split, vocab)
    out: StorageBuffer[f32, "read_write"],      # [row_len] scaled PLE row
    fparams: StorageBuffer[f32, "read"],        # [scale] = sqrt(256)
    dims: StorageBuffer[u32, "read"],           # [row_len, split]
):
    # On-GPU replacement for the CPU-side per-token PLE table gather. The
    # f16 table is 4.7GB — beyond the 4.29GB max storage binding — so it
    # ships as two half-tables and the row is picked by token range.
    i: u32 = gid.x
    row_len: u32 = dims[0]
    split: u32 = dims[1]
    if i >= row_len:
        return
    t: u32 = token[0]
    v: f32 = 0.0
    if t < split:
        v = f32(table_lo[t * row_len + i])
    else:
        v = f32(table_hi[(t - split) * row_len + i])
    out[i] = v * fparams[0]


KERNELS = {
    # unchanged gemma3 kernels
    "embed_scale": embed_scale,
    "embed_scale_f16": embed_scale_f16,
    "matvec_wg": matvec_wg,
    "matvec_wg_packed": matvec_wg_packed,
    "matvec_wg_packed_sg": matvec_wg_packed_sg,
    "rmsnorm_wg_sg": rmsnorm_wg_sg,
    "rmsnorm_add_wg_sg": rmsnorm_add_wg_sg,
    "probe_sg": probe_sg,
    "rmsnorm_wg": rmsnorm_wg,          # fed (w - 1) to realise Gemma 4's w scale
    "rmsnorm_add_wg": rmsnorm_add_wg,  # fed (w - 1), same trick
    "kv_append": kv_append,
    "geglu": geglu,
    "argmax_stage1": argmax_stage1,
    "argmax_stage2": argmax_stage2,
    # gemma4-specific
    "rope_pl": rope_pl,
    "attention_fused_g4": attention_fused_g4,
    "attention_fused_g4_sg": attention_fused_g4_sg,
    "rmsnorm_add_norm_wg": rmsnorm_add_norm_wg,
    "rmsnorm_ns_wg": rmsnorm_ns_wg,
    "softcap": softcap,
    "add_scale": add_scale,
    "combine_scaled": combine_scaled,
    "step_setup_g4": step_setup_g4,
    "ple_gather_f16": ple_gather_f16,
}
