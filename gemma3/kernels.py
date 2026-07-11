"""Gemma 3 GPU kernels, all written in py_shader_lang_wgpu.

Design: single-position (decode-style) forward pass. Prefill runs the same
path one token at a time. All shapes arrive via small u32/f32 params buffers
so the WGSL is model-size agnostic.

GQA note: num_key_value_heads == 1 for both Gemma 3 270M and Gemma 4 E2B,
so the KV cache is laid out [position, head_dim] and every query head reads
the same KV — kernels assume this.
"""

from py_shader_lang_wgpu import (
    kernel, u32, f32, f16, StorageBuffer, Builtin, WorkgroupArray,
)


def gelu(g: f32) -> f32:  # helper — resolved automatically by the DSL
    # gelu_pytorch_tanh: 0.5g(1 + tanh(sqrt(2/pi)(g + 0.044715 g^3)))
    # Clamp: Metal's fast-math tanh computes via exp() and overflows to NaN
    # for |x| >~ 44; tanh(+-20) is already +-1.0 exactly in f32.
    inner: f32 = clamp(0.7978845608028654 * (g + 0.044715 * g * g * g), -20.0, 20.0)
    return 0.5 * g * (1.0 + tanh(inner))


@kernel(workgroup_size=(64,))
def embed_scale(
    gid: Builtin.global_invocation_id,
    token: StorageBuffer[u32, "read"],        # [1] current token id
    table: StorageBuffer[f32, "read"],        # [vocab, hidden] embedding table
    x_out: StorageBuffer[f32, "read_write"],  # [hidden]
    dims: StorageBuffer[u32, "read"],         # [hidden]
):
    i: u32 = gid.x
    hidden: u32 = dims[0]
    if i >= hidden:
        return
    tok: u32 = token[0]
    # Gemma scales embeddings by sqrt(hidden_size)
    x_out[i] = table[tok * hidden + i] * sqrt(f32(hidden))


@kernel(workgroup_size=(64,))
def rmsnorm(
    gid: Builtin.global_invocation_id,
    x_in: StorageBuffer[f32, "read"],         # [n_rows, row_len]
    w: StorageBuffer[f32, "read"],            # [row_len]
    x_out: StorageBuffer[f32, "read_write"],  # [n_rows, row_len]
    dims: StorageBuffer[u32, "read"],         # [n_rows, row_len]
):
    idx: u32 = gid.x
    n_rows: u32 = dims[0]
    row_len: u32 = dims[1]
    if idx >= n_rows * row_len:
        return
    row: u32 = idx / row_len
    col: u32 = idx % row_len

    ss: f32 = 0.0
    for j in range(row_len):
        v: f32 = x_in[row * row_len + j]
        ss += v * v
    inv: f32 = 1.0 / sqrt(ss / f32(row_len) + 1e-6)
    # Gemma-style RMSNorm: scale by (1 + weight)
    x_out[idx] = x_in[idx] * inv * (1.0 + w[col])


@kernel(workgroup_size=(64,))
def matvec(
    gid: Builtin.global_invocation_id,
    w_mat: StorageBuffer[f32, "read"],        # [n_out, n_in] row-major
    x_in: StorageBuffer[f32, "read"],         # [n_in]
    y_out: StorageBuffer[f32, "read_write"],  # [n_out]
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in]
):
    r: u32 = gid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    if r >= n_out:
        return
    acc: f32 = 0.0
    for j in range(n_in):
        acc += w_mat[r * n_in + j] * x_in[j]
    y_out[r] = acc


@kernel(workgroup_size=(64,))
def rope(
    gid: Builtin.global_invocation_id,
    x_io: StorageBuffer[f32, "read_write"],   # [n_heads, head_dim] q or k
    fparams: StorageBuffer[f32, "read"],      # [theta, position]
    dims: StorageBuffer[u32, "read"],         # [n_heads, head_dim]
):
    idx: u32 = gid.x
    n_heads: u32 = dims[0]
    head_dim: u32 = dims[1]
    half: u32 = head_dim / 2
    if idx >= n_heads * half:
        return
    h: u32 = idx / half
    i: u32 = idx % half

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


@kernel(workgroup_size=(64,))
def kv_append(
    gid: Builtin.global_invocation_id,
    src: StorageBuffer[f32, "read"],           # [vec_len] new k or v
    cache: StorageBuffer[f32, "read_write"],   # [max_seq, vec_len]
    dims: StorageBuffer[u32, "read"],          # [vec_len, position]
):
    i: u32 = gid.x
    vec_len: u32 = dims[0]
    pos: u32 = dims[1]
    if i >= vec_len:
        return
    cache[pos * vec_len + i] = src[i]


@kernel(workgroup_size=(8, 8))
def attn_scores(
    gid: Builtin.global_invocation_id,
    q: StorageBuffer[f32, "read"],             # [n_heads, head_dim]
    k_cache: StorageBuffer[f32, "read"],       # [max_seq, head_dim]
    scores: StorageBuffer[f32, "read_write"],  # [n_heads, max_seq]
    dims: StorageBuffer[u32, "read"],          # [n_heads, head_dim, kv_len, start, max_seq]
):
    t: u32 = gid.x
    h: u32 = gid.y
    n_heads: u32 = dims[0]
    head_dim: u32 = dims[1]
    kv_len: u32 = dims[2]
    start: u32 = dims[3]
    max_seq: u32 = dims[4]
    if h >= n_heads or t >= kv_len:
        return
    if t < start:
        # outside the sliding window
        scores[h * max_seq + t] = -1e9
        return
    dot: f32 = 0.0
    for j in range(head_dim):
        dot += q[h * head_dim + j] * k_cache[t * head_dim + j]
    # query_pre_attn_scalar == head_dim for Gemma 3 → 1/sqrt(head_dim)
    scores[h * max_seq + t] = dot / sqrt(f32(head_dim))


@kernel(workgroup_size=(4,))
def attn_softmax(
    gid: Builtin.global_invocation_id,
    scores: StorageBuffer[f32, "read_write"],  # [n_heads, max_seq]
    dims: StorageBuffer[u32, "read"],          # [n_heads, kv_len, max_seq]
):
    h: u32 = gid.x
    n_heads: u32 = dims[0]
    kv_len: u32 = dims[1]
    max_seq: u32 = dims[2]
    if h >= n_heads:
        return
    base: u32 = h * max_seq

    m: f32 = scores[base]
    for t in range(1, kv_len):
        m = max(m, scores[base + t])

    ssum: f32 = 0.0
    for t in range(kv_len):
        e: f32 = exp(scores[base + t] - m)
        scores[base + t] = e
        ssum += e

    for t in range(kv_len):
        scores[base + t] = scores[base + t] / ssum


@kernel(workgroup_size=(64,))
def attn_output(
    gid: Builtin.global_invocation_id,
    scores: StorageBuffer[f32, "read"],        # [n_heads, max_seq] softmaxed
    v_cache: StorageBuffer[f32, "read"],       # [max_seq, head_dim]
    out_vec: StorageBuffer[f32, "read_write"], # [n_heads, head_dim]
    dims: StorageBuffer[u32, "read"],          # [n_heads, head_dim, kv_len, max_seq]
):
    idx: u32 = gid.x
    n_heads: u32 = dims[0]
    head_dim: u32 = dims[1]
    kv_len: u32 = dims[2]
    max_seq: u32 = dims[3]
    if idx >= n_heads * head_dim:
        return
    h: u32 = idx / head_dim
    d: u32 = idx % head_dim
    acc: f32 = 0.0
    for t in range(kv_len):
        acc += scores[h * max_seq + t] * v_cache[t * head_dim + d]
    out_vec[idx] = acc


@kernel(workgroup_size=(64,))
def geglu(
    gid: Builtin.global_invocation_id,
    gate: StorageBuffer[f32, "read"],          # [n] gate_proj output
    up: StorageBuffer[f32, "read"],            # [n] up_proj output
    h_out: StorageBuffer[f32, "read_write"],   # [n]
    dims: StorageBuffer[u32, "read"],          # [n]
):
    i: u32 = gid.x
    n: u32 = dims[0]
    if i >= n:
        return
    h_out[i] = gelu(gate[i]) * up[i]


@kernel(workgroup_size=(64,))
def add_vec(
    gid: Builtin.global_invocation_id,
    a_io: StorageBuffer[f32, "read_write"],    # [n] accumulator (residual)
    b_in: StorageBuffer[f32, "read"],          # [n]
    dims: StorageBuffer[u32, "read"],          # [n]
):
    i: u32 = gid.x
    if i >= dims[0]:
        return
    a_io[i] = a_io[i] + b_in[i]


@kernel(workgroup_size=(64,))
def matvec_wg(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    w_mat: StorageBuffer[f32, "read"],        # [n_out, n_in] row-major
    x_in: StorageBuffer[f32, "read"],         # [n_in]
    y_out: StorageBuffer[f32, "read_write"],  # [n_out]
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in]
    partial: WorkgroupArray[f32, 64],
):
    # One workgroup per output row; 64 threads stride over the input dim,
    # then tree-reduce partial sums in shared memory.
    r: u32 = wid.x
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]

    acc: f32 = 0.0
    if r < n_out:
        for j in range(li, n_in, 64):
            acc += w_mat[r * n_in + j] * x_in[j]
    partial[li] = acc
    barrier()

    s: u32 = 32
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2

    if li == 0 and r < n_out:
        y_out[r] = partial[0]


@kernel(workgroup_size=(64,))
def rmsnorm_wg(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    x_in: StorageBuffer[f32, "read"],         # [n_rows, row_len]
    w: StorageBuffer[f32, "read"],            # [row_len]
    x_out: StorageBuffer[f32, "read_write"],  # [n_rows, row_len]
    dims: StorageBuffer[u32, "read"],         # [n_rows, row_len]
    partial: WorkgroupArray[f32, 64],
):
    # One workgroup per row: phase 1 reduces sum-of-squares in shared
    # memory, phase 2 has all 64 threads write the normalised row.
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
            x_out[row * row_len + j] = x_in[row * row_len + j] * inv * (1.0 + w[j])


@kernel(workgroup_size=(64,))
def rmsnorm_add_wg(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    x_in: StorageBuffer[f32, "read"],         # [n_rows, row_len] block output
    w: StorageBuffer[f32, "read"],            # [row_len] norm weight
    x_io: StorageBuffer[f32, "read_write"],   # [n_rows, row_len] residual accumulator
    dims: StorageBuffer[u32, "read"],         # [n_rows, row_len]
    partial: WorkgroupArray[f32, 64],
):
    # Fused post-norm + residual add: x_io += rmsnorm(x_in) * (1 + w)
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
            idx: u32 = row * row_len + j
            x_io[idx] = x_io[idx] + x_in[idx] * inv * (1.0 + w[j])


@kernel(workgroup_size=(256,))
def attention_fused(
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
    # One workgroup per query head: scores → stable softmax → weighted V sum,
    # without intermediate dispatches. GQA with a single shared KV head.
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
        scores[base + t] = dot / sqrt(f32(head_dim))
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


# --- f16-weight variants: weights stored as f16, math still in f32 ---
# Halves weight bandwidth, which bounds decode throughput. Activations,
# KV cache, and accumulation stay f32.

@kernel(workgroup_size=(64,))
def embed_scale_f16(
    gid: Builtin.global_invocation_id,
    token: StorageBuffer[u32, "read"],
    table: StorageBuffer[f16, "read"],        # [vocab, hidden]
    x_out: StorageBuffer[f32, "read_write"],  # [hidden]
    dims: StorageBuffer[u32, "read"],         # [hidden]
):
    i: u32 = gid.x
    hidden: u32 = dims[0]
    if i >= hidden:
        return
    tok: u32 = token[0]
    x_out[i] = f32(table[tok * hidden + i]) * sqrt(f32(hidden))


@kernel(workgroup_size=(64,))
def matvec_f16(
    gid: Builtin.global_invocation_id,
    w_mat: StorageBuffer[f16, "read"],        # [n_out, n_in]
    x_in: StorageBuffer[f32, "read"],
    y_out: StorageBuffer[f32, "read_write"],
    dims: StorageBuffer[u32, "read"],
):
    r: u32 = gid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    if r >= n_out:
        return
    acc: f32 = 0.0
    for j in range(n_in):
        acc += f32(w_mat[r * n_in + j]) * x_in[j]
    y_out[r] = acc


@kernel(workgroup_size=(64,))
def matvec_wg_f16(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    w_mat: StorageBuffer[f16, "read"],        # [n_out, n_in]
    x_in: StorageBuffer[f32, "read"],
    y_out: StorageBuffer[f32, "read_write"],
    dims: StorageBuffer[u32, "read"],
    partial: WorkgroupArray[f32, 64],
):
    r: u32 = wid.x
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]

    acc: f32 = 0.0
    if r < n_out:
        for j in range(li, n_in, 64):
            acc += f32(w_mat[r * n_in + j]) * x_in[j]
    partial[li] = acc
    barrier()

    s: u32 = 32
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2

    if li == 0 and r < n_out:
        y_out[r] = partial[0]


@kernel(workgroup_size=(64,))
def matvec_packed(
    gid: Builtin.global_invocation_id,
    w_mat: StorageBuffer[u32, "read"],        # [n_out, n_in/2] packed f16 pairs
    x_in: StorageBuffer[f32, "read"],
    y_out: StorageBuffer[f32, "read_write"],
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in] (n_in even)
):
    r: u32 = gid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    if r >= n_out:
        return
    half: u32 = n_in / 2
    acc: f32 = 0.0
    for j in range(half):
        pair = unpack2x16float(w_mat[r * half + j])
        acc += pair.x * x_in[2 * j] + pair.y * x_in[2 * j + 1]
    y_out[r] = acc


@kernel(workgroup_size=(64,))
def matvec_wg_packed(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    w_mat: StorageBuffer[u32, "read"],        # [n_out, n_in/2] packed f16 pairs
    x_in: StorageBuffer[f32, "read"],
    y_out: StorageBuffer[f32, "read_write"],
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in] (n_in even)
    partial: WorkgroupArray[f32, 64],
):
    r: u32 = wid.x
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    half: u32 = n_in / 2

    acc: f32 = 0.0
    if r < n_out:
        for j in range(li, half, 64):
            pair = unpack2x16float(w_mat[r * half + j])
            acc += pair.x * x_in[2 * j] + pair.y * x_in[2 * j + 1]
    partial[li] = acc
    barrier()

    s: u32 = 32
    while s > 0:
        if li < s:
            partial[li] = partial[li] + partial[li + s]
        barrier()
        s = s / 2

    if li == 0 and r < n_out:
        y_out[r] = partial[0]




# --- subgroup variants: reductions via simd ops, no barrier trees ---
# Correct only when subgroup_size >= workgroup width (32); the runner
# verifies this at startup with probe_sg.

@kernel(workgroup_size=(32,))
def probe_sg(
    gid: Builtin.global_invocation_id,
    sgs: Builtin.subgroup_size,
    out: StorageBuffer[u32, "read_write"],
):
    if gid.x == 0:
        out[0] = sgs


@kernel(workgroup_size=(32,))
def matvec_wg_packed_sg(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    w_mat: StorageBuffer[u32, "read"],        # [n_out, n_in/2] packed f16 pairs
    x_in: StorageBuffer[f32, "read"],
    y_out: StorageBuffer[f32, "read_write"],
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in] (n_in even)
):
    r: u32 = wid.x
    li: u32 = lid.x
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    half: u32 = n_in / 2

    acc: f32 = 0.0
    if r < n_out:
        for j in range(li, half, 32):
            pair = unpack2x16float(w_mat[r * half + j])
            acc += pair.x * x_in[2 * j] + pair.y * x_in[2 * j + 1]
    total: f32 = subgroupAdd(acc)
    if li == 0 and r < n_out:
        y_out[r] = total


@kernel(workgroup_size=(32,))
def rmsnorm_wg_sg(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    x_in: StorageBuffer[f32, "read"],
    w: StorageBuffer[f32, "read"],
    x_out: StorageBuffer[f32, "read_write"],
    dims: StorageBuffer[u32, "read"],         # [n_rows, row_len]
):
    row: u32 = wid.x
    li: u32 = lid.x
    n_rows: u32 = dims[0]
    row_len: u32 = dims[1]

    acc: f32 = 0.0
    if row < n_rows:
        for j in range(li, row_len, 32):
            v: f32 = x_in[row * row_len + j]
            acc += v * v
    total: f32 = subgroupAdd(acc)  # every lane receives the sum
    inv: f32 = 1.0 / sqrt(total / f32(row_len) + 1e-6)
    if row < n_rows:
        for j in range(li, row_len, 32):
            x_out[row * row_len + j] = x_in[row * row_len + j] * inv * (1.0 + w[j])


@kernel(workgroup_size=(32,))
def rmsnorm_add_wg_sg(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    x_in: StorageBuffer[f32, "read"],
    w: StorageBuffer[f32, "read"],
    x_io: StorageBuffer[f32, "read_write"],   # residual accumulator
    dims: StorageBuffer[u32, "read"],         # [n_rows, row_len]
):
    row: u32 = wid.x
    li: u32 = lid.x
    n_rows: u32 = dims[0]
    row_len: u32 = dims[1]

    acc: f32 = 0.0
    if row < n_rows:
        for j in range(li, row_len, 32):
            v: f32 = x_in[row * row_len + j]
            acc += v * v
    total: f32 = subgroupAdd(acc)
    inv: f32 = 1.0 / sqrt(total / f32(row_len) + 1e-6)
    if row < n_rows:
        for j in range(li, row_len, 32):
            idx: u32 = row * row_len + j
            x_io[idx] = x_io[idx] + x_in[idx] * inv * (1.0 + w[j])


@kernel(workgroup_size=(256,))
def attention_fused_sg(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    lane: Builtin.subgroup_invocation_id,
    sg_size: Builtin.subgroup_size,
    q: StorageBuffer[f32, "read"],
    k_cache: StorageBuffer[f32, "read"],
    v_cache: StorageBuffer[f32, "read"],
    scores: StorageBuffer[f32, "read_write"],
    out_vec: StorageBuffer[f32, "read_write"],
    dims: StorageBuffer[u32, "read"],          # [n_heads, head_dim, kv_len, start, max_seq]
    smem: WorkgroupArray[f32, 8],              # one slot per subgroup (256/32)
):
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
        scores[base + t] = dot / sqrt(f32(head_dim))
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




# --- GPU-resident decode: argmax + on-GPU step parameter updates ---

@kernel(workgroup_size=(64,))
def argmax_stage1(
    wid: Builtin.workgroup_id,
    lid: Builtin.local_invocation_id,
    logits: StorageBuffer[f32, "read"],        # [n]
    part_val: StorageBuffer[f32, "read_write"],# [n_wgs]
    part_idx: StorageBuffer[u32, "read_write"],# [n_wgs]
    dims: StorageBuffer[u32, "read"],          # [n, n_wgs]
    sv: WorkgroupArray[f32, 64],
    si: WorkgroupArray[u32, 64],
):
    li: u32 = lid.x
    n: u32 = dims[0]
    n_wgs: u32 = dims[1]
    stride: u32 = n_wgs * 64

    best_v: f32 = -3.0e38
    best_i: u32 = 0
    for i in range(wid.x * 64 + li, n, stride):
        v: f32 = logits[i]
        if v > best_v:
            best_v = v
            best_i = i
    sv[li] = best_v
    si[li] = best_i
    barrier()

    s: u32 = 32
    while s > 0:
        if li < s:
            if sv[li + s] > sv[li]:
                sv[li] = sv[li + s]
                si[li] = si[li + s]
        barrier()
        s = s / 2

    if li == 0:
        part_val[wid.x] = sv[0]
        part_idx[wid.x] = si[0]


@kernel(workgroup_size=(64,))
def argmax_stage2(
    lid: Builtin.local_invocation_id,
    part_val: StorageBuffer[f32, "read"],       # [n_parts]
    part_idx: StorageBuffer[u32, "read"],       # [n_parts]
    token: StorageBuffer[u32, "read_write"],    # [1] next-token feedback
    out_tokens: StorageBuffer[u32, "read_write"],  # generated token log
    counter: StorageBuffer[u32, "read_write"],  # [1]
    dims: StorageBuffer[u32, "read"],           # [n_parts]
    sv: WorkgroupArray[f32, 64],
    si: WorkgroupArray[u32, 64],
):
    li: u32 = lid.x
    n_parts: u32 = dims[0]

    best_v: f32 = -3.0e38
    best_i: u32 = 0
    for i in range(li, n_parts, 64):
        if part_val[i] > best_v:
            best_v = part_val[i]
            best_i = part_idx[i]
    sv[li] = best_v
    si[li] = best_i
    barrier()

    s: u32 = 32
    while s > 0:
        if li < s:
            if sv[li + s] > sv[li]:
                sv[li] = sv[li + s]
                si[li] = si[li + s]
        barrier()
        s = s / 2

    if li == 0:
        tok: u32 = si[0]
        token[0] = tok
        out_tokens[counter[0]] = tok
        counter[0] = counter[0] + 1


@kernel(workgroup_size=(1,))
def step_setup(
    gid: Builtin.global_invocation_id,
    pos_buf: StorageBuffer[u32, "read_write"],  # [1] this step's position
    kv_dims: StorageBuffer[u32, "read_write"],  # [vec_len, pos]
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

    kv_dims[1] = pos
    sc_slide[2] = kv_len
    if kv_len > window:
        sc_slide[3] = kv_len - window
    else:
        sc_slide[3] = 0
    sc_full[2] = kv_len
    rope_l[1] = f32(pos)
    rope_g[1] = f32(pos)
    pos_buf[0] = pos + 1


# All kernels, keyed by name, for the runner and tests
KERNELS = {
    "embed_scale": embed_scale,
    "rmsnorm": rmsnorm,
    "matvec": matvec,
    "rope": rope,
    "kv_append": kv_append,
    "attn_scores": attn_scores,
    "attn_softmax": attn_softmax,
    "attn_output": attn_output,
    "geglu": geglu,
    "add_vec": add_vec,
    "matvec_wg": matvec_wg,
    "rmsnorm_wg": rmsnorm_wg,
    "rmsnorm_add_wg": rmsnorm_add_wg,
    "attention_fused": attention_fused,
    "embed_scale_f16": embed_scale_f16,
    "matvec_f16": matvec_f16,
    "matvec_wg_f16": matvec_wg_f16,
    "matvec_packed": matvec_packed,
    "matvec_wg_packed": matvec_wg_packed,
    "probe_sg": probe_sg,
    "matvec_wg_packed_sg": matvec_wg_packed_sg,
    "rmsnorm_wg_sg": rmsnorm_wg_sg,
    "rmsnorm_add_wg_sg": rmsnorm_add_wg_sg,
    "attention_fused_sg": attention_fused_sg,
    "argmax_stage1": argmax_stage1,
    "argmax_stage2": argmax_stage2,
    "step_setup": step_setup,
}


if __name__ == "__main__":
    for name, k in KERNELS.items():
        print(f"// ==== {name} ====")
        print(k.wgsl)
        print()
