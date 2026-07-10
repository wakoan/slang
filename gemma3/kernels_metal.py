"""Metal-backend Gemma kernels, written in the same DSL.

The metalgpu runtime dispatches 1-D grids with threadsPerThreadgroup =
min(n, 1024) — threadgroup width is not controllable. These kernels
therefore avoid threadgroup memory entirely: every reduction happens at
simdgroup scope (32 lanes), and the row index derives from gid.x / 32.
Weights are f16 pairs packed in u32 (metalgpu has no f16 buffer type).

Elementwise kernels (rope, kv_append, geglu, add_vec) are reused from
kernels.py via their .msl translation.
"""

from py_shader_lang_wgpu import kernel, u32, f32, StorageBuffer, Builtin


@kernel(workgroup_size=(64,))
def embed_scale_packed(
    gid: Builtin.global_invocation_id,
    token: StorageBuffer[u32, "read"],        # [1]
    table: StorageBuffer[u32, "read"],        # [vocab, hidden/2] packed f16
    x_out: StorageBuffer[f32, "read_write"],  # [hidden]
    dims: StorageBuffer[u32, "read"],         # [hidden]
):
    j: u32 = gid.x                            # pair index
    hidden: u32 = dims[0]
    if j >= hidden / 2:
        return
    tok: u32 = token[0]
    pair = unpack2x16float(table[tok * (hidden / 2) + j])
    scale: f32 = sqrt(f32(hidden))
    x_out[2 * j] = pair.x * scale
    x_out[2 * j + 1] = pair.y * scale


@kernel(workgroup_size=(32,))
def matvec_simd_packed(
    gid: Builtin.global_invocation_id,
    lane: Builtin.subgroup_invocation_id,
    w_mat: StorageBuffer[u32, "read"],        # [n_out, n_in/2] packed f16
    x_in: StorageBuffer[f32, "read"],
    y_out: StorageBuffer[f32, "read_write"],
    dims: StorageBuffer[u32, "read"],         # [n_out, n_in]
):
    r: u32 = gid.x / 32                       # one simdgroup per output row
    n_out: u32 = dims[0]
    n_in: u32 = dims[1]
    half_n: u32 = n_in / 2

    acc: f32 = 0.0
    if r < n_out:
        for j in range(lane, half_n, 32):
            pair = unpack2x16float(w_mat[r * half_n + j])
            acc += pair.x * x_in[2 * j] + pair.y * x_in[2 * j + 1]
    total: f32 = subgroupAdd(acc)
    if lane == 0 and r < n_out:
        y_out[r] = total


@kernel(workgroup_size=(32,))
def rmsnorm_simd(
    gid: Builtin.global_invocation_id,
    lane: Builtin.subgroup_invocation_id,
    x_in: StorageBuffer[f32, "read"],
    w: StorageBuffer[f32, "read"],
    x_out: StorageBuffer[f32, "read_write"],
    dims: StorageBuffer[u32, "read"],         # [n_rows, row_len]
):
    row: u32 = gid.x / 32
    n_rows: u32 = dims[0]
    row_len: u32 = dims[1]

    acc: f32 = 0.0
    if row < n_rows:
        for j in range(lane, row_len, 32):
            v: f32 = x_in[row * row_len + j]
            acc += v * v
    total: f32 = subgroupAdd(acc)
    inv: f32 = 1.0 / sqrt(total / f32(row_len) + 1e-6)
    if row < n_rows:
        for j in range(lane, row_len, 32):
            x_out[row * row_len + j] = x_in[row * row_len + j] * inv * (1.0 + w[j])


@kernel(workgroup_size=(32,))
def rmsnorm_add_simd(
    gid: Builtin.global_invocation_id,
    lane: Builtin.subgroup_invocation_id,
    x_in: StorageBuffer[f32, "read"],
    w: StorageBuffer[f32, "read"],
    x_io: StorageBuffer[f32, "read_write"],   # residual accumulator
    dims: StorageBuffer[u32, "read"],
):
    row: u32 = gid.x / 32
    n_rows: u32 = dims[0]
    row_len: u32 = dims[1]

    acc: f32 = 0.0
    if row < n_rows:
        for j in range(lane, row_len, 32):
            v: f32 = x_in[row * row_len + j]
            acc += v * v
    total: f32 = subgroupAdd(acc)
    inv: f32 = 1.0 / sqrt(total / f32(row_len) + 1e-6)
    if row < n_rows:
        for j in range(lane, row_len, 32):
            idx: u32 = row * row_len + j
            x_io[idx] = x_io[idx] + x_in[idx] * inv * (1.0 + w[j])


@kernel(workgroup_size=(32,))
def attention_simd(
    gid: Builtin.global_invocation_id,
    lane: Builtin.subgroup_invocation_id,
    q: StorageBuffer[f32, "read"],
    k_cache: StorageBuffer[f32, "read"],
    v_cache: StorageBuffer[f32, "read"],
    scores: StorageBuffer[f32, "read_write"],
    out_vec: StorageBuffer[f32, "read_write"],
    dims: StorageBuffer[u32, "read"],          # [n_heads, head_dim, kv_len, start, max_seq]
):
    h: u32 = gid.x / 32                        # one simdgroup per head
    n_heads: u32 = dims[0]
    head_dim: u32 = dims[1]
    kv_len: u32 = dims[2]
    start: u32 = dims[3]
    max_seq: u32 = dims[4]
    if h >= n_heads:
        return
    base: u32 = h * max_seq

    for t in range(start + lane, kv_len, 32):
        dot: f32 = 0.0
        for j in range(head_dim):
            dot += q[h * head_dim + j] * k_cache[t * head_dim + j]
        scores[base + t] = dot / sqrt(f32(head_dim))
    subgroup_barrier()

    m_local: f32 = -3.0e38
    for t in range(start + lane, kv_len, 32):
        m_local = max(m_local, scores[base + t])
    m: f32 = subgroupMax(m_local)

    sum_local: f32 = 0.0
    for t in range(start + lane, kv_len, 32):
        e: f32 = exp(scores[base + t] - m)
        scores[base + t] = e
        sum_local += e
    denom: f32 = subgroupAdd(sum_local)
    subgroup_barrier()

    for d in range(lane, head_dim, 32):
        acc: f32 = 0.0
        for t in range(start, kv_len):
            acc += scores[base + t] * v_cache[t * head_dim + d]
        out_vec[h * head_dim + d] = acc / denom


METAL_KERNELS = {
    "embed_scale_packed": embed_scale_packed,
    "matvec_simd_packed": matvec_simd_packed,
    "rmsnorm_simd": rmsnorm_simd,
    "rmsnorm_add_simd": rmsnorm_add_simd,
    "attention_simd": attention_simd,
}
