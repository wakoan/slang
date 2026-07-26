"""DSL-authored kernels for gemma4_150 — one Python source, WGSL and MSL out.

This is the seed of the unified kernel set. Today the fast path keeps two
hand-maintained copies of every shader (111 captured `.wgsl` driving wgpu and
the browser, 32 hand-translated `.metal` driving PyObjC/Swift/Rust); anything
added here is written once and emitted for both.

Starting with the GEMM because it is where the backends genuinely diverge.
Prefill's 1208 tok/s comes from `MPSMatrixMultiplication`, an Apple library
call with no WGSL equivalent — so Metal keeps MPS and every other backend needs
a real tiled GEMM of its own. `PrefillGPU` picks between them (see `_gemm`);
the kernel source stays single.

Shapes are literals because the DSL resolves helper *functions* lexically but
does not capture closure *values*. The tile constants below are duplicated as
Python ints for the host side to compute dispatch geometry from.
"""
from __future__ import annotations

from py_shader_lang_wgpu import (
    kernel, u32, f32, f16, vec4, StorageBuffer, AtomicBuffer, Uniform,
    WorkgroupArray, Builtin,
)

# Tile geometry. TILE_M x TILE_N output tile per workgroup, TILE_K deep,
# 16x16 threads each holding a 2x2 register block.
TILE_M = 32
TILE_N = 32
TILE_K = 16
WG_X = 16
WG_Y = 16


@kernel(workgroup_size=(16, 16, 1))
def gemm_tiled(
    a: StorageBuffer[f16, "read"],       # [M, K] with row stride lda
    b: StorageBuffer[f16, "read"],       # [N, K] if tr else [K, N], row stride ldb
    c: StorageBuffer[f16],               # [M, N] with row stride ldc
    p: Uniform[vec4[u32]],               # (M, N, K, bitcast(alpha))
    s: Uniform[vec4[u32]],               # (lda, ldb, ldc, tr)
    As: WorkgroupArray[f16, 512],        # TILE_M x TILE_K
    Bs: WorkgroupArray[f16, 512],        # TILE_N x TILE_K
    lid: Builtin.local_invocation_id,
    wg: Builtin.workgroup_id,
):
    """C = alpha * A @ (B^T if tr else B), f16 in/out, f32 accumulation.

    Accumulating in f32 matches what MPS does internally; f16 accumulation over
    K=1536 loses far too much.

    Both orientations live in one kernel because the transpose only changes how
    the B TILE is staged — once in workgroup memory it is always [n][k], so the
    inner product loop is untouched and the branch costs nothing per MAC. tr=1
    (weights, stored row-per-output) is the MPS transposeRight form; tr=0 is
    attention's P @ V.

    Explicit row strides let a matrix be a window on a wider buffer, which the
    score matrix needs: its rows are padded to keep them 16-byte aligned while
    the operation covers only the real keys.
    """
    M: u32 = p.x
    N: u32 = p.y
    K: u32 = p.z
    alpha: f32 = bitcast_f32(p.w)
    lda: u32 = s.x
    ldb: u32 = s.y
    ldc: u32 = s.z
    tr: u32 = s.w

    m0: u32 = wg.y * u32(32)
    n0: u32 = wg.x * u32(32)
    tx: u32 = lid.x
    ty: u32 = lid.y
    tid: u32 = ty * u32(16) + tx

    acc00: f32 = f32(0.0)
    acc01: f32 = f32(0.0)
    acc10: f32 = f32(0.0)
    acc11: f32 = f32(0.0)

    nkt: u32 = (K + u32(15)) / u32(16)
    for kt in range(nkt):
        kbase: u32 = u32(kt) * u32(16)
        # Stage both tiles. 512 elements, 256 threads -> two loads each.
        for l in range(2):
            idx: u32 = tid + u32(l) * u32(256)
            r: u32 = idx / u32(16)
            kk: u32 = idx % u32(16)
            gk: u32 = kbase + kk
            va: f16 = f16(0.0)
            if m0 + r < M:
                if gk < K:
                    va = a[(m0 + r) * lda + gk]
            As[idx] = va
            vb: f16 = f16(0.0)
            if n0 + r < N:
                if gk < K:
                    if tr == u32(1):
                        vb = b[(n0 + r) * ldb + gk]        # B is [N, K]
                    else:
                        vb = b[gk * ldb + n0 + r]          # B is [K, N]
            Bs[idx] = vb
        barrier()
        for kk2 in range(16):
            av0: f32 = f32(As[(ty * u32(2)) * u32(16) + u32(kk2)])
            av1: f32 = f32(As[(ty * u32(2) + u32(1)) * u32(16) + u32(kk2)])
            bv0: f32 = f32(Bs[(tx * u32(2)) * u32(16) + u32(kk2)])
            bv1: f32 = f32(Bs[(tx * u32(2) + u32(1)) * u32(16) + u32(kk2)])
            acc00 = acc00 + av0 * bv0
            acc01 = acc01 + av0 * bv1
            acc10 = acc10 + av1 * bv0
            acc11 = acc11 + av1 * bv1
        barrier()

    mr: u32 = m0 + ty * u32(2)
    nc: u32 = n0 + tx * u32(2)
    if mr < M:
        if nc < N:
            c[mr * ldc + nc] = f16(alpha * acc00)
        if nc + u32(1) < N:
            c[mr * ldc + nc + u32(1)] = f16(alpha * acc01)
    if mr + u32(1) < M:
        if nc < N:
            c[(mr + u32(1)) * ldc + nc] = f16(alpha * acc10)
        if nc + u32(1) < N:
            c[(mr + u32(1)) * ldc + nc + u32(1)] = f16(alpha * acc11)


def gemm_groups(M: int, N: int) -> tuple[int, int]:
    """Workgroup grid for gemm_tiled at this output size (x = N, y = M)."""
    return ((N + TILE_N - 1) // TILE_N, (M + TILE_M - 1) // TILE_M)


# ---------------------------------------------------------------------------
# Decode kernels. Each one replaces a hand-written pair (a .metal in
# kernels_msl/ and a .wgsl in reference/webml_gemma4_kernels/) and is verified
# BIT-EXACT against the kernel it replaces before the pair is retired.
# ---------------------------------------------------------------------------

def srq(x: f32, s: f32) -> f32:
    """Symmetric per-row quantization. s == 0 means the tensor is unquantized."""
    if s == f32(0.0):
        return x
    return clamp(round(x / s), f32(-128.0), f32(127.0)) * s


@kernel(workgroup_size=(256, 1, 1))
def rmssrq_69(
    x: StorageBuffer[f32, "read"],
    w: StorageBuffer[f32, "read"],
    y: StorageBuffer[f32],
    sum_a: StorageBuffer[f32],
    p: Uniform[vec4[u32]],               # (rows, rowStride, bitcast(inScale), _)
    sgp: WorkgroupArray[f32, 8],
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """Fused weighted RMSNorm + SRQ + sum-of-quantized-activations, one row per
    workgroup.

    The sum is not incidental: the quantized matmuls need sum(a) to undo the
    zero-point offset, so computing it here saves a whole pass over the row.

    The cross-subgroup combine is inlined rather than factored into a helper
    because DSL helpers take scalars only — a threadgroup array cannot be passed.
    """
    rows: u32 = p.x
    stride: u32 = p.y
    if stride == u32(0):
        stride = rows
    row: u32 = wg.x + wg.y * stride
    if row < rows:
        inScale: f32 = bitcast_f32(p.z)
        base: u32 = row * u32(1536)

        acc: f32 = f32(0.0)
        for i in range(tid, 1536, 256):
            v: f32 = x[base + i]
            acc = acc + v * v
        s1: f32 = subgroupAdd(acc)
        if (tid & u32(31)) == u32(0):
            sgp[tid >> u32(5)] = s1
        barrier()
        t1: f32 = f32(0.0)
        for k in range(8):
            t1 = t1 + sgp[k]
        barrier()
        sc: f32 = inverseSqrt(t1 / f32(1536.0) + f32(1e-6))

        qAcc: f32 = f32(0.0)
        for j in range(tid, 1536, 256):
            q: f32 = srq(x[base + j] * sc * w[j], inScale)
            y[base + j] = q
            qAcc = qAcc + q
        s2: f32 = subgroupAdd(qAcc)
        if (tid & u32(31)) == u32(0):
            sgp[tid >> u32(5)] = s2
        barrier()
        t2: f32 = f32(0.0)
        for k2 in range(8):
            t2 = t2 + sgp[k2]
        barrier()
        if tid == u32(0):
            sum_a[row] = t2


def tanh_safe(x: f32) -> f32:
    """tanh saturated outside +-10 — matches the reference kernels, and keeps
    the polynomial from overflowing before tanh flattens anyway."""
    if x > f32(10.0):
        return f32(1.0)
    if x < f32(-10.0):
        return f32(-1.0)
    return tanh(x)


def gelu_tanh(v: f32) -> f32:
    return f32(0.5) * v * (f32(1.0) + tanh_safe(
        f32(0.7978845608028654) * (v + f32(0.044715) * v * v * v)))


@kernel(workgroup_size=(256, 1, 1))
def srqh_b(
    x: StorageBuffer[f32, "read"],
    y: StorageBuffer[f16],
    p: Uniform[vec4[u32]],               # (bitcast(scale), n, _, _)
    gid: Builtin.global_invocation_id,
):
    """f32 activations -> f16 for a GEMM's A matrix, optionally SRQ'd.
    scale == 0 means convert only."""
    if gid.x < p.y:
        y[gid.x] = f16(srq(x[gid.x], bitcast_f32(p.x)))


@kernel(workgroup_size=(256, 1, 1))
def srq_b(
    x: StorageBuffer[f16, "read"],
    y: StorageBuffer[f32],
    p: Uniform[vec4[u32]],               # (bitcast(scale), n, _, _)
    gid: Builtin.global_invocation_id,
):
    """f16 GEMM output -> f32, applying the SRQ the decode kernel applied to its
    own result."""
    if gid.x < p.y:
        y[gid.x] = srq(f32(x[gid.x]), bitcast_f32(p.x))


@kernel(workgroup_size=(256, 1, 1))
def geglu_b(
    gate: StorageBuffer[f16, "read"],
    up: StorageBuffer[f16, "read"],
    out: StorageBuffer[f16],
    lut: StorageBuffer[f32, "read"],
    p: Uniform[vec4[u32]],               # (gateOutScale, upOutScale, outQuantScale, n)
    gid: Builtin.global_invocation_id,
):
    """The geglu tail, once gate/up are GEMMs.

    gate/up arrive already scaled and zero-point corrected (that folds into the
    dequantized weights), so only SRQ + gelu + product remain. When the gate is
    quantized the gelu comes from a 256-entry LUT indexed by the quantization
    grid — exact for every representable input, and cheaper than the polynomial.
    The LUT read is inlined because DSL helpers cannot take a buffer.
    """
    if gid.x < p.w:
        gs: f32 = bitcast_f32(p.x)
        g: f32 = srq(f32(gate[gid.x]), gs)
        u: f32 = srq(f32(up[gid.x]), bitcast_f32(p.y))
        gv: f32 = f32(0.0)
        if gs == f32(0.0):
            gv = gelu_tanh(g)
        else:
            gv = lut[u32(clamp(round(g / gs), f32(-128.0), f32(127.0)) + f32(128.0))]
        dq: f32 = gv * u
        qs: f32 = bitcast_f32(p.z)
        if qs == f32(0.0):
            out[gid.x] = f16(dq)
        else:
            out[gid.x] = f16(clamp(round(dq / qs), f32(-128.0), f32(127.0)))


@kernel(workgroup_size=(256, 1, 1))
def combine(
    ctx: StorageBuffer[f32, "read"],
    ple: StorageBuffer[f32, "read"],
    nw: StorageBuffer[f32, "read"],
    outp: StorageBuffer[f32],
    red: WorkgroupArray[f32, 256],
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """PLE input: per-row RMSNorm(ctx * H^-0.5) + ple, scaled by 2^-0.5.

    HINV and RS2 stay literals so this is a drop-in replacement for the
    hand-written kernel's binding layout; promoting them to a uniform would be
    an improvement but changes the runner's call.
    """
    base: u32 = wg.x * u32(256) + tid
    c: f32 = ctx[base] * f32(2.5515046504e-02)
    red[tid] = c * c
    barrier()
    s: u32 = u32(128)
    while s > u32(0):
        if tid < s:
            red[tid] = red[tid] + red[tid + s]
        barrier()
        s = s / u32(2)
    rms: f32 = inverseSqrt(red[0] / f32(256.0) + f32(1e-6))
    outp[base] = (c * rms * nw[tid] + ple[base]) * f32(0.7071067811865476)


@kernel(workgroup_size=(256, 1, 1))
def down_75(
    a: StorageBuffer[vec4[f16], "read"],     # activations, 2 half4 per packed word
    bits_buf: StorageBuffer[u32, "read"],    # 4-bit weights, WPR words per row
    pp: AtomicBuffer[u32],                   # [0,OUT_F) partials, [CTR] ticket
    scale: StorageBuffer[f32, "read"],
    hidden: StorageBuffer[f32],
    nw: StorageBuffer[f32, "read"],
    params: Uniform[vec4[u32]],              # (inScale, outScale, _, _) bitcast
    sgq: WorkgroupArray[vec4[f32], 8],
    sgs: WorkgroupArray[f32, 8],
    dsh: WorkgroupArray[f32, 1536],
    lastFlag: WorkgroupArray[u32, 1],
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """Fused 4-bit down-projection + post-FFN residual norm-add, in ONE dispatch.

    Each workgroup owns N_ROWS=4 output rows, publishes them through the atomic
    buffer, and takes a ticket; whichever workgroup draws the last ticket
    (TOTAL_WGS-1) re-reads every partial and finishes the RMSNorm-add itself.
    That is what saves a second dispatch and a round trip — and it is the reason
    the DSL needed atomics at all.

    The counter self-resets at [CTR] so the next token starts clean without a
    host-side clear.

    N_ROWS is unrolled into four accumulators rather than a loop over a local
    array: the DSL has no local arrays, and the four sums are independent, so
    unrolling preserves the arithmetic exactly.
    """
    rowBase: u32 = wg.x * u32(4)
    inScale: f32 = bitcast_f32(params.x)
    q0: f32 = f32(0.0)
    q1: f32 = f32(0.0)
    q2: f32 = f32(0.0)
    q3: f32 = f32(0.0)
    sumA: f32 = f32(0.0)

    for w in range(tid, 768, 256):
        av0: vec4[f32] = vec4[f32](a[w * u32(2)])
        av1: vec4[f32] = vec4[f32](a[w * u32(2) + u32(1)])
        sumA = sumA + ((av0.x + av0.y + av0.z + av0.w)
                       + (av1.x + av1.y + av1.z + av1.w))
        if rowBase < u32(1536):
            p0: u32 = bits_buf[rowBase * u32(768) + w]
            lo0: vec4[f32] = unpack4x8unorm(p0 & u32(0x0F0F0F0F))
            hi0: vec4[f32] = unpack4x8unorm((p0 >> u32(4)) & u32(0x0F0F0F0F))
            q0 = q0 + (dot(vec4[f32](lo0.x, hi0.x, lo0.y, hi0.y), av0)
                     + dot(vec4[f32](lo0.z, hi0.z, lo0.w, hi0.w), av1))
        if rowBase + u32(1) < u32(1536):
            p1: u32 = bits_buf[(rowBase + u32(1)) * u32(768) + w]
            lo1: vec4[f32] = unpack4x8unorm(p1 & u32(0x0F0F0F0F))
            hi1: vec4[f32] = unpack4x8unorm((p1 >> u32(4)) & u32(0x0F0F0F0F))
            q1 = q1 + (dot(vec4[f32](lo1.x, hi1.x, lo1.y, hi1.y), av0)
                     + dot(vec4[f32](lo1.z, hi1.z, lo1.w, hi1.w), av1))
        if rowBase + u32(2) < u32(1536):
            p2: u32 = bits_buf[(rowBase + u32(2)) * u32(768) + w]
            lo2: vec4[f32] = unpack4x8unorm(p2 & u32(0x0F0F0F0F))
            hi2: vec4[f32] = unpack4x8unorm((p2 >> u32(4)) & u32(0x0F0F0F0F))
            q2 = q2 + (dot(vec4[f32](lo2.x, hi2.x, lo2.y, hi2.y), av0)
                     + dot(vec4[f32](lo2.z, hi2.z, lo2.w, hi2.w), av1))
        if rowBase + u32(3) < u32(1536):
            p3: u32 = bits_buf[(rowBase + u32(3)) * u32(768) + w]
            lo3: vec4[f32] = unpack4x8unorm(p3 & u32(0x0F0F0F0F))
            hi3: vec4[f32] = unpack4x8unorm((p3 >> u32(4)) & u32(0x0F0F0F0F))
            q3 = q3 + (dot(vec4[f32](lo3.x, hi3.x, lo3.y, hi3.y), av0)
                     + dot(vec4[f32](lo3.z, hi3.z, lo3.w, hi3.w), av1))

    red: vec4[f32] = subgroupAdd(vec4[f32](q0, q1, q2, q3))
    redA: f32 = subgroupAdd(sumA)
    if (tid & u32(31)) == u32(0):
        sgq[tid >> u32(5)] = red
        sgs[tid >> u32(5)] = redA
    barrier()

    if tid == u32(0):
        tot: vec4[f32] = vec4[f32](f32(0.0), f32(0.0), f32(0.0), f32(0.0))
        aSum: f32 = f32(0.0)
        for i in range(8):
            tot = tot + sgq[i]
            aSum = aSum + sgs[i]
        outScale: f32 = bitcast_f32(params.y)
        zpA: f32 = f32(8.0) * aSum
        if rowBase < u32(1536):
            d0: f32 = srq(scale[rowBase] * (inScale * fma(tot.x, f32(255.0), -zpA)), outScale)
            atomicStore(pp, rowBase, bitcast_u32(d0))
        if rowBase + u32(1) < u32(1536):
            d1: f32 = srq(scale[rowBase + u32(1)] * (inScale * fma(tot.y, f32(255.0), -zpA)), outScale)
            atomicStore(pp, rowBase + u32(1), bitcast_u32(d1))
        if rowBase + u32(2) < u32(1536):
            d2: f32 = srq(scale[rowBase + u32(2)] * (inScale * fma(tot.z, f32(255.0), -zpA)), outScale)
            atomicStore(pp, rowBase + u32(2), bitcast_u32(d2))
        if rowBase + u32(3) < u32(1536):
            d3: f32 = srq(scale[rowBase + u32(3)] * (inScale * fma(tot.w, f32(255.0), -zpA)), outScale)
            atomicStore(pp, rowBase + u32(3), bitcast_u32(d3))

    storageBarrier()
    if tid == u32(0):
        ticket: u32 = atomicAdd(pp, u32(1536), u32(1))
        lastFlag[0] = u32(0)
        if ticket == u32(383):
            lastFlag[0] = u32(1)
    barrier()
    if lastFlag[0] == u32(1):
        if tid == u32(0):
            atomicStore(pp, u32(1536), u32(0))
        acc: f32 = f32(0.0)
        for o2 in range(tid, 1536, 256):
            d: f32 = bitcast_f32(atomicLoad(pp, o2))
            dsh[o2] = d
            acc = acc + d * d
        s1: f32 = subgroupAdd(acc)
        if (tid & u32(31)) == u32(0):
            sgs[tid >> u32(5)] = s1
        barrier()
        t1: f32 = f32(0.0)
        for i2 in range(8):
            t1 = t1 + sgs[i2]
        barrier()
        rms: f32 = inverseSqrt(t1 / f32(1536.0) + f32(1e-6))
        for o3 in range(tid, 1536, 256):
            hidden[o3] = hidden[o3] + dsh[o3] * rms * nw[o3]


@kernel(workgroup_size=(64, 1, 1))
def embed_00(
    ids: StorageBuffer[u32, "read"],
    bits_buf: StorageBuffer[u32, "read"],
    scale: StorageBuffer[f32, "read"],
    y: StorageBuffer[f32],
    params: Uniform[vec4[u32]],          # (seq, _, _, _)
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """2-bit embedding gather, one workgroup per token."""
    t: u32 = wg.x
    if t < params.x:
        id: u32 = ids[t]
        if id < u32(262144):
            for w in range(tid, 96, 64):
                packed: u32 = bits_buf[id * u32(96) + w]
                s: f32 = scale[id]                 # NUM_GROUPS == 1
                for v in range(16):
                    q: f32 = f32((packed >> (u32(v) * u32(2))) & u32(3))
                    y[t * u32(1536) + w * u32(16) + u32(v)] = \
                        f32(39.191835884530846) * s * (q - f32(2.0))


@kernel(workgroup_size=(64, 1, 1))
def plegather_01(
    ids: StorageBuffer[u32, "read"],
    bits_buf: StorageBuffer[u32, "read"],
    scale: StorageBuffer[f32, "read"],
    y: StorageBuffer[f32],
    params: Uniform[vec4[u32]],          # (seq, _, _, _)
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """4-bit per-layer-embedding gather. 35 scale groups of 256 per row, so the
    scale index moves with the column — unlike embed_00's single group."""
    t: u32 = wg.x
    if t < params.x:
        id: u32 = ids[t]
        if id < u32(262144):
            for w in range(tid, 1120, 64):
                packed: u32 = bits_buf[id * u32(1120) + w]
                for v in range(8):
                    c: u32 = w * u32(8) + u32(v)
                    s: f32 = scale[id * u32(35) + c / u32(256)]
                    q: f32 = f32((packed >> (u32(v) * u32(4))) & u32(15))
                    y[t * u32(8960) + c] = f32(16.0) * s * (q - f32(8.0))


@kernel(workgroup_size=(256, 1, 1))
def argmax1_34(
    x: StorageBuffer[f32, "read"],
    cand_val: StorageBuffer[f32],
    cand_idx: StorageBuffer[u32],
    wgVal: WorkgroupArray[f32, 256],
    wgIdx: WorkgroupArray[u32, 256],
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """Argmax pass 1: best candidate per 1024-wide slice of the logits.

    Ties break toward the LOWER index, matching the reference — with 262144
    logits, ties on -inf rows are common enough that an unspecified rule would
    make the sampled token depend on scheduling.
    """
    base: u32 = wg.x * u32(1024)
    end: u32 = min(base + u32(1024), u32(262144))
    bv: f32 = f32(-3.4028234663852886e38)
    bi: u32 = u32(0)
    i: u32 = base + tid
    while i < end:
        v: f32 = x[i]
        if v > bv:
            bv = v
            bi = i
        i = i + u32(256)
    wgVal[tid] = bv
    wgIdx[tid] = bi
    barrier()
    stride: u32 = u32(128)
    while stride > u32(0):
        if tid < stride:
            o: u32 = tid + stride
            if wgVal[o] > wgVal[tid]:
                wgVal[tid] = wgVal[o]
                wgIdx[tid] = wgIdx[o]
            else:
                if wgVal[o] == wgVal[tid]:
                    if wgIdx[o] < wgIdx[tid]:
                        wgIdx[tid] = wgIdx[o]
        barrier()
        stride = stride / u32(2)
    if tid == u32(0):
        cand_val[wg.x] = wgVal[0]
        cand_idx[wg.x] = wgIdx[0]


@kernel(workgroup_size=(256, 1, 1))
def argmax2_35(
    cand_val: StorageBuffer[f32, "read"],
    cand_idx: StorageBuffer[u32, "read"],
    out: StorageBuffer[u32],
    wgVal: WorkgroupArray[f32, 256],
    wgIdx: WorkgroupArray[u32, 256],
    tid: Builtin.local_invocation_index,
):
    """Argmax pass 2: the winner among the 256 candidates."""
    bv: f32 = f32(-3.4028234663852886e38)
    bi: u32 = u32(0)
    i: u32 = tid
    while i < u32(256):
        v: f32 = cand_val[i]
        idx: u32 = cand_idx[i]
        if v > bv:
            bv = v
            bi = idx
        else:
            if v == bv:
                if idx < bi:
                    bi = idx
        i = i + u32(256)
    wgVal[tid] = bv
    wgIdx[tid] = bi
    barrier()
    stride: u32 = u32(128)
    while stride > u32(0):
        if tid < stride:
            o: u32 = tid + stride
            if wgVal[o] > wgVal[tid]:
                wgVal[tid] = wgVal[o]
                wgIdx[tid] = wgIdx[o]
            else:
                if wgVal[o] == wgVal[tid]:
                    if wgIdx[o] < wgIdx[tid]:
                        wgIdx[tid] = wgIdx[o]
        barrier()
        stride = stride / u32(2)
    if tid == u32(0):
        out[0] = wgIdx[0]


@kernel(workgroup_size=(256, 1, 1), consts={"HD": 256, "HALF": 128})
def kvnorm(
    ink: StorageBuffer[f32, "read"],
    inv: StorageBuffer[f32, "read"],
    knorm: StorageBuffer[f32, "read"],
    cosT: StorageBuffer[f32, "read"],
    sinT: StorageBuffer[f32, "read"],
    kcache: StorageBuffer[f32],
    vcache: StorageBuffer[f32],
    p: Uniform[vec4[u32]],               # (cacheOffset, _, _, _)
    rk: WorkgroupArray[f32, 512],        # sized for the widest head_dim
    rv: WorkgroupArray[f32, 512],
    tid: Builtin.local_invocation_index,
):
    """k: RMSNorm * knorm + split-half RoPE; v: scale-free RMSNorm; -> caches.

    The first SHAPE-PARAMETERIZED kernel: head_dim is 256 on sliding layers and
    512 on full ones. HD/HALF are `consts`, folded at translate time, so both
    variants come from this one source — replacing the runner's string-patching
    of the shader text, where "HD" could match a substring.

    The workgroup arrays are sized for the widest variant because an annotation
    is evaluated once at definition; the narrow variant simply uses a prefix.
    """
    ko: f32 = ink[tid]
    vo: f32 = inv[tid]
    rk[tid] = ko * ko
    rv[tid] = vo * vo
    barrier()
    s: u32 = u32(HD) / u32(2)
    while s > u32(0):
        if tid < s:
            rk[tid] = rk[tid] + rk[tid + s]
            rv[tid] = rv[tid] + rv[tid + s]
        barrier()
        s = s / u32(2)
    rmsk: f32 = inverseSqrt(rk[0] / f32(HD) + f32(1e-6))
    rmsv: f32 = inverseSqrt(rv[0] / f32(HD) + f32(1e-6))
    vcache[p.x + tid] = vo * rmsv
    if tid < u32(HALF):
        n0: f32 = ink[tid] * rmsk * knorm[tid]
        n1: f32 = ink[tid + u32(HALF)] * rmsk * knorm[tid + u32(HALF)]
        c: f32 = cosT[tid]
        sn: f32 = sinT[tid]
        kcache[p.x + tid] = n0 * c - n1 * sn
        kcache[p.x + tid + u32(HALF)] = n1 * c + n0 * sn


@kernel(workgroup_size=(256, 1, 1))
def rmsadd_b(
    x: StorageBuffer[f16, "read"],
    nw: StorageBuffer[f32, "read"],
    hidden: StorageBuffer[f32],
    p: Uniform[vec4[u32]],               # (inScale, sv, dim, _)
    sgp: WorkgroupArray[f32, 8],
    dsh: WorkgroupArray[f32, 1536],
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """Residual-add + RMSNorm tail, as its own pass over S rows.

    In decode this tail hides behind down_75/oproj_73/pleproj_77's atomic
    last-arriver counter, which saves a dispatch. Batching that would need S
    counters, so prefill pays for a separate pass instead.
    """
    inScale: f32 = bitcast_f32(p.x)
    sv: f32 = bitcast_f32(p.y)
    base: u32 = wg.x * u32(1536)
    acc: f32 = f32(0.0)
    for j in range(tid, 1536, 256):
        d: f32 = srq(f32(x[base + j]), inScale)
        dsh[j] = d
        acc = acc + d * d
    s1: f32 = subgroupAdd(acc)
    if (tid & u32(31)) == u32(0):
        sgp[tid >> u32(5)] = s1
    barrier()
    t1: f32 = f32(0.0)
    for i in range(8):
        t1 = t1 + sgp[i]
    barrier()
    rms: f32 = inverseSqrt(t1 / f32(1536.0) + f32(1e-6))
    for j2 in range(tid, 1536, 256):
        hidden[base + j2] = (hidden[base + j2] + dsh[j2] * rms * nw[j2]) * sv


@kernel(workgroup_size=(256, 1, 1))
def rmssrqh_b(
    hidden: StorageBuffer[f32, "read"],
    w: StorageBuffer[f32, "read"],
    y: StorageBuffer[f16],
    sum_a: StorageBuffer[f32],
    p: Uniform[vec4[u32]],               # (inScale, dim, _, _)
    sgp: WorkgroupArray[f32, 8],
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """Pre-FFN RMSNorm + SRQ over S rows, emitting HALF.

    rmssrq_69 already handles S rows but writes f32, which is right before
    attention and wrong here — gateup reads half4. Reproduces oproj_73's DOUBLE
    rounding exactly, f16(srq(f32(f16(n2)), inScale)): the value is narrowed to
    f16 before quantizing and again after, and collapsing that to a single round
    changes results on the boundary.
    """
    inScale: f32 = bitcast_f32(p.x)
    base: u32 = wg.x * u32(1536)
    acc: f32 = f32(0.0)
    for j in range(tid, 1536, 256):
        v: f32 = hidden[base + j]
        acc = acc + v * v
    s1: f32 = subgroupAdd(acc)
    if (tid & u32(31)) == u32(0):
        sgp[tid >> u32(5)] = s1
    barrier()
    t1: f32 = f32(0.0)
    for i in range(8):
        t1 = t1 + sgp[i]
    barrier()
    rms: f32 = inverseSqrt(t1 / f32(1536.0) + f32(1e-6))
    qAcc: f32 = f32(0.0)
    for j2 in range(tid, 1536, 256):
        n2: f32 = hidden[base + j2] * rms * w[j2]
        qv: f16 = f16(srq(f32(f16(n2)), inScale))
        y[base + j2] = qv
        qAcc = qAcc + f32(qv)
    s2: f32 = subgroupAdd(qAcc)
    if (tid & u32(31)) == u32(0):
        sgp[tid >> u32(5)] = s2
    barrier()
    t2: f32 = f32(0.0)
    for i2 in range(8):
        t2 = t2 + sgp[i2]
    barrier()
    if tid == u32(0):
        sum_a[wg.x] = t2


@kernel(workgroup_size=(256, 1, 1))
def combine_b(
    ctx: StorageBuffer[f32, "read"],
    ple: StorageBuffer[f32, "read"],
    nw: StorageBuffer[f32, "read"],
    outp: StorageBuffer[f32],
    p: Uniform[vec4[u32]],               # (nL, _, _, _)
    red: WorkgroupArray[f32, 256],
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """`combine` over S tokens: one workgroup per (layer, token)."""
    base: u32 = (wg.y * p.x + wg.x) * u32(256) + tid
    c: f32 = ctx[base] * f32(2.5515046504e-02)
    red[tid] = c * c
    barrier()
    s: u32 = u32(128)
    while s > u32(0):
        if tid < s:
            red[tid] = red[tid] + red[tid + s]
        barrier()
        s = s / u32(2)
    rms: f32 = inverseSqrt(red[0] / f32(256.0) + f32(1e-6))
    outp[base] = (c * rms * nw[tid] + ple[base]) * f32(0.7071067811865476)


def srq4(x: vec4[f32], s: f32) -> vec4[f32]:
    """Vector SRQ. s == 0 means the tensor is unquantized."""
    if s == f32(0.0):
        return x
    return clamp(round(x / s),
                 vec4[f32](f32(-128.0), f32(-128.0), f32(-128.0), f32(-128.0)),
                 vec4[f32](f32(127.0), f32(127.0), f32(127.0), f32(127.0))) * s


@kernel(workgroup_size=(32, 1, 1))
def proj_68(
    a: StorageBuffer[f32, "read"],
    wt: StorageBuffer[f16, "read"],
    out: StorageBuffer[f32],
    params: Uniform[vec4[u32]],          # (inScale, outScale, _, _)
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """Dense f16 GEMV for per_layer_model_projection, 8 output rows per
    workgroup. One subgroup (32 threads) per workgroup, so the reduction is a
    single subgroupAdd with no threadgroup memory at all.

    The 8 rows are unrolled into 8 accumulators — the DSL has no local arrays,
    and the sums are independent, so the arithmetic is unchanged.
    """
    inScale: f32 = bitcast_f32(params.x)
    outScale: f32 = bitcast_f32(params.y)
    rowBase: u32 = (wg.y * u32(1120) + wg.x) * u32(8)
    if rowBase < u32(8960):
        a0: f32 = f32(0.0)
        a1: f32 = f32(0.0)
        a2: f32 = f32(0.0)
        a3: f32 = f32(0.0)
        a4v: f32 = f32(0.0)
        a5: f32 = f32(0.0)
        a6: f32 = f32(0.0)
        a7: f32 = f32(0.0)
        for k4 in range(tid, 384, 32):
            kb: u32 = k4 * u32(4)
            av: vec4[f32] = srq4(vec4[f32](a[kb], a[kb + u32(1)],
                                           a[kb + u32(2)], a[kb + u32(3)]), inScale)
            for r in range(8):
                o: u32 = rowBase + u32(r)
                if o < u32(8960):
                    wb: u32 = o * u32(1536) + kb
                    w4: vec4[f32] = vec4[f32](f32(wt[wb]), f32(wt[wb + u32(1)]),
                                              f32(wt[wb + u32(2)]), f32(wt[wb + u32(3)]))
                    d: f32 = dot(w4, av)
                    if r == 0:
                        a0 = a0 + d
                    if r == 1:
                        a1 = a1 + d
                    if r == 2:
                        a2 = a2 + d
                    if r == 3:
                        a3 = a3 + d
                    if r == 4:
                        a4v = a4v + d
                    if r == 5:
                        a5 = a5 + d
                    if r == 6:
                        a6 = a6 + d
                    if r == 7:
                        a7 = a7 + d
        s0: f32 = subgroupAdd(a0)
        s1: f32 = subgroupAdd(a1)
        s2: f32 = subgroupAdd(a2)
        s3: f32 = subgroupAdd(a3)
        s4: f32 = subgroupAdd(a4v)
        s5: f32 = subgroupAdd(a5)
        s6: f32 = subgroupAdd(a6)
        s7: f32 = subgroupAdd(a7)
        if tid == u32(0):
            if rowBase < u32(8960):
                out[rowBase] = srq(s0, outScale)
            if rowBase + u32(1) < u32(8960):
                out[rowBase + u32(1)] = srq(s1, outScale)
            if rowBase + u32(2) < u32(8960):
                out[rowBase + u32(2)] = srq(s2, outScale)
            if rowBase + u32(3) < u32(8960):
                out[rowBase + u32(3)] = srq(s3, outScale)
            if rowBase + u32(4) < u32(8960):
                out[rowBase + u32(4)] = srq(s4, outScale)
            if rowBase + u32(5) < u32(8960):
                out[rowBase + u32(5)] = srq(s5, outScale)
            if rowBase + u32(6) < u32(8960):
                out[rowBase + u32(6)] = srq(s6, outScale)
            if rowBase + u32(7) < u32(8960):
                out[rowBase + u32(7)] = srq(s7, outScale)


@kernel(workgroup_size=(32, 1, 1))
def plegate_76(
    a: StorageBuffer[f32, "read"],
    codes: StorageBuffer[u32, "read"],
    row_scale: StorageBuffer[f32, "read"],
    ple: StorageBuffer[f32, "read"],
    out: StorageBuffer[f32],
    gelu_lut: StorageBuffer[f32, "read"],
    params: Uniform[vec4[u32]],          # (inScale, linOutScale, pleOffset, _)
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """PLE input gate: int8 (+128-biased) GEMV -> gelu-LUT -> * ple.

    The +128 bias is undone in the epilogue via fma(s, 255, -128*aSum) rather
    than per weight — unpack_unorm4x8 divides by 255, so both corrections fold
    into one expression over the row sum.
    """
    inScale: f32 = bitcast_f32(params.x)
    linOutScale: f32 = bitcast_f32(params.y)
    o: u32 = wg.y * u32(256) + wg.x
    if o < u32(256):
        acc: f32 = f32(0.0)
        aAcc: f32 = f32(0.0)
        for wd in range(tid, 384, 32):
            kb: u32 = wd * u32(4)
            av: vec4[f32] = srq4(vec4[f32](a[kb], a[kb + u32(1)],
                                           a[kb + u32(2)], a[kb + u32(3)]), inScale)
            aAcc = aAcc + ((av.x + av.y) + (av.z + av.w))
            acc = acc + dot(unpack4x8unorm(codes[o * u32(384) + wd]), av)
        aSum: f32 = subgroupAdd(aAcc)
        s: f32 = subgroupAdd(acc)
        if tid == u32(0):
            v: f32 = row_scale[o] * fma(s, f32(255.0), f32(-128.0) * aSum)
            qv: f32 = srq(v, linOutScale)
            gv: f32 = f32(0.0)
            if linOutScale == f32(0.0):
                gv = gelu_tanh(qv)
            else:
                gv = gelu_lut[u32(clamp(round(qv / linOutScale),
                                        f32(-128.0), f32(127.0)) + f32(128.0))]
            out[o] = gv * ple[params.z + o]


@kernel(workgroup_size=(256, 1, 1), consts={"WPR": 256})
def oproj_73(
    a: StorageBuffer[vec4[f32], "read"],
    bits_buf: StorageBuffer[u32, "read"],
    scale: StorageBuffer[f32, "read"],
    pp: AtomicBuffer[u32],
    hidden: StorageBuffer[f32],
    w12: StorageBuffer[f32, "read"],
    y2: StorageBuffer[f16],
    sum2: StorageBuffer[f32],
    params: Uniform[vec4[u32]],          # (outScale, inScale2, _, _)
    sgp: WorkgroupArray[f32, 8],
    lastFlag: WorkgroupArray[u32, 1],
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """o-proj (4-bit) + post-attn norm-add + pre-FFN norm, all in one dispatch.

    One subgroup per output row for the GEMV, then the last-arriver runs BOTH
    norms: RMSNorm the projection, add it to the residual, then RMSNorm again
    for the FFN input. Two fused norms behind one atomic counter.

    WPR is a const (256 on sliding layers, 512 on full ones).

    The two norm passes keep their six per-thread residuals in SCALARS, not by
    re-reading hidden[]. The re-read looks equivalent and is not: the reference
    holds hv in a register, and the compiler's contraction there differs from a
    device round-trip by a ulp, which srq's round() turns into a full
    quantization step. That cost a real debugging cycle — it survived 12 random
    seeds and a single-dispatch check on real data, and only showed up as a
    changed token 98 decode steps in (see PORT_NOTES.md). OUT_F/WG is exactly 6,
    so the loops are unrolled sixfold rather than indexed.
    """
    sgId: u32 = tid / u32(32)
    lane: u32 = tid & u32(31)
    outScale: f32 = bitcast_f32(params.x)
    o: u32 = wg.x * u32(8) + sgId
    sumQA: f32 = f32(0.0)
    sumA: f32 = f32(0.0)
    for w in range(lane, WPR, 32):
        avc0: vec4[f32] = a[w * u32(2)]
        avc1: vec4[f32] = a[w * u32(2) + u32(1)]
        # NB the inner parens are load-bearing: the reference writes
        # `sumA += X + Y`, i.e. sumA + (X+Y). Dropping them gives (sumA+X)+Y,
        # which differs in the last ulp — enough to flip an srq rounding
        # eventually, and the model then takes a different path.
        sumA = sumA + ((avc0.x + avc0.y + avc0.z + avc0.w)
                       + (avc1.x + avc1.y + avc1.z + avc1.w))
        if o < u32(1536):
            p: u32 = bits_buf[o * u32(WPR) + w]
            lo: vec4[f32] = vec4[f32](unpack4xU8(p & u32(0x0F0F0F0F)))
            hi: vec4[f32] = vec4[f32](unpack4xU8((p >> u32(4)) & u32(0x0F0F0F0F)))
            sumQA = sumQA + (dot(vec4[f32](lo.x, hi.x, lo.y, hi.y), avc0)
                             + dot(vec4[f32](lo.z, hi.z, lo.w, hi.w), avc1))
    rA: f32 = subgroupAdd(sumA)
    rQA: f32 = subgroupAdd(sumQA)
    if lane == u32(0):
        if o < u32(1536):
            atomicStore(pp, o, bitcast_u32(
                srq(scale[o] * (rQA - f32(8.0) * rA), outScale)))
    storageBarrier()
    if tid == u32(0):
        tk: u32 = atomicAdd(pp, u32(1536), u32(1))
        lastFlag[0] = u32(0)
        if tk == u32(191):
            lastFlag[0] = u32(1)
    barrier()
    if lastFlag[0] == u32(1):
        if tid == u32(0):
            atomicStore(pp, u32(1536), u32(0))
        inScale2: f32 = bitcast_f32(params.y)
        acc1: f32 = f32(0.0)
        for i in range(tid, 1536, 256):
            v: f32 = bitcast_f32(atomicLoad(pp, i))
            acc1 = acc1 + v * v
        r1: f32 = subgroupAdd(acc1)
        if (tid & u32(31)) == u32(0):
            sgp[tid >> u32(5)] = r1
        barrier()
        t1: f32 = f32(0.0)
        for k1 in range(8):
            t1 = t1 + sgp[k1]
        barrier()
        rms1: f32 = inverseSqrt(t1 / f32(1536.0) + f32(1e-6))

        acc2: f32 = f32(0.0)
        h0: f32 = hidden[tid] + bitcast_f32(atomicLoad(pp, tid)) * rms1 * w12[tid]
        hidden[tid] = h0
        acc2 = acc2 + h0 * h0
        i1: u32 = tid + u32(256)
        h1: f32 = hidden[i1] + bitcast_f32(atomicLoad(pp, i1)) * rms1 * w12[i1]
        hidden[i1] = h1
        acc2 = acc2 + h1 * h1
        i2: u32 = tid + u32(512)
        h2: f32 = hidden[i2] + bitcast_f32(atomicLoad(pp, i2)) * rms1 * w12[i2]
        hidden[i2] = h2
        acc2 = acc2 + h2 * h2
        i3: u32 = tid + u32(768)
        h3: f32 = hidden[i3] + bitcast_f32(atomicLoad(pp, i3)) * rms1 * w12[i3]
        hidden[i3] = h3
        acc2 = acc2 + h3 * h3
        i4: u32 = tid + u32(1024)
        h4: f32 = hidden[i4] + bitcast_f32(atomicLoad(pp, i4)) * rms1 * w12[i4]
        hidden[i4] = h4
        acc2 = acc2 + h4 * h4
        i5: u32 = tid + u32(1280)
        h5: f32 = hidden[i5] + bitcast_f32(atomicLoad(pp, i5)) * rms1 * w12[i5]
        hidden[i5] = h5
        acc2 = acc2 + h5 * h5
        r2: f32 = subgroupAdd(acc2)
        if (tid & u32(31)) == u32(0):
            sgp[tid >> u32(5)] = r2
        barrier()
        t2: f32 = f32(0.0)
        for k2 in range(8):
            t2 = t2 + sgp[k2]
        barrier()
        rms2: f32 = inverseSqrt(t2 / f32(1536.0) + f32(1e-6))

        qAcc: f32 = f32(0.0)
        q0v: f16 = f16(srq(f32(f16(h0 * rms2 * w12[u32(1536) + tid])), inScale2))
        y2[tid] = q0v
        qAcc = qAcc + f32(q0v)
        q1v: f16 = f16(srq(f32(f16(h1 * rms2 * w12[u32(1536) + i1])), inScale2))
        y2[i1] = q1v
        qAcc = qAcc + f32(q1v)
        q2v: f16 = f16(srq(f32(f16(h2 * rms2 * w12[u32(1536) + i2])), inScale2))
        y2[i2] = q2v
        qAcc = qAcc + f32(q2v)
        q3v: f16 = f16(srq(f32(f16(h3 * rms2 * w12[u32(1536) + i3])), inScale2))
        y2[i3] = q3v
        qAcc = qAcc + f32(q3v)
        q4v: f16 = f16(srq(f32(f16(h4 * rms2 * w12[u32(1536) + i4])), inScale2))
        y2[i4] = q4v
        qAcc = qAcc + f32(q4v)
        q5v: f16 = f16(srq(f32(f16(h5 * rms2 * w12[u32(1536) + i5])), inScale2))
        y2[i5] = q5v
        qAcc = qAcc + f32(q5v)
        r3: f32 = subgroupAdd(qAcc)
        if (tid & u32(31)) == u32(0):
            sgp[tid >> u32(5)] = r3
        barrier()
        t3: f32 = f32(0.0)
        for k3 in range(8):
            t3 = t3 + sgp[k3]
        barrier()
        if tid == u32(0):
            sum2[0] = t3
