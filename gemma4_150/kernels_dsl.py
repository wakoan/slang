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
    WorkgroupArray, PrivateArray, Builtin,
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

    The four row accumulators live in a PrivateArray, matching the reference's
    `float q[4]`. They were scalars at first, which passes both gates here — but
    it is the same substitution that silently broke oproj_73, and there is no
    gain to offset the risk (see PORT_NOTES.md).
    """
    rowBase: u32 = wg.x * u32(4)
    inScale: f32 = bitcast_f32(params.x)
    q: PrivateArray[f32, 4]
    for r0 in range(4):
        q[r0] = f32(0.0)
    sumA: f32 = f32(0.0)

    for w in range(tid, 768, 256):
        av0: vec4[f32] = vec4[f32](a[w * u32(2)])
        av1: vec4[f32] = vec4[f32](a[w * u32(2) + u32(1)])
        sumA = sumA + ((av0.x + av0.y + av0.z + av0.w)
                       + (av1.x + av1.y + av1.z + av1.w))
        for r in range(4):
            o: u32 = rowBase + u32(r)
            if o < u32(1536):
                p: u32 = bits_buf[o * u32(768) + w]
                # unorm (÷255), NOT unpack4xU8: the epilogue's fma(...,255,...)
                # is what undoes it. oproj_73 uses the raw-byte form and has no
                # 255 factor — the two are not interchangeable.
                lo: vec4[f32] = unpack4x8unorm(p & u32(0x0F0F0F0F))
                hi: vec4[f32] = unpack4x8unorm((p >> u32(4)) & u32(0x0F0F0F0F))
                q[r] = q[r] + (dot(vec4[f32](lo.x, hi.x, lo.y, hi.y), av0)
                               + dot(vec4[f32](lo.z, hi.z, lo.w, hi.w), av1))

    red: vec4[f32] = subgroupAdd(vec4[f32](q[0], q[1], q[2], q[3]))
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
        outScale: f32 = bitcast_f32(params[1])
        zpA: f32 = f32(8.0) * aSum
        for r2 in range(4):
            o2: u32 = rowBase + u32(r2)
            if o2 < u32(1536):
                dv: f32 = srq(scale[o2] * (inScale * fma(tot[r2], f32(255.0), -zpA)),
                              outScale)
                atomicStore(pp, o2, bitcast_u32(dv))

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
    outScale: f32 = bitcast_f32(params[1])
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
    linOutScale: f32 = bitcast_f32(params[1])
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
            out[o] = gv * ple[params[2] + o]


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

    The two norm passes cache their six per-thread residuals in a PrivateArray,
    exactly as the reference does. Neither a device re-read nor six scalars is
    an acceptable substitute: Metal compiles with fast math, so multiply-add
    contraction follows the expression tree, and both substitutions shifted one
    element by a ulp — which srq's round() turned into a full quantization step
    and a different token 98 decode steps later (see PORT_NOTES.md).
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
        inScale2: f32 = bitcast_f32(params[1])
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

        hloc: PrivateArray[f32, 6]
        acc2: f32 = f32(0.0)
        e: u32 = u32(0)
        for j in range(tid, 1536, 256):
            normed: f32 = bitcast_f32(atomicLoad(pp, j)) * rms1 * w12[j]
            hv: f32 = hidden[j] + normed
            hidden[j] = hv
            hloc[e] = hv
            acc2 = acc2 + hv * hv
            e = e + u32(1)
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
        e2: u32 = u32(0)
        for j2 in range(tid, 1536, 256):
            n2: f32 = hloc[e2] * rms2 * w12[u32(1536) + j2]
            qv: f16 = f16(srq(f32(f16(n2)), inScale2))
            y2[j2] = qv
            qAcc = qAcc + f32(qv)
            e2 = e2 + u32(1)
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


@kernel(workgroup_size=(256, 1, 1))
def pleproj_77(
    a: StorageBuffer[f32, "read"],
    codes: StorageBuffer[u32, "read"],
    row_scale: StorageBuffer[f32, "read"],
    pp: AtomicBuffer[u32],
    hidden: StorageBuffer[f32],
    w12s: StorageBuffer[f32, "read"],
    y2: StorageBuffer[f32],
    sum2: StorageBuffer[f32],
    params: Uniform[vec4[u32]],          # (inScale, projInScale, projOutScale, _)
    sgp: WorkgroupArray[f32, 8],
    lastFlag: WorkgroupArray[u32, 1],
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """PLE projection (int8) + norm-add * layer_scalar + next norm, one dispatch.

    Same last-arriver shape as oproj_73, with three per-thread arrays kept as
    PrivateArrays to match the reference's expression tree exactly — see
    PORT_NOTES.md for why substituting scalars or a re-read is not safe here.

    `sv` is the learned per-layer scalar, stored just past the two norm weight
    vectors in w12s.
    """
    projInScale: f32 = bitcast_f32(params[1])
    projOutScale: f32 = bitcast_f32(params[2])
    sgId: u32 = tid / u32(32)
    lane: u32 = tid & u32(31)
    rowBase: u32 = wg.x * u32(16) + sgId * u32(2)

    av: PrivateArray[vec4[f32], 2]
    aAcc: f32 = f32(0.0)
    for ki in range(2):
        k4: u32 = lane + u32(ki) * u32(32)
        av[ki] = vec4[f32](f32(0.0), f32(0.0), f32(0.0), f32(0.0))
        if k4 < u32(64):
            kb: u32 = k4 * u32(4)
            av[ki] = srq4(vec4[f32](a[kb], a[kb + u32(1)], a[kb + u32(2)],
                                    a[kb + u32(3)]), projInScale)
            aAcc = aAcc + ((av[ki].x + av[ki].y) + (av[ki].z + av[ki].w))

    accs: PrivateArray[f32, 2]
    for r in range(2):
        o: u32 = rowBase + u32(r)
        acc: f32 = f32(0.0)
        if o < u32(1536):
            for ki2 in range(2):
                k42: u32 = lane + u32(ki2) * u32(32)
                if k42 < u32(64):
                    acc = acc + dot(unpack4x8unorm(codes[o * u32(64) + k42]), av[ki2])
        accs[r] = acc
    aSum: f32 = subgroupAdd(aAcc)
    for r2 in range(2):
        s: f32 = subgroupAdd(accs[r2])
        o2: u32 = rowBase + u32(r2)
        if lane == u32(0):
            if o2 < u32(1536):
                atomicStore(pp, o2, bitcast_u32(srq(
                    row_scale[o2] * fma(s, f32(255.0), f32(-128.0) * aSum),
                    projOutScale)))
    storageBarrier()
    if tid == u32(0):
        tk: u32 = atomicAdd(pp, u32(1536), u32(1))
        lastFlag[0] = u32(0)
        if tk == u32(95):
            lastFlag[0] = u32(1)
    barrier()
    if lastFlag[0] == u32(1):
        if tid == u32(0):
            atomicStore(pp, u32(1536), u32(0))
        inScale: f32 = bitcast_f32(params.x)
        sv: f32 = w12s[u32(3072)]
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

        hloc: PrivateArray[f32, 6]
        acc2: f32 = f32(0.0)
        e: u32 = u32(0)
        for j in range(tid, 1536, 256):
            normed: f32 = bitcast_f32(atomicLoad(pp, j)) * rms1 * w12s[j]
            hv: f32 = (hidden[j] + normed) * sv
            hidden[j] = hv
            hloc[e] = hv
            acc2 = acc2 + hv * hv
            e = e + u32(1)
        r2v: f32 = subgroupAdd(acc2)
        if (tid & u32(31)) == u32(0):
            sgp[tid >> u32(5)] = r2v
        barrier()
        t2: f32 = f32(0.0)
        for k2 in range(8):
            t2 = t2 + sgp[k2]
        barrier()
        rms2: f32 = inverseSqrt(t2 / f32(1536.0) + f32(1e-6))

        qAcc: f32 = f32(0.0)
        e2: u32 = u32(0)
        for j2 in range(tid, 1536, 256):
            n2: f32 = hloc[e2] * rms2 * w12s[u32(1536) + j2]
            qv: f32 = srq(n2, inScale)
            y2[j2] = qv
            qAcc = qAcc + qv
            e2 = e2 + u32(1)
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


@kernel(workgroup_size=(256, 1, 1))
def down_96(
    a: StorageBuffer[vec4[f16], "read"],
    bits_buf: StorageBuffer[u32, "read"],
    pp: AtomicBuffer[u32],
    scale: StorageBuffer[f32, "read"],
    hidden: StorageBuffer[f32],
    nw: StorageBuffer[f32, "read"],
    params: Uniform[vec4[u32]],
    sgq: WorkgroupArray[vec4[f32], 8],
    sgs: WorkgroupArray[f32, 8],
    dsh: WorkgroupArray[f32, 1536],
    lastFlag: WorkgroupArray[u32, 1],
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """down_75's 2-bit twin for the wide-MLP layers (intermediate 12288).

    Four 2-bit fields per byte instead of two 4-bit ones, so four activation
    chunks and a ZP of 2 rather than 8. Same last-arriver merge.
    """
    rowBase: u32 = wg.x * u32(4)
    inScale: f32 = bitcast_f32(params.x)
    q: PrivateArray[f32, 4]
    for r0 in range(4):
        q[r0] = f32(0.0)
    sumA: f32 = f32(0.0)

    for w in range(tid, 768, 256):
        av0: vec4[f32] = vec4[f32](a[w * u32(4)])
        av1: vec4[f32] = vec4[f32](a[w * u32(4) + u32(1)])
        av2: vec4[f32] = vec4[f32](a[w * u32(4) + u32(2)])
        av3: vec4[f32] = vec4[f32](a[w * u32(4) + u32(3)])
        sumA = sumA + ((av0.x + av0.y + av0.z + av0.w)
                       + (av1.x + av1.y + av1.z + av1.w)
                       + (av2.x + av2.y + av2.z + av2.w)
                       + (av3.x + av3.y + av3.z + av3.w))
        for r in range(4):
            o: u32 = rowBase + u32(r)
            if o < u32(1536):
                p: u32 = bits_buf[o * u32(768) + w]
                c0: vec4[f32] = unpack4x8unorm(p & u32(0x03030303))
                c1: vec4[f32] = unpack4x8unorm((p >> u32(2)) & u32(0x03030303))
                c2: vec4[f32] = unpack4x8unorm((p >> u32(4)) & u32(0x03030303))
                c3: vec4[f32] = unpack4x8unorm((p >> u32(6)) & u32(0x03030303))
                q[r] = q[r] + (dot(vec4[f32](c0.x, c1.x, c2.x, c3.x), av0)
                               + dot(vec4[f32](c0.y, c1.y, c2.y, c3.y), av1)
                               + dot(vec4[f32](c0.z, c1.z, c2.z, c3.z), av2)
                               + dot(vec4[f32](c0.w, c1.w, c2.w, c3.w), av3))

    red: vec4[f32] = subgroupAdd(vec4[f32](q[0], q[1], q[2], q[3]))
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
        outScale: f32 = bitcast_f32(params[1])
        zpA: f32 = f32(2.0) * aSum
        for r2 in range(4):
            o2: u32 = rowBase + u32(r2)
            if o2 < u32(1536):
                dv: f32 = srq(scale[o2] * (inScale * fma(tot[r2], f32(255.0), -zpA)),
                              outScale)
                atomicStore(pp, o2, bitcast_u32(dv))
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
        for o3 in range(tid, 1536, 256):
            d: f32 = bitcast_f32(atomicLoad(pp, o3))
            dsh[o3] = d
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
        for o4 in range(tid, 1536, 256):
            hidden[o4] = hidden[o4] + dsh[o4] * rms * nw[o4]


@kernel(workgroup_size=(64, 1, 1))
def gateup_74(
    hidden: StorageBuffer[vec4[f16], "read"],
    gate_bits: StorageBuffer[u32, "read"],
    gate_scale: StorageBuffer[f32, "read"],
    up_bits: StorageBuffer[u32, "read"],
    up_scale: StorageBuffer[f32, "read"],
    sum_a: StorageBuffer[f32, "read"],
    out: StorageBuffer[f16],
    gelu_lut: StorageBuffer[f32, "read"],
    params: Uniform[vec4[u32]],
    lidx: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """Fused gate/up + geglu (4-bit), virtual-subgroup GEMV.

    Two subgroups per workgroup, each owning 4 output rows; gate and up are
    accumulated together so the shared activation chunk is read once for both.
    """
    gOut: f32 = bitcast_f32(params.x)
    uOut: f32 = bitcast_f32(params[1])
    sgId: u32 = lidx / u32(32)
    tid: u32 = lidx & u32(31)
    rowBase: u32 = (wg.x * u32(2) + sgId) * u32(4)
    gAcc: PrivateArray[f32, 4]
    uAcc: PrivateArray[f32, 4]
    for r0 in range(4):
        gAcc[r0] = f32(0.0)
        uAcc[r0] = f32(0.0)
    for wd in range(tid, 192, 32):
        a0: vec4[f32] = vec4[f32](hidden[wd * u32(2)])
        a1: vec4[f32] = vec4[f32](hidden[wd * u32(2) + u32(1)])
        for r in range(4):
            o: u32 = rowBase + u32(r)
            if o < u32(6144):
                pg: u32 = gate_bits[o * u32(192) + wd]
                pu: u32 = up_bits[o * u32(192) + wd]
                glo: vec4[f32] = unpack4x8unorm(pg & u32(0x0F0F0F0F))
                ghi: vec4[f32] = unpack4x8unorm((pg >> u32(4)) & u32(0x0F0F0F0F))
                gAcc[r] = gAcc[r] + (dot(vec4[f32](glo.x, ghi.x, glo.y, ghi.y), a0)
                                     + dot(vec4[f32](glo.z, ghi.z, glo.w, ghi.w), a1))
                ulo: vec4[f32] = unpack4x8unorm(pu & u32(0x0F0F0F0F))
                uhi: vec4[f32] = unpack4x8unorm((pu >> u32(4)) & u32(0x0F0F0F0F))
                uAcc[r] = uAcc[r] + (dot(vec4[f32](ulo.x, uhi.x, ulo.y, uhi.y), a0)
                                     + dot(vec4[f32](ulo.z, uhi.z, ulo.w, uhi.w), a1))
    aSum: f32 = sum_a[0]
    for r2 in range(4):
        gS: f32 = subgroupAdd(gAcc[r2])
        uS: f32 = subgroupAdd(uAcc[r2])
        if tid == u32(0):
            o2: u32 = rowBase + u32(r2)
            if o2 < u32(6144):
                g: f32 = srq(gate_scale[o2] * fma(gS, f32(255.0),
                                                  -(f32(8.0) * aSum)), gOut)
                u: f32 = srq(up_scale[o2] * fma(uS, f32(255.0),
                                                -(f32(8.0) * aSum)), uOut)
                gv: f32 = f32(0.0)
                if gOut == f32(0.0):
                    gv = gelu_tanh(g)
                else:
                    gv = gelu_lut[u32(clamp(round(g / gOut),
                                            f32(-128.0), f32(127.0)) + f32(128.0))]
                dq: f32 = gv * u
                qs: f32 = bitcast_f32(params[2])
                if qs == f32(0.0):
                    out[o2] = f16(dq)
                else:
                    out[o2] = f16(clamp(round(dq / qs), f32(-128.0), f32(127.0)))
    barrier()


@kernel(workgroup_size=(64, 1, 1))
def gateup_95(
    hidden: StorageBuffer[vec4[f16], "read"],
    gate_bits: StorageBuffer[u32, "read"],
    gate_scale: StorageBuffer[f32, "read"],
    up_bits: StorageBuffer[u32, "read"],
    up_scale: StorageBuffer[f32, "read"],
    sum_a: StorageBuffer[f32, "read"],
    out: StorageBuffer[f16],
    gelu_lut: StorageBuffer[f32, "read"],
    params: Uniform[vec4[u32]],
    lidx: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """gateup_74's 2-bit twin for the wide-MLP layers (intermediate 12288).

    Two rows per subgroup instead of four — the rows are twice as many, so the
    grid grows rather than the per-thread work.
    """
    gOut: f32 = bitcast_f32(params.x)
    uOut: f32 = bitcast_f32(params[1])
    sgId: u32 = lidx / u32(32)
    tid: u32 = lidx & u32(31)
    rowBase: u32 = (wg.x * u32(2) + sgId) * u32(2)
    gAcc: PrivateArray[f32, 2]
    uAcc: PrivateArray[f32, 2]
    for r0 in range(2):
        gAcc[r0] = f32(0.0)
        uAcc[r0] = f32(0.0)
    for wd in range(tid, 96, 32):
        a0: vec4[f32] = vec4[f32](hidden[wd * u32(4)])
        a1: vec4[f32] = vec4[f32](hidden[wd * u32(4) + u32(1)])
        a2: vec4[f32] = vec4[f32](hidden[wd * u32(4) + u32(2)])
        a3: vec4[f32] = vec4[f32](hidden[wd * u32(4) + u32(3)])
        for r in range(2):
            o: u32 = rowBase + u32(r)
            if o < u32(12288):
                pg: u32 = gate_bits[o * u32(96) + wd]
                pu: u32 = up_bits[o * u32(96) + wd]
                g0: vec4[f32] = unpack4x8unorm(pg & u32(0x03030303))
                g1: vec4[f32] = unpack4x8unorm((pg >> u32(2)) & u32(0x03030303))
                g2: vec4[f32] = unpack4x8unorm((pg >> u32(4)) & u32(0x03030303))
                g3: vec4[f32] = unpack4x8unorm((pg >> u32(6)) & u32(0x03030303))
                gAcc[r] = gAcc[r] + (dot(vec4[f32](g0.x, g1.x, g2.x, g3.x), a0)
                                     + dot(vec4[f32](g0.y, g1.y, g2.y, g3.y), a1)
                                     + dot(vec4[f32](g0.z, g1.z, g2.z, g3.z), a2)
                                     + dot(vec4[f32](g0.w, g1.w, g2.w, g3.w), a3))
                u0: vec4[f32] = unpack4x8unorm(pu & u32(0x03030303))
                u1: vec4[f32] = unpack4x8unorm((pu >> u32(2)) & u32(0x03030303))
                u2: vec4[f32] = unpack4x8unorm((pu >> u32(4)) & u32(0x03030303))
                u3: vec4[f32] = unpack4x8unorm((pu >> u32(6)) & u32(0x03030303))
                uAcc[r] = uAcc[r] + (dot(vec4[f32](u0.x, u1.x, u2.x, u3.x), a0)
                                     + dot(vec4[f32](u0.y, u1.y, u2.y, u3.y), a1)
                                     + dot(vec4[f32](u0.z, u1.z, u2.z, u3.z), a2)
                                     + dot(vec4[f32](u0.w, u1.w, u2.w, u3.w), a3))
    aSum: f32 = sum_a[0]
    for r2 in range(2):
        gS: f32 = subgroupAdd(gAcc[r2])
        uS: f32 = subgroupAdd(uAcc[r2])
        if tid == u32(0):
            o2: u32 = rowBase + u32(r2)
            if o2 < u32(12288):
                g: f32 = srq(gate_scale[o2] * fma(gS, f32(255.0),
                                                  -(f32(2.0) * aSum)), gOut)
                u: f32 = srq(up_scale[o2] * fma(uS, f32(255.0),
                                                -(f32(2.0) * aSum)), uOut)
                gv: f32 = f32(0.0)
                if gOut == f32(0.0):
                    gv = gelu_tanh(g)
                else:
                    gv = gelu_lut[u32(clamp(round(g / gOut),
                                            f32(-128.0), f32(127.0)) + f32(128.0))]
                dq: f32 = gv * u
                qs: f32 = bitcast_f32(params[2])
                if qs == f32(0.0):
                    out[o2] = f16(dq)
                else:
                    out[o2] = f16(clamp(round(dq / qs), f32(-128.0), f32(127.0)))
    barrier()


@kernel(workgroup_size=(32, 1, 1),
        consts={"Q_OUT": 2048, "KV_OUT": 256, "Q_WGS": 1024, "KV_WGS": 128,
                "TOTAL_WGS": 1280, "GRID_X": 1280})
def qkv_70(
    a: StorageBuffer[vec4[f32], "read"],
    q_bits: StorageBuffer[u32, "read"],
    k_bits: StorageBuffer[u32, "read"],
    v_bits: StorageBuffer[u32, "read"],
    scales: StorageBuffer[f32, "read"],
    sum_a: StorageBuffer[f32, "read"],
    out_q: StorageBuffer[f32],
    out_k: StorageBuffer[f32],
    out_v: StorageBuffer[f32],
    params: Uniform[vec4[u32]],          # (qOutScale, kOutScale, vOutScale, _)
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """q, k and v projections in ONE dispatch, split by workgroup id.

    The three matrices have different row counts, so rather than three dispatches
    the grid is partitioned: the first Q_WGS workgroups do q, the next KV_WGS do
    k, the rest do v. All six shape constants are per-layer consts.

    Raw-byte unpacking here (no /255) with the matching zero-point subtraction
    `rQA - ZP*rA` — unlike the MLP kernels, which use the unorm form and undo it
    with fma(...,255,...).
    """
    wgId: u32 = wg.y * u32(GRID_X) + wg.x
    if wgId < u32(TOTAL_WGS):
        sumQA: PrivateArray[f32, 2]
        for r0 in range(2):
            sumQA[r0] = f32(0.0)
        if wgId < u32(Q_WGS):
            rowBase: u32 = wgId * u32(2)
            for w in range(tid, 192, 32):
                avc0: vec4[f32] = a[w * u32(2)]
                avc1: vec4[f32] = a[w * u32(2) + u32(1)]
                for r in range(2):
                    o: u32 = rowBase + u32(r)
                    if o < u32(Q_OUT):
                        p: u32 = q_bits[o * u32(192) + w]
                        lo: vec4[f32] = vec4[f32](unpack4xU8(p & u32(0x0F0F0F0F)))
                        hi: vec4[f32] = vec4[f32](unpack4xU8((p >> u32(4)) & u32(0x0F0F0F0F)))
                        sumQA[r] = sumQA[r] + (dot(vec4[f32](lo.x, hi.x, lo.y, hi.y), avc0)
                                               + dot(vec4[f32](lo.z, hi.z, lo.w, hi.w), avc1))
            rA: f32 = sum_a[0]
            for r2 in range(2):
                rQA: f32 = subgroupAdd(sumQA[r2])
                o2: u32 = rowBase + u32(r2)
                if tid == u32(0):
                    if o2 < u32(Q_OUT):
                        out_q[o2] = srq(scales[o2] * (rQA - f32(8.0) * rA),
                                        bitcast_f32(params.x))
        else:
            if wgId < u32(Q_WGS) + u32(KV_WGS):
                rowBaseK: u32 = (wgId - u32(Q_WGS)) * u32(2)
                for wk in range(tid, 192, 32):
                    kvc0: vec4[f32] = a[wk * u32(2)]
                    kvc1: vec4[f32] = a[wk * u32(2) + u32(1)]
                    for rk in range(2):
                        ok: u32 = rowBaseK + u32(rk)
                        if ok < u32(KV_OUT):
                            pk: u32 = k_bits[ok * u32(192) + wk]
                            klo: vec4[f32] = vec4[f32](unpack4xU8(pk & u32(0x0F0F0F0F)))
                            khi: vec4[f32] = vec4[f32](unpack4xU8((pk >> u32(4)) & u32(0x0F0F0F0F)))
                            sumQA[rk] = sumQA[rk] + (dot(vec4[f32](klo.x, khi.x, klo.y, khi.y), kvc0)
                                                     + dot(vec4[f32](klo.z, khi.z, klo.w, khi.w), kvc1))
                rAk: f32 = sum_a[0]
                for rk2 in range(2):
                    rQAk: f32 = subgroupAdd(sumQA[rk2])
                    ok2: u32 = rowBaseK + u32(rk2)
                    if tid == u32(0):
                        if ok2 < u32(KV_OUT):
                            out_k[ok2] = srq(scales[u32(Q_OUT) + ok2] * (rQAk - f32(8.0) * rAk),
                                             bitcast_f32(params[1]))
            else:
                rowBaseV: u32 = (wgId - u32(Q_WGS) - u32(KV_WGS)) * u32(2)
                for wv in range(tid, 192, 32):
                    vvc0: vec4[f32] = a[wv * u32(2)]
                    vvc1: vec4[f32] = a[wv * u32(2) + u32(1)]
                    for rv in range(2):
                        ov: u32 = rowBaseV + u32(rv)
                        if ov < u32(KV_OUT):
                            pv: u32 = v_bits[ov * u32(192) + wv]
                            vlo: vec4[f32] = vec4[f32](unpack4xU8(pv & u32(0x0F0F0F0F)))
                            vhi: vec4[f32] = vec4[f32](unpack4xU8((pv >> u32(4)) & u32(0x0F0F0F0F)))
                            sumQA[rv] = sumQA[rv] + (dot(vec4[f32](vlo.x, vhi.x, vlo.y, vhi.y), vvc0)
                                                     + dot(vec4[f32](vlo.z, vhi.z, vlo.w, vhi.w), vvc1))
                rAv: f32 = sum_a[0]
                for rv2 in range(2):
                    rQAv: f32 = subgroupAdd(sumQA[rv2])
                    ov2: u32 = rowBaseV + u32(rv2)
                    if tid == u32(0):
                        if ov2 < u32(KV_OUT):
                            out_v[ov2] = srq(scales[u32(Q_OUT) + u32(KV_OUT) + ov2]
                                             * (rQAv - f32(8.0) * rAv),
                                             bitcast_f32(params[2]))


@kernel(workgroup_size=(128, 1, 1))
def logits_33(
    a: StorageBuffer[f32, "read"],
    bits_buf: StorageBuffer[vec4[u32], "read"],
    scale: StorageBuffer[f32, "read"],
    out: StorageBuffer[f32],
    params: Uniform[vec4[u32]],          # (inScale, outScale, _, _)
    at: WorkgroupArray[f32, 1536],
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """Dense 2-bit block-major logits GEMV, one thread per output column.

    The weights are block-major — 24 blocks of 64 values, each block a uint4 —
    so a thread reads its whole column with no reduction at all, which is why
    262144 outputs are affordable. The activation row is staged in workgroup
    memory once and reused by all 128 columns.

    block_dot is inlined because DSL helpers cannot take a threadgroup pointer.
    """
    inScale: f32 = bitcast_f32(params.x)
    col: u32 = (wg.y * u32(2048) + wg.x) * u32(128) + tid
    for i in range(tid, 1536, 128):
        at[i] = srq(a[i], inScale)
    barrier()
    if col < u32(262144):
        acc: f32 = f32(0.0)
        for blk in range(24):
            bv: vec4[u32] = bits_buf[u32(blk) * u32(262144) + col]
            aBase: u32 = u32(blk) * u32(64)
            s: f32 = f32(0.0)
            for j in range(4):
                packed: u32 = bv[j]
                d0: vec4[f32] = vec4[f32](unpack4xU8(packed & u32(0x03030303))) \
                    - vec4[f32](f32(2.0), f32(2.0), f32(2.0), f32(2.0))
                d1: vec4[f32] = vec4[f32](unpack4xU8((packed >> u32(2)) & u32(0x03030303))) \
                    - vec4[f32](f32(2.0), f32(2.0), f32(2.0), f32(2.0))
                d2: vec4[f32] = vec4[f32](unpack4xU8((packed >> u32(4)) & u32(0x03030303))) \
                    - vec4[f32](f32(2.0), f32(2.0), f32(2.0), f32(2.0))
                d3: vec4[f32] = vec4[f32](unpack4xU8((packed >> u32(6)) & u32(0x03030303))) \
                    - vec4[f32](f32(2.0), f32(2.0), f32(2.0), f32(2.0))
                b: u32 = aBase + u32(j) * u32(16)
                s = s + (dot(vec4[f32](d0.x, d1.x, d2.x, d3.x),
                             vec4[f32](at[b], at[b + u32(1)], at[b + u32(2)], at[b + u32(3)]))
                         + dot(vec4[f32](d0.y, d1.y, d2.y, d3.y),
                               vec4[f32](at[b + u32(4)], at[b + u32(5)], at[b + u32(6)], at[b + u32(7)]))
                         + dot(vec4[f32](d0.z, d1.z, d2.z, d3.z),
                               vec4[f32](at[b + u32(8)], at[b + u32(9)], at[b + u32(10)], at[b + u32(11)]))
                         + dot(vec4[f32](d0.w, d1.w, d2.w, d3.w),
                               vec4[f32](at[b + u32(12)], at[b + u32(13)], at[b + u32(14)], at[b + u32(15)])))
            acc = acc + s
        out[col] = srq(scale[col] * acc, bitcast_f32(params[1]))


@kernel(workgroup_size=(256, 1, 1),
        consts={"HEAD_DIM": 512, "HALF_DIM": 256, "HD4": 128, "J_GROUPS": 2,
                "PP_COUNTER_BASE": 131584, "OUT_Q": 0.014886821620166302})
def attn_101(
    q: StorageBuffer[f32, "read"],
    w: StorageBuffer[f32, "read"],
    cosTbl: StorageBuffer[f32, "read"],
    sinTbl: StorageBuffer[f32, "read"],
    k: StorageBuffer[vec4[f32], "read"],
    v: StorageBuffer[vec4[f32], "read"],
    partials: AtomicBuffer[u32],
    out: StorageBuffer[f32],
    # P101 is EIGHT words (seqQ, keyLen, qOffset, qHeads, kvHeads, window, _, _)
    # and the runner binds it as one buffer. Uniform[vec4[u32]] is only four, so
    # this is a read-only storage binding rather than two uniforms — same single
    # binding slot, which is what the host requires.
    params: StorageBuffer[u32, "read"],
    qn_sh: WorkgroupArray[f32, 512],
    out_acc: WorkgroupArray[f32, 512],
    probs: WorkgroupArray[f32, 256],
    sval_sh: WorkgroupArray[f32, 256],
    red: WorkgroupArray[f32, 256],
    wgt_sh: WorkgroupArray[f32, 32],
    vacc_sh: WorkgroupArray[vec4[f32], 256],
    st: WorkgroupArray[f32, 2],          # running_max, running_denom
    lastFlag: WorkgroupArray[u32, 1],
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    """Decode flash attention: q-norm + RoPE, online softmax, last-arriver merge.

    One query, so parallelism has to come from splitting the KEY axis: each of
    nActive chunks runs its own online softmax over a slice of the keys and
    publishes (partial output, running max, running denom) through the atomic
    buffer. The last chunk to finish rescales every partial by exp(m_c - m) and
    combines them — a full flash-attention merge inside a single dispatch.

    HEAD_DIM/HALF_DIM/HD4/J_GROUPS/PP_COUNTER_BASE/OUT_Q are consts; the sliding
    layers are 256-wide and the full ones 512.
    """
    h: u32 = wg.x
    ci: u32 = wg.y
    if h >= params[3]:
        return
    hKv: u32 = h / (params[3] / params[4])
    qPos: u32 = params[2]
    qBase: u32 = h * u32(HEAD_DIM)
    maxKj: u32 = min(params[1], qPos + u32(1))
    minKj: u32 = u32(0)
    if params[5] > u32(0):
        if qPos + u32(1) > params[5]:
            minKj = qPos + u32(1) - params[5]
    activeKeys: u32 = maxKj - minKj
    nActive: u32 = clamp((activeKeys + u32(63)) / u32(64), u32(8), u32(32))
    if ci >= nActive:
        return

    ss: f32 = f32(0.0)
    for d in range(tid, HEAD_DIM, 256):
        vv: f32 = q[qBase + d]
        ss = ss + vv * vv
    s0: f32 = subgroupAdd(ss)
    if (tid & u32(31)) == u32(0):
        red[tid >> u32(5)] = s0
    barrier()
    t0: f32 = f32(0.0)
    for i0 in range(8):
        t0 = t0 + red[i0]
    barrier()
    nscale: f32 = inverseSqrt(t0 / f32(HEAD_DIM) + f32(1e-6))

    for p in range(tid, HALF_DIM, 256):
        n0: f32 = q[qBase + p] * nscale * w[p]
        n1: f32 = q[qBase + p + u32(HALF_DIM)] * nscale * w[p + u32(HALF_DIM)]
        c: f32 = cosTbl[p]
        sn: f32 = sinTbl[p]
        qn_sh[p] = n0 * c - n1 * sn
        qn_sh[p + u32(HALF_DIM)] = n1 * c + n0 * sn
    for i1 in range(tid, HEAD_DIM, 256):
        out_acc[i1] = f32(0.0)
    if tid == u32(0):
        st[0] = f32(-3.4028234663852886e38)
        st[1] = f32(0.0)
    barrier()

    chunkLen: u32 = (activeKeys + nActive - u32(1)) / nActive
    start: u32 = minKj + ci * chunkLen
    end: u32 = min(start + chunkLen, maxKj)
    tile: u32 = start
    while tile < end:
        kj: u32 = tile + tid
        tileCountS: u32 = min(u32(256), end - tile)
        sgRounds: u32 = (tileCountS + u32(7)) / u32(8)
        for rr in range(sgRounds):
            j: u32 = u32(rr) * u32(8) + (tid / u32(32))
            accS: f32 = f32(0.0)
            if j < tileCountS:
                kBase4: u32 = ((tile + j) * params[4] + hKv) * u32(HD4)
                for d4 in range(tid & u32(31), HD4, 32):
                    kv4: vec4[f32] = k[kBase4 + d4]
                    accS = accS + dot(vec4[f32](qn_sh[d4 * u32(4)],
                                                qn_sh[d4 * u32(4) + u32(1)],
                                                qn_sh[d4 * u32(4) + u32(2)],
                                                qn_sh[d4 * u32(4) + u32(3)]), kv4)
            sj: f32 = subgroupAdd(accS)
            if (tid & u32(31)) == u32(0):
                if j < tileCountS:
                    sval_sh[j] = sj * f32(1.0)
        barrier()
        sval: f32 = f32(-3.4028234663852886e38)
        if kj < end:
            sval = sval_sh[tid]
        m0: f32 = subgroupMax(sval)
        if (tid & u32(31)) == u32(0):
            red[tid >> u32(5)] = m0
        barrier()
        tileMax: f32 = f32(-3.4028234663852886e38)
        for i2 in range(8):
            tileMax = max(tileMax, red[i2])
        barrier()
        newMax: f32 = max(st[0], tileMax)
        correction: f32 = exp(st[0] - newMax)
        pr: f32 = f32(0.0)
        if kj < end:
            pr = exp(sval - newMax)
        probs[tid] = pr
        d0: f32 = subgroupAdd(pr)
        if (tid & u32(31)) == u32(0):
            red[tid >> u32(5)] = d0
        barrier()
        tileDenom: f32 = f32(0.0)
        for i3 in range(8):
            tileDenom = tileDenom + red[i3]
        barrier()
        if tid == u32(0):
            st[1] = st[1] * correction + tileDenom
            st[0] = newMax
        barrier()
        tileCount: u32 = min(u32(256), end - tile)
        jg: u32 = tid / u32(HD4)
        d4v: u32 = tid % u32(HD4)
        vacc: vec4[f32] = vec4[f32](f32(0.0), f32(0.0), f32(0.0), f32(0.0))
        jj: u32 = jg
        while jj < tileCount:
            vBase4: u32 = ((tile + jj) * params[4] + hKv) * u32(HD4)
            vacc = vacc + probs[jj] * v[vBase4 + d4v]
            jj = jj + u32(J_GROUPS)
        vacc_sh[tid] = vacc
        barrier()
        for d4b in range(tid, HD4, 256):
            a4: vec4[f32] = vec4[f32](out_acc[d4b * u32(4)],
                                      out_acc[d4b * u32(4) + u32(1)],
                                      out_acc[d4b * u32(4) + u32(2)],
                                      out_acc[d4b * u32(4) + u32(3)]) * correction
            for g in range(J_GROUPS):
                a4 = a4 + vacc_sh[u32(g) * u32(HD4) + d4b]
            out_acc[d4b * u32(4)] = a4.x
            out_acc[d4b * u32(4) + u32(1)] = a4.y
            out_acc[d4b * u32(4) + u32(2)] = a4.z
            out_acc[d4b * u32(4) + u32(3)] = a4.w
        barrier()
        tile = tile + u32(256)

    pBase: u32 = (h * u32(32) + ci) * (u32(HEAD_DIM) + u32(2))
    for i4 in range(tid, HEAD_DIM, 256):
        atomicStore(partials, pBase + i4, bitcast_u32(out_acc[i4]))
    if tid == u32(0):
        atomicStore(partials, pBase + u32(HEAD_DIM), bitcast_u32(st[0]))
        atomicStore(partials, pBase + u32(HEAD_DIM) + u32(1), bitcast_u32(st[1]))
    storageBarrier()
    if tid == u32(0):
        tk: u32 = atomicAdd(partials, u32(PP_COUNTER_BASE) + h, u32(1))
        lastFlag[0] = u32(0)
        if tk == nActive - u32(1):
            lastFlag[0] = u32(1)
    barrier()
    if lastFlag[0] != u32(1):
        return
    if tid == u32(0):
        atomicStore(partials, u32(PP_COUNTER_BASE) + h, u32(0))

    mloc: f32 = f32(-3.4028234663852886e38)
    lloc: f32 = f32(0.0)
    if tid < nActive:
        pb: u32 = (h * u32(32) + tid) * (u32(HEAD_DIM) + u32(2))
        mloc = bitcast_f32(atomicLoad(partials, pb + u32(HEAD_DIM)))
        lloc = bitcast_f32(atomicLoad(partials, pb + u32(HEAD_DIM) + u32(1)))
    mm: f32 = subgroupMax(mloc)
    if (tid & u32(31)) == u32(0):
        red[tid >> u32(5)] = mm
    barrier()
    newM: f32 = f32(-3.4028234663852886e38)
    for i5 in range(8):
        newM = max(newM, red[i5])
    barrier()
    wloc: f32 = f32(0.0)
    if tid < nActive:
        wloc = exp(mloc - newM)
        wgt_sh[tid] = wloc
    dd: f32 = subgroupAdd(lloc * wloc)
    if (tid & u32(31)) == u32(0):
        red[tid >> u32(5)] = dd
    barrier()
    denom: f32 = f32(0.0)
    for i6 in range(8):
        denom = denom + red[i6]
    barrier()
    invd: f32 = f32(1.0) / denom
    for d5 in range(tid, HEAD_DIM, 256):
        acc: f32 = f32(0.0)
        for c2 in range(nActive):
            acc = acc + bitcast_f32(atomicLoad(
                partials, (h * u32(32) + u32(c2)) * (u32(HEAD_DIM) + u32(2)) + d5)) * wgt_sh[c2]
        out[h * u32(HEAD_DIM) + d5] = srq(acc * invd, f32(OUT_Q))
