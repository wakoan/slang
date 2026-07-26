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
    kernel, u32, f32, f16, vec4, StorageBuffer, Uniform, WorkgroupArray, Builtin,
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
