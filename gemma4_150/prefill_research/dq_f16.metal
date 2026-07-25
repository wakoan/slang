#include <metal_stdlib>
using namespace metal;
// Dequantize a QAT weight matrix to a plain f16 [n_out][n_in] matrix, so a GEMM
// library (MPS) or a simdgroup_matrix kernel can consume it. Research-only:
// lives in prefill_research/, NOT kernels_msl/, because every backend compiles
// that whole directory at startup.
//
// One kernel covers all three widths; patch BITS to 2, 4 or 8.
//
//     w = row_scale[o] * (q - ZP),   ZP = 1 << (BITS-1)   -> 2 / 8 / 128
//
// The GEMV kernels reach the same value by different routes — 2-bit and 8-bit
// via unpack_unorm4x8 (which divides by 255, undone by a later fma(...,255,...)),
// 4-bit via as_type<uchar4> on raw nibbles — but the packing is one scheme:
// values are byte-major, then low-bits-first within the byte, so value i sits at
//
//     bit  8*(i / (8/BITS))  +  BITS*(i % (8/BITS))
//
// which reduces to 8*(i/4)+2*(i%4), 8*(i/2)+4*(i%2), and 8*i respectively. That
// ordering is what makes the float4 shuffle (c0.x,c1.x,c2.x,c3.x) in
// gateup_95/down_96 line up with activations 0..3; get it wrong and the dot
// products silently pair the wrong operands.
//
// One thread per packed word.
// dispatch with threadsPerThreadgroup = (256) ; groups = ceil(n_out*WPR/256)
struct PDQ { uint n_in; uint n_out; uint p0; uint p1; };

kernel void dq_f16(
    device const uint*  bits  [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device       half*  out   [[buffer(2)]],
    constant     PDQ&   p     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint BITS = 2u;
    const uint VPW = 32u / BITS;              // values per packed word
    const uint PER_BYTE = 8u / BITS;
    const float ZP = float(1u << (BITS - 1u));
    const uint MASK = (1u << BITS) - 1u;

    uint WPR = p.n_in / VPW;
    if (gid >= p.n_out * WPR) return;
    uint o = gid / WPR, w = gid % WPR;
    uint packed = bits[gid];
    float s = scale[o];
    uint base = o * p.n_in + w * VPW;
    for (uint i = 0u; i < VPW; i++) {
        uint q = (packed >> (8u * (i / PER_BYTE) + BITS * (i % PER_BYTE))) & MASK;
        out[base + i] = half(s * (float(q) - ZP));
    }
}
