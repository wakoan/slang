#include <metal_stdlib>
using namespace metal;
// Dequantize a 2-bit QAT weight matrix to a plain f16 [n_out][n_in] matrix, so a
// GEMM library (MPS) or a simdgroup_matrix kernel can consume it. Research-only:
// lives in prefill_research/, NOT kernels_msl/, because every backend compiles
// that whole directory at startup.
//
// The QAT math is  w = row_scale[o] * (q - 2)  with q in {0,1,2,3}: the GEMV
// kernels reach it as scale*(sum(q*a)*255/255 - ZP*sum(a)) after
// unpack_unorm4x8 divides by 255. Value i of a word sits at bit
// 8*(i/4) + 2*(i%4) — byte-major, then 2-bit lane — matching the float4 shuffle
// (c0.x,c1.x,c2.x,c3.x) that gateup_95/down_96 dot against activations 0..3.
//
// One thread per packed word (16 weights).
// dispatch with threadsPerThreadgroup = (256) ; groups = n_out*WPR/256
struct PDQ { uint n_in; uint n_out; uint p0; uint p1; };

kernel void dq2_f16(
    device const uint*  bits  [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device       half*  out   [[buffer(2)]],
    constant     PDQ&   p     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint WPR = p.n_in / 16u;
    uint total = p.n_out * WPR;
    if (gid >= total) return;
    uint o = gid / WPR, w = gid % WPR;
    uint packed = bits[gid];
    float s = scale[o];
    uint base = o * p.n_in + w * 16u;
    for (uint i = 0u; i < 16u; i++) {
        uint q = (packed >> (8u * (i / 4u) + 2u * (i % 4u))) & 3u;
        out[base + i] = half(s * (float(q) - 2.0f));
    }
}
