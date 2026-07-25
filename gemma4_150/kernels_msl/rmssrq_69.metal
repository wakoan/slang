#include <metal_stdlib>
using namespace metal;
// translated from reference/webml_gemma4_kernels/69_sg_sum.wgsl
// fused weighted RMSNorm + SRQ + sum-of-quantized-activations
// dispatch with threadsPerThreadgroup = (256) ; one threadgroup per row
struct P69 { uint rows; uint rowStride; float inScale; uint p; };

inline float srq(float x, float s) {
    if (s == 0.0f) return x;
    return clamp(round(x / s), -128.0f, 127.0f) * s;
}
// subgroupAdd (32-lane) + cross-subgroup combine via threadgroup memory
inline float reduce_sum(float v, uint tid, threadgroup float* sgp) {
    float s = simd_sum(v);
    if ((tid & 31u) == 0u) sgp[tid >> 5u] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0u; i < 256u / 32u; i++) total += sgp[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

kernel void rmssrq_69(
    device const float* x      [[buffer(0)]],
    device const float* w      [[buffer(1)]],
    device       float* y      [[buffer(2)]],
    device       float* sum_a  [[buffer(3)]],
    constant     P69&   params [[buffer(4)]],
    uint  tid [[thread_index_in_threadgroup]],
    uint3 wg  [[threadgroup_position_in_grid]])
{
    const uint DIM = 1536u, WG = 256u;
    const float EPS = 1e-6f;
    threadgroup float sgp[WG / 32u];

    uint rowStride = (params.rowStride == 0u) ? params.rows : params.rowStride;
    uint row = wg.x + wg.y * rowStride;
    if (row >= params.rows) return;
    uint base = row * DIM;
    float inScale = params.inScale;

    float acc = 0.0f;
    for (uint i = tid; i < DIM; i += WG) { float v = x[base + i]; acc += v * v; }
    float sc = rsqrt(reduce_sum(acc, tid, sgp) / float(DIM) + EPS);

    float qAcc = 0.0f;
    for (uint j = tid; j < DIM; j += WG) {
        float q = srq(x[base + j] * sc * w[j], inScale);
        y[base + j] = q;
        qAcc += q;
    }
    float qSum = reduce_sum(qAcc, tid, sgp);
    if (tid == 0u) sum_a[row] = qSum;
}
