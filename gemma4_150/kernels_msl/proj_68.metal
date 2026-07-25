#include <metal_stdlib>
using namespace metal;
// from 68_reduce.wgsl — dense f16 GEMV (per_layer_model_projection)
// dispatch with threadsPerThreadgroup = (32) ; groups = OUT/N_ROWS = 1120
struct P68 { float inScale; float outScale; uint p0; uint p1; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float4 srq4(float4 x, float s){ if(s==0.0f) return x; return clamp(round(x/s), float4(-128.0f), float4(127.0f))*s; }
kernel void proj_68(
    device const float* a [[buffer(0)]], device const half* wt [[buffer(1)]],
    device float* out [[buffer(2)]], constant P68& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint IN=1536u, OUT=8960u, WG=32u, N_ROWS=8u, KV4=1536u/4u;
    uint rowBase = (wg.y*1120u + wg.x) * N_ROWS;
    if (rowBase >= OUT) return;
    float acc[N_ROWS];
    for (uint r=0u;r<N_ROWS;r++) acc[r]=0.0f;
    for (uint k4=tid; k4<KV4; k4+=WG) {
        uint kb=k4*4u;
        float4 a4 = srq4(float4(a[kb],a[kb+1u],a[kb+2u],a[kb+3u]), params.inScale);
        for (uint r=0u;r<N_ROWS;r++){ uint o=rowBase+r;
            if(o<OUT){ uint wb=o*IN+kb; float4 w4=float4(float(wt[wb]),float(wt[wb+1u]),float(wt[wb+2u]),float(wt[wb+3u])); acc[r]+=dot(w4,a4); } }
    }
    for (uint r=0u;r<N_ROWS;r++){ float s=simd_sum(acc[r]); uint o=rowBase+r; if(tid==0u&&o<OUT) out[o]=srq(s,params.outScale); }
}
