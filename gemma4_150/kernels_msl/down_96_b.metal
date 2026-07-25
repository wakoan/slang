#include <metal_stdlib>
using namespace metal;
// BATCHED (prefill) down-projection, 2-bit, INTER=12288 -> H=1536.
// Pairs with gateup_95_b to make the whole MLP block batched.
//
// Same math as down_96 but DELIBERATELY UNFUSED: down_96 fuses the residual add
// + post-FFN RMSNorm behind a global atomic last-arriver counter, which would
// need S counters/accumulators to batch. That fusion is a decode-latency
// optimisation; prefill is compute-bound with S tokens per dispatch, so we emit
// plain y[S][H] and let a separate batched norm consume it.
//
// One simdgroup owns one output row and holds S accumulators: each packed weight
// word is read ONCE and dotted against all S token vectors. sum_a is accumulated
// in-kernel (free — those activations are already being loaded for the dots).
// a is f16 [S][INTER]; y is f32 [S][H].
// dispatch with threadsPerThreadgroup = (64) ; groups = H/SG_COUNT
struct P96B { float inScale; float outScale; uint p0; uint p1; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }

kernel void down_96_b(
    device const half4* a      [[buffer(0)]],
    device const uint*  bits   [[buffer(1)]],
    device const float* scale  [[buffer(2)]],
    device       float* y      [[buffer(3)]],
    constant     P96B&  params [[buffer(4)]],
    uint lidx [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint OUT_F=1536u, INTER=12288u, CHUNKS=4u, WPR=768u, SG_COUNT=2u, SEQ=8u;
    const uint ASTRIDE = INTER/4u;            // half4 per token row
    const float ZP=2.0f;

    uint sgId=lidx/32u, tid=lidx&31u;
    uint o = wg.x*SG_COUNT + sgId;            // one output row per simdgroup
    if(o>=OUT_F) return;

    float acc[SEQ], sumA[SEQ];
    for(uint s=0u;s<SEQ;s++){ acc[s]=0.0f; sumA[s]=0.0f; }

    for(uint w=tid; w<WPR; w+=32u){
        uint p = bits[o*WPR+w];                // ONE weight read, reused for S tokens
        float4 c0=unpack_unorm4x8_to_float(p&0x03030303u), c1=unpack_unorm4x8_to_float((p>>2u)&0x03030303u);
        float4 c2=unpack_unorm4x8_to_float((p>>4u)&0x03030303u), c3=unpack_unorm4x8_to_float((p>>6u)&0x03030303u);
        float4 w0=float4(c0.x,c1.x,c2.x,c3.x), w1=float4(c0.y,c1.y,c2.y,c3.y);
        float4 w2=float4(c0.z,c1.z,c2.z,c3.z), w3=float4(c0.w,c1.w,c2.w,c3.w);
        for(uint s=0u;s<SEQ;s++){
            uint ab = s*ASTRIDE + w*CHUNKS;
            float4 av0=float4(a[ab+0u]), av1=float4(a[ab+1u]), av2=float4(a[ab+2u]), av3=float4(a[ab+3u]);
            acc[s] += dot(w0,av0)+dot(w1,av1)+dot(w2,av2)+dot(w3,av3);
            sumA[s] += (av0.x+av0.y+av0.z+av0.w)+(av1.x+av1.y+av1.z+av1.w)
                     + (av2.x+av2.y+av2.z+av2.w)+(av3.x+av3.y+av3.z+av3.w);
        }
    }
    float inScale=params.inScale, outScale=params.outScale;
    for(uint s=0u;s<SEQ;s++){
        float tot=simd_sum(acc[s]), aSum=simd_sum(sumA[s]);
        if(tid==0u)
            y[s*OUT_F+o] = srq(scale[o]*(inScale*fma(tot,255.0f,-(ZP*aSum))), outScale);
    }
}
