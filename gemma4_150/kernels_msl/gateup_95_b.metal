#include <metal_stdlib>
using namespace metal;
// BATCHED (prefill) variant of gateup_95: same math, but one weight read is
// reused across S tokens. Each simdgroup owns ONE output row and holds S
// accumulators; the weight word `pg`/`pu` is fetched once per (row, word) and
// dotted against all S activation vectors — that S-fold cut in weight traffic
// is the whole point (decode is bandwidth-bound on weight reads).
// hidden/out are [S][*] row-major; sum_a is [S].
// dispatch with threadsPerThreadgroup = (64) ; groups = INTER/SG_COUNT
struct P95B { float gateOutScale; float upOutScale; float outQuantScale; uint p; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float tanh_safe(float x){ if(x>10.0f) return 1.0f; if(x<-10.0f) return -1.0f; return tanh(x); }
inline float gelu_tanh(float v){ return 0.5f*v*(1.0f+tanh_safe(0.7978845608028654f*(v+0.044715f*v*v*v))); }
inline float gelu_grid(float g, float s, device const float* lut){ if(s==0.0f) return gelu_tanh(g); return lut[uint(clamp(round(g/s),-128.0f,127.0f)+128.0f)]; }
kernel void gateup_95_b(
    device const half4* hidden [[buffer(0)]], device const uint* gate_bits [[buffer(1)]],
    device const float* gate_scale [[buffer(2)]], device const uint* up_bits [[buffer(3)]],
    device const float* up_scale [[buffer(4)]], device const float* sum_a [[buffer(5)]],
    device half* out [[buffer(6)]], device const float* gelu_lut [[buffer(7)]],
    constant P95B& params [[buffer(8)]],
    uint lidx [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint INTER=12288u, WPR=96u, CHUNKS=4u, SG_COUNT=2u, SEQ=8u;
    const uint HSTRIDE=384u;              // half4 per token row (1536 halves)
    const float ZP=2.0f;
    uint sgId=lidx/32u, tid=lidx&31u;
    uint o=wg.x*SG_COUNT+sgId;            // one output row per simdgroup
    if(o>=INTER) return;
    float gOut=params.gateOutScale, uOut=params.upOutScale;
    float gAcc[SEQ], uAcc[SEQ];
    for(uint s=0u;s<SEQ;s++){gAcc[s]=0.0f;uAcc[s]=0.0f;}

    for(uint wd=tid; wd<WPR; wd+=32u){
        uint pg=gate_bits[o*WPR+wd], pu=up_bits[o*WPR+wd];   // ONE weight read...
        float4 g0=unpack_unorm4x8_to_float(pg&0x03030303u), g1=unpack_unorm4x8_to_float((pg>>2u)&0x03030303u);
        float4 g2=unpack_unorm4x8_to_float((pg>>4u)&0x03030303u), g3=unpack_unorm4x8_to_float((pg>>6u)&0x03030303u);
        float4 u0=unpack_unorm4x8_to_float(pu&0x03030303u), u1=unpack_unorm4x8_to_float((pu>>2u)&0x03030303u);
        float4 u2=unpack_unorm4x8_to_float((pu>>4u)&0x03030303u), u3=unpack_unorm4x8_to_float((pu>>6u)&0x03030303u);
        float4 gx=float4(g0.x,g1.x,g2.x,g3.x), gy=float4(g0.y,g1.y,g2.y,g3.y);
        float4 gz=float4(g0.z,g1.z,g2.z,g3.z), gw=float4(g0.w,g1.w,g2.w,g3.w);
        float4 ux=float4(u0.x,u1.x,u2.x,u3.x), uy=float4(u0.y,u1.y,u2.y,u3.y);
        float4 uz=float4(u0.z,u1.z,u2.z,u3.z), uw=float4(u0.w,u1.w,u2.w,u3.w);
        for(uint s=0u;s<SEQ;s++){                            // ...reused for S tokens
            uint hb=s*HSTRIDE+wd*CHUNKS;
            float4 a0=float4(hidden[hb+0u]), a1=float4(hidden[hb+1u]);
            float4 a2=float4(hidden[hb+2u]), a3=float4(hidden[hb+3u]);
            gAcc[s]+=dot(gx,a0)+dot(gy,a1)+dot(gz,a2)+dot(gw,a3);
            uAcc[s]+=dot(ux,a0)+dot(uy,a1)+dot(uz,a2)+dot(uw,a3);
        }
    }
    for(uint s=0u;s<SEQ;s++){
        float gS=simd_sum(gAcc[s]), uS=simd_sum(uAcc[s]);
        if(tid==0u){
            float aSum=sum_a[s];
            float g=srq(gate_scale[o]*fma(gS,255.0f,-(ZP*aSum)),gOut);
            float u=srq(up_scale[o]*fma(uS,255.0f,-(ZP*aSum)),uOut);
            float dq=gelu_grid(g,gOut,gelu_lut)*u; float qs=params.outQuantScale;
            out[s*INTER+o]=half((qs==0.0f)?dq:clamp(round(dq/qs),-128.0f,127.0f));
        }
    }
}
