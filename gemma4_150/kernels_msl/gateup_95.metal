#include <metal_stdlib>
using namespace metal;
// from 95_sg_sum.wgsl — fused gate/up geglu (2-bit, double-wide INTER=12288)
// dispatch with threadsPerThreadgroup = (64) ; groups = GRID_X = 3072
struct P95 { float gateOutScale; float upOutScale; float outQuantScale; uint p; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float tanh_safe(float x){ if(x>10.0f) return 1.0f; if(x<-10.0f) return -1.0f; return tanh(x); }
inline float gelu_tanh(float v){ return 0.5f*v*(1.0f+tanh_safe(0.7978845608028654f*(v+0.044715f*v*v*v))); }
inline float gelu_grid(float g, float s, device const float* lut){ if(s==0.0f) return gelu_tanh(g); return lut[uint(clamp(round(g/s),-128.0f,127.0f)+128.0f)]; }
kernel void gateup_95(
    device const half4* hidden [[buffer(0)]], device const uint* gate_bits [[buffer(1)]],
    device const float* gate_scale [[buffer(2)]], device const uint* up_bits [[buffer(3)]],
    device const float* up_scale [[buffer(4)]], device const float* sum_a [[buffer(5)]],
    device half* out [[buffer(6)]], device const float* gelu_lut [[buffer(7)]],
    constant P95& params [[buffer(8)]],
    uint lidx [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint INTER=12288u, WPR=96u, CHUNKS=4u, SG_COUNT=2u, N_ROWS=2u, GRID_X=3072u;
    const float ZP=2.0f;
    uint sgId=lidx/32u, tid=lidx&31u;
    uint rowBase=((wg.y*GRID_X+wg.x)*SG_COUNT+sgId)*N_ROWS;
    float gOut=params.gateOutScale, uOut=params.upOutScale;
    float gAcc[N_ROWS], uAcc[N_ROWS];
    for(uint r=0u;r<N_ROWS;r++){gAcc[r]=0.0f;uAcc[r]=0.0f;}
    for(uint wd=tid; wd<WPR; wd+=32u){
        float4 a0=float4(hidden[wd*CHUNKS+0u]), a1=float4(hidden[wd*CHUNKS+1u]), a2=float4(hidden[wd*CHUNKS+2u]), a3=float4(hidden[wd*CHUNKS+3u]);
        for(uint r=0u;r<N_ROWS;r++){ uint o=rowBase+r; if(o<INTER){
            uint pg=gate_bits[o*WPR+wd], pu=up_bits[o*WPR+wd];
            float4 g0=unpack_unorm4x8_to_float(pg&0x03030303u), g1=unpack_unorm4x8_to_float((pg>>2u)&0x03030303u), g2=unpack_unorm4x8_to_float((pg>>4u)&0x03030303u), g3=unpack_unorm4x8_to_float((pg>>6u)&0x03030303u);
            gAcc[r]+=dot(float4(g0.x,g1.x,g2.x,g3.x),a0)+dot(float4(g0.y,g1.y,g2.y,g3.y),a1)+dot(float4(g0.z,g1.z,g2.z,g3.z),a2)+dot(float4(g0.w,g1.w,g2.w,g3.w),a3);
            float4 u0=unpack_unorm4x8_to_float(pu&0x03030303u), u1=unpack_unorm4x8_to_float((pu>>2u)&0x03030303u), u2=unpack_unorm4x8_to_float((pu>>4u)&0x03030303u), u3=unpack_unorm4x8_to_float((pu>>6u)&0x03030303u);
            uAcc[r]+=dot(float4(u0.x,u1.x,u2.x,u3.x),a0)+dot(float4(u0.y,u1.y,u2.y,u3.y),a1)+dot(float4(u0.z,u1.z,u2.z,u3.z),a2)+dot(float4(u0.w,u1.w,u2.w,u3.w),a3);
        } }
    }
    float aSum=sum_a[0];
    for(uint r=0u;r<N_ROWS;r++){ float gS=simd_sum(gAcc[r]), uS=simd_sum(uAcc[r]);
        if(tid==0u){ uint o=rowBase+r; if(o<INTER){
            float g=srq(gate_scale[o]*fma(gS,255.0f,-(ZP*aSum)),gOut);
            float u=srq(up_scale[o]*fma(uS,255.0f,-(ZP*aSum)),uOut);
            float dq=gelu_grid(g,gOut,gelu_lut)*u; float qs=params.outQuantScale;
            out[o]=half((qs==0.0f)?dq:clamp(round(dq/qs),-128.0f,127.0f)); } }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
