#include <metal_stdlib>
using namespace metal;
// from 74_sg_sum.wgsl — fused gate/up geglu (4-bit), virtual-subgroup GEMV
// dispatch with threadsPerThreadgroup = (64) ; groups = GRID_X = 768
struct P74 { float gateOutScale; float upOutScale; float outQuantScale; uint p; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float tanh_safe(float x){ if(x>10.0f) return 1.0f; if(x<-10.0f) return -1.0f; return tanh(x); }
inline float gelu_tanh(float v){ return 0.5f*v*(1.0f+tanh_safe(0.7978845608028654f*(v+0.044715f*v*v*v))); }
inline float gelu_grid(float g, float s, device const float* lut){ if(s==0.0f) return gelu_tanh(g); return lut[uint(clamp(round(g/s),-128.0f,127.0f)+128.0f)]; }
kernel void gateup_74(
    device const half4* hidden [[buffer(0)]], device const uint* gate_bits [[buffer(1)]],
    device const float* gate_scale [[buffer(2)]], device const uint* up_bits [[buffer(3)]],
    device const float* up_scale [[buffer(4)]], device const float* sum_a [[buffer(5)]],
    device half* out [[buffer(6)]], device const float* gelu_lut [[buffer(7)]],
    constant P74& params [[buffer(8)]],
    uint lidx [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint INTER=6144u, WPR=192u, CHUNKS=2u, SG_COUNT=2u, N_ROWS=4u, GRID_X=768u;
    const float ZP=8.0f;
    uint sgId=lidx/32u, tid=lidx&31u;
    uint rowBase=((wg.y*GRID_X+wg.x)*SG_COUNT+sgId)*N_ROWS;
    float gOut=params.gateOutScale, uOut=params.upOutScale;
    float gAcc[N_ROWS], uAcc[N_ROWS];
    for(uint r=0u;r<N_ROWS;r++){gAcc[r]=0.0f;uAcc[r]=0.0f;}
    for(uint wd=tid; wd<WPR; wd+=32u){
        float4 a0=float4(hidden[wd*CHUNKS+0u]), a1=float4(hidden[wd*CHUNKS+1u]);
        for(uint r=0u;r<N_ROWS;r++){ uint o=rowBase+r;
            if(o<INTER){
                uint pg=gate_bits[o*WPR+wd], pu=up_bits[o*WPR+wd];
                float4 glo=unpack_unorm4x8_to_float(pg&0x0F0F0F0Fu), ghi=unpack_unorm4x8_to_float((pg>>4u)&0x0F0F0F0Fu);
                gAcc[r]+=dot(float4(glo.x,ghi.x,glo.y,ghi.y),a0)+dot(float4(glo.z,ghi.z,glo.w,ghi.w),a1);
                float4 ulo=unpack_unorm4x8_to_float(pu&0x0F0F0F0Fu), uhi=unpack_unorm4x8_to_float((pu>>4u)&0x0F0F0F0Fu);
                uAcc[r]+=dot(float4(ulo.x,uhi.x,ulo.y,uhi.y),a0)+dot(float4(ulo.z,uhi.z,ulo.w,uhi.w),a1);
            }
        }
    }
    float aSum=sum_a[0];
    for(uint r=0u;r<N_ROWS;r++){
        float gS=simd_sum(gAcc[r]), uS=simd_sum(uAcc[r]);
        if(tid==0u){ uint o=rowBase+r;
            if(o<INTER){
                float g=srq(gate_scale[o]*fma(gS,255.0f,-(ZP*aSum)),gOut);
                float u=srq(up_scale[o]*fma(uS,255.0f,-(ZP*aSum)),uOut);
                float dq=gelu_grid(g,gOut,gelu_lut)*u;
                float qs=params.outQuantScale;
                out[o]=half((qs==0.0f)?dq:clamp(round(dq/qs),-128.0f,127.0f));
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
