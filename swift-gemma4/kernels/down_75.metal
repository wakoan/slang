#include <metal_stdlib>
using namespace metal;
// from 75_srq.wgsl — fused down (4-bit) + post-FFN residual norm-add (atomic last-arriver)
// dispatch with threadsPerThreadgroup = (256) ; groups = OUT_F/N_ROWS = 384
struct P75 { float inScale; float outScale; uint p0; uint p1; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float reduce_sum(float v, uint tid, threadgroup float* sgs){
    float s=simd_sum(v); if((tid&31u)==0u) sgs[tid>>5u]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    float t=0.0f; for(uint i=0u;i<256u/32u;i++) t+=sgs[i]; threadgroup_barrier(mem_flags::mem_threadgroup); return t; }
kernel void down_75(
    device const half4* a [[buffer(0)]], device const uint* bits_buf [[buffer(1)]],
    device atomic_uint* pp [[buffer(2)]], device const float* scale [[buffer(3)]],
    device float* hidden [[buffer(4)]], device const float* nw [[buffer(5)]],
    constant P75& params [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint OUT_F=1536u, CHUNKS=2u, WPR=768u, WG=256u, N_ROWS=4u, TOTAL_WGS=384u, CTR=1536u;
    const float ZP=8.0f, EPS=1e-6f;
    threadgroup float dsh[OUT_F];
    threadgroup float4 sgq[WG/32u];
    threadgroup float sgs[WG/32u];
    threadgroup uint lastFlag;
    uint rowBase=wg.x*N_ROWS; float inScale=params.inScale;
    float q[4]={0,0,0,0}; float sumA=0.0f;
    for(uint w=tid; w<WPR; w+=WG){
        float4 av0=float4(a[w*CHUNKS+0u]), av1=float4(a[w*CHUNKS+1u]);
        sumA += (av0.x+av0.y+av0.z+av0.w)+(av1.x+av1.y+av1.z+av1.w);
        for(uint r=0u;r<N_ROWS;r++){ uint o=rowBase+r; if(o<OUT_F){
            uint p=bits_buf[o*WPR+w]; float4 lo=unpack_unorm4x8_to_float(p&0x0F0F0F0Fu), hi=unpack_unorm4x8_to_float((p>>4u)&0x0F0F0F0Fu);
            q[r]+=dot(float4(lo.x,hi.x,lo.y,hi.y),av0)+dot(float4(lo.z,hi.z,lo.w,hi.w),av1); } }
    }
    float4 red=simd_sum(float4(q[0],q[1],q[2],q[3])); float redA=simd_sum(sumA);
    if((tid&31u)==0u){ sgq[tid>>5u]=red; sgs[tid>>5u]=redA; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0u){
        float4 tot=float4(0.0f); float aSum=0.0f;
        for(uint i=0u;i<WG/32u;i++){ tot+=sgq[i]; aSum+=sgs[i]; }
        float outScale=params.outScale, zpA=ZP*aSum;
        for(uint r=0u;r<N_ROWS;r++){ uint o=rowBase+r; if(o<OUT_F){
            float d=srq(scale[o]*(inScale*fma(tot[r],255.0f,-zpA)),outScale);
            atomic_store_explicit(&pp[o], as_type<uint>(d), memory_order_relaxed); } }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0u){ uint ticket=atomic_fetch_add_explicit(&pp[CTR],1u,memory_order_relaxed); lastFlag=(ticket==TOTAL_WGS-1u)?1u:0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lastFlag!=1u) return;
    if(tid==0u) atomic_store_explicit(&pp[CTR],0u,memory_order_relaxed);
    float acc=0.0f;
    for(uint o2=tid; o2<OUT_F; o2+=WG){ float d=as_type<float>(atomic_load_explicit(&pp[o2],memory_order_relaxed)); dsh[o2]=d; acc+=d*d; }
    float rms=rsqrt(reduce_sum(acc,tid,sgs)/float(OUT_F)+EPS);
    for(uint o2=tid; o2<OUT_F; o2+=WG){ hidden[o2]=hidden[o2]+dsh[o2]*rms*nw[o2]; }
}
