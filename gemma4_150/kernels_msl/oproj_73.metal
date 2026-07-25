#include <metal_stdlib>
using namespace metal;
// from 73_sg_sum.wgsl — o-proj (4-bit) + post-attn norm-add + pre-FFN norm (atomic tail)
// shape consts (IN_FEATURES, WORDS_PER_ROW) patched per layer.
// dispatch with threadsPerThreadgroup = (256) ; groups = 192
struct P73 { float outScale; float inScale2; uint p0; uint p1; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float reduce_sum(float v, uint tid, threadgroup float* sgp){
    float s=simd_sum(v); if((tid&31u)==0u) sgp[tid>>5u]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    float t=0.0f; for(uint i=0u;i<256u/32u;i++) t+=sgp[i]; threadgroup_barrier(mem_flags::mem_threadgroup); return t; }
kernel void oproj_73(
    device const float4* a [[buffer(0)]], device const uint* bits_buf [[buffer(1)]],
    device const float* scale [[buffer(2)]], device atomic_uint* pp [[buffer(3)]],
    device float* hidden [[buffer(4)]], device const float* w12 [[buffer(5)]],
    device half* y2 [[buffer(6)]], device float* sum2 [[buffer(7)]],
    constant P73& params [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint IN_FEATURES=2048u, OUT_F=1536u, WPR=256u, CHUNKS=2u, WG=256u, ROWS_PER_WG=8u, TOTAL_WGS=192u;
    const float ZP=8.0f, EPS=1e-6f;
    threadgroup float sgp[WG/32u]; threadgroup uint lastFlag;
    uint sgId=tid/32u, lane=tid&31u;
    float outScale=params.outScale;
    uint o=wg.x*ROWS_PER_WG + sgId;    // SG_ROWS=1
    float sumQA=0.0f, sumA=0.0f;
    for(uint w=lane; w<WPR; w+=32u){
        float4 avc0=a[w*CHUNKS+0u], avc1=a[w*CHUNKS+1u];
        sumA += (avc0.x+avc0.y+avc0.z+avc0.w) + (avc1.x+avc1.y+avc1.z+avc1.w);
        if(o<OUT_F){ uint p=bits_buf[o*WPR+w];
            float4 lo=float4(as_type<uchar4>(p&0x0F0F0F0Fu)), hi=float4(as_type<uchar4>((p>>4u)&0x0F0F0F0Fu));
            sumQA += dot(float4(lo.x,hi.x,lo.y,hi.y),avc0)+dot(float4(lo.z,hi.z,lo.w,hi.w),avc1); }
    }
    float rA=simd_sum(sumA), rQA=simd_sum(sumQA);
    if(lane==0u && o<OUT_F) atomic_store_explicit(&pp[o], as_type<uint>(srq(scale[o]*(rQA-ZP*rA),outScale)), memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0u){ uint tk=atomic_fetch_add_explicit(&pp[OUT_F],1u,memory_order_relaxed); lastFlag=(tk==TOTAL_WGS-1u)?1u:0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lastFlag!=1u) return;
    if(tid==0u) atomic_store_explicit(&pp[OUT_F],0u,memory_order_relaxed);
    float inScale2=params.inScale2;
    float acc1=0.0f;
    for(uint i=tid;i<OUT_F;i+=WG){ float v=as_type<float>(atomic_load_explicit(&pp[i],memory_order_relaxed)); acc1+=v*v; }
    float rms1=rsqrt(reduce_sum(acc1,tid,sgp)/float(OUT_F)+EPS);
    float hloc[6]; float acc2=0.0f;
    { uint e=0u; for(uint j=tid;j<OUT_F;j+=WG){ float normed=as_type<float>(atomic_load_explicit(&pp[j],memory_order_relaxed))*rms1*w12[j]; float hv=hidden[j]+normed; hidden[j]=hv; hloc[e]=hv; acc2+=hv*hv; e++; } }
    float rms2=rsqrt(reduce_sum(acc2,tid,sgp)/float(OUT_F)+EPS);
    float qAcc=0.0f;
    { uint e=0u; for(uint j=tid;j<OUT_F;j+=WG){ float n2=hloc[e]*rms2*w12[OUT_F+j]; half qv=half(srq(float(half(n2)),inScale2)); y2[j]=qv; qAcc+=float(qv); e++; } }
    float qSum=reduce_sum(qAcc,tid,sgp); if(tid==0u) sum2[0]=qSum;
}
