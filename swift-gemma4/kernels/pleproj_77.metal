#include <metal_stdlib>
using namespace metal;
// from 77_sg_sum.wgsl — PLE projection (int8 unpack_unorm4x8) + norm-add*sv + next norm (atomic tail)
// dispatch with threadsPerThreadgroup = (256) ; groups = 96
struct P77 { float inScale; float projInScale; float projOutScale; uint p; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float4 srq4(float4 x, float s){ if(s==0.0f) return x; return clamp(round(x/s),float4(-128.0f),float4(127.0f))*s; }
inline float reduce_sum(float v, uint tid, threadgroup float* sgp){
    float s=simd_sum(v); if((tid&31u)==0u) sgp[tid>>5u]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    float t=0.0f; for(uint i=0u;i<256u/32u;i++) t+=sgp[i]; threadgroup_barrier(mem_flags::mem_threadgroup); return t; }
kernel void pleproj_77(
    device const float* a [[buffer(0)]], device const uint* codes [[buffer(1)]],
    device const float* row_scale [[buffer(2)]], device atomic_uint* pp [[buffer(3)]],
    device float* hidden [[buffer(4)]], device const float* w12s [[buffer(5)]],
    device float* y2 [[buffer(6)]], device float* sum2 [[buffer(7)]],
    constant P77& params [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint IN_F=256u, OUT_F=1536u, KV4=64u, K_ITER=2u, WG=256u, SG_ROWS=2u, ROWS_PER_WG=16u, TOTAL_WGS=96u;
    const float EPS=1e-6f;
    threadgroup float sgp[WG/32u]; threadgroup uint lastFlag;
    uint sgId=tid/32u, lane=tid&31u;
    uint rowBase=wg.x*ROWS_PER_WG + sgId*SG_ROWS;
    float4 av[K_ITER]; float aAcc=0.0f;
    for(uint ki=0u;ki<K_ITER;ki++){ uint k4=lane+ki*32u; av[ki]=float4(0.0f);
        if(k4<KV4){ uint kb=k4*4u; av[ki]=srq4(float4(a[kb],a[kb+1u],a[kb+2u],a[kb+3u]), params.projInScale); aAcc += (av[ki].x+av[ki].y)+(av[ki].z+av[ki].w); } }
    float accs[SG_ROWS];
    for(uint r=0u;r<SG_ROWS;r++){ uint o=rowBase+r; float acc=0.0f;
        if(o<OUT_F){ for(uint ki=0u;ki<K_ITER;ki++){ uint k4=lane+ki*32u; if(k4<KV4) acc += dot(unpack_unorm4x8_to_float(codes[o*KV4+k4]), av[ki]); } }
        accs[r]=acc; }
    float aSum=simd_sum(aAcc);
    for(uint r=0u;r<SG_ROWS;r++){ float s=simd_sum(accs[r]); uint o=rowBase+r;
        if(lane==0u && o<OUT_F) atomic_store_explicit(&pp[o], as_type<uint>(srq(row_scale[o]*fma(s,255.0f,-128.0f*aSum), params.projOutScale)), memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0u){ uint tk=atomic_fetch_add_explicit(&pp[OUT_F],1u,memory_order_relaxed); lastFlag=(tk==TOTAL_WGS-1u)?1u:0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lastFlag!=1u) return;
    if(tid==0u) atomic_store_explicit(&pp[OUT_F],0u,memory_order_relaxed);
    float inScale=params.inScale; float sv=w12s[2u*OUT_F];
    float acc1=0.0f;
    for(uint i=tid;i<OUT_F;i+=WG){ float v=as_type<float>(atomic_load_explicit(&pp[i],memory_order_relaxed)); acc1+=v*v; }
    float rms1=rsqrt(reduce_sum(acc1,tid,sgp)/float(OUT_F)+EPS);
    float hloc[6]; float acc2=0.0f;
    { uint e=0u; for(uint j=tid;j<OUT_F;j+=WG){ float normed=as_type<float>(atomic_load_explicit(&pp[j],memory_order_relaxed))*rms1*w12s[j]; float hv=(hidden[j]+normed)*sv; hidden[j]=hv; hloc[e]=hv; acc2+=hv*hv; e++; } }
    float rms2=rsqrt(reduce_sum(acc2,tid,sgp)/float(OUT_F)+EPS);
    float qAcc=0.0f;
    { uint e=0u; for(uint j=tid;j<OUT_F;j+=WG){ float n2=hloc[e]*rms2*w12s[OUT_F+j]; float qv=srq(n2,inScale); y2[j]=qv; qAcc+=qv; e++; } }
    float qSum=reduce_sum(qAcc,tid,sgp); if(tid==0u) sum2[0]=qSum;
}
