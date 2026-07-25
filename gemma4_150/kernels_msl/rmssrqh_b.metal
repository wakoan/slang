#include <metal_stdlib>
using namespace metal;
// PREFILL: pre-FFN RMSNorm + SRQ over S rows, emitting HALF (gateup_74/95 read
// half4). rmssrq_69 already handles S rows but writes f32, which is right for
// the pre-attention norm and wrong here. Reproduces oproj_73's double rounding
// exactly: half(srq(float(half(n2)), inScale)).
// dispatch with threadsPerThreadgroup = (256) ; groups = S
struct PRH { float inScale; uint dim; uint p0; uint p1; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float reduce_sum(float v, uint tid, threadgroup float* sgp){
    float s=simd_sum(v); if((tid&31u)==0u) sgp[tid>>5u]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    float t=0.0f; for(uint i=0u;i<256u/32u;i++) t+=sgp[i]; threadgroup_barrier(mem_flags::mem_threadgroup); return t; }
kernel void rmssrqh_b(
    device const float* hidden [[buffer(0)]], device const float* w [[buffer(1)]],
    device half* y [[buffer(2)]], device float* sum_a [[buffer(3)]],
    constant PRH& p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint DIM=1536u, WG=256u; const float EPS=1e-6f;
    threadgroup float sgp[WG/32u];
    uint base=wg.x*DIM;
    float acc=0.0f;
    for(uint j=tid;j<DIM;j+=WG){ float v=hidden[base+j]; acc+=v*v; }
    float rms=rsqrt(reduce_sum(acc,tid,sgp)/float(DIM)+EPS);
    float qAcc=0.0f;
    for(uint j=tid;j<DIM;j+=WG){
        float n2=hidden[base+j]*rms*w[j];
        half qv=half(srq(float(half(n2)), p.inScale));
        y[base+j]=qv; qAcc+=float(qv); }
    float qs=reduce_sum(qAcc,tid,sgp);
    if(tid==0u) sum_a[wg.x]=qs;
}
