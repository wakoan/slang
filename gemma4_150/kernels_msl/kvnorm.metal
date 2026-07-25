#include <metal_stdlib>
using namespace metal;
// custom (runner.py _KVNORM) — k: RMSNorm*knorm + split-half RoPE; v: scale-free RMSNorm; -> caches
// HD/HALF patched per layer-type. dispatch with threadsPerThreadgroup = (256) ; groups = 1
kernel void kvnorm(
    device const float* ink [[buffer(0)]], device const float* inv [[buffer(1)]],
    device const float* knorm [[buffer(2)]], device const float* cosT [[buffer(3)]],
    device const float* sinT [[buffer(4)]], device float* kcache [[buffer(5)]],
    device float* vcache [[buffer(6)]], constant uint4& p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint HD=256u, HALF=128u; const float EPS=1e-6f;
    threadgroup float rk[HD]; threadgroup float rv[HD];
    float ko=ink[tid], vo=inv[tid];
    rk[tid]=ko*ko; rv[tid]=vo*vo; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=HD/2u;s>0u;s/=2u){ if(tid<s){rk[tid]+=rk[tid+s]; rv[tid]+=rv[tid+s];} threadgroup_barrier(mem_flags::mem_threadgroup); }
    float rmsk=rsqrt(rk[0]/float(HD)+EPS), rmsv=rsqrt(rv[0]/float(HD)+EPS);
    vcache[p.x+tid]=vo*rmsv;
    if(tid<HALF){ float n0=ink[tid]*rmsk*knorm[tid], n1=ink[tid+HALF]*rmsk*knorm[tid+HALF];
        float c=cosT[tid], sn=sinT[tid]; kcache[p.x+tid]=n0*c-n1*sn; kcache[p.x+tid+HALF]=n1*c+n0*sn; }
}
