#include <metal_stdlib>
using namespace metal;
// PREFILL: kvnorm for S positions at once. Same math as kvnorm (k: RMSNorm*knorm
// + split-half RoPE, v: scale-free RMSNorm) but one workgroup per token, with the
// SRQ that qkv_70 applied to its k/v outputs folded in, and the rope tables
// indexed by absolute position instead of pre-offset.
// HD/HALF patched per layer-type. dispatch with threadsPerThreadgroup = (HD) ; groups = S
struct PKV { uint startPos; float kScale; float vScale; uint p1; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
kernel void kvnorm_b(
    device const half* ink [[buffer(0)]], device const half* inv [[buffer(1)]],
    device const float* knorm [[buffer(2)]], device const float* cosT [[buffer(3)]],
    device const float* sinT [[buffer(4)]], device float* kcache [[buffer(5)]],
    device float* vcache [[buffer(6)]], constant PKV& p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint HD=256u, HALF=128u; const float EPS=1e-6f;
    threadgroup float rk[HD], rv[HD], ksh[HD];
    uint s=wg.x, ib=s*HD;
    uint pos=p.startPos+s, co=pos*HALF, cache=pos*HD;
    float ko=srq(float(ink[ib+tid]), p.kScale), vo=srq(float(inv[ib+tid]), p.vScale);
    ksh[tid]=ko;
    rk[tid]=ko*ko; rv[tid]=vo*vo; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=HD/2u;st>0u;st/=2u){ if(tid<st){rk[tid]+=rk[tid+st]; rv[tid]+=rv[tid+st];} threadgroup_barrier(mem_flags::mem_threadgroup); }
    float rmsk=rsqrt(rk[0]/float(HD)+EPS), rmsv=rsqrt(rv[0]/float(HD)+EPS);
    vcache[cache+tid]=vo*rmsv;
    if(tid<HALF){ float n0=ksh[tid]*rmsk*knorm[tid], n1=ksh[tid+HALF]*rmsk*knorm[tid+HALF];
        float c=cosT[co+tid], sn=sinT[co+tid];
        kcache[cache+tid]=n0*c-n1*sn; kcache[cache+tid+HALF]=n1*c+n0*sn; }
}
