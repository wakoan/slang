#include <metal_stdlib>
using namespace metal;
// custom (runner.py _COMBINE) — PLE input: per-row RMSNorm(ctx*H^-0.5) + ple, *2^-0.5. HINV patched.
// dispatch with threadsPerThreadgroup = (256) ; groups = nL = 35
kernel void combine(
    device const float* ctx [[buffer(0)]], device const float* ple [[buffer(1)]],
    device const float* nw [[buffer(2)]], device float* outp [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint D=256u; const float HINV=2.5515046504e-02f; const float EPS=1e-6f; const float RS2=0.7071067811865476f;
    threadgroup float red[D];
    uint base=wg.x*D + tid;
    float c=ctx[base]*HINV;
    red[tid]=c*c; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=D/2u;s>0u;s/=2u){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float rms=rsqrt(red[0]/float(D)+EPS);
    outp[base]=(c*rms*nw[tid]+ple[base])*RS2;
}
