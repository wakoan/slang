#include <metal_stdlib>
using namespace metal;
// PREFILL: `combine` over S tokens. ctx/ple are [S][nL*D]; one workgroup per
// (layer, token). HINV patched, as in combine.
// dispatch with threadsPerThreadgroup = (256) ; groups = (nL, S)
struct PCB { uint nL; uint p0; uint p1; uint p2; };
kernel void combine_b(
    device const float* ctx [[buffer(0)]], device const float* ple [[buffer(1)]],
    device const float* nw [[buffer(2)]], device float* outp [[buffer(3)]],
    constant PCB& p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint D=256u; const float HINV=2.5515046504e-02f, EPS=1e-6f, RS2=0.7071067811865476f;
    threadgroup float red[D];
    uint base=(wg.y*p.nL + wg.x)*D + tid;
    float c=ctx[base]*HINV;
    red[tid]=c*c; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=D/2u;s>0u;s/=2u){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float rms=rsqrt(red[0]/float(D)+EPS);
    outp[base]=(c*rms*nw[tid]+ple[base])*RS2;
}
