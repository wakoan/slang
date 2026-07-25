#include <metal_stdlib>
using namespace metal;
// PREFILL: the residual-add + RMSNorm tail that oproj_73 / down_96 / pleproj_77
// each hide behind a global atomic last-arriver counter. Those fusions exist to
// save a decode dispatch; batching them would need S counters, so prefill runs
// the tail as its own pass over S rows.
//     d = srq(x, inScale);  hidden = (hidden + d*rms(d)*nw) * sv
// sv is pleproj_77's learned per-layer scalar (1.0 elsewhere).
// dispatch with threadsPerThreadgroup = (256) ; groups = S
struct PRA { float inScale; float sv; uint dim; uint p1; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float reduce_sum(float v, uint tid, threadgroup float* sgp){
    float s=simd_sum(v); if((tid&31u)==0u) sgp[tid>>5u]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    float t=0.0f; for(uint i=0u;i<256u/32u;i++) t+=sgp[i]; threadgroup_barrier(mem_flags::mem_threadgroup); return t; }
kernel void rmsadd_b(
    device const half* x [[buffer(0)]], device const float* nw [[buffer(1)]],
    device float* hidden [[buffer(2)]], constant PRA& p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint DIM=1536u, WG=256u; const float EPS=1e-6f;
    threadgroup float sgp[WG/32u], dsh[DIM];
    uint base=wg.x*DIM;
    float acc=0.0f;
    for(uint j=tid;j<DIM;j+=WG){ float d=srq(float(x[base+j]), p.inScale); dsh[j]=d; acc+=d*d; }
    float rms=rsqrt(reduce_sum(acc,tid,sgp)/float(DIM)+EPS);
    for(uint j=tid;j<DIM;j+=WG) hidden[base+j]=(hidden[base+j]+dsh[j]*rms*nw[j])*p.sv;
}
