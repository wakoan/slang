#include <metal_stdlib>
using namespace metal;
// from 34_main.wgsl — argmax pass 1 (per-slice candidate). groups = COUNT/SLICE = 256
// dispatch with threadsPerThreadgroup = (256)
kernel void argmax1_34(
    device const float* x [[buffer(0)]], device float* cand_val [[buffer(1)]],
    device uint* cand_idx [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint COUNT=262144u, SLICE=1024u, WG=256u; const float NEG_INF=-3.4028234663852886e38;
    threadgroup float wgVal[WG]; threadgroup uint wgIdx[WG];
    uint base=wg.x*SLICE, end=min(base+SLICE, COUNT);
    float bv=NEG_INF; uint bi=0u;
    for(uint i=base+tid;i<end;i+=WG){ float v=x[i]; if(v>bv){bv=v;bi=i;} }
    wgVal[tid]=bv; wgIdx[tid]=bi; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=WG/2u; stride>0u; stride/=2u){
        if(tid<stride){ uint o=tid+stride; if(wgVal[o]>wgVal[tid]||(wgVal[o]==wgVal[tid]&&wgIdx[o]<wgIdx[tid])){wgVal[tid]=wgVal[o];wgIdx[tid]=wgIdx[o];} }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0u){ cand_val[wg.x]=wgVal[0]; cand_idx[wg.x]=wgIdx[0]; }
}
