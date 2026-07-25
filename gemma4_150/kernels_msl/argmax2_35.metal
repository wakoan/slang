#include <metal_stdlib>
using namespace metal;
// from 35_main.wgsl — argmax pass 2 (winner among candidates). groups = 1
// dispatch with threadsPerThreadgroup = (256)
kernel void argmax2_35(
    device const float* cand_val [[buffer(0)]], device const uint* cand_idx [[buffer(1)]],
    device uint* out [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint NCAND=256u, WG=256u; const float NEG_INF=-3.4028234663852886e38;
    threadgroup float wgVal[WG]; threadgroup uint wgIdx[WG];
    float bv=NEG_INF; uint bi=0u;
    for(uint i=tid;i<NCAND;i+=WG){ float v=cand_val[i]; uint idx=cand_idx[i]; if(v>bv||(v==bv&&idx<bi)){bv=v;bi=idx;} }
    wgVal[tid]=bv; wgIdx[tid]=bi; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=WG/2u; stride>0u; stride/=2u){
        if(tid<stride){ uint o=tid+stride; if(wgVal[o]>wgVal[tid]||(wgVal[o]==wgVal[tid]&&wgIdx[o]<wgIdx[tid])){wgVal[tid]=wgVal[o];wgIdx[tid]=wgIdx[o];} }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0u) out[0]=wgIdx[0];
}
