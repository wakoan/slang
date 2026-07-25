#include <metal_stdlib>
using namespace metal;
// PREFILL: q RMSNorm + RoPE for all S*qHeads query vectors, emitted as f16.
//
// This is the head of attn_prefill split out so the score matrix can become a
// GEMM. The key observation is that kvHeads==1 for this model, so QK^T over ALL
// heads and ALL query positions is ONE matrix product: q is laid out
// [S][qHeads][HEAD_DIM], whose rows viewed flat are exactly [S*qHeads][HEAD_DIM],
// and every one of those rows multiplies the same K cache. No per-head batching.
//
// out has the same [s][h][d] layout as q, just half instead of float.
// HEAD_DIM/HALF_DIM patched per layer-type.
// dispatch with threadsPerThreadgroup = (512) ; groups = (qHeads, S)
struct PQP { uint startPos; uint qHeads; uint p0; uint p1; };
kernel void qprep_b(
    device const float* q [[buffer(0)]], device const float* w [[buffer(1)]],
    device const float* cosT [[buffer(2)]], device const float* sinT [[buffer(3)]],
    device half* out [[buffer(4)]], constant PQP& p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint HEAD_DIM=512u, HALF_DIM=256u; const float EPS=1e-6f;
    threadgroup float red[HEAD_DIM], qsh[HEAD_DIM];
    uint h=wg.x, s=wg.y;
    uint base=(s*p.qHeads+h)*HEAD_DIM;
    float v=q[base+tid]; qsh[tid]=v; red[tid]=v*v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=HEAD_DIM/2u; st>0u; st/=2u){
        if(tid<st) red[tid]+=red[tid+st];
        threadgroup_barrier(mem_flags::mem_threadgroup); }
    float ns=rsqrt(red[0]/float(HEAD_DIM)+EPS);
    uint rb=(p.startPos+s)*HALF_DIM;              // absolute-position rope lookup
    if(tid<HALF_DIM){
        float n0=qsh[tid]*ns*w[tid], n1=qsh[tid+HALF_DIM]*ns*w[tid+HALF_DIM];
        float c=cosT[rb+tid], sn=sinT[rb+tid];
        out[base+tid]=half(n0*c-n1*sn);
        out[base+tid+HALF_DIM]=half(n1*c+n0*sn); }
}
