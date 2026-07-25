#include <metal_stdlib>
using namespace metal;
// PREFILL: causal + sliding-window softmax over a GEMM-produced score matrix.
//
// sc is [S*qHeads][stride] f16; row r belongs to query token r/qHeads (any head,
// since the mask depends only on the position). The mask is not a buffer and is
// not materialised: it is the loop bound, exactly as in attn_prefill.
//
// The normalisation is deliberately NOT applied here. Dividing would cost a
// third pass over the row; instead each row's denominator is written to denom[]
// and folded into attnout_b, which already has to touch every output element.
// Unnormalised probabilities are still <= 1 (exp of a non-positive number), so
// nothing overflows f16.
//
// Columns outside [0, T) are never written and never read: the PV GEMM is
// encoded with interiorColumns = T, so the row padding that keeps rowBytes
// 16-byte aligned stays untouched garbage.
// dispatch with threadsPerThreadgroup = (256) ; groups = S*qHeads
struct PSM { uint startPos; uint qHeads; uint window; uint T; uint stride; uint p0; uint p1; uint p2; };
kernel void smax_b(
    device half* sc [[buffer(0)]], device float* denom [[buffer(1)]],
    constant PSM& p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint WG=256u; const float NEG_INF=-3.4028234663852886e38f;
    threadgroup float red[WG];
    uint r=wg.x, s=r/p.qHeads;
    uint qPos=p.startPos+s;
    uint maxKj=qPos+1u, minKj=0u;                 // causal bound == the mask
    if(p.window>0u && qPos+1u>p.window) minKj=qPos+1u-p.window;
    if(maxKj>p.T) maxKj=p.T;
    device half* row = sc + (ulong)r*(ulong)p.stride;

    float m=NEG_INF;
    for(uint j=minKj+tid; j<maxKj; j+=WG) m=max(m, float(row[j]));
    red[tid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=WG/2u; st>0u; st/=2u){
        if(tid<st) red[tid]=max(red[tid], red[tid+st]);
        threadgroup_barrier(mem_flags::mem_threadgroup); }
    m=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum=0.0f;
    for(uint j=tid; j<p.T; j+=WG){
        float e=0.0f;
        if(j>=minKj && j<maxKj){ e=exp(float(row[j])-m); sum+=e; }
        row[j]=half(e); }
    red[tid]=sum; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint st=WG/2u; st>0u; st/=2u){
        if(tid<st) red[tid]+=red[tid+st];
        threadgroup_barrier(mem_flags::mem_threadgroup); }
    if(tid==0u) denom[r]=red[0];
}
