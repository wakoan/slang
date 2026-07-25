#include <metal_stdlib>
using namespace metal;
// BATCHED (prefill) causal attention: S query positions in one dispatch.
//
// attn_101 (decode) has only nH workgroups of parallelism for a single query, so
// it splits the KEY axis into NCHUNK chunks and merges them with an atomic
// last-arriver pass. Prefill has S*nH workgroups already, so that machinery is
// pure overhead here: one workgroup owns one (head, query token) and walks all
// of its keys with a plain online softmax. No partials buffer, no atomics, no
// cross-workgroup merge — which also means no counter to reset between layers.
//
// Causality is per-token: query s sits at absolute position startPos+s and
// attends to keys [minKj, startPos+s], with minKj cutting in the sliding window.
// There is no mask buffer; the key loop bound IS the mask.
//
// q and out are [S][qHeads][HEAD_DIM]; k/v are the flat [max_seq][HEAD_DIM]
// caches (kvHeads==1 for this model). cos/sin are the FULL tables, indexed by
// absolute position — decode binds them pre-offset for its one position, which
// does not generalise to a batch.
// HEAD_DIM/HALF_DIM/OUT_Q are patched per layer, as in attn_101.
// dispatch with threadsPerThreadgroup = (256) ; groups = (qHeads, S)
struct PAP { uint startPos; uint qHeads; uint kvHeads; uint window; uint seq; uint p0; uint p1; uint p2; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float reduce_sum(float v, uint tid, threadgroup float* red){
    float s=simd_sum(v); if((tid&31u)==0u) red[tid>>5u]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    float t=0.0f; for(uint i=0u;i<256u/32u;i++) t+=red[i]; threadgroup_barrier(mem_flags::mem_threadgroup); return t; }
inline float reduce_max(float v, uint tid, threadgroup float* red){
    float s=simd_max(v); if((tid&31u)==0u) red[tid>>5u]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    float t=-3.4028234663852886e38f; for(uint i=0u;i<256u/32u;i++) t=max(t,red[i]); threadgroup_barrier(mem_flags::mem_threadgroup); return t; }

kernel void attn_prefill(
    device const float* q [[buffer(0)]], device const float* w [[buffer(1)]],
    device const float* cosTbl [[buffer(2)]], device const float* sinTbl [[buffer(3)]],
    device const float4* k [[buffer(4)]], device const float4* v [[buffer(5)]],
    device float* out [[buffer(6)]], constant PAP& params [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint HEAD_DIM=512u, HALF_DIM=256u, WG=256u;
    const float EPS=1e-6f, SCALE=1.0f, NEG_INF=-3.4028234663852886e38f;
    const float OUT_Q=0.014886821620166302f;
    const uint HD4=HEAD_DIM/4u, J_GROUPS=WG/(HEAD_DIM/4u);
    threadgroup float qn_sh[HEAD_DIM], out_acc[HEAD_DIM], probs[WG], sval_sh[WG], red[WG];
    threadgroup float4 vacc_sh[WG];
    threadgroup float running_max, running_denom;

    uint h=wg.x, s=wg.y;
    if(h>=params.qHeads || s>=params.seq) return;
    uint hKv=h/(params.qHeads/params.kvHeads);
    uint qPos=params.startPos+s;
    uint qBase=(s*params.qHeads+h)*HEAD_DIM;
    uint maxKj=qPos+1u, minKj=0u;                     // causal bound == the mask
    if(params.window>0u && qPos+1u>params.window) minKj=qPos+1u-params.window;

    float ss=0.0f; for(uint d=tid;d<HEAD_DIM;d+=WG){ float vv=q[qBase+d]; ss+=vv*vv; }
    float nscale=rsqrt(reduce_sum(ss,tid,red)/float(HEAD_DIM)+EPS);
    uint rb=qPos*HALF_DIM;                            // absolute-position rope lookup
    for(uint p=tid;p<HALF_DIM;p+=WG){
        float n0=q[qBase+p]*nscale*w[p], n1=q[qBase+p+HALF_DIM]*nscale*w[p+HALF_DIM];
        float c=cosTbl[rb+p], sn=sinTbl[rb+p];
        qn_sh[p]=n0*c-n1*sn; qn_sh[p+HALF_DIM]=n1*c+n0*sn; }
    for(uint i=tid;i<HEAD_DIM;i+=WG) out_acc[i]=0.0f;
    if(tid==0u){ running_max=NEG_INF; running_denom=0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(uint tile=minKj; tile<maxKj; tile+=WG){
        uint kj=tile+tid; uint tileCount=min(WG, maxKj-tile);
        uint sgRounds=(tileCount+(WG/32u)-1u)/(WG/32u);
        for(uint rr=0u;rr<sgRounds;rr++){
            uint j=rr*(WG/32u)+(tid/32u); float accS=0.0f;
            if(j<tileCount){ uint kBase4=((tile+j)*params.kvHeads+hKv)*HD4;
                for(uint d4=(tid&31u); d4<HD4; d4+=32u){ float4 kv4=k[kBase4+d4];
                    accS+=dot(float4(qn_sh[d4*4u],qn_sh[d4*4u+1u],qn_sh[d4*4u+2u],qn_sh[d4*4u+3u]),kv4); } }
            float sj=simd_sum(accS); if((tid&31u)==0u && j<tileCount) sval_sh[j]=sj*SCALE; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sval=NEG_INF; if(kj<maxKj) sval=sval_sh[tid];
        float tileMax=reduce_max(sval,tid,red);
        float newMax=max(running_max, tileMax);
        float correction=exp(running_max-newMax);
        float pr=0.0f; if(kj<maxKj) pr=exp(sval-newMax); probs[tid]=pr;
        float tileDenom=reduce_sum(pr,tid,red);
        if(tid==0u){ running_denom=running_denom*correction+tileDenom; running_max=newMax; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint jg=tid/HD4, d4v=tid%HD4;
        float4 vacc=float4(0.0f);
        for(uint jj=jg; jj<tileCount; jj+=J_GROUPS){ uint vBase4=((tile+jj)*params.kvHeads+hKv)*HD4;
            vacc+=probs[jj]*v[vBase4+d4v]; }
        vacc_sh[tid]=vacc; threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint d4=tid; d4<HD4; d4+=WG){
            float4 a4=float4(out_acc[d4*4u],out_acc[d4*4u+1u],out_acc[d4*4u+2u],out_acc[d4*4u+3u])*correction;
            for(uint g=0u;g<J_GROUPS;g++) a4+=vacc_sh[g*HD4+d4];
            out_acc[d4*4u]=a4.x; out_acc[d4*4u+1u]=a4.y; out_acc[d4*4u+2u]=a4.z; out_acc[d4*4u+3u]=a4.w; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float invd=1.0f/running_denom;
    for(uint d=tid; d<HEAD_DIM; d+=WG) out[qBase+d]=srq(out_acc[d]*invd, OUT_Q);
}
