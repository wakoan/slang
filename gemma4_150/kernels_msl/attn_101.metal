#include <metal_stdlib>
using namespace metal;
// from 101_srq.wgsl — decode flash attention (q-norm+rope, online softmax, last-arriver merge)
// HEAD_DIM/HALF_DIM/OUT_Q patched per layer. dispatch 2-D groups = (qHeads, NCHUNK); tpg=(256)
// dispatch with threadsPerThreadgroup = (256)
struct P101 { uint seqQ; uint keyLen; uint qOffset; uint qHeads; uint kvHeads; uint window; uint p0; uint p1; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float reduce_sum(float v, uint tid, threadgroup float* red){
    float s=simd_sum(v); if((tid&31u)==0u) red[tid>>5u]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    float t=0.0f; for(uint i=0u;i<256u/32u;i++) t+=red[i]; threadgroup_barrier(mem_flags::mem_threadgroup); return t; }
inline float reduce_max(float v, uint tid, threadgroup float* red){
    float s=simd_max(v); if((tid&31u)==0u) red[tid>>5u]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
    float t=-3.4028234663852886e38f; for(uint i=0u;i<256u/32u;i++) t=max(t,red[i]); threadgroup_barrier(mem_flags::mem_threadgroup); return t; }
kernel void attn_101(
    device const float* q [[buffer(0)]], device const float* w [[buffer(1)]],
    device const float* cosTbl [[buffer(2)]], device const float* sinTbl [[buffer(3)]],
    device const float4* k [[buffer(4)]], device const float4* v [[buffer(5)]],
    device atomic_uint* partials [[buffer(6)]], device float* out [[buffer(7)]],
    constant P101& params [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint HEAD_DIM=512u, HALF_DIM=256u, NCHUNK=32u, WG=256u;
    const float EPS=1e-6f, SCALE=1.0f, NEG_INF=-3.4028234663852886e38f, OUT_Q=0.014886821620166302f;
    const uint PP_COUNTER_BASE=8u*NCHUNK*(HEAD_DIM+2u), HD4=HEAD_DIM/4u, J_GROUPS=WG/(HEAD_DIM/4u);
    threadgroup float qn_sh[HEAD_DIM], out_acc[HEAD_DIM], probs[WG], sval_sh[WG], red[WG], wgt_sh[NCHUNK];
    threadgroup float4 vacc_sh[WG];
    threadgroup float running_max, running_denom; threadgroup uint lastFlag;
    uint h=wg.x, ci=wg.y;
    if(h>=params.qHeads) return;
    uint hKv=h/(params.qHeads/params.kvHeads);
    uint qPos=params.qOffset, qBase=h*HEAD_DIM;
    uint maxKj=min(params.keyLen, qPos+1u), minKj=0u;
    if(params.window>0u && qPos+1u>params.window) minKj=qPos+1u-params.window;
    uint activeKeys=maxKj-minKj;
    uint nActive=clamp((activeKeys+63u)/64u, 8u, NCHUNK);
    if(ci>=nActive) return;
    float ss=0.0f; for(uint d=tid;d<HEAD_DIM;d+=WG){ float vv=q[qBase+d]; ss+=vv*vv; }
    float nscale=rsqrt(reduce_sum(ss,tid,red)/float(HEAD_DIM)+EPS);
    for(uint p=tid;p<HALF_DIM;p+=WG){ float n0=q[qBase+p]*nscale*w[p]; float n1=q[qBase+p+HALF_DIM]*nscale*w[p+HALF_DIM];
        float c=cosTbl[p], s=sinTbl[p]; qn_sh[p]=n0*c-n1*s; qn_sh[p+HALF_DIM]=n1*c+n0*s; }
    for(uint i=tid;i<HEAD_DIM;i+=WG) out_acc[i]=0.0f;
    if(tid==0u){ running_max=NEG_INF; running_denom=0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint chunkLen=(activeKeys+nActive-1u)/nActive;
    uint start=minKj+ci*chunkLen, end=min(start+chunkLen, maxKj);
    for(uint tile=start; tile<end; tile+=WG){
        uint kj=tile+tid; uint tileCountS=min(WG, end-tile);
        uint sgRounds=(tileCountS+(WG/32u)-1u)/(WG/32u);
        for(uint rr=0u;rr<sgRounds;rr++){ uint j=rr*(WG/32u)+(tid/32u); float accS=0.0f;
            if(j<tileCountS){ uint kBase4=((tile+j)*params.kvHeads+hKv)*HD4;
                for(uint d4=(tid&31u); d4<HD4; d4+=32u){ float4 kv4=k[kBase4+d4]; accS+=dot(float4(qn_sh[d4*4u],qn_sh[d4*4u+1u],qn_sh[d4*4u+2u],qn_sh[d4*4u+3u]),kv4); } }
            float sj=simd_sum(accS); if((tid&31u)==0u && j<tileCountS) sval_sh[j]=sj*SCALE; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sval=NEG_INF; if(kj<end) sval=sval_sh[tid];
        float tileMax=reduce_max(sval,tid,red);
        float newMax=max(running_max, tileMax);
        float correction=exp(running_max-newMax);
        float pr=0.0f; if(kj<end) pr=exp(sval-newMax); probs[tid]=pr;
        float tileDenom=reduce_sum(pr,tid,red);
        if(tid==0u){ running_denom=running_denom*correction+tileDenom; running_max=newMax; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tileCount=min(WG, end-tile);
        uint jg=tid/HD4, d4v=tid%HD4;
        float4 vacc=float4(0.0f);
        for(uint jj=jg; jj<tileCount; jj+=J_GROUPS){ uint vBase4=((tile+jj)*params.kvHeads+hKv)*HD4; vacc+=probs[jj]*v[vBase4+d4v]; }
        vacc_sh[tid]=vacc; threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint d4=tid; d4<HD4; d4+=WG){ float4 a4=float4(out_acc[d4*4u],out_acc[d4*4u+1u],out_acc[d4*4u+2u],out_acc[d4*4u+3u])*correction;
            for(uint g=0u;g<J_GROUPS;g++) a4+=vacc_sh[g*HD4+d4];
            out_acc[d4*4u]=a4.x; out_acc[d4*4u+1u]=a4.y; out_acc[d4*4u+2u]=a4.z; out_acc[d4*4u+3u]=a4.w; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint pBase=(h*NCHUNK+ci)*(HEAD_DIM+2u);
    for(uint i=tid;i<HEAD_DIM;i+=WG) atomic_store_explicit(&partials[pBase+i], as_type<uint>(out_acc[i]), memory_order_relaxed);
    if(tid==0u){ atomic_store_explicit(&partials[pBase+HEAD_DIM], as_type<uint>(running_max), memory_order_relaxed);
        atomic_store_explicit(&partials[pBase+HEAD_DIM+1u], as_type<uint>(running_denom), memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0u){ uint tk=atomic_fetch_add_explicit(&partials[PP_COUNTER_BASE+h],1u,memory_order_relaxed); lastFlag=(tk==nActive-1u)?1u:0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lastFlag!=1u) return;
    if(tid==0u) atomic_store_explicit(&partials[PP_COUNTER_BASE+h],0u,memory_order_relaxed);
    float mloc=NEG_INF, lloc=0.0f;
    if(tid<nActive){ uint pb=(h*NCHUNK+tid)*(HEAD_DIM+2u); mloc=as_type<float>(atomic_load_explicit(&partials[pb+HEAD_DIM],memory_order_relaxed)); lloc=as_type<float>(atomic_load_explicit(&partials[pb+HEAD_DIM+1u],memory_order_relaxed)); }
    float newM=reduce_max(mloc,tid,red);
    float wloc=0.0f; if(tid<nActive){ wloc=exp(mloc-newM); wgt_sh[tid]=wloc; }
    float denom=reduce_sum(lloc*wloc,tid,red); float invd=1.0f/denom;
    for(uint d=tid; d<HEAD_DIM; d+=WG){ float acc=0.0f;
        for(uint c=0u;c<nActive;c++) acc+=as_type<float>(atomic_load_explicit(&partials[(h*NCHUNK+c)*(HEAD_DIM+2u)+d],memory_order_relaxed))*wgt_sh[c];
        out[h*HEAD_DIM+d]=srq(acc*invd, OUT_Q); }
}
