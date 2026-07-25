#include <metal_stdlib>
using namespace metal;
// BATCHED (prefill) down-projection, 2-bit, INTER=12288 -> H=1536.
// Pairs with gateup_95_b to make the whole MLP block batched.
//
// Same math as down_96 but DELIBERATELY UNFUSED: down_96 fuses the residual add
// + post-FFN RMSNorm behind a global atomic last-arriver counter, which would
// need S counters/accumulators to batch. That fusion is a decode-latency
// optimisation; prefill is compute-bound with S tokens per dispatch, so we emit
// plain y[S][H] and let a separate batched norm consume it.
//
// TWO-DIMENSIONAL register blocking (this is the whole point). A thread holds a
// micro-tile of N_ROWS output rows x SEQ tokens, so per weight-word it issues
// N_ROWS weight loads + SEQ*4 activation loads to do N_ROWS*SEQ*16 MACs:
//
//     MAC / issued-load  =  16*N_ROWS*SEQ / (N_ROWS + 4*SEQ)
//
// These GEMVs are load-ISSUE bound, so that ratio — not bytes moved — is what
// sets the speed. Blocking only tokens (N_ROWS=1, SEQ=8) scores 3.9, *worse*
// than decode's N_ROWS=4/SEQ=1 at 8.0, which is exactly why the first attempt
// at this kernel measured no win at all.
// Activations are hoisted into registers once per word (s-outer) and reused
// across all N_ROWS weight rows; SEQ=4 keeps that hoist affordable (SEQ=8
// doubles it to 128 registers and spills, measuring 0.27-0.41x).
//
// Measured warm (prefill_research/bench_batched.py), vs S x down_96 in ONE
// command buffer: 1.45x at S=8, 1.80x at S=16. Do not trust a cold GPU — the
// baseline alone swings 2.8x between cold and warm clocks.
// a is half [S][INTER]; y is f32 [S][OUT_F]. sum_a is accumulated in-kernel.
// dispatch with threadsPerThreadgroup = (256) ; groups = (OUT_F/N_ROWS, S/SEQ)
struct P96B { float inScale; float outScale; uint p0; uint p1; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }

kernel void down_96_b(
    device const half4* a      [[buffer(0)]],
    device const uint*  bits   [[buffer(1)]],
    device const float* scale  [[buffer(2)]],
    device       float* y      [[buffer(3)]],
    constant     P96B&  params [[buffer(4)]],
    uint lidx [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint OUT_F=1536u, INTER=12288u, CHUNKS=4u, WPR=768u, WG=256u;
    const uint N_ROWS=16u, SEQ=4u;
    const uint ASTRIDE = INTER/4u;            // half4 per token row
    const float ZP=2.0f;
    threadgroup float red[(WG/32u)*N_ROWS*SEQ], redA[(WG/32u)*SEQ];

    uint sg=lidx/32u, lane=lidx&31u;
    uint rowBase = wg.x*N_ROWS, sBase = wg.y*SEQ;

    float acc[N_ROWS*SEQ], sumA[SEQ];
    for(uint i=0u;i<N_ROWS*SEQ;i++) acc[i]=0.0f;
    for(uint s=0u;s<SEQ;s++) sumA[s]=0.0f;

    for(uint w=lidx; w<WPR; w+=WG){
        float4 av[SEQ*4u];                    // activations hoisted once...
        for(uint s=0u;s<SEQ;s++){
            uint ab=(sBase+s)*ASTRIDE + w*CHUNKS;
            float4 v0=float4(a[ab+0u]), v1=float4(a[ab+1u]);
            float4 v2=float4(a[ab+2u]), v3=float4(a[ab+3u]);
            av[s*4u+0u]=v0; av[s*4u+1u]=v1; av[s*4u+2u]=v2; av[s*4u+3u]=v3;
            sumA[s] += (v0.x+v0.y+v0.z+v0.w)+(v1.x+v1.y+v1.z+v1.w)
                     + (v2.x+v2.y+v2.z+v2.w)+(v3.x+v3.y+v3.z+v3.w);
        }
        for(uint r=0u;r<N_ROWS;r++){          // ...and reused for every weight row
            uint p = bits[(rowBase+r)*WPR+w];
            float4 c0=unpack_unorm4x8_to_float(p&0x03030303u), c1=unpack_unorm4x8_to_float((p>>2u)&0x03030303u);
            float4 c2=unpack_unorm4x8_to_float((p>>4u)&0x03030303u), c3=unpack_unorm4x8_to_float((p>>6u)&0x03030303u);
            float4 w0=float4(c0.x,c1.x,c2.x,c3.x), w1=float4(c0.y,c1.y,c2.y,c3.y);
            float4 w2=float4(c0.z,c1.z,c2.z,c3.z), w3=float4(c0.w,c1.w,c2.w,c3.w);
            for(uint s=0u;s<SEQ;s++)
                acc[r*SEQ+s] += dot(w0,av[s*4u+0u])+dot(w1,av[s*4u+1u])
                              + dot(w2,av[s*4u+2u])+dot(w3,av[s*4u+3u]);
        }
    }
    for(uint i=0u;i<N_ROWS*SEQ;i++){ float v=simd_sum(acc[i]); if(lane==0u) red[sg*N_ROWS*SEQ+i]=v; }
    for(uint s=0u;s<SEQ;s++){ float v=simd_sum(sumA[s]); if(lane==0u) redA[sg*SEQ+s]=v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(lidx < N_ROWS*SEQ){
        uint r=lidx/SEQ, s=lidx%SEQ;
        float tot=0.0f, aSum=0.0f;
        for(uint g=0u;g<WG/32u;g++){ tot+=red[g*N_ROWS*SEQ+lidx]; aSum+=redA[g*SEQ+s]; }
        uint o=rowBase+r;
        y[(sBase+s)*OUT_F+o] = srq(scale[o]*(params.inScale*fma(tot,255.0f,-(ZP*aSum))), params.outScale);
    }
}
