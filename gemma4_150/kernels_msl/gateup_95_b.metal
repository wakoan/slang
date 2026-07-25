#include <metal_stdlib>
using namespace metal;
// BATCHED (prefill) variant of gateup_95 (2-bit, INTER=12288).
//
// TWO-DIMENSIONAL register blocking, same principle as down_96_b: a simdgroup
// owns a micro-tile of N_ROWS output rows x SEQ tokens. Per weight-word it
// issues 2*N_ROWS weight loads (gate+up) + 4*SEQ activation loads to do
// 32*N_ROWS*SEQ MACs:
//
//     MAC / issued-load  =  32*N_ROWS*SEQ / (2*N_ROWS + 4*SEQ)
//
// Blocking tokens alone (N_ROWS=1, SEQ=8) scores 7.5 — BELOW decode's
// N_ROWS=2/SEQ=1 at 8.0 — and duly measures 0.72x/0.55x (S=8/16), i.e. the
// first version of this kernel was a regression, not the 2.49x once recorded
// against an unfair per-token-command-buffer baseline.
// Activations are hoisted into registers once per word (s-outer) and reused
// across every weight row; the gate/up weight pair is streamed one row at a
// time so only 32 weight registers stay live.
//
// The ratio is necessary but NOT sufficient: measured, N_ROWS=2/SEQ=4 (ratio
// 12.8) beats N_ROWS=8/SEQ=4 (ratio 32.0) 1.21x to 0.43x, because n_in=1536 is
// only 96 words = 3 per lane, so a big micro-tile costs registers/occupancy it
// never amortises over such a short k-loop. Contrast down_96_b, whose 768-word
// k-loop does pay for N_ROWS=16. Re-sweep with prefill_research/bench_batched.py
// rather than reasoning from the ratio alone.
//
// Measured warm, vs S x gateup_95 in ONE command buffer: 1.21x at S=8, 1.23x at
// S=16. Modest — this kernel is the 56% of prefill FLOPs, so it caps the whole
// batching path at ~1.3x end-to-end. Parity with LiteRT needs a real GEMM.
// hidden is half [S][H]; out is half [S][INTER]; sum_a is f32 [S].
// dispatch with threadsPerThreadgroup = (64) ; groups = (INTER/(SG_COUNT*N_ROWS), S/SEQ)
struct P95B { float gateOutScale; float upOutScale; float outQuantScale; uint p; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float tanh_safe(float x){ if(x>10.0f) return 1.0f; if(x<-10.0f) return -1.0f; return tanh(x); }
inline float gelu_tanh(float v){ return 0.5f*v*(1.0f+tanh_safe(0.7978845608028654f*(v+0.044715f*v*v*v))); }
inline float gelu_grid(float g, float s, device const float* lut){ if(s==0.0f) return gelu_tanh(g); return lut[uint(clamp(round(g/s),-128.0f,127.0f)+128.0f)]; }

kernel void gateup_95_b(
    device const half4* hidden [[buffer(0)]], device const uint* gate_bits [[buffer(1)]],
    device const float* gate_scale [[buffer(2)]], device const uint* up_bits [[buffer(3)]],
    device const float* up_scale [[buffer(4)]], device const float* sum_a [[buffer(5)]],
    device half* out [[buffer(6)]], device const float* gelu_lut [[buffer(7)]],
    constant P95B& params [[buffer(8)]],
    uint lidx [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint INTER=12288u, WPR=96u, CHUNKS=4u, SG_COUNT=2u;
    const uint N_ROWS=2u, SEQ=4u;
    const uint HSTRIDE=384u;              // half4 per token row (1536 halves)
    const float ZP=2.0f;

    uint sgId=lidx/32u, tid=lidx&31u;
    uint rowBase=(wg.x*SG_COUNT+sgId)*N_ROWS, sBase=wg.y*SEQ;
    float gOut=params.gateOutScale, uOut=params.upOutScale;

    float gAcc[N_ROWS*SEQ], uAcc[N_ROWS*SEQ];
    for(uint i=0u;i<N_ROWS*SEQ;i++){ gAcc[i]=0.0f; uAcc[i]=0.0f; }

    for(uint wd=tid; wd<WPR; wd+=32u){
        float4 av[SEQ*4u];                                   // hoisted once...
        for(uint s=0u;s<SEQ;s++){
            uint hb=(sBase+s)*HSTRIDE+wd*CHUNKS;
            av[s*4u+0u]=float4(hidden[hb+0u]); av[s*4u+1u]=float4(hidden[hb+1u]);
            av[s*4u+2u]=float4(hidden[hb+2u]); av[s*4u+3u]=float4(hidden[hb+3u]);
        }
        for(uint r=0u;r<N_ROWS;r++){                         // ...reused per row
            uint o=rowBase+r;
            uint pg=gate_bits[o*WPR+wd], pu=up_bits[o*WPR+wd];
            float4 g0=unpack_unorm4x8_to_float(pg&0x03030303u), g1=unpack_unorm4x8_to_float((pg>>2u)&0x03030303u);
            float4 g2=unpack_unorm4x8_to_float((pg>>4u)&0x03030303u), g3=unpack_unorm4x8_to_float((pg>>6u)&0x03030303u);
            float4 u0=unpack_unorm4x8_to_float(pu&0x03030303u), u1=unpack_unorm4x8_to_float((pu>>2u)&0x03030303u);
            float4 u2=unpack_unorm4x8_to_float((pu>>4u)&0x03030303u), u3=unpack_unorm4x8_to_float((pu>>6u)&0x03030303u);
            float4 gx=float4(g0.x,g1.x,g2.x,g3.x), gy=float4(g0.y,g1.y,g2.y,g3.y);
            float4 gz=float4(g0.z,g1.z,g2.z,g3.z), gw=float4(g0.w,g1.w,g2.w,g3.w);
            float4 ux=float4(u0.x,u1.x,u2.x,u3.x), uy=float4(u0.y,u1.y,u2.y,u3.y);
            float4 uz=float4(u0.z,u1.z,u2.z,u3.z), uw=float4(u0.w,u1.w,u2.w,u3.w);
            for(uint s=0u;s<SEQ;s++){
                float4 a0=av[s*4u+0u], a1=av[s*4u+1u], a2=av[s*4u+2u], a3=av[s*4u+3u];
                gAcc[r*SEQ+s]+=dot(gx,a0)+dot(gy,a1)+dot(gz,a2)+dot(gw,a3);
                uAcc[r*SEQ+s]+=dot(ux,a0)+dot(uy,a1)+dot(uz,a2)+dot(uw,a3);
            }
        }
    }
    float qs=params.outQuantScale;
    for(uint r=0u;r<N_ROWS;r++){
        uint o=rowBase+r;
        for(uint s=0u;s<SEQ;s++){
            float gS=simd_sum(gAcc[r*SEQ+s]), uS=simd_sum(uAcc[r*SEQ+s]);
            if(tid==0u){
                float aSum=sum_a[sBase+s];
                float g=srq(gate_scale[o]*fma(gS,255.0f,-(ZP*aSum)),gOut);
                float u=srq(up_scale[o]*fma(uS,255.0f,-(ZP*aSum)),uOut);
                float dq=gelu_grid(g,gOut,gelu_lut)*u;
                out[(sBase+s)*INTER+o]=half((qs==0.0f)?dq:clamp(round(dq/qs),-128.0f,127.0f));
            }
        }
    }
}
