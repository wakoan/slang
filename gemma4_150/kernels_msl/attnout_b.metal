#include <metal_stdlib>
using namespace metal;
// PREFILL: tail of the GEMM attention path — divide the PV product by the
// softmax denominator smax_b deferred, then apply the SRQ that the o-projection
// expects on its input (attn_prefill folded both into its own epilogue).
// Input and output are both f16 [S*qHeads][HEAD_DIM]; hd is passed so a row's
// denominator can be found from a flat element index.
// dispatch with threadsPerThreadgroup = (256) ; groups = ceil(n/256)
struct PAO { float scale; uint n; uint hd; uint p1; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
kernel void attnout_b(
    device const half* x [[buffer(0)]], device const float* denom [[buffer(1)]],
    device half* y [[buffer(2)]], constant PAO& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if(gid>=p.n) return;
    y[gid]=half(srq(float(x[gid])/denom[gid/p.hd], p.scale));
}
