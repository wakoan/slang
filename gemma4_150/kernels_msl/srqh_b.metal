#include <metal_stdlib>
using namespace metal;
// PREFILL: f32 activations -> f16 for an MPS GEMM's A matrix, optionally SRQ'd.
// Several decode kernels quantize their input inline (plegate_76/pleproj_77/
// proj_68 call srq4 on the fly); once the matmul is MPS this has to become a
// real pass. scale==0 means "convert only".
// dispatch with threadsPerThreadgroup = (256) ; groups = ceil(n/256)
struct PSQ { float scale; uint n; uint p0; uint p1; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
kernel void srqh_b(device const float* x [[buffer(0)]], device half* y [[buffer(1)]],
                   constant PSQ& p [[buffer(2)]], uint gid [[thread_position_in_grid]])
{ if(gid>=p.n) return; y[gid]=half(srq(x[gid], p.scale)); }
