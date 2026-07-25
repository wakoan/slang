#include <metal_stdlib>
using namespace metal;
// PREFILL: f16 MPS GEMM output -> f32, with the SRQ the decode kernel applied to
// its own result (e.g. qkv_70's srq(...,qOutScale)).
// dispatch with threadsPerThreadgroup = (256) ; groups = ceil(n/256)
struct PSQ2 { float scale; uint n; uint p0; uint p1; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
kernel void srq_b(device const half* x [[buffer(0)]], device float* y [[buffer(1)]],
                  constant PSQ2& p [[buffer(2)]], uint gid [[thread_position_in_grid]])
{ if(gid>=p.n) return; y[gid]=srq(float(x[gid]), p.scale); }
