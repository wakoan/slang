#include <metal_stdlib>
using namespace metal;
// PREFILL: the tail of plegate_76 once its matmul is an MPS GEMM.
// out[s][o] = gelu_grid(srq(lin)) * ple[s][pleOffset+o]
// dispatch with threadsPerThreadgroup = (256) ; groups = ceil(S*OUT/256)
struct PPG { float linOutScale; uint pleOffset; uint pleStride; uint n; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float tanh_safe(float x){ if(x>10.0f) return 1.0f; if(x<-10.0f) return -1.0f; return tanh(x); }
inline float gelu_tanh(float v){ return 0.5f*v*(1.0f+tanh_safe(0.7978845608028654f*(v+0.044715f*v*v*v))); }
inline float gelu_grid(float g, float s, device const float* lut){ if(s==0.0f) return gelu_tanh(g); return lut[uint(clamp(round(g/s),-128.0f,127.0f)+128.0f)]; }
kernel void plegate_b(
    device const half* lin [[buffer(0)]], device const float* ple [[buffer(1)]],
    device float* out [[buffer(2)]], device const float* lut [[buffer(3)]],
    constant PPG& p [[buffer(4)]], uint gid [[thread_position_in_grid]])
{
    const uint OUT=256u;
    if(gid>=p.n) return;
    uint s=gid/OUT, o=gid%OUT;
    float v=srq(float(lin[gid]), p.linOutScale);
    out[gid]=gelu_grid(v, p.linOutScale, lut) * ple[s*p.pleStride + p.pleOffset + o];
}
