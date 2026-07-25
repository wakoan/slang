#include <metal_stdlib>
using namespace metal;
// PREFILL: the geglu tail of gateup_74/95, once the two matmuls are MPS GEMMs.
// gate/up arrive already scaled and zero-point-corrected (that folds into the
// dequantized weights), so only the SRQ + gelu-LUT + product remain.
// dispatch with threadsPerThreadgroup = (256) ; groups = ceil(n/256)
struct PGG { float gateOutScale; float upOutScale; float outQuantScale; uint n; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float tanh_safe(float x){ if(x>10.0f) return 1.0f; if(x<-10.0f) return -1.0f; return tanh(x); }
inline float gelu_tanh(float v){ return 0.5f*v*(1.0f+tanh_safe(0.7978845608028654f*(v+0.044715f*v*v*v))); }
inline float gelu_grid(float g, float s, device const float* lut){ if(s==0.0f) return gelu_tanh(g); return lut[uint(clamp(round(g/s),-128.0f,127.0f)+128.0f)]; }
kernel void geglu_b(
    device const half* gate [[buffer(0)]], device const half* up [[buffer(1)]],
    device half* out [[buffer(2)]], device const float* lut [[buffer(3)]],
    constant PGG& p [[buffer(4)]], uint gid [[thread_position_in_grid]])
{
    if(gid>=p.n) return;
    float g=srq(float(gate[gid]), p.gateOutScale);
    float u=srq(float(up[gid]), p.upOutScale);
    float dq=gelu_grid(g, p.gateOutScale, lut)*u;
    float qs=p.outQuantScale;
    out[gid]=half((qs==0.0f)?dq:clamp(round(dq/qs),-128.0f,127.0f));
}
