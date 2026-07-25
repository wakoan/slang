#include <metal_stdlib>
using namespace metal;
// from 76_reduce.wgsl — PLE input gate (int8 +128-biased) + gelu-LUT * ple
// dispatch with threadsPerThreadgroup = (32) ; groups = OUT_FEATURES = 256
struct P76 { float inScale; float linOutScale; uint pleOffset; uint p; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float4 srq4(float4 x, float s){ if(s==0.0f) return x; return clamp(round(x/s),float4(-128.0f),float4(127.0f))*s; }
inline float tanh_safe(float x){ if(x>10.0f) return 1.0f; if(x<-10.0f) return -1.0f; return tanh(x); }
inline float gelu_tanh(float v){ return 0.5f*v*(1.0f+tanh_safe(0.7978845608028654f*(v+0.044715f*v*v*v))); }
inline float gelu_grid(float g, float s, device const float* lut){ if(s==0.0f) return gelu_tanh(g); return lut[uint(clamp(round(g/s),-128.0f,127.0f)+128.0f)]; }
kernel void plegate_76(
    device const float* a [[buffer(0)]], device const uint* codes [[buffer(1)]],
    device const float* row_scale [[buffer(2)]], device const float* ple [[buffer(3)]],
    device float* out [[buffer(4)]], device const float* gelu_lut [[buffer(5)]],
    constant P76& params [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint OUT=256u, WG=32u, WPR=384u, GRID_X=256u;
    uint o = (wg.y*GRID_X+wg.x);          // N_ROWS=1
    if(o>=OUT) return;
    float acc=0.0f, aAcc=0.0f;
    for(uint wd=tid; wd<WPR; wd+=WG){ uint kb=wd*4u;
        float4 a4=srq4(float4(a[kb],a[kb+1u],a[kb+2u],a[kb+3u]), params.inScale);
        aAcc += (a4.x+a4.y)+(a4.z+a4.w);
        acc += dot(unpack_unorm4x8_to_float(codes[o*WPR+wd]), a4);
    }
    float aSum=simd_sum(aAcc); float s=simd_sum(acc);
    if(tid==0u){ float v=row_scale[o]*fma(s,255.0f,-128.0f*aSum);
        out[o]=gelu_grid(srq(v,params.linOutScale),params.linOutScale,gelu_lut) * ple[params.pleOffset+o]; }
}
