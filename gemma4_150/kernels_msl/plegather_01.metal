#include <metal_stdlib>
using namespace metal;
// from 01_main.wgsl — 4-bit per-layer-embedding gather
// dispatch with threadsPerThreadgroup = (64) ; groups = seq
struct P01 { uint seq; uint p0; uint p1; uint p2; };
kernel void plegather_01(
    device const uint* ids [[buffer(0)]], device const uint* bits_buf [[buffer(1)]],
    device const float* scale [[buffer(2)]], device float* y [[buffer(3)]],
    constant P01& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint HIDDEN=8960u, VOCAB=262144u, GROUP=256u, NUM_GROUPS=35u, WPR=1120u, VPW=8u, BITS=4u, MASK=15u, WG=64u;
    const float ZP=8.0f, EMBED_SCALE=16.0f;
    uint t=wg.x; if(t>=params.seq) return;
    uint id=ids[t]; if(id>=VOCAB) return;
    uint rwb=id*WPR, rsb=id*NUM_GROUPS;
    for(uint w=tid;w<WPR;w+=WG){ uint packed=bits_buf[rwb+w]; uint cb=w*VPW;
        for(uint v=0u;v<VPW;v++){ uint c=cb+v; float s=scale[rsb+c/GROUP]; float q=float((packed>>(v*BITS))&MASK); y[t*HIDDEN+c]=EMBED_SCALE*s*(q-ZP); } }
}
