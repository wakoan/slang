#include <metal_stdlib>
using namespace metal;
// from 33_srq.wgsl — dense 2-bit block-major logits GEMV (thread-per-column)
// dispatch with threadsPerThreadgroup = (128) ; groups = N/128 = 2048
struct P33 { float inScale; float outScale; uint p0; uint p1; };
inline float srq33(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
inline float block_dot(uint4 bv, uint aBase, threadgroup float* at) {
    float s=0.0f;
    for(uint j=0u;j<4u;j++){ uint packed=bv[j];
        float4 d0=float4(as_type<uchar4>(packed & 0x03030303u)) - float4(2.0f);
        float4 d1=float4(as_type<uchar4>((packed>>2u)&0x03030303u)) - float4(2.0f);
        float4 d2=float4(as_type<uchar4>((packed>>4u)&0x03030303u)) - float4(2.0f);
        float4 d3=float4(as_type<uchar4>((packed>>6u)&0x03030303u)) - float4(2.0f);
        uint b=aBase+j*16u;
        s += dot(float4(d0.x,d1.x,d2.x,d3.x), float4(at[b],at[b+1u],at[b+2u],at[b+3u]))
           + dot(float4(d0.y,d1.y,d2.y,d3.y), float4(at[b+4u],at[b+5u],at[b+6u],at[b+7u]))
           + dot(float4(d0.z,d1.z,d2.z,d3.z), float4(at[b+8u],at[b+9u],at[b+10u],at[b+11u]))
           + dot(float4(d0.w,d1.w,d2.w,d3.w), float4(at[b+12u],at[b+13u],at[b+14u],at[b+15u]));
    }
    return s;
}
kernel void logits_33(
    device const float* a [[buffer(0)]], device const uint4* bits_buf [[buffer(1)]],
    device const float* scale [[buffer(2)]], device float* out [[buffer(3)]],
    constant P33& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint K=1536u, N=262144u, TILE_N=128u, VPV=64u, NUM_BLK=24u, GRID_X=2048u;
    threadgroup float at[K];
    uint col=(wg.y*GRID_X+wg.x)*TILE_N+tid;
    for(uint id=tid;id<K;id+=TILE_N) at[id]=srq33(a[id], params.inScale);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(col<N){ float acc=0.0f; for(uint blk=0u;blk<NUM_BLK;blk++) acc+=block_dot(bits_buf[blk*N+col], blk*VPV, at); out[col]=srq33(scale[col]*acc, params.outScale); }
}
