#include <metal_stdlib>
using namespace metal;
// from 70_srq.wgsl — fused qkv (4-bit) GEMV. shape consts patched per layer.
// dispatch with threadsPerThreadgroup = (32) ; groups = TOTAL_WGS
struct P70 { float qOutScale; float kOutScale; float vOutScale; uint p; };
inline float srq(float x, float s){ if(s==0.0f) return x; return clamp(round(x/s),-128.0f,127.0f)*s; }
kernel void qkv_70(
    device const float4* a [[buffer(0)]], device const uint* q_bits [[buffer(1)]],
    device const uint* k_bits [[buffer(2)]], device const uint* v_bits [[buffer(3)]],
    device const float* scales [[buffer(4)]], device const float* sum_a [[buffer(5)]],
    device float* out_q [[buffer(6)]], device float* out_k [[buffer(7)]], device float* out_v [[buffer(8)]],
    constant P70& params [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]], uint3 wg [[threadgroup_position_in_grid]])
{
    const uint Q_OUT=2048u, KV_OUT=256u, WPR=192u, CHUNKS=2u, WG=32u, N_ROWS=2u;
    const uint Q_WGS=1024u, KV_WGS=128u, TOTAL_WGS=1280u, GRID_X=1280u;
    const float ZP=8.0f;
    uint wgId=wg.y*GRID_X+wg.x;
    if(wgId>=TOTAL_WGS) return;
    float sumQA[N_ROWS]; for(uint r=0u;r<N_ROWS;r++) sumQA[r]=0.0f;
    float rA;
    if(wgId<Q_WGS){
        uint rowBase=wgId*N_ROWS;
        for(uint w=tid; w<WPR; w+=WG){ float4 avc0=a[w*CHUNKS+0u], avc1=a[w*CHUNKS+1u];
            for(uint r=0u;r<N_ROWS;r++){ uint o=rowBase+r; if(o<Q_OUT){ uint p=q_bits[o*WPR+w];
                float4 lo=float4(as_type<uchar4>(p&0x0F0F0F0Fu)), hi=float4(as_type<uchar4>((p>>4u)&0x0F0F0F0Fu));
                sumQA[r]+=dot(float4(lo.x,hi.x,lo.y,hi.y),avc0)+dot(float4(lo.z,hi.z,lo.w,hi.w),avc1); } } }
        rA=sum_a[0];
        for(uint r=0u;r<N_ROWS;r++){ float rQA=simd_sum(sumQA[r]); uint o=rowBase+r; if(tid==0u&&o<Q_OUT) out_q[o]=srq(scales[o]*(rQA-ZP*rA),params.qOutScale); }
    } else if(wgId<Q_WGS+KV_WGS){
        uint rowBase=(wgId-Q_WGS)*N_ROWS;
        for(uint w=tid; w<WPR; w+=WG){ float4 avc0=a[w*CHUNKS+0u], avc1=a[w*CHUNKS+1u];
            for(uint r=0u;r<N_ROWS;r++){ uint o=rowBase+r; if(o<KV_OUT){ uint p=k_bits[o*WPR+w];
                float4 lo=float4(as_type<uchar4>(p&0x0F0F0F0Fu)), hi=float4(as_type<uchar4>((p>>4u)&0x0F0F0F0Fu));
                sumQA[r]+=dot(float4(lo.x,hi.x,lo.y,hi.y),avc0)+dot(float4(lo.z,hi.z,lo.w,hi.w),avc1); } } }
        rA=sum_a[0];
        for(uint r=0u;r<N_ROWS;r++){ float rQA=simd_sum(sumQA[r]); uint o=rowBase+r; if(tid==0u&&o<KV_OUT) out_k[o]=srq(scales[Q_OUT+o]*(rQA-ZP*rA),params.kOutScale); }
    } else {
        uint rowBase=(wgId-Q_WGS-KV_WGS)*N_ROWS;
        for(uint w=tid; w<WPR; w+=WG){ float4 avc0=a[w*CHUNKS+0u], avc1=a[w*CHUNKS+1u];
            for(uint r=0u;r<N_ROWS;r++){ uint o=rowBase+r; if(o<KV_OUT){ uint p=v_bits[o*WPR+w];
                float4 lo=float4(as_type<uchar4>(p&0x0F0F0F0Fu)), hi=float4(as_type<uchar4>((p>>4u)&0x0F0F0F0Fu));
                sumQA[r]+=dot(float4(lo.x,hi.x,lo.y,hi.y),avc0)+dot(float4(lo.z,hi.z,lo.w,hi.w),avc1); } } }
        rA=sum_a[0];
        for(uint r=0u;r<N_ROWS;r++){ float rQA=simd_sum(sumQA[r]); uint o=rowBase+r; if(tid==0u&&o<KV_OUT) out_v[o]=srq(scales[Q_OUT+KV_OUT+o]*(rQA-ZP*rA),params.vOutScale); }
    }
}
