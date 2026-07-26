// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

float srq(float x, float s) {
  // Symmetric per-row quantization. s == 0 means the tensor is unquantized.
  if (s == float(0.0)) {
    return x;
  }
  return clamp(round(x / s), float(-128.0), float(127.0)) * s;
}

// dispatch with threadsPerThreadgroup = (128)
kernel void logits_33(
    device const float* a [[buffer(0)]],
    device const uint4* bits_buf [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  threadgroup float at[1536];
  // Dense 2-bit block-major logits GEMV, one thread per output column.
  //
  // The weights are block-major — 24 blocks of 64 values, each block a uint4 —
  // so a thread reads its whole column with no reduction at all, which is why
  // 262144 outputs are affordable. The activation row is staged in workgroup
  // memory once and reused by all 128 columns.
  //
  // block_dot is inlined because DSL helpers cannot take a threadgroup pointer.
  float inScale = as_type<float>(params.x);
  uint col = (wg.y * uint(2048) + wg.x) * uint(128) + tid;
  for (uint i = tid; i < 1536; i += 128) {
    at[i] = srq(a[i], inScale);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (col < uint(262144)) {
    float acc = float(0.0);
    for (uint blk = 0; blk < 24; blk++) {
      uint4 bv = bits_buf[uint(blk) * uint(262144) + col];
      uint aBase = uint(blk) * uint(64);
      float s = float(0.0);
      for (uint j = 0; j < 4; j++) {
        uint packed = bv[j];
        float4 d0 = float4(float4(as_type<uchar4>((packed & uint(50529027))))) - float4(float(2.0), float(2.0), float(2.0), float(2.0));
        float4 d1 = float4(float4(as_type<uchar4>(((packed >> uint(2)) & uint(50529027))))) - float4(float(2.0), float(2.0), float(2.0), float(2.0));
        float4 d2 = float4(float4(as_type<uchar4>(((packed >> uint(4)) & uint(50529027))))) - float4(float(2.0), float(2.0), float(2.0), float(2.0));
        float4 d3 = float4(float4(as_type<uchar4>(((packed >> uint(6)) & uint(50529027))))) - float4(float(2.0), float(2.0), float(2.0), float(2.0));
        uint b = aBase + uint(j) * uint(16);
        s = s + (dot(float4(d0.x, d1.x, d2.x, d3.x), float4(at[b], at[b + uint(1)], at[b + uint(2)], at[b + uint(3)])) + dot(float4(d0.y, d1.y, d2.y, d3.y), float4(at[b + uint(4)], at[b + uint(5)], at[b + uint(6)], at[b + uint(7)])) + dot(float4(d0.z, d1.z, d2.z, d3.z), float4(at[b + uint(8)], at[b + uint(9)], at[b + uint(10)], at[b + uint(11)])) + dot(float4(d0.w, d1.w, d2.w, d3.w), float4(at[b + uint(12)], at[b + uint(13)], at[b + uint(14)], at[b + uint(15)])));
      }
      acc = acc + s;
    }
    out[col] = srq(scale[col] * acc, as_type<float>(params[1]));
  }
}