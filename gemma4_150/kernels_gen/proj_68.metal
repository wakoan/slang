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

float4 srq4(float4 x, float s) {
  // Vector SRQ. s == 0 means the tensor is unquantized.
  if (s == float(0.0)) {
    return x;
  }
  return clamp(round(x / s), float4(float(-128.0), float(-128.0), float(-128.0), float(-128.0)), float4(float(127.0), float(127.0), float(127.0), float(127.0))) * s;
}

// dispatch with threadsPerThreadgroup = (32)
kernel void proj_68(
    device const float* a [[buffer(0)]],
    device const half* wt [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint4& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  // Dense f16 GEMV for per_layer_model_projection, 8 output rows per
  // workgroup. One subgroup (32 threads) per workgroup, so the reduction is a
  // single subgroupAdd with no threadgroup memory at all.
  //
  // The 8 rows are unrolled into 8 accumulators — the DSL has no local arrays,
  // and the sums are independent, so the arithmetic is unchanged.
  float inScale = as_type<float>(params.x);
  float outScale = as_type<float>(params[1]);
  uint rowBase = (wg.y * uint(1120) + wg.x) * uint(8);
  if (rowBase < uint(8960)) {
    float a0 = float(0.0);
    float a1 = float(0.0);
    float a2 = float(0.0);
    float a3 = float(0.0);
    float a4v = float(0.0);
    float a5 = float(0.0);
    float a6 = float(0.0);
    float a7 = float(0.0);
    for (uint k4 = tid; k4 < 384; k4 += 32) {
      uint kb = k4 * uint(4);
      float4 av = srq4(float4(a[kb], a[kb + uint(1)], a[kb + uint(2)], a[kb + uint(3)]), inScale);
      for (uint r = 0; r < 8; r++) {
        uint o = rowBase + uint(r);
        if (o < uint(8960)) {
          uint wb = o * uint(1536) + kb;
          float4 w4 = float4(float(wt[wb]), float(wt[wb + uint(1)]), float(wt[wb + uint(2)]), float(wt[wb + uint(3)]));
          float d = dot(w4, av);
          if (r == 0) {
            a0 = a0 + d;
          }
          if (r == 1) {
            a1 = a1 + d;
          }
          if (r == 2) {
            a2 = a2 + d;
          }
          if (r == 3) {
            a3 = a3 + d;
          }
          if (r == 4) {
            a4v = a4v + d;
          }
          if (r == 5) {
            a5 = a5 + d;
          }
          if (r == 6) {
            a6 = a6 + d;
          }
          if (r == 7) {
            a7 = a7 + d;
          }
        }
      }
    }
    float s0 = simd_sum(a0);
    float s1 = simd_sum(a1);
    float s2 = simd_sum(a2);
    float s3 = simd_sum(a3);
    float s4 = simd_sum(a4v);
    float s5 = simd_sum(a5);
    float s6 = simd_sum(a6);
    float s7 = simd_sum(a7);
    if (tid == uint(0)) {
      if (rowBase < uint(8960)) {
        out[rowBase] = srq(s0, outScale);
      }
      if (rowBase + uint(1) < uint(8960)) {
        out[rowBase + uint(1)] = srq(s1, outScale);
      }
      if (rowBase + uint(2) < uint(8960)) {
        out[rowBase + uint(2)] = srq(s2, outScale);
      }
      if (rowBase + uint(3) < uint(8960)) {
        out[rowBase + uint(3)] = srq(s3, outScale);
      }
      if (rowBase + uint(4) < uint(8960)) {
        out[rowBase + uint(4)] = srq(s4, outScale);
      }
      if (rowBase + uint(5) < uint(8960)) {
        out[rowBase + uint(5)] = srq(s5, outScale);
      }
      if (rowBase + uint(6) < uint(8960)) {
        out[rowBase + uint(6)] = srq(s6, outScale);
      }
      if (rowBase + uint(7) < uint(8960)) {
        out[rowBase + uint(7)] = srq(s7, outScale);
      }
    }
  }
}