// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

float tanh_safe(float x) {
  // tanh saturated outside +-10 — matches the reference kernels, and keeps
  // the polynomial from overflowing before tanh flattens anyway.
  if (x > float(10.0)) {
    return float(1.0);
  }
  if (x < float(-10.0)) {
    return float(-1.0);
  }
  return tanh(x);
}

float gelu_tanh(float v) {
  return float(0.5) * v * (float(1.0) + tanh_safe(float(0.7978845608028654) * (v + float(0.044715) * v * v * v)));
}

float srq(float x, float s) {
  // Symmetric per-row quantization. s == 0 means the tensor is unquantized.
  if (s == float(0.0)) {
    return x;
  }
  return clamp(round(x / s), float(-128.0), float(127.0)) * s;
}

// dispatch with threadsPerThreadgroup = (64)
kernel void gateup_74(
    device const half4* hidden [[buffer(0)]],
    device const uint* gate_bits [[buffer(1)]],
    device const float* gate_scale [[buffer(2)]],
    device const uint* up_bits [[buffer(3)]],
    device const float* up_scale [[buffer(4)]],
    device const float* sum_a [[buffer(5)]],
    device half* out [[buffer(6)]],
    device const float* gelu_lut [[buffer(7)]],
    constant uint4& params [[buffer(8)]],
    uint lidx [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  // Fused gate/up + geglu (4-bit), virtual-subgroup GEMV.
  //
  // Two subgroups per workgroup, each owning 4 output rows; gate and up are
  // accumulated together so the shared activation chunk is read once for both.
  float gOut = as_type<float>(params.x);
  float uOut = as_type<float>(params[1]);
  uint sgId = lidx / uint(32);
  uint tid = (lidx & uint(31));
  uint rowBase = (wg.x * uint(2) + sgId) * uint(4);
  float gAcc[4];
  float uAcc[4];
  for (uint r0 = 0; r0 < 4; r0++) {
    gAcc[r0] = float(0.0);
    uAcc[r0] = float(0.0);
  }
  for (uint wd = tid; wd < 192; wd += 32) {
    float4 a0 = float4(hidden[wd * uint(2)]);
    float4 a1 = float4(hidden[wd * uint(2) + uint(1)]);
    for (uint r = 0; r < 4; r++) {
      uint o = rowBase + uint(r);
      if (o < uint(6144)) {
        uint pg = gate_bits[o * uint(192) + wd];
        uint pu = up_bits[o * uint(192) + wd];
        float4 glo = unpack_unorm4x8_to_float((pg & uint(252645135)));
        float4 ghi = unpack_unorm4x8_to_float(((pg >> uint(4)) & uint(252645135)));
        gAcc[r] = gAcc[r] + (dot(float4(glo.x, ghi.x, glo.y, ghi.y), a0) + dot(float4(glo.z, ghi.z, glo.w, ghi.w), a1));
        float4 ulo = unpack_unorm4x8_to_float((pu & uint(252645135)));
        float4 uhi = unpack_unorm4x8_to_float(((pu >> uint(4)) & uint(252645135)));
        uAcc[r] = uAcc[r] + (dot(float4(ulo.x, uhi.x, ulo.y, uhi.y), a0) + dot(float4(ulo.z, uhi.z, ulo.w, uhi.w), a1));
      }
    }
  }
  float aSum = sum_a[0];
  for (uint r2 = 0; r2 < 4; r2++) {
    float gS = simd_sum(gAcc[r2]);
    float uS = simd_sum(uAcc[r2]);
    if (tid == uint(0)) {
      uint o2 = rowBase + uint(r2);
      if (o2 < uint(6144)) {
        float g = srq(gate_scale[o2] * fma(gS, float(255.0), -(float(8.0) * aSum)), gOut);
        float u = srq(up_scale[o2] * fma(uS, float(255.0), -(float(8.0) * aSum)), uOut);
        float gv = float(0.0);
        if (gOut == float(0.0)) {
          gv = gelu_tanh(g);
        } else {
          gv = gelu_lut[uint(clamp(round(g / gOut), float(-128.0), float(127.0)) + float(128.0))];
        }
        float dq = gv * u;
        float qs = as_type<float>(params[2]);
        if (qs == float(0.0)) {
          out[o2] = half(dq);
        } else {
          out[o2] = half(clamp(round(dq / qs), float(-128.0), float(127.0)));
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}