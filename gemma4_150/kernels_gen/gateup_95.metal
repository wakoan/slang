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
kernel void gateup_95(
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
  // gateup_74's 2-bit twin for the wide-MLP layers (intermediate 12288).
  //
  // Two rows per subgroup instead of four — the rows are twice as many, so the
  // grid grows rather than the per-thread work.
  float gOut = as_type<float>(params.x);
  float uOut = as_type<float>(params[1]);
  uint sgId = lidx / uint(32);
  uint tid = (lidx & uint(31));
  uint rowBase = (wg.x * uint(2) + sgId) * uint(2);
  float gAcc[2];
  float uAcc[2];
  for (uint r0 = 0; r0 < 2; r0++) {
    gAcc[r0] = float(0.0);
    uAcc[r0] = float(0.0);
  }
  for (uint wd = tid; wd < 96; wd += 32) {
    float4 a0 = float4(hidden[wd * uint(4)]);
    float4 a1 = float4(hidden[wd * uint(4) + uint(1)]);
    float4 a2 = float4(hidden[wd * uint(4) + uint(2)]);
    float4 a3 = float4(hidden[wd * uint(4) + uint(3)]);
    for (uint r = 0; r < 2; r++) {
      uint o = rowBase + uint(r);
      if (o < uint(12288)) {
        uint pg = gate_bits[o * uint(96) + wd];
        uint pu = up_bits[o * uint(96) + wd];
        float4 g0 = unpack_unorm4x8_to_float((pg & uint(50529027)));
        float4 g1 = unpack_unorm4x8_to_float(((pg >> uint(2)) & uint(50529027)));
        float4 g2 = unpack_unorm4x8_to_float(((pg >> uint(4)) & uint(50529027)));
        float4 g3 = unpack_unorm4x8_to_float(((pg >> uint(6)) & uint(50529027)));
        gAcc[r] = gAcc[r] + (dot(float4(g0.x, g1.x, g2.x, g3.x), a0) + dot(float4(g0.y, g1.y, g2.y, g3.y), a1) + dot(float4(g0.z, g1.z, g2.z, g3.z), a2) + dot(float4(g0.w, g1.w, g2.w, g3.w), a3));
        float4 u0 = unpack_unorm4x8_to_float((pu & uint(50529027)));
        float4 u1 = unpack_unorm4x8_to_float(((pu >> uint(2)) & uint(50529027)));
        float4 u2 = unpack_unorm4x8_to_float(((pu >> uint(4)) & uint(50529027)));
        float4 u3 = unpack_unorm4x8_to_float(((pu >> uint(6)) & uint(50529027)));
        uAcc[r] = uAcc[r] + (dot(float4(u0.x, u1.x, u2.x, u3.x), a0) + dot(float4(u0.y, u1.y, u2.y, u3.y), a1) + dot(float4(u0.z, u1.z, u2.z, u3.z), a2) + dot(float4(u0.w, u1.w, u2.w, u3.w), a3));
      }
    }
  }
  float aSum = sum_a[0];
  for (uint r2 = 0; r2 < 2; r2++) {
    float gS = simd_sum(gAcc[r2]);
    float uS = simd_sum(uAcc[r2]);
    if (tid == uint(0)) {
      uint o2 = rowBase + uint(r2);
      if (o2 < uint(12288)) {
        float g = srq(gate_scale[o2] * fma(gS, float(255.0), -(float(2.0) * aSum)), gOut);
        float u = srq(up_scale[o2] * fma(uS, float(255.0), -(float(2.0) * aSum)), uOut);
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