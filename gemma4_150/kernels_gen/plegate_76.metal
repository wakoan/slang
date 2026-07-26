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

float4 srq4(float4 x, float s) {
  // Vector SRQ. s == 0 means the tensor is unquantized.
  if (s == float(0.0)) {
    return x;
  }
  return clamp(round(x / s), float4(float(-128.0), float(-128.0), float(-128.0), float(-128.0)), float4(float(127.0), float(127.0), float(127.0), float(127.0))) * s;
}

// dispatch with threadsPerThreadgroup = (32)
kernel void plegate_76(
    device const float* a [[buffer(0)]],
    device const uint* codes [[buffer(1)]],
    device const float* row_scale [[buffer(2)]],
    device const float* ple [[buffer(3)]],
    device float* out [[buffer(4)]],
    device const float* gelu_lut [[buffer(5)]],
    constant uint4& params [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  // PLE input gate: int8 (+128-biased) GEMV -> gelu-LUT -> * ple.
  //
  // The +128 bias is undone in the epilogue via fma(s, 255, -128*aSum) rather
  // than per weight — unpack_unorm4x8 divides by 255, so both corrections fold
  // into one expression over the row sum.
  float inScale = as_type<float>(params.x);
  float linOutScale = as_type<float>(params[1]);
  uint o = wg.y * uint(256) + wg.x;
  if (o < uint(256)) {
    float acc = float(0.0);
    float aAcc = float(0.0);
    for (uint wd = tid; wd < 384; wd += 32) {
      uint kb = wd * uint(4);
      float4 av = srq4(float4(a[kb], a[kb + uint(1)], a[kb + uint(2)], a[kb + uint(3)]), inScale);
      aAcc = aAcc + (av.x + av.y + (av.z + av.w));
      acc = acc + dot(unpack_unorm4x8_to_float(codes[o * uint(384) + wd]), av);
    }
    float aSum = simd_sum(aAcc);
    float s = simd_sum(acc);
    if (tid == uint(0)) {
      float v = row_scale[o] * fma(s, float(255.0), float(-128.0) * aSum);
      float qv = srq(v, linOutScale);
      float gv = float(0.0);
      if (linOutScale == float(0.0)) {
        gv = gelu_tanh(qv);
      } else {
        gv = gelu_lut[uint(clamp(round(qv / linOutScale), float(-128.0), float(127.0)) + float(128.0))];
      }
      out[o] = gv * ple[params[2] + o];
    }
  }
}