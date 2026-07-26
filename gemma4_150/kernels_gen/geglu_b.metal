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

// dispatch with threadsPerThreadgroup = (256)
kernel void geglu_b(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* out [[buffer(2)]],
    device const float* lut [[buffer(3)]],
    constant uint4& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
  // The geglu tail, once gate/up are GEMMs.
  //
  // gate/up arrive already scaled and zero-point corrected (that folds into the
  // dequantized weights), so only SRQ + gelu + product remain. When the gate is
  // quantized the gelu comes from a 256-entry LUT indexed by the quantization
  // grid — exact for every representable input, and cheaper than the polynomial.
  // The LUT read is inlined because DSL helpers cannot take a buffer.
  if (gid.x < p.w) {
    float gs = as_type<float>(p.x);
    float g = srq(float(gate[gid.x]), gs);
    float u = srq(float(up[gid.x]), as_type<float>(p.y));
    float gv = float(0.0);
    if (gs == float(0.0)) {
      gv = gelu_tanh(g);
    } else {
      gv = lut[uint(clamp(round(g / gs), float(-128.0), float(127.0)) + float(128.0))];
    }
    float dq = gv * u;
    float qs = as_type<float>(p.z);
    if (qs == float(0.0)) {
      out[gid.x] = half(dq);
    } else {
      out[gid.x] = half(clamp(round(dq / qs), float(-128.0), float(127.0)));
    }
  }
}