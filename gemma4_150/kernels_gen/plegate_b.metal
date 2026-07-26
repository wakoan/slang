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
kernel void plegate_b(
    device const half* lin [[buffer(0)]],
    device const float* ple [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const float* lut [[buffer(3)]],
    constant uint4& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
  // Tail of plegate_76 once its matmul is a GEMM: SRQ, gelu, times ple.
  if (gid.x < p.w) {
    uint s = gid.x / uint(256);
    uint o = gid.x % uint(256);
    float ls = as_type<float>(p.x);
    float v = srq(float(lin[gid.x]), ls);
    float gv = float(0.0);
    if (ls == float(0.0)) {
      gv = gelu_tanh(v);
    } else {
      gv = lut[uint(clamp(round(v / ls), float(-128.0), float(127.0)) + float(128.0))];
    }
    out[gid.x] = gv * ple[s * p.z + p.y + o];
  }
}