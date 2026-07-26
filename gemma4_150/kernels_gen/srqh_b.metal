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

// dispatch with threadsPerThreadgroup = (256)
kernel void srqh_b(
    device const float* x [[buffer(0)]],
    device half* y [[buffer(1)]],
    constant uint4& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
  // f32 activations -> f16 for a GEMM's A matrix, optionally SRQ'd.
  // scale == 0 means convert only.
  if (gid.x < p.y) {
    y[gid.x] = half(srq(x[gid.x], as_type<float>(p.x)));
  }
}