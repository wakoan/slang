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
kernel void attnout_b(
    device const half* x [[buffer(0)]],
    device const float* denom [[buffer(1)]],
    device half* y [[buffer(2)]],
    constant uint4& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
  // Tail of the GEMM attention path: divide by the softmax denominator smax_b
  // deferred, then apply the SRQ the o-projection expects.
  if (gid.x < p.y) {
    y[gid.x] = half(srq(float(x[gid.x]) / denom[gid.x / p.z], as_type<float>(p.x)));
  }
}