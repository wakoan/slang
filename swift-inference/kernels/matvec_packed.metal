// Auto-generated from matvec_packed
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void matvec_packed(
    device const uint* w_mat [[buffer(0)]],
    device const float* x_in [[buffer(1)]],
    device float* y_out [[buffer(2)]],
    device const uint* dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
  uint r = gid.x;
  uint n_out = dims[0];
  uint n_in = dims[1];
  if (r >= n_out) {
    return;
  }
  uint half_ = n_in / 2;
  float acc = 0.0;
  for (uint j = 0; j < half_; j++) {
    const auto pair = float2(as_type<half2>(w_mat[r * half_ + j]));
    acc += pair.x * x_in[2 * j] + pair.y * x_in[2 * j + 1];
  }
  y_out[r] = acc;
}