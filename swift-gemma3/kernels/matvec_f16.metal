// Auto-generated from matvec_f16
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void matvec_f16(
    device const half* w_mat [[buffer(0)]],
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
  float acc = 0.0;
  for (uint j = 0; j < n_in; j++) {
    acc += float(w_mat[r * n_in + j]) * x_in[j];
  }
  y_out[r] = acc;
}