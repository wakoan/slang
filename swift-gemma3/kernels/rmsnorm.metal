// Auto-generated from rmsnorm
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void rmsnorm(
    device const float* x_in [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* x_out [[buffer(2)]],
    device const uint* dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
  uint idx = gid.x;
  uint n_rows = dims[0];
  uint row_len = dims[1];
  if (idx >= n_rows * row_len) {
    return;
  }
  uint row = idx / row_len;
  uint col = idx % row_len;
  float ss = 0.0;
  for (uint j = 0; j < row_len; j++) {
    float v = x_in[row * row_len + j];
    ss += v * v;
  }
  float inv = 1.0 / sqrt(ss / float(row_len) + 1e-06);
  x_out[idx] = x_in[idx] * inv * (1.0 + w[col]);
}