// Auto-generated from rmsnorm_add_simd
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (32)
kernel void rmsnorm_add_simd(
    device const float* x_in [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* x_io [[buffer(2)]],
    device const uint* dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
  uint row = gid.x / 32;
  uint n_rows = dims[0];
  uint row_len = dims[1];
  float acc = 0.0;
  if (row < n_rows) {
    for (uint j = lane; j < row_len; j += 32) {
      float v = x_in[row * row_len + j];
      acc += v * v;
    }
  }
  float total = simd_sum(acc);
  float inv = 1.0 / sqrt(total / float(row_len) + 1e-06);
  if (row < n_rows) {
    for (uint j = lane; j < row_len; j += 32) {
      uint idx = row * row_len + j;
      x_io[idx] = x_io[idx] + x_in[idx] * inv * (1.0 + w[j]);
    }
  }
}