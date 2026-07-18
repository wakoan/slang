// Auto-generated from rmsnorm_wg
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void rmsnorm_wg(
    device const float* x_in [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* x_out [[buffer(2)]],
    device const uint* dims [[buffer(3)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
  threadgroup float partial[64];
  uint row = wid.x;
  uint li = lid.x;
  uint n_rows = dims[0];
  uint row_len = dims[1];
  float acc = 0.0;
  if (row < n_rows) {
    for (uint j = li; j < row_len; j += 64) {
      float v = x_in[row * row_len + j];
      acc += v * v;
    }
  }
  partial[li] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint s = 32;
  while (s > 0) {
    if (li < s) {
      partial[li] = partial[li] + partial[li + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = s / 2;
  }
  float inv = 1.0 / sqrt(partial[0] / float(row_len) + 1e-06);
  if (row < n_rows) {
    for (uint j = li; j < row_len; j += 64) {
      x_out[row * row_len + j] = x_in[row * row_len + j] * inv * (1.0 + w[j]);
    }
  }
}