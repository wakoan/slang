// Auto-generated from matvec_wg
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void matvec_wg(
    device const float* w_mat [[buffer(0)]],
    device const float* x_in [[buffer(1)]],
    device float* y_out [[buffer(2)]],
    device const uint* dims [[buffer(3)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
  threadgroup float partial[64];
  uint r = wid.x;
  uint li = lid.x;
  uint n_out = dims[0];
  uint n_in = dims[1];
  float acc = 0.0;
  if (r < n_out) {
    for (uint j = li; j < n_in; j += 64) {
      acc += w_mat[r * n_in + j] * x_in[j];
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
  if (li == 0 && r < n_out) {
    y_out[r] = partial[0];
  }
}