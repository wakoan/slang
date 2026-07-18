// Auto-generated from matvec_wg_packed
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void matvec_wg_packed(
    device const uint* w_mat [[buffer(0)]],
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
  uint half_ = n_in / 2;
  float acc = 0.0;
  if (r < n_out) {
    for (uint j = li; j < half_; j += 64) {
      const auto pair = float2(as_type<half2>(w_mat[r * half_ + j]));
      acc += pair.x * x_in[2 * j] + pair.y * x_in[2 * j + 1];
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