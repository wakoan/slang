// Auto-generated from matvec_simd_packed
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (32)
kernel void matvec_simd_packed(
    device const uint* w_mat [[buffer(0)]],
    device const float* x_in [[buffer(1)]],
    device float* y_out [[buffer(2)]],
    device const uint* dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
  uint r = gid.x / 32;
  uint n_out = dims[0];
  uint n_in = dims[1];
  uint half_n = n_in / 2;
  float acc = 0.0;
  if (r < n_out) {
    for (uint j = lane; j < half_n; j += 32) {
      const auto pair = float2(as_type<half2>(w_mat[r * half_n + j]));
      acc += pair.x * x_in[2 * j] + pair.y * x_in[2 * j + 1];
    }
  }
  float total = simd_sum(acc);
  if (lane == 0 && r < n_out) {
    y_out[r] = total;
  }
}