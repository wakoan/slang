// Auto-generated from add_vec
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void add_vec(
    device float* a_io [[buffer(0)]],
    device const float* b_in [[buffer(1)]],
    device const uint* dims [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
  uint i = gid.x;
  if (i >= dims[0]) {
    return;
  }
  a_io[i] = a_io[i] + b_in[i];
}