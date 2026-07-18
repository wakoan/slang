// Auto-generated from embed_scale_f16
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void embed_scale_f16(
    device const uint* token [[buffer(0)]],
    device const half* table [[buffer(1)]],
    device float* x_out [[buffer(2)]],
    device const uint* dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
  uint i = gid.x;
  uint hidden = dims[0];
  if (i >= hidden) {
    return;
  }
  uint tok = token[0];
  x_out[i] = float(table[tok * hidden + i]) * sqrt(float(hidden));
}