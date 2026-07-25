// Auto-generated from embed_scale_packed
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void embed_scale_packed(
    device const uint* token [[buffer(0)]],
    device const uint* table [[buffer(1)]],
    device float* x_out [[buffer(2)]],
    device const uint* dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
  uint j = gid.x;
  uint hidden = dims[0];
  if (j >= hidden / 2) {
    return;
  }
  uint tok = token[0];
  const auto pair = float2(as_type<half2>(table[tok * (hidden / 2) + j]));
  float scale = sqrt(float(hidden));
  x_out[2 * j] = pair.x * scale;
  x_out[2 * j + 1] = pair.y * scale;
}