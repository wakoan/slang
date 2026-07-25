// Auto-generated from geglu
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

float gelu(float g) {
  float inner = clamp(0.7978845608028654 * (g + 0.044715 * g * g * g), -20.0, 20.0);
  return 0.5 * g * (1.0 + tanh(inner));
}

// dispatch with threadsPerThreadgroup = (64)
kernel void geglu(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* h_out [[buffer(2)]],
    device const uint* dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
  uint i = gid.x;
  uint n = dims[0];
  if (i >= n) {
    return;
  }
  h_out[i] = gelu(gate[i]) * up[i];
}