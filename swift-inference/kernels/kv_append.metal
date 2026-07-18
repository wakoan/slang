// Auto-generated from kv_append
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void kv_append(
    device const float* src [[buffer(0)]],
    device float* cache [[buffer(1)]],
    device const uint* dims [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
  uint i = gid.x;
  uint vec_len = dims[0];
  uint pos = dims[1];
  if (i >= vec_len) {
    return;
  }
  cache[pos * vec_len + i] = src[i];
}