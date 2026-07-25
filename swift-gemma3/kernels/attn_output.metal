// Auto-generated from attn_output
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void attn_output(
    device const float* scores [[buffer(0)]],
    device const float* v_cache [[buffer(1)]],
    device float* out_vec [[buffer(2)]],
    device const uint* dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
  uint idx = gid.x;
  uint n_heads = dims[0];
  uint head_dim = dims[1];
  uint kv_len = dims[2];
  uint max_seq = dims[3];
  if (idx >= n_heads * head_dim) {
    return;
  }
  uint h = idx / head_dim;
  uint d = idx % head_dim;
  float acc = 0.0;
  for (uint t = 0; t < kv_len; t++) {
    acc += scores[h * max_seq + t] * v_cache[t * head_dim + d];
  }
  out_vec[idx] = acc;
}