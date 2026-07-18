// Auto-generated from attn_scores
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (8, 8)
kernel void attn_scores(
    device const float* q [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],
    device float* scores [[buffer(2)]],
    device const uint* dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
  uint t = gid.x;
  uint h = gid.y;
  uint n_heads = dims[0];
  uint head_dim = dims[1];
  uint kv_len = dims[2];
  uint start = dims[3];
  uint max_seq = dims[4];
  if (h >= n_heads || t >= kv_len) {
    return;
  }
  if (t < start) {
    scores[h * max_seq + t] = -1000000000.0;
    return;
  }
  float dot = 0.0;
  for (uint j = 0; j < head_dim; j++) {
    dot += q[h * head_dim + j] * k_cache[t * head_dim + j];
  }
  scores[h * max_seq + t] = dot / sqrt(float(head_dim));
}