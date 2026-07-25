// Auto-generated from attn_softmax
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (4)
kernel void attn_softmax(
    device float* scores [[buffer(0)]],
    device const uint* dims [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
  uint h = gid.x;
  uint n_heads = dims[0];
  uint kv_len = dims[1];
  uint max_seq = dims[2];
  if (h >= n_heads) {
    return;
  }
  uint base = h * max_seq;
  float m = scores[base];
  for (uint t = 1; t < kv_len; t++) {
    m = max(m, scores[base + t]);
  }
  float ssum = 0.0;
  for (uint t = 0; t < kv_len; t++) {
    float e = exp(scores[base + t] - m);
    scores[base + t] = e;
    ssum += e;
  }
  for (uint t = 0; t < kv_len; t++) {
    scores[base + t] = scores[base + t] / ssum;
  }
}