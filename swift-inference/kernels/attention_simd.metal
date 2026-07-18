// Auto-generated from attention_simd
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (32)
kernel void attention_simd(
    device const float* q [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    device float* scores [[buffer(3)]],
    device float* out_vec [[buffer(4)]],
    device const uint* dims [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
  uint h = gid.x / 32;
  uint n_heads = dims[0];
  uint head_dim = dims[1];
  uint kv_len = dims[2];
  uint start = dims[3];
  uint max_seq = dims[4];
  if (h >= n_heads) {
    return;
  }
  uint base = h * max_seq;
  for (uint t = start + lane; t < kv_len; t += 32) {
    float dot = 0.0;
    for (uint j = 0; j < head_dim; j++) {
      dot += q[h * head_dim + j] * k_cache[t * head_dim + j];
    }
    scores[base + t] = dot / sqrt(float(head_dim));
  }
  simdgroup_barrier(mem_flags::mem_device);
  float m_local = -3e+38;
  for (uint t = start + lane; t < kv_len; t += 32) {
    m_local = max(m_local, scores[base + t]);
  }
  float m = simd_max(m_local);
  float sum_local = 0.0;
  for (uint t = start + lane; t < kv_len; t += 32) {
    float e = exp(scores[base + t] - m);
    scores[base + t] = e;
    sum_local += e;
  }
  float denom = simd_sum(sum_local);
  simdgroup_barrier(mem_flags::mem_device);
  for (uint d = lane; d < head_dim; d += 32) {
    float acc = 0.0;
    for (uint t = start; t < kv_len; t++) {
      acc += scores[base + t] * v_cache[t * head_dim + d];
    }
    out_vec[h * head_dim + d] = acc / denom;
  }
}