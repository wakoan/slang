// Auto-generated from attention_fused
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (256)
kernel void attention_fused(
    device const float* q [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    device float* scores [[buffer(3)]],
    device float* out_vec [[buffer(4)]],
    device const uint* dims [[buffer(5)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
  threadgroup float smem[256];
  uint h = wid.x;
  uint li = lid.x;
  uint head_dim = dims[1];
  uint kv_len = dims[2];
  uint start = dims[3];
  uint max_seq = dims[4];
  uint base = h * max_seq;
  for (uint t = start + li; t < kv_len; t += 256) {
    float dot = 0.0;
    for (uint j = 0; j < head_dim; j++) {
      dot += q[h * head_dim + j] * k_cache[t * head_dim + j];
    }
    scores[base + t] = dot / sqrt(float(head_dim));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m_local = -1e+30;
  for (uint t = start + li; t < kv_len; t += 256) {
    m_local = max(m_local, scores[base + t]);
  }
  smem[li] = m_local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint s = 128;
  while (s > 0) {
    if (li < s) {
      smem[li] = max(smem[li], smem[li + s]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = s / 2;
  }
  float m = smem[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum_local = 0.0;
  for (uint t = start + li; t < kv_len; t += 256) {
    float e = exp(scores[base + t] - m);
    scores[base + t] = e;
    sum_local += e;
  }
  smem[li] = sum_local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  s = 128;
  while (s > 0) {
    if (li < s) {
      smem[li] = smem[li] + smem[li + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = s / 2;
  }
  float denom = smem[0];
  for (uint d = li; d < head_dim; d += 256) {
    float acc = 0.0;
    for (uint t = start; t < kv_len; t++) {
      acc += scores[base + t] * v_cache[t * head_dim + d];
    }
    out_vec[h * head_dim + d] = acc / denom;
  }
}