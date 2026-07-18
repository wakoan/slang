// Auto-generated from attention_fused_sg
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (256)
kernel void attention_fused_sg(
    device const float* q [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    device float* scores [[buffer(3)]],
    device float* out_vec [[buffer(4)]],
    device const uint* dims [[buffer(5)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg_size [[threads_per_simdgroup]]
) {
  threadgroup float smem[8];
  uint h = wid.x;
  uint li = lid.x;
  uint head_dim = dims[1];
  uint kv_len = dims[2];
  uint start = dims[3];
  uint max_seq = dims[4];
  uint base = h * max_seq;
  uint sg_id = li / sg_size;
  uint n_sg = 256 / sg_size;
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
  float m_sg = simd_max(m_local);
  if (lane == 0) {
    smem[sg_id] = m_sg;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = smem[0];
  for (uint i = 1; i < n_sg; i++) {
    m = max(m, smem[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum_local = 0.0;
  for (uint t = start + li; t < kv_len; t += 256) {
    float e = exp(scores[base + t] - m);
    scores[base + t] = e;
    sum_local += e;
  }
  float s_sg = simd_sum(sum_local);
  if (lane == 0) {
    smem[sg_id] = s_sg;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float denom = smem[0];
  for (uint i = 1; i < n_sg; i++) {
    denom += smem[i];
  }
  for (uint d = li; d < head_dim; d += 256) {
    float acc = 0.0;
    for (uint t = start; t < kv_len; t++) {
      acc += scores[base + t] * v_cache[t * head_dim + d];
    }
    out_vec[h * head_dim + d] = acc / denom;
  }
}