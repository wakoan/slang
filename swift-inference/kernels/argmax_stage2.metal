// Auto-generated from argmax_stage2
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void argmax_stage2(
    device const float* part_val [[buffer(0)]],
    device const uint* part_idx [[buffer(1)]],
    device uint* token [[buffer(2)]],
    device uint* out_tokens [[buffer(3)]],
    device uint* counter [[buffer(4)]],
    device const uint* dims [[buffer(5)]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
  threadgroup float sv[64];
  threadgroup uint si[64];
  uint li = lid.x;
  uint n_parts = dims[0];
  float best_v = -3e+38;
  uint best_i = 0;
  for (uint i = li; i < n_parts; i += 64) {
    if (part_val[i] > best_v) {
      best_v = part_val[i];
      best_i = part_idx[i];
    }
  }
  sv[li] = best_v;
  si[li] = best_i;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint s = 32;
  while (s > 0) {
    if (li < s) {
      if (sv[li + s] > sv[li]) {
        sv[li] = sv[li + s];
        si[li] = si[li + s];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = s / 2;
  }
  if (li == 0) {
    uint tok = si[0];
    token[0] = tok;
    out_tokens[counter[0]] = tok;
    counter[0] = counter[0] + 1;
  }
}