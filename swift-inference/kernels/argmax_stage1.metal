// Auto-generated from argmax_stage1
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void argmax_stage1(
    device const float* logits [[buffer(0)]],
    device float* part_val [[buffer(1)]],
    device uint* part_idx [[buffer(2)]],
    device const uint* dims [[buffer(3)]],
    uint3 wid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
  threadgroup float sv[64];
  threadgroup uint si[64];
  uint li = lid.x;
  uint n = dims[0];
  uint n_wgs = dims[1];
  uint stride = n_wgs * 64;
  float best_v = -3e+38;
  uint best_i = 0;
  for (uint i = wid.x * 64 + li; i < n; i += stride) {
    float v = logits[i];
    if (v > best_v) {
      best_v = v;
      best_i = i;
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
    part_val[wid.x] = sv[0];
    part_idx[wid.x] = si[0];
  }
}