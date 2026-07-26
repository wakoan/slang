// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (256)
kernel void argmax2_35(
    device const float* cand_val [[buffer(0)]],
    device const uint* cand_idx [[buffer(1)]],
    device uint* out [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {
  threadgroup float wgVal[256];
  threadgroup uint wgIdx[256];
  // Argmax pass 2: the winner among the 256 candidates.
  float bv = float(-3.4028234663852886e+38);
  uint bi = uint(0);
  uint i = tid;
  while (i < uint(256)) {
    float v = cand_val[i];
    uint idx = cand_idx[i];
    if (v > bv) {
      bv = v;
      bi = idx;
    } else if (v == bv) {
      if (idx < bi) {
        bi = idx;
      }
    }
    i = i + uint(256);
  }
  wgVal[tid] = bv;
  wgIdx[tid] = bi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stride = uint(128);
  while (stride > uint(0)) {
    if (tid < stride) {
      uint o = tid + stride;
      if (wgVal[o] > wgVal[tid]) {
        wgVal[tid] = wgVal[o];
        wgIdx[tid] = wgIdx[o];
      } else if (wgVal[o] == wgVal[tid]) {
        if (wgIdx[o] < wgIdx[tid]) {
          wgIdx[tid] = wgIdx[o];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stride = stride / uint(2);
  }
  if (tid == uint(0)) {
    out[0] = wgIdx[0];
  }
}