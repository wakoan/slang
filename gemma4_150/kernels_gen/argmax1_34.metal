// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (256)
kernel void argmax1_34(
    device const float* x [[buffer(0)]],
    device float* cand_val [[buffer(1)]],
    device uint* cand_idx [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  threadgroup float wgVal[256];
  threadgroup uint wgIdx[256];
  // Argmax pass 1: best candidate per 1024-wide slice of the logits.
  //
  // Ties break toward the LOWER index, matching the reference — with 262144
  // logits, ties on -inf rows are common enough that an unspecified rule would
  // make the sampled token depend on scheduling.
  uint base = wg.x * uint(1024);
  uint end = min(base + uint(1024), uint(262144));
  float bv = float(-3.4028234663852886e+38);
  uint bi = uint(0);
  uint i = base + tid;
  while (i < end) {
    float v = x[i];
    if (v > bv) {
      bv = v;
      bi = i;
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
    cand_val[wg.x] = wgVal[0];
    cand_idx[wg.x] = wgIdx[0];
  }
}