// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (256)
kernel void smax_b(
    device half* sc [[buffer(0)]],
    device float* denom [[buffer(1)]],
    device const uint* p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  threadgroup float red[256];
  // Causal + sliding-window softmax over a GEMM-produced score matrix.
  //
  // Normalisation is deliberately NOT applied: the per-row denominator goes to
  // denom[] and attnout_b folds it in, saving a third pass over the row.
  uint r = wg.x;
  uint s = r / p[1];
  uint qPos = p[0] + s;
  uint maxKj = qPos + uint(1);
  uint minKj = uint(0);
  if (p[2] > uint(0)) {
    if (qPos + uint(1) > p[2]) {
      minKj = qPos + uint(1) - p[2];
    }
  }
  if (maxKj > p[3]) {
    maxKj = p[3];
  }
  uint base = r * p[4];
  float m = float(-3.4028234663852886e+38);
  uint j = minKj + tid;
  while (j < maxKj) {
    m = max(m, float(sc[base + j]));
    j = j + uint(256);
  }
  red[tid] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint st = uint(128);
  while (st > uint(0)) {
    if (tid < st) {
      red[tid] = max(red[tid], red[tid + st]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    st = st / uint(2);
  }
  m = red[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = float(0.0);
  uint j2 = tid;
  while (j2 < p[3]) {
    float e = float(0.0);
    if (j2 >= minKj) {
      if (j2 < maxKj) {
        e = exp(float(sc[base + j2]) - m);
        tot = tot + e;
      }
    }
    sc[base + j2] = half(e);
    j2 = j2 + uint(256);
  }
  red[tid] = tot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint st2 = uint(128);
  while (st2 > uint(0)) {
    if (tid < st2) {
      red[tid] = red[tid] + red[tid + st2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    st2 = st2 / uint(2);
  }
  if (tid == uint(0)) {
    denom[r] = red[0];
  }
}