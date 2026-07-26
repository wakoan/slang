// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

constant uint HEAD_DIM=512u;
constant uint HALF_DIM=256u;

// dispatch with threadsPerThreadgroup = (512)
kernel void qprep_b(
    device const float* q [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* cosT [[buffer(2)]],
    device const float* sinT [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant uint4& p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  threadgroup float red[512];
  threadgroup float qsh[512];
  // q RMSNorm + RoPE for all S*qHeads query vectors, emitted as f16 so the
  // score matrix can be a single GEMM.
  uint h = wg.x;
  uint s = wg.y;
  uint base = (s * p.y + h) * uint(HEAD_DIM);
  float v = q[base + tid];
  qsh[tid] = v;
  red[tid] = v * v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint st = uint(HEAD_DIM) / uint(2);
  while (st > uint(0)) {
    if (tid < st) {
      red[tid] = red[tid] + red[tid + st];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    st = st / uint(2);
  }
  float ns = rsqrt(red[0] / float(HEAD_DIM) + float(1e-06));
  uint rb = (p.x + s) * uint(HALF_DIM);
  if (tid < uint(HALF_DIM)) {
    float n0 = qsh[tid] * ns * w[tid];
    float n1 = qsh[tid + uint(HALF_DIM)] * ns * w[tid + uint(HALF_DIM)];
    float c = cosT[rb + tid];
    float sn = sinT[rb + tid];
    out[base + tid] = half(n0 * c - n1 * sn);
    out[base + tid + uint(HALF_DIM)] = half(n1 * c + n0 * sn);
  }
}