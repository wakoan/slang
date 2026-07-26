// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

constant uint HD=256u;
constant uint HALF=128u;

float srq(float x, float s) {
  // Symmetric per-row quantization. s == 0 means the tensor is unquantized.
  if (s == float(0.0)) {
    return x;
  }
  return clamp(round(x / s), float(-128.0), float(127.0)) * s;
}

// dispatch with threadsPerThreadgroup = (256)
kernel void kvnorm_b(
    device const half* ink [[buffer(0)]],
    device const half* inv [[buffer(1)]],
    device const float* knorm [[buffer(2)]],
    device const float* cosT [[buffer(3)]],
    device const float* sinT [[buffer(4)]],
    device float* kcache [[buffer(5)]],
    device float* vcache [[buffer(6)]],
    constant uint4& p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  threadgroup float rk[512];
  threadgroup float rv[512];
  threadgroup float ksh[512];
  // kvnorm for S positions at once — one workgroup per token, rope indexed by
  // absolute position rather than pre-offset.
  uint s = wg.x;
  uint ib = s * uint(HD);
  uint pos = p.x + s;
  uint co = pos * uint(HALF);
  uint cache = pos * uint(HD);
  float ko = srq(float(ink[ib + tid]), as_type<float>(p.y));
  float vo = srq(float(inv[ib + tid]), as_type<float>(p.z));
  ksh[tid] = ko;
  rk[tid] = ko * ko;
  rv[tid] = vo * vo;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint st = uint(HD) / uint(2);
  while (st > uint(0)) {
    if (tid < st) {
      rk[tid] = rk[tid] + rk[tid + st];
      rv[tid] = rv[tid] + rv[tid + st];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    st = st / uint(2);
  }
  float rmsk = rsqrt(rk[0] / float(HD) + float(1e-06));
  float rmsv = rsqrt(rv[0] / float(HD) + float(1e-06));
  vcache[cache + tid] = vo * rmsv;
  if (tid < uint(HALF)) {
    float n0 = ksh[tid] * rmsk * knorm[tid];
    float n1 = ksh[tid + uint(HALF)] * rmsk * knorm[tid + uint(HALF)];
    float c = cosT[co + tid];
    float sn = sinT[co + tid];
    kcache[cache + tid] = n0 * c - n1 * sn;
    kcache[cache + tid + uint(HALF)] = n1 * c + n0 * sn;
  }
}