// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

constant uint HD=256u;
constant uint HALF=128u;

// dispatch with threadsPerThreadgroup = (256)
kernel void kvnorm(
    device const float* ink [[buffer(0)]],
    device const float* inv [[buffer(1)]],
    device const float* knorm [[buffer(2)]],
    device const float* cosT [[buffer(3)]],
    device const float* sinT [[buffer(4)]],
    device float* kcache [[buffer(5)]],
    device float* vcache [[buffer(6)]],
    constant uint4& p [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]]
) {
  threadgroup float rk[512];
  threadgroup float rv[512];
  // k: RMSNorm * knorm + split-half RoPE; v: scale-free RMSNorm; -> caches.
  //
  // The first SHAPE-PARAMETERIZED kernel: head_dim is 256 on sliding layers and
  // 512 on full ones. HD/HALF are `consts`, folded at translate time, so both
  // variants come from this one source — replacing the runner's string-patching
  // of the shader text, where "HD" could match a substring.
  //
  // The workgroup arrays are sized for the widest variant because an annotation
  // is evaluated once at definition; the narrow variant simply uses a prefix.
  float ko = ink[tid];
  float vo = inv[tid];
  rk[tid] = ko * ko;
  rv[tid] = vo * vo;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint s = uint(HD) / uint(2);
  while (s > uint(0)) {
    if (tid < s) {
      rk[tid] = rk[tid] + rk[tid + s];
      rv[tid] = rv[tid] + rv[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = s / uint(2);
  }
  float rmsk = rsqrt(rk[0] / float(HD) + float(1e-06));
  float rmsv = rsqrt(rv[0] / float(HD) + float(1e-06));
  vcache[p.x + tid] = vo * rmsv;
  if (tid < uint(HALF)) {
    float n0 = ink[tid] * rmsk * knorm[tid];
    float n1 = ink[tid + uint(HALF)] * rmsk * knorm[tid + uint(HALF)];
    float c = cosT[tid];
    float sn = sinT[tid];
    kcache[p.x + tid] = n0 * c - n1 * sn;
    kcache[p.x + tid + uint(HALF)] = n1 * c + n0 * sn;
  }
}