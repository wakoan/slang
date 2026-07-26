// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

float srq(float x, float s) {
  // Symmetric per-row quantization. s == 0 means the tensor is unquantized.
  if (s == float(0.0)) {
    return x;
  }
  return clamp(round(x / s), float(-128.0), float(127.0)) * s;
}

// dispatch with threadsPerThreadgroup = (256)
kernel void rmssrqh_b(
    device const float* hidden [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device half* y [[buffer(2)]],
    device float* sum_a [[buffer(3)]],
    constant uint4& p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  threadgroup float sgp[8];
  // Pre-FFN RMSNorm + SRQ over S rows, emitting HALF.
  //
  // rmssrq_69 already handles S rows but writes f32, which is right before
  // attention and wrong here — gateup reads half4. Reproduces oproj_73's DOUBLE
  // rounding exactly, f16(srq(f32(f16(n2)), inScale)): the value is narrowed to
  // f16 before quantizing and again after, and collapsing that to a single round
  // changes results on the boundary.
  float inScale = as_type<float>(p.x);
  uint base = wg.x * uint(1536);
  float acc = float(0.0);
  for (uint j = tid; j < 1536; j += 256) {
    float v = hidden[base + j];
    acc = acc + v * v;
  }
  float s1 = simd_sum(acc);
  if ((tid & uint(31)) == uint(0)) {
    sgp[(tid >> uint(5))] = s1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t1 = float(0.0);
  for (uint i = 0; i < 8; i++) {
    t1 = t1 + sgp[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float rms = rsqrt(t1 / float(1536.0) + float(1e-06));
  float qAcc = float(0.0);
  for (uint j2 = tid; j2 < 1536; j2 += 256) {
    float n2 = hidden[base + j2] * rms * w[j2];
    half qv = half(srq(float(half(n2)), inScale));
    y[base + j2] = qv;
    qAcc = qAcc + float(qv);
  }
  float s2 = simd_sum(qAcc);
  if ((tid & uint(31)) == uint(0)) {
    sgp[(tid >> uint(5))] = s2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t2 = float(0.0);
  for (uint i2 = 0; i2 < 8; i2++) {
    t2 = t2 + sgp[i2];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == uint(0)) {
    sum_a[wg.x] = t2;
  }
}