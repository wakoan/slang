// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

template <typename T> inline T _wgUniformLoad(threadgroup T& v) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return v;
}

float srq(float x, float s) {
  // Symmetric per-row quantization. s == 0 means the tensor is unquantized.
  if (s == float(0.0)) {
    return x;
  }
  return clamp(round(x / s), float(-128.0), float(127.0)) * s;
}

// dispatch with threadsPerThreadgroup = (256)
kernel void down_75(
    device const half4* a [[buffer(0)]],
    device const uint* bits_buf [[buffer(1)]],
    device atomic_uint* pp [[buffer(2)]],
    device const float* scale [[buffer(3)]],
    device float* hidden [[buffer(4)]],
    device const float* nw [[buffer(5)]],
    constant uint4& params [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  threadgroup float4 sgq[8];
  threadgroup float sgs[8];
  threadgroup float dsh[1536];
  threadgroup uint lastFlag[1];
  // Fused 4-bit down-projection + post-FFN residual norm-add, in ONE dispatch.
  //
  // Each workgroup owns N_ROWS=4 output rows, publishes them through the atomic
  // buffer, and takes a ticket; whichever workgroup draws the last ticket
  // (TOTAL_WGS-1) re-reads every partial and finishes the RMSNorm-add itself.
  // That is what saves a second dispatch and a round trip — and it is the reason
  // the DSL needed atomics at all.
  //
  // The counter self-resets at [CTR] so the next token starts clean without a
  // host-side clear.
  //
  // The four row accumulators live in a PrivateArray, matching the reference's
  // `float q[4]`. They were scalars at first, which passes both gates here — but
  // it is the same substitution that silently broke oproj_73, and there is no
  // gain to offset the risk (see PORT_NOTES.md).
  uint rowBase = wg.x * uint(4);
  float inScale = as_type<float>(params.x);
  float q[4];
  for (uint r0 = 0; r0 < 4; r0++) {
    q[r0] = float(0.0);
  }
  float sumA = float(0.0);
  for (uint w = tid; w < 768; w += 256) {
    float4 av0 = float4(a[w * uint(2)]);
    float4 av1 = float4(a[w * uint(2) + uint(1)]);
    sumA = sumA + (av0.x + av0.y + av0.z + av0.w + (av1.x + av1.y + av1.z + av1.w));
    for (uint r = 0; r < 4; r++) {
      uint o = rowBase + uint(r);
      if (o < uint(1536)) {
        uint p = bits_buf[o * uint(768) + w];
        float4 lo = unpack_unorm4x8_to_float((p & uint(252645135)));
        float4 hi = unpack_unorm4x8_to_float(((p >> uint(4)) & uint(252645135)));
        q[r] = q[r] + (dot(float4(lo.x, hi.x, lo.y, hi.y), av0) + dot(float4(lo.z, hi.z, lo.w, hi.w), av1));
      }
    }
  }
  float4 red = simd_sum(float4(q[0], q[1], q[2], q[3]));
  float redA = simd_sum(sumA);
  if ((tid & uint(31)) == uint(0)) {
    sgq[(tid >> uint(5))] = red;
    sgs[(tid >> uint(5))] = redA;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == uint(0)) {
    float4 tot = float4(float(0.0), float(0.0), float(0.0), float(0.0));
    float aSum = float(0.0);
    for (uint i = 0; i < 8; i++) {
      tot = tot + sgq[i];
      aSum = aSum + sgs[i];
    }
    float outScale = as_type<float>(params[1]);
    float zpA = float(8.0) * aSum;
    for (uint r2 = 0; r2 < 4; r2++) {
      uint o2 = rowBase + uint(r2);
      if (o2 < uint(1536)) {
        float dv = srq(scale[o2] * (inScale * fma(tot[r2], float(255.0), -zpA)), outScale);
        atomic_store_explicit(&pp[o2], as_type<uint>(dv), memory_order_relaxed);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == uint(0)) {
    uint ticket = atomic_fetch_add_explicit(&pp[uint(1536)], uint(1), memory_order_relaxed);
    lastFlag[0] = uint(0);
    if (ticket == uint(383)) {
      lastFlag[0] = uint(1);
    }
  }
  if (_wgUniformLoad(lastFlag[0]) == uint(1)) {
    if (tid == uint(0)) {
      atomic_store_explicit(&pp[uint(1536)], uint(0), memory_order_relaxed);
    }
    float acc = float(0.0);
    for (uint o2 = tid; o2 < 1536; o2 += 256) {
      float d = as_type<float>(atomic_load_explicit(&pp[o2], memory_order_relaxed));
      dsh[o2] = d;
      acc = acc + d * d;
    }
    float s1 = simd_sum(acc);
    if ((tid & uint(31)) == uint(0)) {
      sgs[(tid >> uint(5))] = s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t1 = float(0.0);
    for (uint i2 = 0; i2 < 8; i2++) {
      t1 = t1 + sgs[i2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms = rsqrt(t1 / float(1536.0) + float(1e-06));
    for (uint o3 = tid; o3 < 1536; o3 += 256) {
      hidden[o3] = hidden[o3] + dsh[o3] * rms * nw[o3];
    }
  }
}