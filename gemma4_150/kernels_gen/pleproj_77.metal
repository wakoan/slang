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

float4 srq4(float4 x, float s) {
  // Vector SRQ. s == 0 means the tensor is unquantized.
  if (s == float(0.0)) {
    return x;
  }
  return clamp(round(x / s), float4(float(-128.0), float(-128.0), float(-128.0), float(-128.0)), float4(float(127.0), float(127.0), float(127.0), float(127.0))) * s;
}

// dispatch with threadsPerThreadgroup = (256)
kernel void pleproj_77(
    device const float* a [[buffer(0)]],
    device const uint* codes [[buffer(1)]],
    device const float* row_scale [[buffer(2)]],
    device atomic_uint* pp [[buffer(3)]],
    device float* hidden [[buffer(4)]],
    device const float* w12s [[buffer(5)]],
    device float* y2 [[buffer(6)]],
    device float* sum2 [[buffer(7)]],
    constant uint4& params [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  threadgroup float sgp[8];
  threadgroup uint lastFlag[1];
  // PLE projection (int8) + norm-add * layer_scalar + next norm, one dispatch.
  //
  // Same last-arriver shape as oproj_73, with three per-thread arrays kept as
  // PrivateArrays to match the reference's expression tree exactly — see
  // PORT_NOTES.md for why substituting scalars or a re-read is not safe here.
  //
  // `sv` is the learned per-layer scalar, stored just past the two norm weight
  // vectors in w12s.
  float projInScale = as_type<float>(params[1]);
  float projOutScale = as_type<float>(params[2]);
  uint sgId = tid / uint(32);
  uint lane = (tid & uint(31));
  uint rowBase = wg.x * uint(16) + sgId * uint(2);
  float4 av[2];
  float aAcc = float(0.0);
  for (uint ki = 0; ki < 2; ki++) {
    uint k4 = lane + uint(ki) * uint(32);
    av[ki] = float4(float(0.0), float(0.0), float(0.0), float(0.0));
    if (k4 < uint(64)) {
      uint kb = k4 * uint(4);
      av[ki] = srq4(float4(a[kb], a[kb + uint(1)], a[kb + uint(2)], a[kb + uint(3)]), projInScale);
      aAcc = aAcc + (av[ki].x + av[ki].y + (av[ki].z + av[ki].w));
    }
  }
  float accs[2];
  for (uint r = 0; r < 2; r++) {
    uint o = rowBase + uint(r);
    float acc = float(0.0);
    if (o < uint(1536)) {
      for (uint ki2 = 0; ki2 < 2; ki2++) {
        uint k42 = lane + uint(ki2) * uint(32);
        if (k42 < uint(64)) {
          acc = acc + dot(unpack_unorm4x8_to_float(codes[o * uint(64) + k42]), av[ki2]);
        }
      }
    }
    accs[r] = acc;
  }
  float aSum = simd_sum(aAcc);
  for (uint r2 = 0; r2 < 2; r2++) {
    float s = simd_sum(accs[r2]);
    uint o2 = rowBase + uint(r2);
    if (lane == uint(0)) {
      if (o2 < uint(1536)) {
        atomic_store_explicit(&pp[o2], as_type<uint>(srq(row_scale[o2] * fma(s, float(255.0), float(-128.0) * aSum), projOutScale)), memory_order_relaxed);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == uint(0)) {
    uint tk = atomic_fetch_add_explicit(&pp[uint(1536)], uint(1), memory_order_relaxed);
    lastFlag[0] = uint(0);
    if (tk == uint(95)) {
      lastFlag[0] = uint(1);
    }
  }
  if (_wgUniformLoad(lastFlag[0]) == uint(1)) {
    if (tid == uint(0)) {
      atomic_store_explicit(&pp[uint(1536)], uint(0), memory_order_relaxed);
    }
    float inScale = as_type<float>(params.x);
    float sv = w12s[uint(3072)];
    float acc1 = float(0.0);
    for (uint i = tid; i < 1536; i += 256) {
      float v = as_type<float>(atomic_load_explicit(&pp[i], memory_order_relaxed));
      acc1 = acc1 + v * v;
    }
    float r1 = simd_sum(acc1);
    if ((tid & uint(31)) == uint(0)) {
      sgp[(tid >> uint(5))] = r1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t1 = float(0.0);
    for (uint k1 = 0; k1 < 8; k1++) {
      t1 = t1 + sgp[k1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms1 = rsqrt(t1 / float(1536.0) + float(1e-06));
    float hloc[6];
    float acc2 = float(0.0);
    uint e = uint(0);
    for (uint j = tid; j < 1536; j += 256) {
      float normed = as_type<float>(atomic_load_explicit(&pp[j], memory_order_relaxed)) * rms1 * w12s[j];
      float hv = (hidden[j] + normed) * sv;
      hidden[j] = hv;
      hloc[e] = hv;
      acc2 = acc2 + hv * hv;
      e = e + uint(1);
    }
    float r2v = simd_sum(acc2);
    if ((tid & uint(31)) == uint(0)) {
      sgp[(tid >> uint(5))] = r2v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t2 = float(0.0);
    for (uint k2 = 0; k2 < 8; k2++) {
      t2 = t2 + sgp[k2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms2 = rsqrt(t2 / float(1536.0) + float(1e-06));
    float qAcc = float(0.0);
    uint e2 = uint(0);
    for (uint j2 = tid; j2 < 1536; j2 += 256) {
      float n2 = hloc[e2] * rms2 * w12s[uint(1536) + j2];
      float qv = srq(n2, inScale);
      y2[j2] = qv;
      qAcc = qAcc + qv;
      e2 = e2 + uint(1);
    }
    float r3 = simd_sum(qAcc);
    if ((tid & uint(31)) == uint(0)) {
      sgp[(tid >> uint(5))] = r3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t3 = float(0.0);
    for (uint k3 = 0; k3 < 8; k3++) {
      t3 = t3 + sgp[k3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == uint(0)) {
      sum2[0] = t3;
    }
  }
}