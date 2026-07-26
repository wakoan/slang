// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

constant uint WPR=256u;

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
kernel void oproj_73(
    device const float4* a [[buffer(0)]],
    device const uint* bits_buf [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device atomic_uint* pp [[buffer(3)]],
    device float* hidden [[buffer(4)]],
    device const float* w12 [[buffer(5)]],
    device half* y2 [[buffer(6)]],
    device float* sum2 [[buffer(7)]],
    constant uint4& params [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  threadgroup float sgp[8];
  threadgroup uint lastFlag[1];
  // o-proj (4-bit) + post-attn norm-add + pre-FFN norm, all in one dispatch.
  //
  // One subgroup per output row for the GEMV, then the last-arriver runs BOTH
  // norms: RMSNorm the projection, add it to the residual, then RMSNorm again
  // for the FFN input. Two fused norms behind one atomic counter.
  //
  // WPR is a const (256 on sliding layers, 512 on full ones).
  //
  // The two norm passes cache their six per-thread residuals in a PrivateArray,
  // exactly as the reference does. Neither a device re-read nor six scalars is
  // an acceptable substitute: Metal compiles with fast math, so multiply-add
  // contraction follows the expression tree, and both substitutions shifted one
  // element by a ulp — which srq's round() turned into a full quantization step
  // and a different token 98 decode steps later (see PORT_NOTES.md).
  uint sgId = tid / uint(32);
  uint lane = (tid & uint(31));
  float outScale = as_type<float>(params.x);
  uint o = wg.x * uint(8) + sgId;
  float sumQA = float(0.0);
  float sumA = float(0.0);
  for (uint w = lane; w < WPR; w += 32) {
    float4 avc0 = a[w * uint(2)];
    float4 avc1 = a[w * uint(2) + uint(1)];
    sumA = sumA + (avc0.x + avc0.y + avc0.z + avc0.w + (avc1.x + avc1.y + avc1.z + avc1.w));
    if (o < uint(1536)) {
      uint p = bits_buf[o * uint(WPR) + w];
      float4 lo = float4(float4(as_type<uchar4>((p & uint(252645135)))));
      float4 hi = float4(float4(as_type<uchar4>(((p >> uint(4)) & uint(252645135)))));
      sumQA = sumQA + (dot(float4(lo.x, hi.x, lo.y, hi.y), avc0) + dot(float4(lo.z, hi.z, lo.w, hi.w), avc1));
    }
  }
  float rA = simd_sum(sumA);
  float rQA = simd_sum(sumQA);
  if (lane == uint(0)) {
    if (o < uint(1536)) {
      atomic_store_explicit(&pp[o], as_type<uint>(srq(scale[o] * (rQA - float(8.0) * rA), outScale)), memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == uint(0)) {
    uint tk = atomic_fetch_add_explicit(&pp[uint(1536)], uint(1), memory_order_relaxed);
    lastFlag[0] = uint(0);
    if (tk == uint(191)) {
      lastFlag[0] = uint(1);
    }
  }
  if (_wgUniformLoad(lastFlag[0]) == uint(1)) {
    if (tid == uint(0)) {
      atomic_store_explicit(&pp[uint(1536)], uint(0), memory_order_relaxed);
    }
    float inScale2 = as_type<float>(params[1]);
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
      float normed = as_type<float>(atomic_load_explicit(&pp[j], memory_order_relaxed)) * rms1 * w12[j];
      float hv = hidden[j] + normed;
      hidden[j] = hv;
      hloc[e] = hv;
      acc2 = acc2 + hv * hv;
      e = e + uint(1);
    }
    float r2 = simd_sum(acc2);
    if ((tid & uint(31)) == uint(0)) {
      sgp[(tid >> uint(5))] = r2;
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
      float n2 = hloc[e2] * rms2 * w12[uint(1536) + j2];
      half qv = half(srq(float(half(n2)), inScale2));
      y2[j2] = qv;
      qAcc = qAcc + float(qv);
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