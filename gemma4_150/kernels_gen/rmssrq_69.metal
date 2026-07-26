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
kernel void rmssrq_69(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    device float* sum_a [[buffer(3)]],
    constant uint4& p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  threadgroup float sgp[8];
  // Fused weighted RMSNorm + SRQ + sum-of-quantized-activations, one row per
  // workgroup.
  //
  // The sum is not incidental: the quantized matmuls need sum(a) to undo the
  // zero-point offset, so computing it here saves a whole pass over the row.
  //
  // The cross-subgroup combine is inlined rather than factored into a helper
  // because DSL helpers take scalars only — a threadgroup array cannot be passed.
  uint rows = p.x;
  uint stride = p.y;
  if (stride == uint(0)) {
    stride = rows;
  }
  uint row = wg.x + wg.y * stride;
  if (row < rows) {
    float inScale = as_type<float>(p.z);
    uint base = row * uint(1536);
    float acc = float(0.0);
    for (uint i = tid; i < 1536; i += 256) {
      float v = x[base + i];
      acc = acc + v * v;
    }
    float s1 = simd_sum(acc);
    if ((tid & uint(31)) == uint(0)) {
      sgp[(tid >> uint(5))] = s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t1 = float(0.0);
    for (uint k = 0; k < 8; k++) {
      t1 = t1 + sgp[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc = rsqrt(t1 / float(1536.0) + float(1e-06));
    float qAcc = float(0.0);
    for (uint j = tid; j < 1536; j += 256) {
      float q = srq(x[base + j] * sc * w[j], inScale);
      y[base + j] = q;
      qAcc = qAcc + q;
    }
    float s2 = simd_sum(qAcc);
    if ((tid & uint(31)) == uint(0)) {
      sgp[(tid >> uint(5))] = s2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t2 = float(0.0);
    for (uint k2 = 0; k2 < 8; k2++) {
      t2 = t2 + sgp[k2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == uint(0)) {
      sum_a[row] = t2;
    }
  }
}