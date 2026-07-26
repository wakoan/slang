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
kernel void rmsadd_b(
    device const half* x [[buffer(0)]],
    device const float* nw [[buffer(1)]],
    device float* hidden [[buffer(2)]],
    constant uint4& p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  threadgroup float sgp[8];
  threadgroup float dsh[1536];
  // Residual-add + RMSNorm tail, as its own pass over S rows.
  //
  // In decode this tail hides behind down_75/oproj_73/pleproj_77's atomic
  // last-arriver counter, which saves a dispatch. Batching that would need S
  // counters, so prefill pays for a separate pass instead.
  float inScale = as_type<float>(p.x);
  float sv = as_type<float>(p.y);
  uint base = wg.x * uint(1536);
  float acc = float(0.0);
  for (uint j = tid; j < 1536; j += 256) {
    float d = srq(float(x[base + j]), inScale);
    dsh[j] = d;
    acc = acc + d * d;
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
  for (uint j2 = tid; j2 < 1536; j2 += 256) {
    hidden[base + j2] = (hidden[base + j2] + dsh[j2] * rms * nw[j2]) * sv;
  }
}