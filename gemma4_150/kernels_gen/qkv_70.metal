// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

constant uint Q_OUT=2048u;
constant uint KV_OUT=256u;
constant uint Q_WGS=1024u;
constant uint KV_WGS=128u;
constant uint TOTAL_WGS=1280u;
constant uint GRID_X=1280u;

float srq(float x, float s) {
  // Symmetric per-row quantization. s == 0 means the tensor is unquantized.
  if (s == float(0.0)) {
    return x;
  }
  return clamp(round(x / s), float(-128.0), float(127.0)) * s;
}

// dispatch with threadsPerThreadgroup = (32)
kernel void qkv_70(
    device const float4* a [[buffer(0)]],
    device const uint* q_bits [[buffer(1)]],
    device const uint* k_bits [[buffer(2)]],
    device const uint* v_bits [[buffer(3)]],
    device const float* scales [[buffer(4)]],
    device const float* sum_a [[buffer(5)]],
    device float* out_q [[buffer(6)]],
    device float* out_k [[buffer(7)]],
    device float* out_v [[buffer(8)]],
    constant uint4& params [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  // q, k and v projections in ONE dispatch, split by workgroup id.
  //
  // The three matrices have different row counts, so rather than three dispatches
  // the grid is partitioned: the first Q_WGS workgroups do q, the next KV_WGS do
  // k, the rest do v. All six shape constants are per-layer consts.
  //
  // Raw-byte unpacking here (no /255) with the matching zero-point subtraction
  // `rQA - ZP*rA` — unlike the MLP kernels, which use the unorm form and undo it
  // with fma(...,255,...).
  uint wgId = wg.y * uint(GRID_X) + wg.x;
  if (wgId < uint(TOTAL_WGS)) {
    float sumQA[2];
    for (uint r0 = 0; r0 < 2; r0++) {
      sumQA[r0] = float(0.0);
    }
    if (wgId < uint(Q_WGS)) {
      uint rowBase = wgId * uint(2);
      for (uint w = tid; w < 192; w += 32) {
        float4 avc0 = a[w * uint(2)];
        float4 avc1 = a[w * uint(2) + uint(1)];
        for (uint r = 0; r < 2; r++) {
          uint o = rowBase + uint(r);
          if (o < uint(Q_OUT)) {
            uint p = q_bits[o * uint(192) + w];
            float4 lo = float4(float4(as_type<uchar4>((p & uint(252645135)))));
            float4 hi = float4(float4(as_type<uchar4>(((p >> uint(4)) & uint(252645135)))));
            sumQA[r] = sumQA[r] + (dot(float4(lo.x, hi.x, lo.y, hi.y), avc0) + dot(float4(lo.z, hi.z, lo.w, hi.w), avc1));
          }
        }
      }
      float rA = sum_a[0];
      for (uint r2 = 0; r2 < 2; r2++) {
        float rQA = simd_sum(sumQA[r2]);
        uint o2 = rowBase + uint(r2);
        if (tid == uint(0)) {
          if (o2 < uint(Q_OUT)) {
            out_q[o2] = srq(scales[o2] * (rQA - float(8.0) * rA), as_type<float>(params.x));
          }
        }
      }
    } else if (wgId < uint(Q_WGS) + uint(KV_WGS)) {
      uint rowBaseK = (wgId - uint(Q_WGS)) * uint(2);
      for (uint wk = tid; wk < 192; wk += 32) {
        float4 kvc0 = a[wk * uint(2)];
        float4 kvc1 = a[wk * uint(2) + uint(1)];
        for (uint rk = 0; rk < 2; rk++) {
          uint ok = rowBaseK + uint(rk);
          if (ok < uint(KV_OUT)) {
            uint pk = k_bits[ok * uint(192) + wk];
            float4 klo = float4(float4(as_type<uchar4>((pk & uint(252645135)))));
            float4 khi = float4(float4(as_type<uchar4>(((pk >> uint(4)) & uint(252645135)))));
            sumQA[rk] = sumQA[rk] + (dot(float4(klo.x, khi.x, klo.y, khi.y), kvc0) + dot(float4(klo.z, khi.z, klo.w, khi.w), kvc1));
          }
        }
      }
      float rAk = sum_a[0];
      for (uint rk2 = 0; rk2 < 2; rk2++) {
        float rQAk = simd_sum(sumQA[rk2]);
        uint ok2 = rowBaseK + uint(rk2);
        if (tid == uint(0)) {
          if (ok2 < uint(KV_OUT)) {
            out_k[ok2] = srq(scales[uint(Q_OUT) + ok2] * (rQAk - float(8.0) * rAk), as_type<float>(params[1]));
          }
        }
      }
    } else {
      uint rowBaseV = (wgId - uint(Q_WGS) - uint(KV_WGS)) * uint(2);
      for (uint wv = tid; wv < 192; wv += 32) {
        float4 vvc0 = a[wv * uint(2)];
        float4 vvc1 = a[wv * uint(2) + uint(1)];
        for (uint rv = 0; rv < 2; rv++) {
          uint ov = rowBaseV + uint(rv);
          if (ov < uint(KV_OUT)) {
            uint pv = v_bits[ov * uint(192) + wv];
            float4 vlo = float4(float4(as_type<uchar4>((pv & uint(252645135)))));
            float4 vhi = float4(float4(as_type<uchar4>(((pv >> uint(4)) & uint(252645135)))));
            sumQA[rv] = sumQA[rv] + (dot(float4(vlo.x, vhi.x, vlo.y, vhi.y), vvc0) + dot(float4(vlo.z, vhi.z, vlo.w, vhi.w), vvc1));
          }
        }
      }
      float rAv = sum_a[0];
      for (uint rv2 = 0; rv2 < 2; rv2++) {
        float rQAv = simd_sum(sumQA[rv2]);
        uint ov2 = rowBaseV + uint(rv2);
        if (tid == uint(0)) {
          if (ov2 < uint(KV_OUT)) {
            out_v[ov2] = srq(scales[uint(Q_OUT) + uint(KV_OUT) + ov2] * (rQAv - float(8.0) * rAv), as_type<float>(params[2]));
          }
        }
      }
    }
  }
}