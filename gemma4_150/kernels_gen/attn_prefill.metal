// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

constant uint HEAD_DIM=512u;
constant uint HALF_DIM=256u;
constant uint HD4=128u;
constant uint J_GROUPS=2u;
constant float OUT_Q=0.014886821620166302f;

float srq(float x, float s) {
  // Symmetric per-row quantization. s == 0 means the tensor is unquantized.
  if (s == float(0.0)) {
    return x;
  }
  return clamp(round(x / s), float(-128.0), float(127.0)) * s;
}

// dispatch with threadsPerThreadgroup = (256)
kernel void attn_prefill(
    device const float* q [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* cosTbl [[buffer(2)]],
    device const float* sinTbl [[buffer(3)]],
    device const float4* k [[buffer(4)]],
    device const float4* v [[buffer(5)]],
    device float* out [[buffer(6)]],
    device const uint* params [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  threadgroup float qn_sh[512];
  threadgroup float out_acc[512];
  threadgroup float probs[256];
  threadgroup float sval_sh[256];
  threadgroup float red[256];
  threadgroup float4 vacc_sh[256];
  threadgroup float st[2];
  // Batched causal attention: S query positions in one dispatch.
  //
  // attn_101 splits the KEY axis because a single query gives only nH workgroups
  // of parallelism. Prefill already has S*nH, so that machinery is pure overhead
  // here — one workgroup owns one (head, query token) and walks its keys with a
  // plain online softmax. No partials, no atomics, no cross-workgroup merge.
  //
  // Causality is the loop bound, not a mask buffer.
  uint h = wg.x;
  uint s = wg.y;
  if (h < params[1]) {
    if (s < params[4]) {
      uint hKv = h / (params[1] / params[2]);
      uint qPos = params[0] + s;
      uint qBase = (s * params[1] + h) * uint(HEAD_DIM);
      uint maxKj = qPos + uint(1);
      uint minKj = uint(0);
      if (params[3] > uint(0)) {
        if (qPos + uint(1) > params[3]) {
          minKj = qPos + uint(1) - params[3];
        }
      }
      float ss = float(0.0);
      for (uint d = tid; d < HEAD_DIM; d += 256) {
        float vv = q[qBase + d];
        ss = ss + vv * vv;
      }
      float s0 = simd_sum(ss);
      if ((tid & uint(31)) == uint(0)) {
        red[(tid >> uint(5))] = s0;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float t0 = float(0.0);
      for (uint i0 = 0; i0 < 8; i0++) {
        t0 = t0 + red[i0];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float nscale = rsqrt(t0 / float(HEAD_DIM) + float(1e-06));
      uint rb = qPos * uint(HALF_DIM);
      for (uint p = tid; p < HALF_DIM; p += 256) {
        float n0 = q[qBase + p] * nscale * w[p];
        float n1 = q[qBase + p + uint(HALF_DIM)] * nscale * w[p + uint(HALF_DIM)];
        float c = cosTbl[rb + p];
        float sn = sinTbl[rb + p];
        qn_sh[p] = n0 * c - n1 * sn;
        qn_sh[p + uint(HALF_DIM)] = n1 * c + n0 * sn;
      }
      for (uint i1 = tid; i1 < HEAD_DIM; i1 += 256) {
        out_acc[i1] = float(0.0);
      }
      if (tid == uint(0)) {
        st[0] = float(-3.4028234663852886e+38);
        st[1] = float(0.0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint tile = minKj;
      while (tile < maxKj) {
        uint kj = tile + tid;
        uint tileCount = min(uint(256), maxKj - tile);
        uint sgRounds = (tileCount + uint(7)) / uint(8);
        for (uint rr = 0; rr < sgRounds; rr++) {
          uint j = uint(rr) * uint(8) + tid / uint(32);
          float accS = float(0.0);
          if (j < tileCount) {
            uint kBase4 = ((tile + j) * params[2] + hKv) * uint(HD4);
            for (uint d4 = (tid & uint(31)); d4 < HD4; d4 += 32) {
              float4 kv4 = k[kBase4 + d4];
              accS = accS + dot(float4(qn_sh[d4 * uint(4)], qn_sh[d4 * uint(4) + uint(1)], qn_sh[d4 * uint(4) + uint(2)], qn_sh[d4 * uint(4) + uint(3)]), kv4);
            }
          }
          float sj = simd_sum(accS);
          if ((tid & uint(31)) == uint(0)) {
            if (j < tileCount) {
              sval_sh[j] = sj * float(1.0);
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sval = float(-3.4028234663852886e+38);
        if (kj < maxKj) {
          sval = sval_sh[tid];
        }
        float m0 = simd_max(sval);
        if ((tid & uint(31)) == uint(0)) {
          red[(tid >> uint(5))] = m0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tileMax = float(-3.4028234663852886e+38);
        for (uint i2 = 0; i2 < 8; i2++) {
          tileMax = max(tileMax, red[i2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float newMax = max(st[0], tileMax);
        float correction = exp(st[0] - newMax);
        float pr = float(0.0);
        if (kj < maxKj) {
          pr = exp(sval - newMax);
        }
        probs[tid] = pr;
        float d0 = simd_sum(pr);
        if ((tid & uint(31)) == uint(0)) {
          red[(tid >> uint(5))] = d0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tileDenom = float(0.0);
        for (uint i3 = 0; i3 < 8; i3++) {
          tileDenom = tileDenom + red[i3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == uint(0)) {
          st[1] = st[1] * correction + tileDenom;
          st[0] = newMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint jg = tid / uint(HD4);
        uint d4v = tid % uint(HD4);
        float4 vacc = float4(float(0.0), float(0.0), float(0.0), float(0.0));
        uint jj = jg;
        while (jj < tileCount) {
          uint vBase4 = ((tile + jj) * params[2] + hKv) * uint(HD4);
          vacc = vacc + probs[jj] * v[vBase4 + d4v];
          jj = jj + uint(J_GROUPS);
        }
        vacc_sh[tid] = vacc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d4b = tid; d4b < HD4; d4b += 256) {
          float4 a4 = float4(out_acc[d4b * uint(4)], out_acc[d4b * uint(4) + uint(1)], out_acc[d4b * uint(4) + uint(2)], out_acc[d4b * uint(4) + uint(3)]) * correction;
          for (uint g = 0; g < J_GROUPS; g++) {
            a4 = a4 + vacc_sh[uint(g) * uint(HD4) + d4b];
          }
          out_acc[d4b * uint(4)] = a4.x;
          out_acc[d4b * uint(4) + uint(1)] = a4.y;
          out_acc[d4b * uint(4) + uint(2)] = a4.z;
          out_acc[d4b * uint(4) + uint(3)] = a4.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile = tile + uint(256);
      }
      float invd = float(1.0) / st[1];
      for (uint d5 = tid; d5 < HEAD_DIM; d5 += 256) {
        out[qBase + d5] = srq(out_acc[d5] * invd, float(OUT_Q));
      }
    }
  }
}