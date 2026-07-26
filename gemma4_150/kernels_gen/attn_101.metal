// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

constant uint HEAD_DIM=512u;
constant uint HALF_DIM=256u;
constant uint HD4=128u;
constant uint J_GROUPS=2u;
constant uint PP_COUNTER_BASE=131584u;
constant float OUT_Q=0.014886821620166302f;

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
kernel void attn_101(
    device const float* q [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* cosTbl [[buffer(2)]],
    device const float* sinTbl [[buffer(3)]],
    device const float4* k [[buffer(4)]],
    device const float4* v [[buffer(5)]],
    device atomic_uint* partials [[buffer(6)]],
    device float* out [[buffer(7)]],
    device const uint* params [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  threadgroup float qn_sh[512];
  threadgroup float out_acc[512];
  threadgroup float probs[256];
  threadgroup float sval_sh[256];
  threadgroup float red[256];
  threadgroup float wgt_sh[32];
  threadgroup float4 vacc_sh[256];
  threadgroup float st[2];
  threadgroup uint lastFlag[1];
  // Decode flash attention: q-norm + RoPE, online softmax, last-arriver merge.
  //
  // One query, so parallelism has to come from splitting the KEY axis: each of
  // nActive chunks runs its own online softmax over a slice of the keys and
  // publishes (partial output, running max, running denom) through the atomic
  // buffer. The last chunk to finish rescales every partial by exp(m_c - m) and
  // combines them — a full flash-attention merge inside a single dispatch.
  //
  // HEAD_DIM/HALF_DIM/HD4/J_GROUPS/PP_COUNTER_BASE/OUT_Q are consts; the sliding
  // layers are 256-wide and the full ones 512.
  uint h = wg.x;
  uint ci = wg.y;
  if (h >= params[3]) {
    return;
  }
  uint hKv = h / (params[3] / params[4]);
  uint qPos = params[2];
  uint qBase = h * uint(HEAD_DIM);
  uint maxKj = min(params[1], qPos + uint(1));
  uint minKj = uint(0);
  if (params[5] > uint(0)) {
    if (qPos + uint(1) > params[5]) {
      minKj = qPos + uint(1) - params[5];
    }
  }
  uint activeKeys = maxKj - minKj;
  uint nActive = clamp((activeKeys + uint(63)) / uint(64), uint(8), uint(32));
  if (ci >= nActive) {
    return;
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
  for (uint p = tid; p < HALF_DIM; p += 256) {
    float n0 = q[qBase + p] * nscale * w[p];
    float n1 = q[qBase + p + uint(HALF_DIM)] * nscale * w[p + uint(HALF_DIM)];
    float c = cosTbl[p];
    float sn = sinTbl[p];
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
  uint chunkLen = (activeKeys + nActive - uint(1)) / nActive;
  uint start = minKj + ci * chunkLen;
  uint end = min(start + chunkLen, maxKj);
  uint tile = start;
  while (tile < end) {
    uint kj = tile + tid;
    uint tileCountS = min(uint(256), end - tile);
    uint sgRounds = (tileCountS + uint(7)) / uint(8);
    for (uint rr = 0; rr < sgRounds; rr++) {
      uint j = uint(rr) * uint(8) + tid / uint(32);
      float accS = float(0.0);
      if (j < tileCountS) {
        uint kBase4 = ((tile + j) * params[4] + hKv) * uint(HD4);
        for (uint d4 = (tid & uint(31)); d4 < HD4; d4 += 32) {
          float4 kv4 = k[kBase4 + d4];
          accS = accS + dot(float4(qn_sh[d4 * uint(4)], qn_sh[d4 * uint(4) + uint(1)], qn_sh[d4 * uint(4) + uint(2)], qn_sh[d4 * uint(4) + uint(3)]), kv4);
        }
      }
      float sj = simd_sum(accS);
      if ((tid & uint(31)) == uint(0)) {
        if (j < tileCountS) {
          sval_sh[j] = sj * float(1.0);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sval = float(-3.4028234663852886e+38);
    if (kj < end) {
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
    if (kj < end) {
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
    uint tileCount = min(uint(256), end - tile);
    uint jg = tid / uint(HD4);
    uint d4v = tid % uint(HD4);
    float4 vacc = float4(float(0.0), float(0.0), float(0.0), float(0.0));
    uint jj = jg;
    while (jj < tileCount) {
      uint vBase4 = ((tile + jj) * params[4] + hKv) * uint(HD4);
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
  uint pBase = (h * uint(32) + ci) * (uint(HEAD_DIM) + uint(2));
  for (uint i4 = tid; i4 < HEAD_DIM; i4 += 256) {
    atomic_store_explicit(&partials[pBase + i4], as_type<uint>(out_acc[i4]), memory_order_relaxed);
  }
  if (tid == uint(0)) {
    atomic_store_explicit(&partials[pBase + uint(HEAD_DIM)], as_type<uint>(st[0]), memory_order_relaxed);
    atomic_store_explicit(&partials[pBase + uint(HEAD_DIM) + uint(1)], as_type<uint>(st[1]), memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == uint(0)) {
    uint tk = atomic_fetch_add_explicit(&partials[uint(PP_COUNTER_BASE) + h], uint(1), memory_order_relaxed);
    lastFlag[0] = uint(0);
    if (tk == nActive - uint(1)) {
      lastFlag[0] = uint(1);
    }
  }
  if (_wgUniformLoad(lastFlag[0]) != uint(1)) {
    return;
  }
  if (tid == uint(0)) {
    atomic_store_explicit(&partials[uint(PP_COUNTER_BASE) + h], uint(0), memory_order_relaxed);
  }
  float mloc = float(-3.4028234663852886e+38);
  float lloc = float(0.0);
  if (tid < nActive) {
    uint pb = (h * uint(32) + tid) * (uint(HEAD_DIM) + uint(2));
    mloc = as_type<float>(atomic_load_explicit(&partials[pb + uint(HEAD_DIM)], memory_order_relaxed));
    lloc = as_type<float>(atomic_load_explicit(&partials[pb + uint(HEAD_DIM) + uint(1)], memory_order_relaxed));
  }
  float mm = simd_max(mloc);
  if ((tid & uint(31)) == uint(0)) {
    red[(tid >> uint(5))] = mm;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float newM = float(-3.4028234663852886e+38);
  for (uint i5 = 0; i5 < 8; i5++) {
    newM = max(newM, red[i5]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float wloc = float(0.0);
  if (tid < nActive) {
    wloc = exp(mloc - newM);
    wgt_sh[tid] = wloc;
  }
  float dd = simd_sum(lloc * wloc);
  if ((tid & uint(31)) == uint(0)) {
    red[(tid >> uint(5))] = dd;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float denom = float(0.0);
  for (uint i6 = 0; i6 < 8; i6++) {
    denom = denom + red[i6];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float invd = float(1.0) / denom;
  for (uint d5 = tid; d5 < HEAD_DIM; d5 += 256) {
    float acc = float(0.0);
    for (uint c2 = 0; c2 < nActive; c2++) {
      acc = acc + as_type<float>(atomic_load_explicit(&partials[(h * uint(32) + uint(c2)) * (uint(HEAD_DIM) + uint(2)) + d5], memory_order_relaxed)) * wgt_sh[c2];
    }
    out[h * uint(HEAD_DIM) + d5] = srq(acc * invd, float(OUT_Q));
  }
}