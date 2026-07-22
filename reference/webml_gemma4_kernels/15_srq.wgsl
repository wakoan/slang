enable subgroups;
struct Params { inScale: f32, outScale: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bits_buf: array<u32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Weight-only QAT matmul: out[m, o] = scale[o] * sum_k (q[o,k] - ZP) * a[m,k]
//   = scale[o] * (sum_k q[o,k]*a[m,k] - ZP * sum_k a[m,k])
// One workgroup (= one subgroup, WG=32) computes N_ROWS output rows. Threads
// split K so adjacent threads read adjacent packed weight words (coalesced);
// the K-reduction uses subgroupAdd (zero barriers). The activation value for
// each column is read once per word and reused across the N_ROWS rows in
// registers, without staging a workgroup activation tile.
// N_ROWS is specialized per output width: 1 for small-outF matmuls, >1 when
// the activation read can be shared across multiple output rows. vec4 unpack:
// 4 values per dot().

const M: u32 = 32u;
const M_TILE: u32 = 8u;
const IN_FEATURES: u32 = 2048u;
const OUT_FEATURES: u32 = 1536u;
const BITS: u32 = 4u;
const VALS_PER_WORD: u32 = 8u;
const CHUNKS: u32 = 2u;
const WORDS_PER_ROW: u32 = 256u;
const MASK: u32 = 15u;
const ZP: f32 = 8.0;
const GRID_X: u32 = 768u;
const WG: u32 = 32u;
const N_ROWS: u32 = 2u;


// Static Range Quantization: round-trip through an int8 grid (no-op when scale==0).
fn srq(x: f32, s: f32) -> f32 {
  if (s == 0.0) {
    return x;
  }
  return clamp(round(x / s), -128.0, 127.0) * s;
}

// Componentwise srq over a vec4 (bit-identical to 4 scalar srq calls).
fn srq4(x: vec4<f32>, s: f32) -> vec4<f32> {
  if (s == 0.0) {
    return x;
  }
  return clamp(round(x / s), vec4<f32>(-128.0), vec4<f32>(127.0)) * s;
}

fn reduce(value: f32, tid: u32) -> f32 {
  return subgroupAdd(value);
}

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wgId = wg.y * GRID_X + wg.x;
  let rowBase = wgId * N_ROWS;
  if (rowBase >= OUT_FEATURES) {
    return;
  }
  let tid = lid.x;
  let inScale = params.inScale;
  let outScale = params.outScale;

  // Word-outer, m-unrolled GEMM tile (prefill): each weight word is read + unpacked once and
  // dotted against all M_TILE input rows. Everything lives in NAMED variables — dynamically
  // indexed local arrays can spill to memory. Per-(m,row)
  // accumulation order is identical to the m-outer GEMV, so results are bit-identical.
  let mStart = wg.z * M_TILE;
  let mOk0 = mStart + 0u < M;
  var sumA_0: f32 = 0.0;
  var sumQA_0_0: f32 = 0.0;
  var sumQA_0_1: f32 = 0.0;
  let mOk1 = mStart + 1u < M;
  var sumA_1: f32 = 0.0;
  var sumQA_1_0: f32 = 0.0;
  var sumQA_1_1: f32 = 0.0;
  let mOk2 = mStart + 2u < M;
  var sumA_2: f32 = 0.0;
  var sumQA_2_0: f32 = 0.0;
  var sumQA_2_1: f32 = 0.0;
  let mOk3 = mStart + 3u < M;
  var sumA_3: f32 = 0.0;
  var sumQA_3_0: f32 = 0.0;
  var sumQA_3_1: f32 = 0.0;
  let mOk4 = mStart + 4u < M;
  var sumA_4: f32 = 0.0;
  var sumQA_4_0: f32 = 0.0;
  var sumQA_4_1: f32 = 0.0;
  let mOk5 = mStart + 5u < M;
  var sumA_5: f32 = 0.0;
  var sumQA_5_0: f32 = 0.0;
  var sumQA_5_1: f32 = 0.0;
  let mOk6 = mStart + 6u < M;
  var sumA_6: f32 = 0.0;
  var sumQA_6_0: f32 = 0.0;
  var sumQA_6_1: f32 = 0.0;
  let mOk7 = mStart + 7u < M;
  var sumA_7: f32 = 0.0;
  var sumQA_7_0: f32 = 0.0;
  var sumQA_7_1: f32 = 0.0;

  var w: u32 = tid;
  loop {
    if (w >= WORDS_PER_ROW) {
      break;
    }
    var packed0: u32 = 0u;
    if (rowBase + 0u < OUT_FEATURES) { packed0 = bits_buf[(rowBase + 0u) * WORDS_PER_ROW + w]; }
    let lo0 = vec4<f32>(unpack4xU8(packed0 & 0x0F0F0F0Fu));
    let hi0 = vec4<f32>(unpack4xU8((packed0 >> 4u) & 0x0F0F0F0Fu));
    let q0_0 = vec4<f32>(lo0.x, hi0.x, lo0.y, hi0.y);
    let q0_1 = vec4<f32>(lo0.z, hi0.z, lo0.w, hi0.w);
    var packed1: u32 = 0u;
    if (rowBase + 1u < OUT_FEATURES) { packed1 = bits_buf[(rowBase + 1u) * WORDS_PER_ROW + w]; }
    let lo1 = vec4<f32>(unpack4xU8(packed1 & 0x0F0F0F0Fu));
    let hi1 = vec4<f32>(unpack4xU8((packed1 >> 4u) & 0x0F0F0F0Fu));
    let q1_0 = vec4<f32>(lo1.x, hi1.x, lo1.y, hi1.y);
    let q1_1 = vec4<f32>(lo1.z, hi1.z, lo1.w, hi1.w);
    if (mOk0) {
      let aV4Base0 = (mStart + 0u) * (IN_FEATURES / 4u) + w * CHUNKS;
      let a0_0 = srq4(vec4<f32>(a[aV4Base0 + 0u]), inScale);
      sumA_0 = sumA_0 + a0_0.x + a0_0.y + a0_0.z + a0_0.w;
      sumQA_0_0 = sumQA_0_0 + dot(q0_0, a0_0);
      sumQA_0_1 = sumQA_0_1 + dot(q1_0, a0_0);
      let a0_1 = srq4(vec4<f32>(a[aV4Base0 + 1u]), inScale);
      sumA_0 = sumA_0 + a0_1.x + a0_1.y + a0_1.z + a0_1.w;
      sumQA_0_0 = sumQA_0_0 + dot(q0_1, a0_1);
      sumQA_0_1 = sumQA_0_1 + dot(q1_1, a0_1);
    }
    if (mOk1) {
      let aV4Base1 = (mStart + 1u) * (IN_FEATURES / 4u) + w * CHUNKS;
      let a1_0 = srq4(vec4<f32>(a[aV4Base1 + 0u]), inScale);
      sumA_1 = sumA_1 + a1_0.x + a1_0.y + a1_0.z + a1_0.w;
      sumQA_1_0 = sumQA_1_0 + dot(q0_0, a1_0);
      sumQA_1_1 = sumQA_1_1 + dot(q1_0, a1_0);
      let a1_1 = srq4(vec4<f32>(a[aV4Base1 + 1u]), inScale);
      sumA_1 = sumA_1 + a1_1.x + a1_1.y + a1_1.z + a1_1.w;
      sumQA_1_0 = sumQA_1_0 + dot(q0_1, a1_1);
      sumQA_1_1 = sumQA_1_1 + dot(q1_1, a1_1);
    }
    if (mOk2) {
      let aV4Base2 = (mStart + 2u) * (IN_FEATURES / 4u) + w * CHUNKS;
      let a2_0 = srq4(vec4<f32>(a[aV4Base2 + 0u]), inScale);
      sumA_2 = sumA_2 + a2_0.x + a2_0.y + a2_0.z + a2_0.w;
      sumQA_2_0 = sumQA_2_0 + dot(q0_0, a2_0);
      sumQA_2_1 = sumQA_2_1 + dot(q1_0, a2_0);
      let a2_1 = srq4(vec4<f32>(a[aV4Base2 + 1u]), inScale);
      sumA_2 = sumA_2 + a2_1.x + a2_1.y + a2_1.z + a2_1.w;
      sumQA_2_0 = sumQA_2_0 + dot(q0_1, a2_1);
      sumQA_2_1 = sumQA_2_1 + dot(q1_1, a2_1);
    }
    if (mOk3) {
      let aV4Base3 = (mStart + 3u) * (IN_FEATURES / 4u) + w * CHUNKS;
      let a3_0 = srq4(vec4<f32>(a[aV4Base3 + 0u]), inScale);
      sumA_3 = sumA_3 + a3_0.x + a3_0.y + a3_0.z + a3_0.w;
      sumQA_3_0 = sumQA_3_0 + dot(q0_0, a3_0);
      sumQA_3_1 = sumQA_3_1 + dot(q1_0, a3_0);
      let a3_1 = srq4(vec4<f32>(a[aV4Base3 + 1u]), inScale);
      sumA_3 = sumA_3 + a3_1.x + a3_1.y + a3_1.z + a3_1.w;
      sumQA_3_0 = sumQA_3_0 + dot(q0_1, a3_1);
      sumQA_3_1 = sumQA_3_1 + dot(q1_1, a3_1);
    }
    if (mOk4) {
      let aV4Base4 = (mStart + 4u) * (IN_FEATURES / 4u) + w * CHUNKS;
      let a4_0 = srq4(vec4<f32>(a[aV4Base4 + 0u]), inScale);
      sumA_4 = sumA_4 + a4_0.x + a4_0.y + a4_0.z + a4_0.w;
      sumQA_4_0 = sumQA_4_0 + dot(q0_0, a4_0);
      sumQA_4_1 = sumQA_4_1 + dot(q1_0, a4_0);
      let a4_1 = srq4(vec4<f32>(a[aV4Base4 + 1u]), inScale);
      sumA_4 = sumA_4 + a4_1.x + a4_1.y + a4_1.z + a4_1.w;
      sumQA_4_0 = sumQA_4_0 + dot(q0_1, a4_1);
      sumQA_4_1 = sumQA_4_1 + dot(q1_1, a4_1);
    }
    if (mOk5) {
      let aV4Base5 = (mStart + 5u) * (IN_FEATURES / 4u) + w * CHUNKS;
      let a5_0 = srq4(vec4<f32>(a[aV4Base5 + 0u]), inScale);
      sumA_5 = sumA_5 + a5_0.x + a5_0.y + a5_0.z + a5_0.w;
      sumQA_5_0 = sumQA_5_0 + dot(q0_0, a5_0);
      sumQA_5_1 = sumQA_5_1 + dot(q1_0, a5_0);
      let a5_1 = srq4(vec4<f32>(a[aV4Base5 + 1u]), inScale);
      sumA_5 = sumA_5 + a5_1.x + a5_1.y + a5_1.z + a5_1.w;
      sumQA_5_0 = sumQA_5_0 + dot(q0_1, a5_1);
      sumQA_5_1 = sumQA_5_1 + dot(q1_1, a5_1);
    }
    if (mOk6) {
      let aV4Base6 = (mStart + 6u) * (IN_FEATURES / 4u) + w * CHUNKS;
      let a6_0 = srq4(vec4<f32>(a[aV4Base6 + 0u]), inScale);
      sumA_6 = sumA_6 + a6_0.x + a6_0.y + a6_0.z + a6_0.w;
      sumQA_6_0 = sumQA_6_0 + dot(q0_0, a6_0);
      sumQA_6_1 = sumQA_6_1 + dot(q1_0, a6_0);
      let a6_1 = srq4(vec4<f32>(a[aV4Base6 + 1u]), inScale);
      sumA_6 = sumA_6 + a6_1.x + a6_1.y + a6_1.z + a6_1.w;
      sumQA_6_0 = sumQA_6_0 + dot(q0_1, a6_1);
      sumQA_6_1 = sumQA_6_1 + dot(q1_1, a6_1);
    }
    if (mOk7) {
      let aV4Base7 = (mStart + 7u) * (IN_FEATURES / 4u) + w * CHUNKS;
      let a7_0 = srq4(vec4<f32>(a[aV4Base7 + 0u]), inScale);
      sumA_7 = sumA_7 + a7_0.x + a7_0.y + a7_0.z + a7_0.w;
      sumQA_7_0 = sumQA_7_0 + dot(q0_0, a7_0);
      sumQA_7_1 = sumQA_7_1 + dot(q1_0, a7_0);
      let a7_1 = srq4(vec4<f32>(a[aV4Base7 + 1u]), inScale);
      sumA_7 = sumA_7 + a7_1.x + a7_1.y + a7_1.z + a7_1.w;
      sumQA_7_0 = sumQA_7_0 + dot(q0_1, a7_1);
      sumQA_7_1 = sumQA_7_1 + dot(q1_1, a7_1);
    }
    w = w + WG;
  }

  if (mOk0) {
    let rA0 = reduce(sumA_0, tid);
    {
      let rQA = reduce(sumQA_0_0, tid);
      let o = rowBase + 0u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 0u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA0), outScale));
      }
    }
    {
      let rQA = reduce(sumQA_0_1, tid);
      let o = rowBase + 1u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 0u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA0), outScale));
      }
    }
  }
  if (mOk1) {
    let rA1 = reduce(sumA_1, tid);
    {
      let rQA = reduce(sumQA_1_0, tid);
      let o = rowBase + 0u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 1u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA1), outScale));
      }
    }
    {
      let rQA = reduce(sumQA_1_1, tid);
      let o = rowBase + 1u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 1u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA1), outScale));
      }
    }
  }
  if (mOk2) {
    let rA2 = reduce(sumA_2, tid);
    {
      let rQA = reduce(sumQA_2_0, tid);
      let o = rowBase + 0u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 2u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA2), outScale));
      }
    }
    {
      let rQA = reduce(sumQA_2_1, tid);
      let o = rowBase + 1u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 2u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA2), outScale));
      }
    }
  }
  if (mOk3) {
    let rA3 = reduce(sumA_3, tid);
    {
      let rQA = reduce(sumQA_3_0, tid);
      let o = rowBase + 0u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 3u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA3), outScale));
      }
    }
    {
      let rQA = reduce(sumQA_3_1, tid);
      let o = rowBase + 1u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 3u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA3), outScale));
      }
    }
  }
  if (mOk4) {
    let rA4 = reduce(sumA_4, tid);
    {
      let rQA = reduce(sumQA_4_0, tid);
      let o = rowBase + 0u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 4u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA4), outScale));
      }
    }
    {
      let rQA = reduce(sumQA_4_1, tid);
      let o = rowBase + 1u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 4u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA4), outScale));
      }
    }
  }
  if (mOk5) {
    let rA5 = reduce(sumA_5, tid);
    {
      let rQA = reduce(sumQA_5_0, tid);
      let o = rowBase + 0u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 5u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA5), outScale));
      }
    }
    {
      let rQA = reduce(sumQA_5_1, tid);
      let o = rowBase + 1u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 5u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA5), outScale));
      }
    }
  }
  if (mOk6) {
    let rA6 = reduce(sumA_6, tid);
    {
      let rQA = reduce(sumQA_6_0, tid);
      let o = rowBase + 0u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 6u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA6), outScale));
      }
    }
    {
      let rQA = reduce(sumQA_6_1, tid);
      let o = rowBase + 1u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 6u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA6), outScale));
      }
    }
  }
  if (mOk7) {
    let rA7 = reduce(sumA_7, tid);
    {
      let rQA = reduce(sumQA_7_0, tid);
      let o = rowBase + 0u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 7u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA7), outScale));
      }
    }
    {
      let rQA = reduce(sumQA_7_1, tid);
      let o = rowBase + 1u;
      if (tid == 0u && o < OUT_FEATURES) {
        out[(mStart + 7u) * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA7), outScale));
      }
    }
  }
}
