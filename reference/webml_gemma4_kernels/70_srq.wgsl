enable subgroups;
struct Params { qOutScale: f32, kOutScale: f32, vOutScale: f32, _pad0: u32 };
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> q_bits: array<u32>;
@group(0) @binding(2) var<storage, read> k_bits: array<u32>;
@group(0) @binding(3) var<storage, read> v_bits: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> sum_a: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_q: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_k: array<f32>;
@group(0) @binding(8) var<storage, read_write> out_v: array<f32>;
@group(0) @binding(9) var<uniform> params: Params;

// Fused decode (M=1) q/k/v projection: one dispatch computes all three QAT GEMVs over the
// same presrq'd activation (q/k/v share input_activation_scale; the producer norm already
// quantized `a` and staged its sum in sum_a). Workgroups are partitioned by output row:
//   [0, Q_WGS)                      -> q rows
//   [Q_WGS, Q_WGS+KV_WGS)           -> k rows
//   [Q_WGS+KV_WGS, TOTAL_WGS)       -> v rows
// Each per-row reduction follows QatMatMul scalar_presrq (WG=32 lane-strided
// words, same chunk/dot order, same subgroupAdd), so q/k/v preserve the
// per-projection rounding contract while sharing the presrq activation read and sum.
// Per-projection output_activation_scale (SRQ) comes from params; per-row weight scales are
// packed [qScale | kScale | vScale] in `scales`.

const IN_FEATURES: u32 = 1536u;
const Q_OUT: u32 = 2048u;
const KV_OUT: u32 = 256u;
const BITS: u32 = 4u;
const VALS_PER_WORD: u32 = 8u;
const CHUNKS: u32 = 2u;
const WORDS_PER_ROW: u32 = 192u;
const MASK: u32 = 15u;
const ZP: f32 = 8.0;
const WG: u32 = 32u;
const N_ROWS: u32 = 2u;
const Q_WGS: u32 = 1024u;
const KV_WGS: u32 = 128u;
const TOTAL_WGS: u32 = 1280u;
const GRID_X: u32 = 1280u;


// Static Range Quantization: round-trip through an int8 grid (no-op when scale==0).
fn srq(x: f32, s: f32) -> f32 {
  if (s == 0.0) {
    return x;
  }
  return clamp(round(x / s), -128.0, 127.0) * s;
}

fn reduce(value: f32, tid: u32) -> f32 {
  return subgroupAdd(value);
}


@compute @workgroup_size(32, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wgId = wg.y * GRID_X + wg.x;
  if (wgId >= TOTAL_WGS) {
    return;
  }
  let tid = lid.x;
  if (wgId < Q_WGS) {
    let rowBase = wgId * N_ROWS;
    // Same structure as QatMatMul scalar_presrq M==1: lane-strided words, the word's
    // activation chunk read once and reused across N_ROWS rows, unpack4xU8 dequant.
    var sumQA: array<f32, N_ROWS>;
    for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) { sumQA[r] = 0.0; }
    var w: u32 = tid;
    loop {
      if (w >= WORDS_PER_ROW) {
        break;
      }
      var avc: array<vec4<f32>, CHUNKS>;
      for (var c: u32 = 0u; c < CHUNKS; c = c + 1u) {
        avc[c] = a[w * CHUNKS + c];
      }
      for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) {
        let o = rowBase + r;
        if (o < Q_OUT) {
          let packed: u32 = q_bits[o * WORDS_PER_ROW + w];
          let lo = vec4<f32>(unpack4xU8(packed & 0x0F0F0F0Fu));
          let hi = vec4<f32>(unpack4xU8((packed >> 4u) & 0x0F0F0F0Fu));
          sumQA[r] = sumQA[r] + dot(vec4<f32>(lo.x, hi.x, lo.y, hi.y), avc[0]) + dot(vec4<f32>(lo.z, hi.z, lo.w, hi.w), avc[1]);
        }
      }
      w = w + WG;
    }
    let rA = sum_a[0];
    for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) {
      let rQA = reduce(sumQA[r], tid);
      let o = rowBase + r;
      if (tid == 0u && o < Q_OUT) {
        out_q[o] = srq(scales[0u + o] * (rQA - ZP * rA), params.qOutScale);
      }
    }

  } else if (wgId < Q_WGS + KV_WGS) {
    let rowBase = (wgId - Q_WGS) * N_ROWS;
    // Same structure as QatMatMul scalar_presrq M==1: lane-strided words, the word's
    // activation chunk read once and reused across N_ROWS rows, unpack4xU8 dequant.
    var sumQA: array<f32, N_ROWS>;
    for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) { sumQA[r] = 0.0; }
    var w: u32 = tid;
    loop {
      if (w >= WORDS_PER_ROW) {
        break;
      }
      var avc: array<vec4<f32>, CHUNKS>;
      for (var c: u32 = 0u; c < CHUNKS; c = c + 1u) {
        avc[c] = a[w * CHUNKS + c];
      }
      for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) {
        let o = rowBase + r;
        if (o < KV_OUT) {
          let packed: u32 = k_bits[o * WORDS_PER_ROW + w];
          let lo = vec4<f32>(unpack4xU8(packed & 0x0F0F0F0Fu));
          let hi = vec4<f32>(unpack4xU8((packed >> 4u) & 0x0F0F0F0Fu));
          sumQA[r] = sumQA[r] + dot(vec4<f32>(lo.x, hi.x, lo.y, hi.y), avc[0]) + dot(vec4<f32>(lo.z, hi.z, lo.w, hi.w), avc[1]);
        }
      }
      w = w + WG;
    }
    let rA = sum_a[0];
    for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) {
      let rQA = reduce(sumQA[r], tid);
      let o = rowBase + r;
      if (tid == 0u && o < KV_OUT) {
        out_k[o] = srq(scales[Q_OUT + o] * (rQA - ZP * rA), params.kOutScale);
      }
    }

  } else {
    let rowBase = (wgId - Q_WGS - KV_WGS) * N_ROWS;
    // Same structure as QatMatMul scalar_presrq M==1: lane-strided words, the word's
    // activation chunk read once and reused across N_ROWS rows, unpack4xU8 dequant.
    var sumQA: array<f32, N_ROWS>;
    for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) { sumQA[r] = 0.0; }
    var w: u32 = tid;
    loop {
      if (w >= WORDS_PER_ROW) {
        break;
      }
      var avc: array<vec4<f32>, CHUNKS>;
      for (var c: u32 = 0u; c < CHUNKS; c = c + 1u) {
        avc[c] = a[w * CHUNKS + c];
      }
      for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) {
        let o = rowBase + r;
        if (o < KV_OUT) {
          let packed: u32 = v_bits[o * WORDS_PER_ROW + w];
          let lo = vec4<f32>(unpack4xU8(packed & 0x0F0F0F0Fu));
          let hi = vec4<f32>(unpack4xU8((packed >> 4u) & 0x0F0F0F0Fu));
          sumQA[r] = sumQA[r] + dot(vec4<f32>(lo.x, hi.x, lo.y, hi.y), avc[0]) + dot(vec4<f32>(lo.z, hi.z, lo.w, hi.w), avc[1]);
        }
      }
      w = w + WG;
    }
    let rA = sum_a[0];
    for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) {
      let rQA = reduce(sumQA[r], tid);
      let o = rowBase + r;
      if (tid == 0u && o < KV_OUT) {
        out_v[o] = srq(scales[Q_OUT + KV_OUT + o] * (rQA - ZP * rA), params.vOutScale);
      }
    }

  }
}