enable subgroups;
struct Params { inScale: f32, outScale: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bits_buf: array<u32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> sum_a: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

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

const M: u32 = 1u;
const M_TILE: u32 = 1u;
const IN_FEATURES: u32 = 1536u;
const OUT_FEATURES: u32 = 2048u;
const BITS: u32 = 4u;
const VALS_PER_WORD: u32 = 8u;
const CHUNKS: u32 = 2u;
const WORDS_PER_ROW: u32 = 192u;
const MASK: u32 = 15u;
const ZP: f32 = 8.0;
const GRID_X: u32 = 1024u;
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

  // M==1 (decode): m-outer GEMV. Hoisting the unpack into a register array
  // indexed in an inner loop is avoided because dynamically-indexed local
  // arrays can spill.
  let mEnd = min((wg.z + 1u) * M_TILE, M);
  for (var m: u32 = wg.z * M_TILE; m < mEnd; m = m + 1u) {
    let aV4Base = m * (IN_FEATURES / 4u);
    var sumQA: array<f32, N_ROWS>;
    for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) { sumQA[r] = 0.0; }
    var sumA: f32 = 0.0;

    var w: u32 = tid;
    loop {
      if (w >= WORDS_PER_ROW) {
        break;
      }
      let colBase: u32 = w * VALS_PER_WORD;
      // presrq: the activation is already srq-quantized (DecodeRmsSrq / DecodeNormAddNorm)
      // and its sum arrives via sum_a — no per-workgroup srq divisions, no sumA reduction.
      var avc: array<vec4<f32>, CHUNKS>;
      for (var c: u32 = 0u; c < CHUNKS; c = c + 1u) {
        avc[c] = vec4<f32>(a[aV4Base + w * CHUNKS + c]);
      }
      for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) {
        let o = rowBase + r;
        if (o < OUT_FEATURES) {
          let packed: u32 = bits_buf[o * WORDS_PER_ROW + w];
          // Dequant via unpack4xU8, which splits one u32 into 4 u8 lanes.
          let lo = vec4<f32>(unpack4xU8(packed & 0x0F0F0F0Fu));
          let hi = vec4<f32>(unpack4xU8((packed >> 4u) & 0x0F0F0F0Fu));
          sumQA[r] = sumQA[r] + dot(vec4<f32>(lo.x, hi.x, lo.y, hi.y), avc[0]) + dot(vec4<f32>(lo.z, hi.z, lo.w, hi.w), avc[1]);
        }
      }
      w = w + WG;
    }

    // Presrq producers provide the activation row sum alongside the quantized row.
    let rA = sum_a[m] + sumA;
    for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) {
      let rQA = reduce(sumQA[r], tid);
      let o = rowBase + r;
      if (tid == 0u && o < OUT_FEATURES) {
        out[m * OUT_FEATURES + o] = f32(srq(scale[o] * (rQA - ZP * rA), outScale));
      }
    }
  }
}
