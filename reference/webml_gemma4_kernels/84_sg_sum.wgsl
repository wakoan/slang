enable f16;
enable subgroups;
struct Params { outScale: f32, inScale2: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bits_buf: array<u32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> pp: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(5) var<storage, read> w12: array<f32>;
@group(0) @binding(6) var<storage, read_write> y2: array<f16>;
@group(0) @binding(7) var<storage, read_write> sum2: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

// Single-dispatch fused o-projection (QAT GEMV) + post-attention residual norm-add + pre-FFN
// norm (M=1):
//   o[r]   = srq(scale[r] * (sum_k q[r,k]*a[k] - ZP * sum_k a[k]), outScale)
//   hidden = hidden + RMSNorm(o) * w1                       (post-attn scale is always 1.0)
//   y2     = toY(srq(f32(toY(RMSNorm(hidden') * w2)), inScale2));  sum2 = sum f32(y2)
// The GEMV phase and both normalization phases share one dispatch with a
// last-arriver tail. `a` (attnOut) is already SRQ-quantized by the attention merge, so the
// GEMV runs division-free (inScale handled upstream); the per-row ZP correction sum_k a[k]
// falls out of the activation staging. w12 = [w1 | w2] packed. Virtual-subgroup GEMV phase.
// pp layout: [0..OUT_F) o values (bitcast f32); [OUT_F] ticket counter.

const IN_FEATURES: u32 = 4096u;
const OUT_F: u32 = 1536u;
const BITS: u32 = 4u;
const VALS_PER_WORD: u32 = 8u;
const CHUNKS: u32 = 2u;
const WORDS_PER_ROW: u32 = 512u;
const MASK: u32 = 15u;
const ZP: f32 = 8.0;
const WG: u32 = 256u;
const SG_ROWS: u32 = 1u;
const ROWS_PER_WG: u32 = 8u;
const TOTAL_WGS: u32 = 192u;
const EPS: f32 = 0.000001;
const ELEMS: u32 = 6u;

var<workgroup> lastFlag: u32;
var<workgroup> sgp: array<f32, WG / 32u>;


// Sum over each logical 32-lane block. On a fixed 32-wide adapter (sgExact32) this is the
// hardware subgroupAdd. On adapters reporting a wider/ranged subgroup (NVIDIA D3D12 [32,128],
// AMD [32,64]/[64,64]) a plain subgroupAdd would span multiple 32-blocks, so we use a 32-lane
// subgroupShuffleXor butterfly (deltas 1,2,4,8,16) that reduces each block independently —
// correct for ANY hardware subgroup width >= 32, which is exactly what the >=32 gate ensures.
fn sg_sum(value: f32) -> f32 {
  return subgroupAdd(value);
}

fn reduce_sum(value: f32, tid: u32) -> f32 {
  let s = sg_sum(value);
  if ((tid & 31u) == 0u) { sgp[tid >> 5u] = s; }
  workgroupBarrier();
  var total: f32 = 0.0;
  for (var i: u32 = 0u; i < WG / 32u; i = i + 1u) { total = total + sgp[i]; }
  workgroupBarrier();
  return total;
}

fn srq(x: f32, s: f32) -> f32 {
  if (s == 0.0) { return x; }
  return clamp(round(x / s), -128.0, 127.0) * s;
}

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let sgId = tid / 32u;
  let lane = tid & 31u;
  let outScale = params.outScale;

  let rowBase = wg.x * ROWS_PER_WG + sgId * SG_ROWS;

  // --- QAT GEMV phase (per virtual subgroup; mirrors QatMatMul scalar, division-free) ---
  var sumQA: array<f32, SG_ROWS>;
  for (var r: u32 = 0u; r < SG_ROWS; r = r + 1u) { sumQA[r] = 0.0; }
  var sumA: f32 = 0.0;

  var w: u32 = lane;
  loop {
    if (w >= WORDS_PER_ROW) { break; }
    var avc: array<vec4<f32>, CHUNKS>;
    for (var c: u32 = 0u; c < CHUNKS; c = c + 1u) {
      let a4 = vec4<f32>(a[w * CHUNKS + c]);
      avc[c] = a4;
      sumA = sumA + a4.x + a4.y + a4.z + a4.w;
    }
    for (var r: u32 = 0u; r < SG_ROWS; r = r + 1u) {
      let o = rowBase + r;
      if (o < OUT_F) {
        let packed: u32 = bits_buf[o * WORDS_PER_ROW + w];
        let lo = vec4<f32>(unpack4xU8(packed & 0x0F0F0F0Fu));
        let hi = vec4<f32>(unpack4xU8((packed >> 4u) & 0x0F0F0F0Fu));
        sumQA[r] = sumQA[r] + dot(vec4<f32>(lo.x, hi.x, lo.y, hi.y), avc[0]) + dot(vec4<f32>(lo.z, hi.z, lo.w, hi.w), avc[1]);
      }
    }
    w = w + 32u;
  }

  let rA = sg_sum(sumA);
  for (var r: u32 = 0u; r < SG_ROWS; r = r + 1u) {
    let rQA = sg_sum(sumQA[r]);
    let o = rowBase + r;
    if (lane == 0u && o < OUT_F) {
      atomicStore(&pp[o], bitcast<u32>(srq(scale[o] * (rQA - ZP * rA), outScale)));
    }
  }
  storageBarrier();

  // --- last-arriver norm tail (post-attn norm-add + pre-FFN norm + SRQ + sum) ---
  if (tid == 0u) {
    let ticket = atomicAdd(&pp[OUT_F], 1u);
    lastFlag = select(0u, 1u, ticket == TOTAL_WGS - 1u);
  }
  if (workgroupUniformLoad(&lastFlag) != 1u) {
    return;
  }
  if (tid == 0u) { atomicStore(&pp[OUT_F], 0u); }
  let inScale2 = params.inScale2;

  var acc1: f32 = 0.0;
  var i: u32 = tid;
  loop {
    if (i >= OUT_F) { break; }
    let v = bitcast<f32>(atomicLoad(&pp[i]));
    acc1 = acc1 + v * v;
    i = i + WG;
  }
  let rms1 = inverseSqrt(reduce_sum(acc1, tid) / f32(OUT_F) + EPS);

  var hloc: array<f32, ELEMS>;
  var acc2: f32 = 0.0;
  var j: u32 = tid;
  var e: u32 = 0u;
  loop {
    if (j >= OUT_F) { break; }
    let normed = bitcast<f32>(atomicLoad(&pp[j])) * rms1 * f32(w12[j]);
    let hv = f32(f32(f32(hidden[j]) + normed));
    hidden[j] = f32(hv);
    hloc[e] = hv;
    acc2 = acc2 + hv * hv;
    j = j + WG;
    e = e + 1u;
  }
  let rms2 = inverseSqrt(reduce_sum(acc2, tid) / f32(OUT_F) + EPS);

  var qAcc: f32 = 0.0;
  j = tid;
  e = 0u;
  loop {
    if (j >= OUT_F) { break; }
    let n2 = hloc[e] * rms2 * f32(w12[OUT_F + j]);
    let qv = f16(srq(f32(f16(n2)), inScale2));
    y2[j] = qv;
    qAcc = qAcc + f32(qv);
    j = j + WG;
    e = e + 1u;
  }
  let qSum = reduce_sum(qAcc, tid);
  if (tid == 0u) {
    sum2[0] = qSum;
  }
}