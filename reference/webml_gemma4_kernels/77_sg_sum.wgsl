enable subgroups;
struct Params { inScale: f32, projInScale: f32, projOutScale: f32, _pad0: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read> row_scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> pp: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(5) var<storage, read> w12s: array<f32>;
@group(0) @binding(6) var<storage, read_write> y2: array<f32>;
@group(0) @binding(7) var<storage, read_write> sum2: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

// Codes path for the single-dispatch fused PLE projection + post-PLE residual
// norm-add + next-layer norm (M=1). The int8 dense projection weight streams as
// packed +128-biased u8 codes (4/u32) plus a per-row scale. unpack4x8unorm
// lanes decode as fl((c+128)/255); bias and unorm decode are undone once per output row:
//   proj[o] = srq(row_scale[o] * (255*sum_k(u_k*a_k) - 128*sum_k(a_k)), projOutScale)
// The a-words and their sum are hoisted per 32-lane subgroup (K_ITER registers) and reused
// across SG_ROWS rows. Norm tail unchanged (last-arriver over pp).
// pp layout: [0..OUT_F) proj values (bitcast f32); [OUT_F] ticket counter.

const IN_F: u32 = 256u;
const OUT_F: u32 = 1536u;
const KV4: u32 = 256u / 4u;
const K_ITER: u32 = 2u;
const WG: u32 = 256u;
const SG_ROWS: u32 = 2u;
const ROWS_PER_WG: u32 = 16u;
const TOTAL_WGS: u32 = 96u;
const EPS: f32 = 0.000001;
const ELEMS: u32 = 6u;

var<workgroup> lastFlag: u32;
var<workgroup> sgp: array<f32, WG / 32u>;

// Sum over each logical 32-lane block. sgExact32 (fixed 32-wide adapter) -> hardware
// subgroupAdd; otherwise a 32-lane subgroupShuffleXor butterfly that reduces each block
// independently, correct for any subgroup width >= 32 (NVIDIA D3D12 [32,128], AMD [32,64]).
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

fn srq4(x: vec4<f32>, s: f32) -> vec4<f32> {
  if (s == 0.0) { return x; }
  return clamp(round(x / s), vec4<f32>(-128.0), vec4<f32>(127.0)) * s;
}

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let sgId = tid / 32u;
  let lane = tid & 31u;
  let rowBase = wg.x * ROWS_PER_WG + sgId * SG_ROWS;

  // --- dense GEMV phase (per virtual subgroup) ---
  // Hoist the activation words (srq'd) + their sum once per subgroup; reuse across rows.
  var av: array<vec4<f32>, K_ITER>;
  var aAcc: f32 = 0.0;
  for (var ki: u32 = 0u; ki < K_ITER; ki = ki + 1u) {
    let k4 = lane + ki * 32u;
    av[ki] = vec4<f32>(0.0);
    if (k4 < KV4) {
      let kb = k4 * 4u;
      // QAT wrapper: srq the projection's input (no-op when scale==0).
      av[ki] = srq4(vec4<f32>(f32(a[kb]), f32(a[kb + 1u]), f32(a[kb + 2u]), f32(a[kb + 3u])), params.projInScale);
      aAcc = aAcc + (av[ki].x + av[ki].y) + (av[ki].z + av[ki].w);
    }
  }
  var accs: array<f32, SG_ROWS>;
  for (var r: u32 = 0u; r < SG_ROWS; r = r + 1u) {
    let o = rowBase + r;
    var acc: f32 = 0.0;
    if (o < OUT_F) {
      for (var ki: u32 = 0u; ki < K_ITER; ki = ki + 1u) {
        let k4 = lane + ki * 32u;
        if (k4 < KV4) {
          acc = acc + dot(unpack4x8unorm(codes[o * KV4 + k4]), av[ki]);
        }
      }
    }
    accs[r] = acc;
  }
  let aSum = sg_sum(aAcc);
  for (var r: u32 = 0u; r < SG_ROWS; r = r + 1u) {
    let s = sg_sum(accs[r]);
    let o = rowBase + r;
    if (lane == 0u && o < OUT_F) {
      // fma(s, 255, -128*aSum) undoes the unorm 1/255 decode and the +128 code bias.
      atomicStore(&pp[o], bitcast<u32>(srq(row_scale[o] * fma(s, 255.0, -128.0 * aSum), params.projOutScale)));
    }
  }
  storageBarrier();

  // --- last-arriver norm tail (all WG threads of the final workgroup) ---
  if (tid == 0u) {
    let ticket = atomicAdd(&pp[OUT_F], 1u);
    lastFlag = select(0u, 1u, ticket == TOTAL_WGS - 1u);
  }
  if (workgroupUniformLoad(&lastFlag) != 1u) {
    return;
  }
  if (tid == 0u) { atomicStore(&pp[OUT_F], 0u); }
  let inScale = params.inScale;
  let sv = w12s[2u * OUT_F];

  // rms over proj
  var acc1: f32 = 0.0;
  var i: u32 = tid;
  loop {
    if (i >= OUT_F) { break; }
    let v = bitcast<f32>(atomicLoad(&pp[i]));
    acc1 = acc1 + v * v;
    i = i + WG;
  }
  let rms1 = inverseSqrt(reduce_sum(acc1, tid) / f32(OUT_F) + EPS);

  // hidden update (kept in registers for the second norm)
  var hloc: array<f32, ELEMS>;
  var acc2: f32 = 0.0;
  var j: u32 = tid;
  var e: u32 = 0u;
  loop {
    if (j >= OUT_F) { break; }
    let normed = bitcast<f32>(atomicLoad(&pp[j])) * rms1 * f32(w12s[j]);
    let hv = f32(f32((f32(hidden[j]) + normed) * sv));
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
    let n2 = hloc[e] * rms2 * f32(w12s[OUT_F + j]);
    let qv = f32(srq(f32(f32(n2)), inScale));
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