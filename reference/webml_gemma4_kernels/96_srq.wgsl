enable f16;
enable subgroups;
struct Params { inScale: f32, outScale: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> a: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> bits_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> pp: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> scale: array<f32>;
@group(0) @binding(4) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(5) var<storage, read> nw: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

// Single-dispatch fused down-projection + post-FFN residual norm-add (M=1 decode).
// N_ROWS output rows per workgroup, each reducing the full K row (no K-split):
// threads stride the row's packed words (lane-coalesced), activation vec4 loads
// are amortized over all N_ROWS rows, and the per-row scale/ZP/SRQ epilogue
// runs in the GEMV phase (one subgroup tree per workgroup). The last-arriver
// tail re-reads only the final OUT_F d values.
//   - each workgroup bumps an atomic ticket counter (in `pp`); the last workgroup merges:
//     hidden = hidden + RMSNorm(d) * w  (d values re-read through the atomics)
// pp layout: [0 .. OUT_F)  final d values (bitcast f32 through atomic u32 — the WGSL memory
//                          model only guarantees cross-workgroup visibility through atomics)
//            [OUT_F]       ticket counter (reset by the merge for the next replay)

const OUT_F: u32 = 1536u;
const CHUNKS: u32 = 4u;
const WORDS_PER_ROW: u32 = 768u;
const ZP: f32 = 2.0;
const WG: u32 = 256u;
const N_ROWS: u32 = 4u;
const TOTAL_WGS: u32 = 384u;
const EPS: f32 = 0.000001;
const COUNTER_IDX: u32 = OUT_F;

var<workgroup> dsh: array<f32, OUT_F>;
var<workgroup> sgq: array<vec4<f32>, WG / 32u>;
var<workgroup> sgs: array<f32, WG / 32u>;
var<workgroup> lastFlag: u32;

fn srq(x: f32, s: f32) -> f32 {
  if (s == 0.0) { return x; }
  return clamp(round(x / s), -128.0, 127.0) * s;
}

fn srq4(x: vec4<f32>, s: f32) -> vec4<f32> {
  if (s == 0.0) { return x; }
  return clamp(round(x / s), vec4<f32>(-128.0), vec4<f32>(127.0)) * s;
}

// Sum over each logical 32-lane block. sgExact32 (fixed 32-wide adapter) -> hardware
// subgroupAdd; otherwise a 32-lane subgroupShuffleXor butterfly that reduces each block
// independently, correct for any subgroup width >= 32 (NVIDIA D3D12 [32,128], AMD [32,64]).
fn sg_sum(value: f32) -> f32 {
  return subgroupAdd(value);
}
fn sg_sum_v4(value: vec4<f32>) -> vec4<f32> {
  return subgroupAdd(value);
}

fn reduce_sum(value: f32, tid: u32) -> f32 {
  // Hybrid 2-barrier reduction (sg_sum within each 32-block + cross-block combine).
  let s = sg_sum(value);
  if ((tid & 31u) == 0u) { sgs[tid >> 5u] = s; }
  workgroupBarrier();
  var total: f32 = 0.0;
  for (var i: u32 = 0u; i < WG / 32u; i = i + 1u) { total = total + sgs[i]; }
  workgroupBarrier();
  return total;
}

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let rowBase = wg.x * N_ROWS;
  let inScale = params.inScale;

  var q0: f32 = 0.0;
  var q1: f32 = 0.0;
  var q2: f32 = 0.0;
  var q3: f32 = 0.0;
  var sumA: f32 = 0.0;
  var w: u32 = tid;
  loop {
    if (w >= WORDS_PER_ROW) { break; }
    // Codes mode: `a` holds int8 SRQ codes (producer-quantized); the grid
    // scale is applied once per row in the epilogue. With the unorm weight
    // lanes below, the dot uses c/255-rounded lanes and has the same small
    // drift profile as the other presrq GEMVs.
    let av0 = vec4<f32>(a[w * CHUNKS + 0u]);
    let av1 = vec4<f32>(a[w * CHUNKS + 1u]);
    let av2 = vec4<f32>(a[w * CHUNKS + 2u]);
    let av3 = vec4<f32>(a[w * CHUNKS + 3u]);
    sumA = sumA + (av0.x + av0.y + av0.z + av0.w) + (av1.x + av1.y + av1.z + av1.w) + (av2.x + av2.y + av2.z + av2.w) + (av3.x + av3.y + av3.z + av3.w);
    {
      let o = rowBase + 0u;
      if (o < OUT_F) {
        let p = bits_buf[o * WORDS_PER_ROW + w];
        // unorm cvt-fold: unpack4x8unorm gives fl(code/255); the x255 decode
        // is undone once per output row in the epilogue.
        let d0 = unpack4x8unorm(p & 0x03030303u);
        let d1 = unpack4x8unorm((p >> 2u) & 0x03030303u);
        let d2 = unpack4x8unorm((p >> 4u) & 0x03030303u);
        let d3 = unpack4x8unorm((p >> 6u) & 0x03030303u);
        q0 = q0 + ((dot(vec4<f32>(d0.x, d1.x, d2.x, d3.x), av0)
                              + dot(vec4<f32>(d0.y, d1.y, d2.y, d3.y), av1))
                             + (dot(vec4<f32>(d0.z, d1.z, d2.z, d3.z), av2)
                              + dot(vec4<f32>(d0.w, d1.w, d2.w, d3.w), av3)));
      }
    }
    {
      let o = rowBase + 1u;
      if (o < OUT_F) {
        let p = bits_buf[o * WORDS_PER_ROW + w];
        // unorm cvt-fold: unpack4x8unorm gives fl(code/255); the x255 decode
        // is undone once per output row in the epilogue.
        let d0 = unpack4x8unorm(p & 0x03030303u);
        let d1 = unpack4x8unorm((p >> 2u) & 0x03030303u);
        let d2 = unpack4x8unorm((p >> 4u) & 0x03030303u);
        let d3 = unpack4x8unorm((p >> 6u) & 0x03030303u);
        q1 = q1 + ((dot(vec4<f32>(d0.x, d1.x, d2.x, d3.x), av0)
                              + dot(vec4<f32>(d0.y, d1.y, d2.y, d3.y), av1))
                             + (dot(vec4<f32>(d0.z, d1.z, d2.z, d3.z), av2)
                              + dot(vec4<f32>(d0.w, d1.w, d2.w, d3.w), av3)));
      }
    }
    {
      let o = rowBase + 2u;
      if (o < OUT_F) {
        let p = bits_buf[o * WORDS_PER_ROW + w];
        // unorm cvt-fold: unpack4x8unorm gives fl(code/255); the x255 decode
        // is undone once per output row in the epilogue.
        let d0 = unpack4x8unorm(p & 0x03030303u);
        let d1 = unpack4x8unorm((p >> 2u) & 0x03030303u);
        let d2 = unpack4x8unorm((p >> 4u) & 0x03030303u);
        let d3 = unpack4x8unorm((p >> 6u) & 0x03030303u);
        q2 = q2 + ((dot(vec4<f32>(d0.x, d1.x, d2.x, d3.x), av0)
                              + dot(vec4<f32>(d0.y, d1.y, d2.y, d3.y), av1))
                             + (dot(vec4<f32>(d0.z, d1.z, d2.z, d3.z), av2)
                              + dot(vec4<f32>(d0.w, d1.w, d2.w, d3.w), av3)));
      }
    }
    {
      let o = rowBase + 3u;
      if (o < OUT_F) {
        let p = bits_buf[o * WORDS_PER_ROW + w];
        // unorm cvt-fold: unpack4x8unorm gives fl(code/255); the x255 decode
        // is undone once per output row in the epilogue.
        let d0 = unpack4x8unorm(p & 0x03030303u);
        let d1 = unpack4x8unorm((p >> 2u) & 0x03030303u);
        let d2 = unpack4x8unorm((p >> 4u) & 0x03030303u);
        let d3 = unpack4x8unorm((p >> 6u) & 0x03030303u);
        q3 = q3 + ((dot(vec4<f32>(d0.x, d1.x, d2.x, d3.x), av0)
                              + dot(vec4<f32>(d0.y, d1.y, d2.y, d3.y), av1))
                             + (dot(vec4<f32>(d0.z, d1.z, d2.z, d3.z), av2)
                              + dot(vec4<f32>(d0.w, d1.w, d2.w, d3.w), av3)));
      }
    }
    w = w + WG;
  }

  let red = sg_sum_v4(vec4<f32>(q0, q1, q2, q3));
  let redA = sg_sum(sumA);
  if ((tid & 31u) == 0u) { sgq[tid >> 5u] = red; sgs[tid >> 5u] = redA; }
  workgroupBarrier();
  if (tid == 0u) {
    var tot = vec4<f32>(0.0);
    var aSum: f32 = 0.0;
    for (var i: u32 = 0u; i < WG / 32u; i = i + 1u) { tot = tot + sgq[i]; aSum = aSum + sgs[i]; }
    let outScale = params.outScale;
    let zpA = ZP * aSum;
    {
      let o = rowBase + 0u;
      if (o < OUT_F) {
        // fma(q, 255, -zpA) undoes the partial phase's unorm 1/255 decode scale.
        let d = srq(scale[o] * (inScale * fma(tot[0u], 255.0, -zpA)), outScale);
        atomicStore(&pp[o], bitcast<u32>(d));
      }
    }
    {
      let o = rowBase + 1u;
      if (o < OUT_F) {
        // fma(q, 255, -zpA) undoes the partial phase's unorm 1/255 decode scale.
        let d = srq(scale[o] * (inScale * fma(tot[1u], 255.0, -zpA)), outScale);
        atomicStore(&pp[o], bitcast<u32>(d));
      }
    }
    {
      let o = rowBase + 2u;
      if (o < OUT_F) {
        // fma(q, 255, -zpA) undoes the partial phase's unorm 1/255 decode scale.
        let d = srq(scale[o] * (inScale * fma(tot[2u], 255.0, -zpA)), outScale);
        atomicStore(&pp[o], bitcast<u32>(d));
      }
    }
    {
      let o = rowBase + 3u;
      if (o < OUT_F) {
        // fma(q, 255, -zpA) undoes the partial phase's unorm 1/255 decode scale.
        let d = srq(scale[o] * (inScale * fma(tot[3u], 255.0, -zpA)), outScale);
        atomicStore(&pp[o], bitcast<u32>(d));
      }
    }
  }
  storageBarrier();

  if (tid == 0u) {
    let ticket = atomicAdd(&pp[COUNTER_IDX], 1u);
    lastFlag = select(0u, 1u, ticket == TOTAL_WGS - 1u);
  }
  // workgroupUniformLoad = implicit barrier + a value the uniformity analysis accepts
  // (the merge tail below contains workgroupBarrier calls).
  if (workgroupUniformLoad(&lastFlag) != 1u) {
    return;
  }

  // ---- norm-add tail (last workgroup, all WG threads) ----
  if (tid == 0u) { atomicStore(&pp[COUNTER_IDX], 0u); }
  var acc: f32 = 0.0;
  var o2: u32 = tid;
  loop {
    if (o2 >= OUT_F) { break; }
    let d = bitcast<f32>(atomicLoad(&pp[o2]));
    dsh[o2] = d;
    acc = acc + d * d;
    o2 = o2 + WG;
  }
  let rms = inverseSqrt(reduce_sum(acc, tid) / f32(OUT_F) + EPS);

  o2 = tid;
  loop {
    if (o2 >= OUT_F) { break; }
    hidden[o2] = f32(f32(hidden[o2]) + dsh[o2] * rms * f32(nw[o2]));
    o2 = o2 + WG;
  }
}