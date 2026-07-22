enable f16;
enable subgroups;
struct Params { inScale: f32, outScale: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f16>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// Dense GEMV (no transpose): out[m, o] = sum_k W[o, k] * a[m, k]. Used for
// per-layer-embedding dense projections with small M. One workgroup (= one
// subgroup, WG=32) computes N_ROWS output rows; threads split K with coalesced
// vec4 weight reads + subgroupAdd reduction; the activation vec4 is read once
// per K-step and reused across the N_ROWS rows. W may be f16 with f32 activation.

const M: u32 = 32u;
const IN_FEATURES: u32 = 1536u;
const OUT_FEATURES: u32 = 8960u;
const WG: u32 = 32u;
const N_ROWS: u32 = 8u;
const KV4: u32 = 1536u / 4u;


fn reduce(value: f32, tid: u32) -> f32 {
  return subgroupAdd(value);
}

fn srq(x: f32, s: f32) -> f32 {
  if (s == 0.0) { return x; }
  return clamp(round(x / s), -128.0, 127.0) * s;
}

fn srq4(x: vec4<f32>, s: f32) -> vec4<f32> {
  if (s == 0.0) { return x; }
  return clamp(round(x / s), vec4<f32>(-128.0), vec4<f32>(127.0)) * s;
}

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wgId = wg.y * 1120u + wg.x;
  let rowBase = wgId * N_ROWS;
  if (rowBase >= OUT_FEATURES) {
    return;
  }
  let tid = lid.x;

  for (var m: u32 = 0u; m < M; m = m + 1u) {
    let aBase = m * IN_FEATURES;
    var acc: array<f32, N_ROWS>;
    for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) { acc[r] = 0.0; }

    var k4: u32 = tid;
    loop {
      if (k4 >= KV4) { break; }
      let kb = k4 * 4u;
      // QAT wrapper semantics: srq the linear's input and output (no-op when scale==0).
      let a4 = srq4(vec4<f32>(f32(a[aBase + kb]), f32(a[aBase + kb + 1u]), f32(a[aBase + kb + 2u]), f32(a[aBase + kb + 3u])), params.inScale);
      for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) {
        let o = rowBase + r;
        if (o < OUT_FEATURES) {
          let wb = o * IN_FEATURES + kb;
          let w4 = vec4<f32>(f32(wt[wb]), f32(wt[wb + 1u]), f32(wt[wb + 2u]), f32(wt[wb + 3u]));
          acc[r] = acc[r] + dot(w4, a4);
        }
      }
      k4 = k4 + WG;
    }

    for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) {
      let s = reduce(acc[r], tid);
      let o = rowBase + r;
      if (tid == 0u && o < OUT_FEATURES) {
        out[m * OUT_FEATURES + o] = f32(srq(s, params.outScale));
      }
    }
  }
}