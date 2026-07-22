enable subgroups;
struct Params { inScale: f32, linOutScale: f32, pleOffset: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read> row_scale: array<f32>;
@group(0) @binding(3) var<storage, read> ple: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> gelu_lut: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

// Codes path for the fused per-layer-input gate: the int8 dense weight streams
// as packed +128-biased u8 codes (4/u32) plus a per-row scale. unpack4x8unorm
// lanes decode as fl((c+128)/255); the bias and the x255 unorm decode are
// undone once per output row in the epilogue:
//   w·a = row_scale[o] * (255*sum_k(u_k*a_k) - 128*sum_k(a_k))
// This matches the unorm decode fold used by the other presrq GEMV kernels.
//   out[m,o] = gelu_grid(srq(w·a, linOutScale)) * ple[pleOffset + m*outF + o]

const M: u32 = 1u;
const IN_FEATURES: u32 = 1536u;
const OUT_FEATURES: u32 = 256u;
const WG: u32 = 32u;
const N_ROWS: u32 = 1u;
const WPR: u32 = 1536u / 4u;
const GRID_X: u32 = 256u;


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

fn tanh_safe(x: f32) -> f32 {
  if (x > 10.0) { return 1.0; }
  if (x < -10.0) { return -1.0; }
  return tanh(x);
}
fn gelu_tanh(v: f32) -> f32 {
  return 0.5 * v * (1.0 + tanh_safe(0.7978845608028654 * (v + 0.044715 * v * v * v)));
}
// gelu over a grid input g = k * S (k in [-128,127]): the host-f64 table fixes
// the rounded activation value for every fused path.
fn gelu_grid(g: f32, s: f32) -> f32 {
  if (s == 0.0) { return gelu_tanh(g); }
  return gelu_lut[u32(clamp(round(g / s), -128.0, 127.0) + 128.0)];
}


@compute @workgroup_size(32, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wgId = wg.y * GRID_X + wg.x;
  let rowBase = wgId * N_ROWS;
  if (rowBase >= OUT_FEATURES) { return; }
  let tid = lid.x;

  for (var m: u32 = 0u; m < M; m = m + 1u) {
    let aBase = m * IN_FEATURES;
    var acc: array<f32, N_ROWS>;
    for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) { acc[r] = 0.0; }
    var aAcc: f32 = 0.0;
    var wd: u32 = tid;
    loop {
      if (wd >= WPR) { break; }
      let kb = wd * 4u;
      // QAT wrapper: srq the gate linear's input (no-op when scale==0).
      let a4 = srq4(vec4<f32>(f32(a[aBase + kb]), f32(a[aBase + kb + 1u]), f32(a[aBase + kb + 2u]), f32(a[aBase + kb + 3u])), params.inScale);
      aAcc = aAcc + (a4.x + a4.y) + (a4.z + a4.w);
      for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) {
        let o = rowBase + r;
        if (o < OUT_FEATURES) {
          acc[r] = acc[r] + dot(unpack4x8unorm(codes[o * WPR + wd]), a4);
        }
      }
      wd = wd + WG;
    }
    let aSum = reduce(aAcc, tid);
    for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) {
      let s = reduce(acc[r], tid);
      let o = rowBase + r;
      if (tid == 0u && o < OUT_FEATURES) {
        // fma(s, 255, -128*aSum) undoes the unorm 1/255 decode and the +128 code bias.
        let v = row_scale[o] * fma(s, 255.0, -128.0 * aSum);
        out[m * OUT_FEATURES + o] = f32(gelu_grid(srq(v, params.linOutScale), params.linOutScale) * f32(ple[params.pleOffset + m * OUT_FEATURES + o]));
      }
    }
  }
}