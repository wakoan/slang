struct Params { seq: u32, heads: u32, dstOffset: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> cosTbl: array<f32>;
@group(0) @binding(3) var<storage, read> sinTbl: array<f32>;
@group(0) @binding(4) var<storage, read_write> yn: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

// Fused q/k RMSNorm + split-half RoPE, one workgroup per (seq, head). The
// normalized q/k row is rotated and written directly to its destination without
// an intermediate buffer. Numerically identical to com.xenova.RMSNorm (f32
// reduction, weight is the full multiplier) followed by com.xenova.Rope1d
// (split-half): yn[d] = nd*cos - nh*sin ; yn[d+half] = nh*cos + nd*sin,
// n = x/sqrt(mean(x^2)+eps)*w.

const HEAD_DIM: u32 = 512u;
const HALF_DIM: u32 = 256u;
const WG: u32 = 128u;
const EPS: f32 = 0.000001;

var<workgroup> red: array<f32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wg.x;
  let h = wg.y;
  if (t >= params.seq || h >= params.heads) { return; }
  let tid = lid.x;
  let base = (t * params.heads + h) * HEAD_DIM;
  // dstOffset lets the output land directly in the KV cache at the per-token position (folds the
  // separate strided cache-write op into this one). 0 for q (writes a plain qn buffer).
  let outBase = params.dstOffset + base;
  let csBase = t * HALF_DIM;

  // RMS reduction over the head dim (f32).
  var ss: f32 = 0.0;
  var d: u32 = tid;
  loop {
    if (d >= HEAD_DIM) { break; }
    let v = f32(x[base + d]);
    ss = ss + v * v;
    d = d + WG;
  }
  red[tid] = ss;
  workgroupBarrier();
  var stride: u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
    stride = stride / 2u;
    workgroupBarrier();
  }
  let scale = inverseSqrt(red[0] / f32(HEAD_DIM) + EPS);

  // Apply norm * weight, then split-half RoPE on pairs (k, k+half).
  var k: u32 = tid;
  loop {
    if (k >= HALF_DIM) { break; }
    let n0 = f32(x[base + k]) * scale * f32(w[k]);
    let n1 = f32(x[base + k + HALF_DIM]) * scale * f32(w[k + HALF_DIM]);
    let c = cosTbl[csBase + k];
    let s = sinTbl[csBase + k];
    yn[outBase + k] = f32(n0 * c - n1 * s);
    yn[outBase + k + HALF_DIM] = f32(n1 * c + n0 * s);
    k = k + WG;
  }
}