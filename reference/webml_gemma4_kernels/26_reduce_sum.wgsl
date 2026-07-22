struct Params { rows: u32, rowStride: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const DIM: u32 = 512u;
const EPS: f32 = 0.000001;
const WG: u32 = 64u;

var<workgroup> partial: array<f32, WG>;

fn reduce_sum(value: f32, tid: u32) -> f32 {
  partial[tid] = value;
  workgroupBarrier();
  var stride = WG / 2u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (tid < stride) {
      partial[tid] = partial[tid] + partial[tid + stride];
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  return partial[0];
}

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let rowStride = select(params.rowStride, params.rows, params.rowStride == 0u);
  let row = wg.x + wg.y * rowStride;
  if (row >= params.rows) {
    return;
  }
  let tid = lid.x;
  let base = row * DIM;

  // Compute sum of squares.
  var acc: f32 = 0.0;
  var i: u32 = tid;
  loop {
    if (i >= DIM) {
      break;
    }
    let v = f32(x[base + i]);
    acc = acc + v * v;
    i = i + WG;
  }
  let scale = inverseSqrt(reduce_sum(acc, tid) / f32(DIM) + EPS);

  // Apply normalization (+ optional weight).
  var j: u32 = tid;
  loop {
    if (j >= DIM) {
      break;
    }
    let xv = f32(x[base + j]);
    y[base + j] = f32(xv * scale);
    j = j + WG;
  }
}