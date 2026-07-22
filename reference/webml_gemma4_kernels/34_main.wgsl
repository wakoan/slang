@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> cand_val: array<f32>;
@group(0) @binding(2) var<storage, read_write> cand_idx: array<u32>;

// Two-pass argmax, pass 1: each workgroup scans a contiguous slice of X and emits its local
// (max, index) candidate. Slices are contiguous and in order, and within a slice the strided
// scan + index tie-break keep first-on-ties semantics, so the final pass over candidates is
// exactly equivalent to the single-workgroup scan.

const COUNT: u32 = 262144u;
const SLICE: u32 = 1024u;
const WG: u32 = 256u;
const NEG_INF: f32 = -3.4028234663852886e38;

var<workgroup> wgVal: array<f32, WG>;
var<workgroup> wgIdx: array<u32, WG>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let base = wg.x * SLICE;
  let end = min(base + SLICE, COUNT);
  var bestVal: f32 = NEG_INF;
  var bestIdx: u32 = 0u;
  var i: u32 = base + tid;
  loop {
    if (i >= end) {
      break;
    }
    let v = f32(x[i]);
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
    i = i + WG;
  }
  wgVal[tid] = bestVal;
  wgIdx[tid] = bestIdx;
  workgroupBarrier();

  var stride: u32 = WG / 2u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (tid < stride) {
      let o = tid + stride;
      if (wgVal[o] > wgVal[tid] || (wgVal[o] == wgVal[tid] && wgIdx[o] < wgIdx[tid])) {
        wgVal[tid] = wgVal[o];
        wgIdx[tid] = wgIdx[o];
      }
    }
    stride = stride / 2u;
    workgroupBarrier();
  }

  if (tid == 0u) {
    cand_val[wg.x] = wgVal[0];
    cand_idx[wg.x] = wgIdx[0];
  }
}