@group(0) @binding(0) var<storage, read> cand_val: array<f32>;
@group(0) @binding(1) var<storage, read> cand_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

// Two-pass argmax, pass 2: pick the winner among the per-slice candidates. Candidates are in
// slice order, so the index tie-break keeps first-on-ties semantics.

const NCAND: u32 = 256u;
const WG: u32 = 256u;
const NEG_INF: f32 = -3.4028234663852886e38;

var<workgroup> wgVal: array<f32, WG>;
var<workgroup> wgIdx: array<u32, WG>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var bestVal: f32 = NEG_INF;
  var bestIdx: u32 = 0u;
  var i: u32 = tid;
  loop {
    if (i >= NCAND) {
      break;
    }
    let v = cand_val[i];
    let idx = cand_idx[i];
    if (v > bestVal || (v == bestVal && idx < bestIdx)) {
      bestVal = v;
      bestIdx = idx;
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
    out[0] = wgIdx[0];
  }
}