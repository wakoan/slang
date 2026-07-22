struct Params { seq: u32, heads: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read> cosTbl: array<f32>;
@group(0) @binding(2) var<storage, read> sinTbl: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const HEAD_DIM: u32 = 512u;
const HALF_DIM: u32 = 256u;
const WG: u32 = 64u;

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wg.x;
  let h = wg.y;
  if (t >= params.seq || h >= params.heads) {
    return;
  }
  let tid = lid.x;
  let qBase = (t * params.heads + h) * HEAD_DIM;
  let csBase = t * HALF_DIM;

  var k: u32 = tid;
  loop {
    if (k >= HALF_DIM) {
      break;
    }
    let c = cosTbl[csBase + k];
    let s = sinTbl[csBase + k];
    let x0 = f32(q[qBase + k]);
    let x1 = f32(q[qBase + k + HALF_DIM]);
    q[qBase + k] = f32(x0 * c - x1 * s);
    q[qBase + k + HALF_DIM] = f32(x1 * c + x0 * s);
    k = k + WG;
  }
}