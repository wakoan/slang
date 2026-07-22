struct Params { count: u32, period: u32, wgY: u32, _pad0: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> factor: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
const WG: u32 = 64u;
@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * params.wgY;
  let i = wg_idx * WG + lid.x;
  if (i >= params.count) {
    return;
  }
  var pIdx = i;
  if (params.period > 0u) {
    pIdx = i % params.period;
  }
  x[i] = f32(f32(x[i]) * f32(factor[pIdx]));
}