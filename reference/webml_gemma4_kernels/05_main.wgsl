struct Params { count: u32, wgY: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 64u;

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * params.wgY;
  let i = wg_idx * WG + lid.x;
  if (i >= params.count) {
    return;
  }

  let yv = f32(y[i]);
  let xv = f32(x[i]);
  y[i] = yv + xv;
}