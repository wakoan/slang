struct Params { rows: u32, copyCols: u32, srcStride: u32, srcStart: u32, dstStride: u32, dstStart: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> s: array<f32>;
@group(0) @binding(1) var<storage, read_write> d: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.x;
  if (r >= p.rows) {
    return;
  }
  var i: u32 = lid.x;
  loop {
    if (i >= p.copyCols) {
      break;
    }
    d[r * p.dstStride + p.dstStart + i] = f32(f32(s[r * p.srcStride + p.srcStart + i]));
    i = i + 64u;
  }
}