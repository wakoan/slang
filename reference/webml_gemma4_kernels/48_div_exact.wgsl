struct Params { scale: f32, invScale: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Elementwise SRQ: y = clamp(round(x/scale), -128, 127) * scale (no-op when scale==0).
// Applied once per activation element so the downstream QatMatMul can skip per-output SRQ.
// The division is a Markstein sequence seeded with the host-computed fl(1/scale);
// native f32 division can be off by ulps and flip round() at exact-.5 grid ties.
const COUNT: u32 = 524288u;

fn div_exact(x: f32, s: f32, t: f32) -> f32 {
  let q0 = x * t;
  let r = fma(-s, q0, x);
  return fma(r, t, q0);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= COUNT) { return; }
  let s = params.scale;
  let v = f32(x[i]);
  let q = select(v, clamp(round(div_exact(v, s, params.invScale)), -128.0, 127.0) * s, s != 0.0);
  y[i] = f32(q);
}