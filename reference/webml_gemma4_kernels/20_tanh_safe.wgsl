struct Params { count: u32, wgY: u32, bOffset: u32, gridScale: f32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> gelu_lut: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Fused GeGLU multiply: y = gelu_tanh(a) * b, used for both the main-MLP gate
// and the per-layer-input gate.
// gelu_tanh / tanh_safe match ai.onnx.Gelu (approximate="tanh"), including
// tanh clamping to +/-1 past |x| > 10.

const WG: u32 = 256u;

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

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wg_idx = wg.x + wg.y * params.wgY;
  let i = wg_idx * WG + lid.x;
  if (i >= params.count) {
    return;
  }
  // b may be a larger tensor read at a fixed offset, such as the per-layer
  // slice of pleNorm.
  y[i] = f32(gelu_grid(f32(a[i]), params.gridScale) * f32(b[params.bOffset + i]));
}