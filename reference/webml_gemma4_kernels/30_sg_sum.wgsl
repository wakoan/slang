enable subgroups;
struct Params { inScale: f32, gateOutScale: f32, upOutScale: f32, _pad0: u32 };
@group(0) @binding(0) var<storage, read> hidden: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> gate_bits: array<u32>;
@group(0) @binding(2) var<storage, read> gate_scale: array<f32>;
@group(0) @binding(3) var<storage, read> up_bits: array<u32>;
@group(0) @binding(4) var<storage, read> up_scale: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<storage, read> gelu_lut: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

// presrq path for the fused gate/up GEMV: `hidden` is already srq-quantized and `sum_a[m]`
// holds its per-row sum (both produced by com.xenova.gemma4.DecodeRmsSrq). This removes the
// per-workgroup srq() over activation elements and the per-workgroup sumA reduction.
//   g   = srq(gate_scale[o] * (sum_k qg*a - ZP*sum_a), gateOutScale)
//   u   = srq(up_scale[o]   * (sum_k qu*a - ZP*sum_a), upOutScale)
//   out[o] = gelu_tanh(g) * u

const M: u32 = 32u;
const M_TILE: u32 = 2u;
const H: u32 = 1536u;
const INTER: u32 = 12288u;
const BITS: u32 = 2u;
const VPW: u32 = 16u;
const CHUNKS: u32 = 4u;
const WPR: u32 = 96u;
const MASK: u32 = 3u;
const ZP: f32 = 2.0;
const WG: u32 = 32u;
const SG_COUNT: u32 = 1u;
const N_ROWS: u32 = 2u;
const GRID_X: u32 = 768u;


// Sum over each logical 32-lane virtual subgroup. sgExact32 (fixed 32-wide adapter) ->
// hardware subgroupAdd; otherwise a 32-lane subgroupShuffleXor butterfly that reduces each
// 32-block independently — correct for any subgroup width >= 32 (NVIDIA D3D12 [32,128],
// AMD [32,64]) where a plain subgroupAdd over the WG would merge the virtual units.
fn sg_sum(value: f32) -> f32 {
  var x = value;
  x = x + subgroupShuffleXor(x, 1u);
  x = x + subgroupShuffleXor(x, 2u);
  x = x + subgroupShuffleXor(x, 4u);
  x = x + subgroupShuffleXor(x, 8u);
  x = x + subgroupShuffleXor(x, 16u);
  return x;
}

fn reduce_sum(value: f32, lidx: u32) -> f32 {
  return sg_sum(value);
}

fn srq(x: f32, s: f32) -> f32 {
  if (s == 0.0) { return x; }
  return clamp(round(x / s), -128.0, 127.0) * s;
}

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


// Register-blocked presrq GEMM tile for prefill (M >= 16): each thread owns an N_PT x M_PT
// (inter-row x token) accumulator block for both the gate and up streams and runs the full
// serial k-loop, so every gate/up weight word is loaded and dequantized once for all M token
// rows in the tile. Two weight streams double the live accumulator/register pressure, so the
// tile shape keeps the gate/up accumulator footprint bounded.
const THREADS_N: u32 = 16u;
const THREADS_M: u32 = 16u;
const N_PT: u32 = 1u;
const M_PT: u32 = 2u;
const TILE_N: u32 = THREADS_N * N_PT;
const TILE_M: u32 = THREADS_M * M_PT;

fn srq4(x: vec4<f32>, s: f32) -> vec4<f32> {
  if (s == 0.0) { return x; }
  return clamp(round(x / s), vec4<f32>(-128.0), vec4<f32>(127.0)) * s;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let wgId = wg.y * GRID_X + wg.x;
  let tid = lid.x;
  let nSub = tid % THREADS_N;
  let mSub = tid / THREADS_N;
  let nBase = wgId * TILE_N + nSub * N_PT;
  let mBase = wg.z * TILE_M + mSub * M_PT;
  let gOut = params.gateOutScale;
  let uOut = params.upOutScale;
  let inScale = params.inScale;

  let ro0 = nBase + 0u;
  let mr0 = mBase + 0u;
  let hBase0 = min(mr0, M - 1u) * (H / 4u);
  let mr1 = mBase + 1u;
  let hBase1 = min(mr1, M - 1u) * (H / 4u);
  var gAcc_0_0: f32 = 0.0;
  var uAcc_0_0: f32 = 0.0;
  var gAcc_0_1: f32 = 0.0;
  var uAcc_0_1: f32 = 0.0;
  var sA_0: f32 = 0.0;
  var sA_1: f32 = 0.0;

  var w: u32 = 0u;
  loop {
    if (w >= WPR) { break; }
    var pg0: u32 = 0u;
    var pu0: u32 = 0u;
    if (ro0 < INTER) {
      pg0 = gate_bits[ro0 * WPR + w];
      pu0 = up_bits[ro0 * WPR + w];
    }
    let g00 = unpack4x8unorm(pg0 & 0x03030303u);
    let g10 = unpack4x8unorm((pg0 >> 2u) & 0x03030303u);
    let g20 = unpack4x8unorm((pg0 >> 4u) & 0x03030303u);
    let g30 = unpack4x8unorm((pg0 >> 6u) & 0x03030303u);
    let qg0_0 = vec4<f32>(g00.x, g10.x, g20.x, g30.x);
    let qg0_1 = vec4<f32>(g00.y, g10.y, g20.y, g30.y);
    let qg0_2 = vec4<f32>(g00.z, g10.z, g20.z, g30.z);
    let qg0_3 = vec4<f32>(g00.w, g10.w, g20.w, g30.w);
    let u00 = unpack4x8unorm(pu0 & 0x03030303u);
    let u10 = unpack4x8unorm((pu0 >> 2u) & 0x03030303u);
    let u20 = unpack4x8unorm((pu0 >> 4u) & 0x03030303u);
    let u30 = unpack4x8unorm((pu0 >> 6u) & 0x03030303u);
    let qu0_0 = vec4<f32>(u00.x, u10.x, u20.x, u30.x);
    let qu0_1 = vec4<f32>(u00.y, u10.y, u20.y, u30.y);
    let qu0_2 = vec4<f32>(u00.z, u10.z, u20.z, u30.z);
    let qu0_3 = vec4<f32>(u00.w, u10.w, u20.w, u30.w);
    {
      let a0_0 = srq4(vec4<f32>(hidden[hBase0 + w * CHUNKS + 0u]), inScale);
      sA_0 = sA_0 + a0_0.x + a0_0.y + a0_0.z + a0_0.w;
      let a0_1 = srq4(vec4<f32>(hidden[hBase0 + w * CHUNKS + 1u]), inScale);
      sA_0 = sA_0 + a0_1.x + a0_1.y + a0_1.z + a0_1.w;
      let a0_2 = srq4(vec4<f32>(hidden[hBase0 + w * CHUNKS + 2u]), inScale);
      sA_0 = sA_0 + a0_2.x + a0_2.y + a0_2.z + a0_2.w;
      let a0_3 = srq4(vec4<f32>(hidden[hBase0 + w * CHUNKS + 3u]), inScale);
      sA_0 = sA_0 + a0_3.x + a0_3.y + a0_3.z + a0_3.w;
      gAcc_0_0 = gAcc_0_0 + dot(qg0_0, a0_0);
      uAcc_0_0 = uAcc_0_0 + dot(qu0_0, a0_0);
      gAcc_0_0 = gAcc_0_0 + dot(qg0_1, a0_1);
      uAcc_0_0 = uAcc_0_0 + dot(qu0_1, a0_1);
      gAcc_0_0 = gAcc_0_0 + dot(qg0_2, a0_2);
      uAcc_0_0 = uAcc_0_0 + dot(qu0_2, a0_2);
      gAcc_0_0 = gAcc_0_0 + dot(qg0_3, a0_3);
      uAcc_0_0 = uAcc_0_0 + dot(qu0_3, a0_3);
    }
    {
      let a1_0 = srq4(vec4<f32>(hidden[hBase1 + w * CHUNKS + 0u]), inScale);
      sA_1 = sA_1 + a1_0.x + a1_0.y + a1_0.z + a1_0.w;
      let a1_1 = srq4(vec4<f32>(hidden[hBase1 + w * CHUNKS + 1u]), inScale);
      sA_1 = sA_1 + a1_1.x + a1_1.y + a1_1.z + a1_1.w;
      let a1_2 = srq4(vec4<f32>(hidden[hBase1 + w * CHUNKS + 2u]), inScale);
      sA_1 = sA_1 + a1_2.x + a1_2.y + a1_2.z + a1_2.w;
      let a1_3 = srq4(vec4<f32>(hidden[hBase1 + w * CHUNKS + 3u]), inScale);
      sA_1 = sA_1 + a1_3.x + a1_3.y + a1_3.z + a1_3.w;
      gAcc_0_1 = gAcc_0_1 + dot(qg0_0, a1_0);
      uAcc_0_1 = uAcc_0_1 + dot(qu0_0, a1_0);
      gAcc_0_1 = gAcc_0_1 + dot(qg0_1, a1_1);
      uAcc_0_1 = uAcc_0_1 + dot(qu0_1, a1_1);
      gAcc_0_1 = gAcc_0_1 + dot(qg0_2, a1_2);
      uAcc_0_1 = uAcc_0_1 + dot(qu0_2, a1_2);
      gAcc_0_1 = gAcc_0_1 + dot(qg0_3, a1_3);
      uAcc_0_1 = uAcc_0_1 + dot(qu0_3, a1_3);
    }
    w = w + 1u;
  }

  if (mr0 < M) {
    let zpA0 = ZP * sA_0;
    if (ro0 < INTER) {
      // fma(x, 255, -zp*sum) undoes the unorm 1/255 decode scale once per (m,o).
      let g = srq(gate_scale[ro0] * fma(gAcc_0_0, 255.0, -zpA0), gOut);
      let u = srq(up_scale[ro0] * fma(uAcc_0_0, 255.0, -zpA0), uOut);
      out[mr0 * INTER + ro0] = f32(gelu_grid(g, gOut) * u);
    }
  }
  if (mr1 < M) {
    let zpA1 = ZP * sA_1;
    if (ro0 < INTER) {
      // fma(x, 255, -zp*sum) undoes the unorm 1/255 decode scale once per (m,o).
      let g = srq(gate_scale[ro0] * fma(gAcc_0_1, 255.0, -zpA1), gOut);
      let u = srq(up_scale[ro0] * fma(uAcc_0_1, 255.0, -zpA1), uOut);
      out[mr1 * INTER + ro0] = f32(gelu_grid(g, gOut) * u);
    }
  }
}
