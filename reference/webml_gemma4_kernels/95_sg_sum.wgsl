enable f16;
enable subgroups;
struct Params { gateOutScale: f32, upOutScale: f32, outQuantScale: f32, _pad0: u32 };
@group(0) @binding(0) var<storage, read> hidden: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> gate_bits: array<u32>;
@group(0) @binding(2) var<storage, read> gate_scale: array<f32>;
@group(0) @binding(3) var<storage, read> up_bits: array<u32>;
@group(0) @binding(4) var<storage, read> up_scale: array<f32>;
@group(0) @binding(5) var<storage, read> sum_a: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f16>;
@group(0) @binding(7) var<storage, read> gelu_lut: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

// presrq path for the fused gate/up GEMV: `hidden` is already srq-quantized and `sum_a[m]`
// holds its per-row sum (both produced by com.xenova.gemma4.DecodeRmsSrq). This removes the
// per-workgroup srq() over activation elements and the per-workgroup sumA reduction.
//   g   = srq(gate_scale[o] * (sum_k qg*a - ZP*sum_a), gateOutScale)
//   u   = srq(up_scale[o]   * (sum_k qu*a - ZP*sum_a), upOutScale)
//   out[o] = gelu_tanh(g) * u

const M: u32 = 1u;
const M_TILE: u32 = 1u;
const H: u32 = 1536u;
const INTER: u32 = 12288u;
const BITS: u32 = 2u;
const VPW: u32 = 16u;
const CHUNKS: u32 = 4u;
const WPR: u32 = 96u;
const MASK: u32 = 3u;
const ZP: f32 = 2.0;
const WG: u32 = 64u;
const SG_COUNT: u32 = 2u;
const N_ROWS: u32 = 2u;
const GRID_X: u32 = 3072u;


// Sum over each logical 32-lane virtual subgroup. sgExact32 (fixed 32-wide adapter) ->
// hardware subgroupAdd; otherwise a 32-lane subgroupShuffleXor butterfly that reduces each
// 32-block independently — correct for any subgroup width >= 32 (NVIDIA D3D12 [32,128],
// AMD [32,64]) where a plain subgroupAdd over the WG would merge the virtual units.
fn sg_sum(value: f32) -> f32 {
  return subgroupAdd(value);
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


@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  // Virtual-subgroup mode: each 32-lane subgroup acts as an independent GEMV unit (own row
  // group), so the dispatch uses WG/32x fewer, wider workgroups. No early return (the
  // trailing barrier must stay in uniform control flow); OOB virtual units idle in guards.
  let sgId = lid.x / 32u;
  let tid = lid.x & 31u;
  let wgId = (wg.y * GRID_X + wg.x) * SG_COUNT + sgId;
  let rowBase = wgId * N_ROWS;
  let gOut = params.gateOutScale;
  let uOut = params.upOutScale;

  let mEnd = min((wg.z + 1u) * M_TILE, M);
  for (var m: u32 = wg.z * M_TILE; m < mEnd; m = m + 1u) {
    let hV4Base = m * (H / 4u);
    var gAcc: array<f32, N_ROWS>;
    var uAcc: array<f32, N_ROWS>;
    for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) { gAcc[r] = 0.0; uAcc[r] = 0.0; }

    var wd: u32 = tid;
    loop {
      if (wd >= WPR) { break; }
      // The activation is already on the int8 grid: read it once (vec4), reuse
      // across rows, and upcast once per word to f32. The dots run against
      // unpack4x8unorm code lanes; the lanes are fl(code/255), and the x255
      // decode is undone once per output row in the epilogue.
      var avc: array<vec4<f16>, CHUNKS>;
      for (var c: u32 = 0u; c < CHUNKS; c = c + 1u) {
        avc[c] = hidden[hV4Base + wd * CHUNKS + c];
      }
      var avcf: array<vec4<f32>, CHUNKS>;
      for (var c: u32 = 0u; c < CHUNKS; c = c + 1u) {
        avcf[c] = vec4<f32>(avc[c]);
      }
      for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) {
        let o = rowBase + r;
        if (o < INTER) {
          let pg = gate_bits[o * WPR + wd];
          let pu = up_bits[o * WPR + wd];
          let g0 = unpack4x8unorm(pg & 0x03030303u);
          let g1 = unpack4x8unorm((pg >> 2u) & 0x03030303u);
          let g2 = unpack4x8unorm((pg >> 4u) & 0x03030303u);
          let g3 = unpack4x8unorm((pg >> 6u) & 0x03030303u);
          gAcc[r] = gAcc[r] + ((dot(vec4<f32>(g0.x, g1.x, g2.x, g3.x), avcf[0])
                              + dot(vec4<f32>(g0.y, g1.y, g2.y, g3.y), avcf[1]))
                             + (dot(vec4<f32>(g0.z, g1.z, g2.z, g3.z), avcf[2])
                              + dot(vec4<f32>(g0.w, g1.w, g2.w, g3.w), avcf[3])));
          let u0 = unpack4x8unorm(pu & 0x03030303u);
          let u1 = unpack4x8unorm((pu >> 2u) & 0x03030303u);
          let u2 = unpack4x8unorm((pu >> 4u) & 0x03030303u);
          let u3 = unpack4x8unorm((pu >> 6u) & 0x03030303u);
          uAcc[r] = uAcc[r] + ((dot(vec4<f32>(u0.x, u1.x, u2.x, u3.x), avcf[0])
                              + dot(vec4<f32>(u0.y, u1.y, u2.y, u3.y), avcf[1]))
                             + (dot(vec4<f32>(u0.z, u1.z, u2.z, u3.z), avcf[2])
                              + dot(vec4<f32>(u0.w, u1.w, u2.w, u3.w), avcf[3])));
        }
      }
      wd = wd + 32u;
    }

    let aSum = sum_a[m];
    for (var r: u32 = 0u; r < N_ROWS; r = r + 1u) {
      let gS = reduce_sum(gAcc[r], lid.x);
      let uS = reduce_sum(uAcc[r], lid.x);
      if (tid == 0u) {
        let o = rowBase + r;
        if (o < INTER) {
          // fma(x, 255, -zp*sum) undoes the unorm 1/255 decode scale once per output row.
          let g = srq(gate_scale[o] * fma(gS, 255.0, -(ZP * aSum)), gOut);
          let u = srq(up_scale[o] * fma(uS, 255.0, -(ZP * aSum)), uOut);
          // Codes mode: emit the down projection's int8 SRQ code
          // (clamp(round(x/s)), exactly representable in f16). The consumer
          // multiplies by the grid scale once per row after its integer-exact
          // reduction, avoiding per-element srq division in the down GEMV
          // without forcing an f32 buffer.
          let dq = gelu_grid(g, gOut) * u;
          let qs = params.outQuantScale;
          var code: f32;
          if (qs == 0.0) { code = dq; } else { code = clamp(round(dq / qs), -128.0, 127.0); }
          out[m * INTER + o] = f16(code);
        }
      }
    }
    workgroupBarrier();
  }
}
