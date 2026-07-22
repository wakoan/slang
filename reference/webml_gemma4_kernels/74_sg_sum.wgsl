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
const INTER: u32 = 6144u;
const BITS: u32 = 4u;
const VPW: u32 = 8u;
const CHUNKS: u32 = 2u;
const WPR: u32 = 192u;
const MASK: u32 = 15u;
const ZP: f32 = 8.0;
const WG: u32 = 64u;
const SG_COUNT: u32 = 2u;
const N_ROWS: u32 = 4u;
const GRID_X: u32 = 768u;


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
          let glo = unpack4x8unorm(pg & 0x0F0F0F0Fu);
          let ghi = unpack4x8unorm((pg >> 4u) & 0x0F0F0F0Fu);
          gAcc[r] = gAcc[r] + (dot(vec4<f32>(glo.x, ghi.x, glo.y, ghi.y), avcf[0])
                             + dot(vec4<f32>(glo.z, ghi.z, glo.w, ghi.w), avcf[1]));
          let ulo = unpack4x8unorm(pu & 0x0F0F0F0Fu);
          let uhi = unpack4x8unorm((pu >> 4u) & 0x0F0F0F0Fu);
          uAcc[r] = uAcc[r] + (dot(vec4<f32>(ulo.x, uhi.x, ulo.y, uhi.y), avcf[0])
                             + dot(vec4<f32>(ulo.z, uhi.z, ulo.w, uhi.w), avcf[1]));
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
