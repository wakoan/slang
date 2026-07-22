enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);

struct Params { inScale: f32, gateOutScale: f32, upOutScale: f32, _pad0: u32 };
@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> gate_bits: array<u32>;
@group(0) @binding(2) var<storage, read> gate_scale: array<f32>;
@group(0) @binding(3) var<storage, read> up_bits: array<u32>;
@group(0) @binding(4) var<storage, read> up_scale: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<storage, read> gelu_lut: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

// Subgroup-matrix (tensor-core) fused gate/up prefill GEMM (M >= 64), integer codes domain —
// the QatMatMul gemm_sgmat structure with TWO weight streams sharing one A tile: per K-tile
// the loaders dequant gate AND up packed words to (code - ZP) f16 tiles, the A loader
// quantizes the activations to int8 codes (round(a / inScale), matching staged-path SRQ),
// and each subgroup accumulates 8 gate + 8 up 8x8 result matrices in f32 (integer-exact:
// |w-ZP| * 127 * K stays far inside 2^24). Epilogue per element:
//   g = srq(gate_scale[o] * (inScale * Cg), gateOut); u likewise; out = gelu_grid(g) * u.
// Tile geometry: 128-thread WG = 4 subgroups, each owning a 16x32 output
// subtile; TILE = 32 M x 64 N x 32 K.

const IN_F:      u32 = 1536u;
const OUT_F:     u32 = 6144u;
const M_TOTAL:   u32 = 128u;
const WPR:       u32 = 192u;
const ZP:        f32 = 8.0;
const TILE_COLS: u32 = 64u;
const TILE_ROWS: u32 = 32u;
const TILE_K:    u32 = 32u;
const SUB_COLS:  u32 = 32u;
const SUB_ROWS:  u32 = 16u;

var<workgroup> tile_A: array<f16, 32 * 32>;
var<workgroup> tile_G: array<f16, 64 * 32>;
var<workgroup> tile_U: array<f16, 64 * 32>;
var<workgroup> scratchG: array<array<f32, 64>, 4>;
var<workgroup> scratchU: array<array<f32, 64>, 4>;

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

// gelu over a grid input g = k * S: host-f64 table lookup gives every fused
// path a fixed rounded activation value.
fn gelu_grid(g: f32, s: f32) -> f32 {
  if (s == 0.0) { return gelu_tanh(g); }
  return gelu_lut[u32(clamp(round(g / s), -128.0, 127.0) + 128.0)];
}

fn loadSHMA(tile_base: u32, k_idx: u32, row: u32, c_idx: u32, invS: f32) {
  let a_global: u32 = tile_base + row;
  let col: u32 = c_idx * 8u;
  for (var col_offset: u32 = 0u; col_offset < 8u; col_offset++) {
    let k: u32 = k_idx + col + col_offset;
    var code: f32 = 0.0;
    if (a_global < M_TOTAL) {
      code = clamp(round(f32(hidden[a_global * IN_F + k]) * invS), -128.0, 127.0);
    }
    tile_A[row * TILE_K + col + col_offset] = f16(code);
  }
}

// Dequant BOTH weight tiles: gate and up words for the same (row, k-chunk) per visit.
fn loadSHMB(tile_base: u32, k_idx: u32, lin: u32) {
  for (var i: u32 = 0u; i < 2u; i++) {
    let lin2 = lin + i * 128u;
    let r = lin2 / 4u;
    let w = lin2 % 4u;
    let wordIdx = (tile_base + r) * WPR + (k_idx / 8u) + w;
    let pg = gate_bits[wordIdx];
    let pu = up_bits[wordIdx];
    let kb = r * TILE_K + w * 8u;
    for (var j: u32 = 0u; j < 8u; j++) {
      let sh = 8u * (j >> 1u) + 4u * (j & 1u);
      tile_G[kb + j] = f16(f32((pg >> sh) & 0xFu) - ZP);
      tile_U[kb + j] = f16(f32((pu >> sh) & 0xFu) - ZP);
    }
  }
}

fn storeOutput(offset: u32, row: u32, col: u32, src_slot: u32, row_limit: i32, col_base: u32, sEff: f32) {
  if (row_limit > 0 && row < u32(row_limit)) {
    let gOut = params.gateOutScale;
    let uOut = params.upOutScale;
    for (var cc: u32 = 0u; cc < 2u; cc++) {
      let o = col_base + col + cc;
      let g = srq(gate_scale[o] * (scratchG[src_slot][row * 8u + col + cc] * sEff), gOut);
      let u = srq(up_scale[o] * (scratchU[src_slot][row * 8u + col + cc] * sEff), uOut);
      out[offset + row * OUT_F + col + cc] = f32(gelu_grid(g, gOut) * u);
    }
  }
}

@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) local_idx: u32,
  @builtin(subgroup_invocation_id) sg_id: u32,
  @builtin(subgroup_size) sg_size: u32
) {
  let a_global_base: u32 = workgroup_id.y * TILE_ROWS;
  let w_global_base: u32 = workgroup_id.x * TILE_COLS;

  let sEff = params.inScale;
  let invS = 1.0 / sEff;

  let subtile_id: u32 = local_idx / sg_size;
  let subtile_idx: u32 = subtile_id / 2u;
  let subtile_idy: u32 = subtile_id % 2u;
  let base_A: u32 = subtile_idy * SUB_ROWS;
  let base_B: u32 = subtile_idx * SUB_COLS;

  var gC00: subgroup_matrix_result<f32, 8, 8>;
  var gC01: subgroup_matrix_result<f32, 8, 8>;
  var gC02: subgroup_matrix_result<f32, 8, 8>;
  var gC03: subgroup_matrix_result<f32, 8, 8>;
  var gC10: subgroup_matrix_result<f32, 8, 8>;
  var gC11: subgroup_matrix_result<f32, 8, 8>;
  var gC12: subgroup_matrix_result<f32, 8, 8>;
  var gC13: subgroup_matrix_result<f32, 8, 8>;
  var uC00: subgroup_matrix_result<f32, 8, 8>;
  var uC01: subgroup_matrix_result<f32, 8, 8>;
  var uC02: subgroup_matrix_result<f32, 8, 8>;
  var uC03: subgroup_matrix_result<f32, 8, 8>;
  var uC10: subgroup_matrix_result<f32, 8, 8>;
  var uC11: subgroup_matrix_result<f32, 8, 8>;
  var uC12: subgroup_matrix_result<f32, 8, 8>;
  var uC13: subgroup_matrix_result<f32, 8, 8>;

  for (var kidx: u32 = 0u; kidx < IN_F; kidx += TILE_K) {
    loadSHMA(a_global_base, kidx, local_idx / 4u, local_idx % 4u, invS);
    loadSHMB(w_global_base, kidx, local_idx);
    workgroupBarrier();

    for (var step: u32 = 0u; step < TILE_K; step += 8u) {
      let matrix_a_offset: u32 = subtile_idy * SUB_ROWS * TILE_K + step;
      var matA0: subgroup_matrix_left<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&tile_A, matrix_a_offset, false, TILE_K);
      var matA1: subgroup_matrix_left<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&tile_A, matrix_a_offset + 8u * TILE_K, false, TILE_K);

      let matrix_b_offset: u32 = subtile_idx * SUB_COLS * TILE_K + step;
      var gB0: subgroup_matrix_right<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_G, matrix_b_offset, true, TILE_K);
      var gB1: subgroup_matrix_right<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_G, matrix_b_offset +  8u * TILE_K, true, TILE_K);
      var gB2: subgroup_matrix_right<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_G, matrix_b_offset + 16u * TILE_K, true, TILE_K);
      var gB3: subgroup_matrix_right<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_G, matrix_b_offset + 24u * TILE_K, true, TILE_K);
      gC00 = subgroupMatrixMultiplyAccumulate(matA0, gB0, gC00);
      gC01 = subgroupMatrixMultiplyAccumulate(matA0, gB1, gC01);
      gC02 = subgroupMatrixMultiplyAccumulate(matA0, gB2, gC02);
      gC03 = subgroupMatrixMultiplyAccumulate(matA0, gB3, gC03);
      gC10 = subgroupMatrixMultiplyAccumulate(matA1, gB0, gC10);
      gC11 = subgroupMatrixMultiplyAccumulate(matA1, gB1, gC11);
      gC12 = subgroupMatrixMultiplyAccumulate(matA1, gB2, gC12);
      gC13 = subgroupMatrixMultiplyAccumulate(matA1, gB3, gC13);

      var uB0: subgroup_matrix_right<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_U, matrix_b_offset, true, TILE_K);
      var uB1: subgroup_matrix_right<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_U, matrix_b_offset +  8u * TILE_K, true, TILE_K);
      var uB2: subgroup_matrix_right<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_U, matrix_b_offset + 16u * TILE_K, true, TILE_K);
      var uB3: subgroup_matrix_right<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_U, matrix_b_offset + 24u * TILE_K, true, TILE_K);
      uC00 = subgroupMatrixMultiplyAccumulate(matA0, uB0, uC00);
      uC01 = subgroupMatrixMultiplyAccumulate(matA0, uB1, uC01);
      uC02 = subgroupMatrixMultiplyAccumulate(matA0, uB2, uC02);
      uC03 = subgroupMatrixMultiplyAccumulate(matA0, uB3, uC03);
      uC10 = subgroupMatrixMultiplyAccumulate(matA1, uB0, uC10);
      uC11 = subgroupMatrixMultiplyAccumulate(matA1, uB1, uC11);
      uC12 = subgroupMatrixMultiplyAccumulate(matA1, uB2, uC12);
      uC13 = subgroupMatrixMultiplyAccumulate(matA1, uB3, uC13);
    }
    workgroupBarrier();
  }

  let row: u32 = sg_id / 4u;
  let col: u32 = (sg_id % 4u) * 2u;
  var matrix_c_offset: u32 = (a_global_base + base_A) * OUT_F + w_global_base + base_B;
  var col_base: u32 = w_global_base + base_B;
  var row_limit: i32 = i32(M_TOTAL) - i32(a_global_base + base_A);
  subgroupMatrixStore(&scratchG[subtile_id], 0u, gC00, false, 8u);
  subgroupMatrixStore(&scratchU[subtile_id], 0u, uC00, false, 8u);
  storeOutput(matrix_c_offset, row, col, subtile_id, row_limit, col_base, sEff);
  subgroupMatrixStore(&scratchG[subtile_id], 0u, gC01, false, 8u);
  subgroupMatrixStore(&scratchU[subtile_id], 0u, uC01, false, 8u);
  storeOutput(matrix_c_offset + 8u, row, col, subtile_id, row_limit, col_base + 8u, sEff);
  subgroupMatrixStore(&scratchG[subtile_id], 0u, gC02, false, 8u);
  subgroupMatrixStore(&scratchU[subtile_id], 0u, uC02, false, 8u);
  storeOutput(matrix_c_offset + 16u, row, col, subtile_id, row_limit, col_base + 16u, sEff);
  subgroupMatrixStore(&scratchG[subtile_id], 0u, gC03, false, 8u);
  subgroupMatrixStore(&scratchU[subtile_id], 0u, uC03, false, 8u);
  storeOutput(matrix_c_offset + 24u, row, col, subtile_id, row_limit, col_base + 24u, sEff);

  matrix_c_offset = matrix_c_offset + 8u * OUT_F;
  row_limit = i32(M_TOTAL) - i32(a_global_base + base_A + 8u);
  subgroupMatrixStore(&scratchG[subtile_id], 0u, gC10, false, 8u);
  subgroupMatrixStore(&scratchU[subtile_id], 0u, uC10, false, 8u);
  storeOutput(matrix_c_offset, row, col, subtile_id, row_limit, col_base, sEff);
  subgroupMatrixStore(&scratchG[subtile_id], 0u, gC11, false, 8u);
  subgroupMatrixStore(&scratchU[subtile_id], 0u, uC11, false, 8u);
  storeOutput(matrix_c_offset + 8u, row, col, subtile_id, row_limit, col_base + 8u, sEff);
  subgroupMatrixStore(&scratchG[subtile_id], 0u, gC12, false, 8u);
  subgroupMatrixStore(&scratchU[subtile_id], 0u, uC12, false, 8u);
  storeOutput(matrix_c_offset + 16u, row, col, subtile_id, row_limit, col_base + 16u, sEff);
  subgroupMatrixStore(&scratchG[subtile_id], 0u, gC13, false, 8u);
  subgroupMatrixStore(&scratchU[subtile_id], 0u, uC13, false, 8u);
  storeOutput(matrix_c_offset + 24u, row, col, subtile_id, row_limit, col_base + 24u, sEff);
}