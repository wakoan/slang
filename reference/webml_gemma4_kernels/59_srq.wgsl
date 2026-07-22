enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
diagnostic(off, chromium.subgroup_matrix_uniformity);

struct Params { inScale: f32, outScale: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// Subgroup-matrix f16 GEMM for prefill per-layer-embedding projections
// (M >= 64). A (the activation, SRQ-quantized per the QAT wrapper then cast to
// f16) and B (the dense weight cast to f16) are staged into workgroup memory,
// then multiplied with 8x8 f16 MMAs and f32 accumulation. Each weight tile is
// loaded once and reused across TILE_ROWS activation rows. Per element:
//   out = srq(sum_k srq(a[m,k], inScale) * w[o,k], outScale)   (srq is a no-op when scale == 0).
// Tile = 32 M x 64 N x 32 K, 128-thread workgroup = 4 subgroups, each owning a 16x32 subtile
// (2x4 of 8x8 result matrices), matching the QAT subgroup-matrix tile geometry.

const IN_F:      u32 = 1536u;
const OUT_F:     u32 = 256u;
const M_TOTAL:   u32 = 256u;
const TILE_COLS: u32 = 64u;
const TILE_ROWS: u32 = 32u;
const TILE_K:    u32 = 32u;
const SUB_COLS:  u32 = 32u;
const SUB_ROWS:  u32 = 16u;

var<workgroup> tile_A: array<f16, 32 * 32>;
var<workgroup> tile_B: array<f16, 64 * 32>;
var<workgroup> scratch: array<array<f32, 64>, 4>;

fn srq(x: f32, s: f32) -> f32 {
  if (s == 0.0) { return x; }
  return clamp(round(x / s), -128.0, 127.0) * s;
}

// A tile: 32 m-rows x 32 k of f16(srq(activation, inScale)). One k-tile column-strip per thread.
fn loadSHMA(tile_base: u32, k_idx: u32, row: u32, c_idx: u32, inScale: f32) {
  let a_global: u32 = tile_base + row;
  let col: u32 = c_idx * 8u;
  for (var col_offset: u32 = 0u; col_offset < 8u; col_offset++) {
    let k: u32 = k_idx + col + col_offset;
    var v: f32 = 0.0;
    if (a_global < M_TOTAL) {
      v = srq(f32(a[a_global * IN_F + k]), inScale);
    }
    tile_A[row * TILE_K + col + col_offset] = f16(v);
  }
}

// B tile: 64 output rows x 32 k of f16(weight). 2048 f16 over 128 threads = 16 each.
fn loadSHMB(tile_base: u32, k_idx: u32, lin: u32) {
  for (var i: u32 = lin; i < TILE_COLS * TILE_K; i += 128u) {
    let r = i / TILE_K;
    let c = i % TILE_K;
    tile_B[i] = f16(f32(wt[(tile_base + r) * IN_F + k_idx + c]));
  }
}

fn storeOutput(offset: u32, row: u32, col: u32, src_slot: u32, row_limit: i32, col_base: u32, outScale: f32) {
  if (row_limit > 0 && row < u32(row_limit)) {
    let c1 = scratch[src_slot][row * 8u + col];
    let c2 = scratch[src_slot][row * 8u + col + 1u];
    out[offset + row * OUT_F + col] = f32(srq(c1, outScale));
    out[offset + row * OUT_F + col + 1u] = f32(srq(c2, outScale));
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
  let inScale = params.inScale;
  let outScale = params.outScale;

  let subtile_id: u32 = local_idx / sg_size;
  let subtile_idx: u32 = subtile_id / 2u;
  let subtile_idy: u32 = subtile_id % 2u;
  let base_A: u32 = subtile_idy * SUB_ROWS;
  let base_B: u32 = subtile_idx * SUB_COLS;

  var matC00: subgroup_matrix_result<f32, 8, 8>;
  var matC01: subgroup_matrix_result<f32, 8, 8>;
  var matC02: subgroup_matrix_result<f32, 8, 8>;
  var matC03: subgroup_matrix_result<f32, 8, 8>;
  var matC10: subgroup_matrix_result<f32, 8, 8>;
  var matC11: subgroup_matrix_result<f32, 8, 8>;
  var matC12: subgroup_matrix_result<f32, 8, 8>;
  var matC13: subgroup_matrix_result<f32, 8, 8>;

  for (var kidx: u32 = 0u; kidx < IN_F; kidx += TILE_K) {
    loadSHMA(a_global_base, kidx, local_idx / 4u, local_idx % 4u, inScale);
    loadSHMB(w_global_base, kidx, local_idx);
    workgroupBarrier();

    for (var step: u32 = 0u; step < TILE_K; step += 8u) {
      let matrix_a_offset: u32 = subtile_idy * SUB_ROWS * TILE_K + step;
      var matA0: subgroup_matrix_left<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&tile_A, matrix_a_offset, false, TILE_K);
      var matA1: subgroup_matrix_left<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&tile_A, matrix_a_offset + 8u * TILE_K, false, TILE_K);

      let matrix_b_offset: u32 = subtile_idx * SUB_COLS * TILE_K + step;
      var matB0: subgroup_matrix_right<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_B, matrix_b_offset, true, TILE_K);
      var matB1: subgroup_matrix_right<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_B, matrix_b_offset +  8u * TILE_K, true, TILE_K);
      var matB2: subgroup_matrix_right<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_B, matrix_b_offset + 16u * TILE_K, true, TILE_K);
      var matB3: subgroup_matrix_right<f16, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&tile_B, matrix_b_offset + 24u * TILE_K, true, TILE_K);

      matC00 = subgroupMatrixMultiplyAccumulate(matA0, matB0, matC00);
      matC01 = subgroupMatrixMultiplyAccumulate(matA0, matB1, matC01);
      matC02 = subgroupMatrixMultiplyAccumulate(matA0, matB2, matC02);
      matC03 = subgroupMatrixMultiplyAccumulate(matA0, matB3, matC03);
      matC10 = subgroupMatrixMultiplyAccumulate(matA1, matB0, matC10);
      matC11 = subgroupMatrixMultiplyAccumulate(matA1, matB1, matC11);
      matC12 = subgroupMatrixMultiplyAccumulate(matA1, matB2, matC12);
      matC13 = subgroupMatrixMultiplyAccumulate(matA1, matB3, matC13);
    }
    workgroupBarrier();
  }

  let row: u32 = sg_id / 4u;
  let col: u32 = (sg_id % 4u) * 2u;
  var matrix_c_offset: u32 = (a_global_base + base_A) * OUT_F + w_global_base + base_B;
  var row_limit: i32 = i32(M_TOTAL) - i32(a_global_base + base_A);
  subgroupMatrixStore(&scratch[subtile_id], 0u, matC00, false, 8u);
  storeOutput(matrix_c_offset, row, col, subtile_id, row_limit, w_global_base + base_B, outScale);
  subgroupMatrixStore(&scratch[subtile_id], 0u, matC01, false, 8u);
  storeOutput(matrix_c_offset + 8u, row, col, subtile_id, row_limit, w_global_base + base_B + 8u, outScale);
  subgroupMatrixStore(&scratch[subtile_id], 0u, matC02, false, 8u);
  storeOutput(matrix_c_offset + 16u, row, col, subtile_id, row_limit, w_global_base + base_B + 16u, outScale);
  subgroupMatrixStore(&scratch[subtile_id], 0u, matC03, false, 8u);
  storeOutput(matrix_c_offset + 24u, row, col, subtile_id, row_limit, w_global_base + base_B + 24u, outScale);

  matrix_c_offset = matrix_c_offset + 8u * OUT_F;
  row_limit = i32(M_TOTAL) - i32(a_global_base + base_A + 8u);
  subgroupMatrixStore(&scratch[subtile_id], 0u, matC10, false, 8u);
  storeOutput(matrix_c_offset, row, col, subtile_id, row_limit, w_global_base + base_B, outScale);
  subgroupMatrixStore(&scratch[subtile_id], 0u, matC11, false, 8u);
  storeOutput(matrix_c_offset + 8u, row, col, subtile_id, row_limit, w_global_base + base_B + 8u, outScale);
  subgroupMatrixStore(&scratch[subtile_id], 0u, matC12, false, 8u);
  storeOutput(matrix_c_offset + 16u, row, col, subtile_id, row_limit, w_global_base + base_B + 16u, outScale);
  subgroupMatrixStore(&scratch[subtile_id], 0u, matC13, false, 8u);
  storeOutput(matrix_c_offset + 24u, row, col, subtile_id, row_limit, w_global_base + base_B + 24u, outScale);
}