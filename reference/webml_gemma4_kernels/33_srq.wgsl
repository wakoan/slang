struct Params { inScale: f32, outScale: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> bits_buf: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// QAT GEMV specialization (M=1 decode):
//   - thread-per-column: each thread computes COLS full output columns with f32 accumulators and
//     no cross-thread reduction;
//   - the packed weight is read as vec4<u32> (128-bit loads) and dequantized with unpack4xU8.
// Per-row scale (one scale per output column) and a fixed integer ZP, matching the QAT checkpoint.
//
// Presrq stream path:
//   - the activation arrives already srq'd with its row sum (sum_a), so the dot uses the
//     unsigned codes (no per-value -ZP vec subs); the ZP correction folds into the epilogue:
//     sum_k (q-ZP)*a = sum_k q*a - ZP*sum_a, matching the scalar presrq algebra;
//   - activations are read as vec4 directly from device memory, so no workgroup
//     activation tile is needed;
//   - the block dot is fully unrolled with per-word partial sums combined as a tree, which
//     shortens the serial FMA dependency chain.
const K: u32 = 1536u;
const N: u32 = 262144u;
const BITS: u32 = 2u;
const ZP: f32 = 2.0;
const TILE_N: u32 = 128u;                  // threads per workgroup
const VPV: u32 = 128u / 2u;            // weight values per vec4<u32> (32 for 4-bit, 64 for 2-bit)
const NUM_BLK: u32 = 96u / 4u; // vec4<u32> blocks per output row (== K / VPV)
const GRID_X: u32 = 2048u;

fn srq(x: f32, s: f32) -> f32 {
  if (s == 0.0) { return x; }
  return clamp(round(x / s), -128.0, 127.0) * s;
}

var<workgroup> a_tile: array<f32, K>;

fn block_dot(bv: vec4<u32>, aBase: u32) -> f32 {
  var s: f32 = 0.0;
  for (var j: u32 = 0u; j < 4u; j = j + 1u) {
    let packed = bv[j];
    let d0 = vec4<f32>(unpack4xU8(packed & 0x03030303u)) - vec4<f32>(ZP);
    let d1 = vec4<f32>(unpack4xU8((packed >> 2u) & 0x03030303u)) - vec4<f32>(ZP);
    let d2 = vec4<f32>(unpack4xU8((packed >> 4u) & 0x03030303u)) - vec4<f32>(ZP);
    let d3 = vec4<f32>(unpack4xU8((packed >> 6u) & 0x03030303u)) - vec4<f32>(ZP);
    let base = aBase + j * 16u;
    s = s + dot(vec4<f32>(d0.x, d1.x, d2.x, d3.x), vec4<f32>(a_tile[base], a_tile[base + 1u], a_tile[base + 2u], a_tile[base + 3u]))
          + dot(vec4<f32>(d0.y, d1.y, d2.y, d3.y), vec4<f32>(a_tile[base + 4u], a_tile[base + 5u], a_tile[base + 6u], a_tile[base + 7u]))
          + dot(vec4<f32>(d0.z, d1.z, d2.z, d3.z), vec4<f32>(a_tile[base + 8u], a_tile[base + 9u], a_tile[base + 10u], a_tile[base + 11u]))
          + dot(vec4<f32>(d0.w, d1.w, d2.w, d3.w), vec4<f32>(a_tile[base + 12u], a_tile[base + 13u], a_tile[base + 14u], a_tile[base + 15u]));
  }
  return s;
}

@compute @workgroup_size(TILE_N, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let col = (wg.y * GRID_X + wg.x) * TILE_N + lid.x;
  let inScale = params.inScale;

  var id: u32 = lid.x;
  loop {
    if (id >= K) { break; }
    a_tile[id] = srq(f32(a[id]), inScale);
    id = id + TILE_N;
  }
  workgroupBarrier();

  if (col < N) {
    // BLOCK-MAJOR weights (repacked at load): block b of every column is contiguous, so the
    // TILE_N threads of this workgroup read consecutive vec4<u32>s — fully coalesced.
    var acc: f32 = 0.0;
    var blk: u32 = 0u;
    loop {
      if (blk >= NUM_BLK) { break; }
      acc = acc + block_dot(bits_buf[blk * N + col], blk * VPV);
      blk = blk + 1u;
    }
    out[col] = f32(srq(scale[col] * acc, params.outScale));
  }
}
