struct Params { inScale: f32, outScale: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bits_buf: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> sum_a: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

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

const COLS: u32 = 1u;                  // output columns per thread
const K4: u32 = K / 4u;


// Unsigned-code dot over one vec4<u32> weight block (VPV values) against vec4
// activation reads straight from device memory. No workgroup a_tile is used.
// aBase is in vec4 units (block * VPV/4).
//
// Unorm conversion fold: code lanes are produced by unpack4x8unorm, which yields
// fl(code / 255). The x255 decode is undone once per column in the epilogue.
// Lane values are c/255-rounded; reference-parity paths use the exact kernels.
fn block_dot(bv: vec4<u32>, aBase: u32) -> f32 {
  let p0 = bv[0u];
  let d00 = unpack4x8unorm(p0 & 0x03030303u);
  let d10 = unpack4x8unorm((p0 >> 2u) & 0x03030303u);
  let d20 = unpack4x8unorm((p0 >> 4u) & 0x03030303u);
  let d30 = unpack4x8unorm((p0 >> 6u) & 0x03030303u);
  let s0 = (dot(vec4<f32>(d00.x, d10.x, d20.x, d30.x), vec4<f32>(a[aBase + 0u]))
                + dot(vec4<f32>(d00.y, d10.y, d20.y, d30.y), vec4<f32>(a[aBase + 1u])))
               + (dot(vec4<f32>(d00.z, d10.z, d20.z, d30.z), vec4<f32>(a[aBase + 2u]))
                + dot(vec4<f32>(d00.w, d10.w, d20.w, d30.w), vec4<f32>(a[aBase + 3u])));
  let p1 = bv[1u];
  let d01 = unpack4x8unorm(p1 & 0x03030303u);
  let d11 = unpack4x8unorm((p1 >> 2u) & 0x03030303u);
  let d21 = unpack4x8unorm((p1 >> 4u) & 0x03030303u);
  let d31 = unpack4x8unorm((p1 >> 6u) & 0x03030303u);
  let s1 = (dot(vec4<f32>(d01.x, d11.x, d21.x, d31.x), vec4<f32>(a[aBase + 4u]))
                + dot(vec4<f32>(d01.y, d11.y, d21.y, d31.y), vec4<f32>(a[aBase + 5u])))
               + (dot(vec4<f32>(d01.z, d11.z, d21.z, d31.z), vec4<f32>(a[aBase + 6u]))
                + dot(vec4<f32>(d01.w, d11.w, d21.w, d31.w), vec4<f32>(a[aBase + 7u])));
  let p2 = bv[2u];
  let d02 = unpack4x8unorm(p2 & 0x03030303u);
  let d12 = unpack4x8unorm((p2 >> 2u) & 0x03030303u);
  let d22 = unpack4x8unorm((p2 >> 4u) & 0x03030303u);
  let d32 = unpack4x8unorm((p2 >> 6u) & 0x03030303u);
  let s2 = (dot(vec4<f32>(d02.x, d12.x, d22.x, d32.x), vec4<f32>(a[aBase + 8u]))
                + dot(vec4<f32>(d02.y, d12.y, d22.y, d32.y), vec4<f32>(a[aBase + 9u])))
               + (dot(vec4<f32>(d02.z, d12.z, d22.z, d32.z), vec4<f32>(a[aBase + 10u]))
                + dot(vec4<f32>(d02.w, d12.w, d22.w, d32.w), vec4<f32>(a[aBase + 11u])));
  let p3 = bv[3u];
  let d03 = unpack4x8unorm(p3 & 0x03030303u);
  let d13 = unpack4x8unorm((p3 >> 2u) & 0x03030303u);
  let d23 = unpack4x8unorm((p3 >> 4u) & 0x03030303u);
  let d33 = unpack4x8unorm((p3 >> 6u) & 0x03030303u);
  let s3 = (dot(vec4<f32>(d03.x, d13.x, d23.x, d33.x), vec4<f32>(a[aBase + 12u]))
                + dot(vec4<f32>(d03.y, d13.y, d23.y, d33.y), vec4<f32>(a[aBase + 13u])))
               + (dot(vec4<f32>(d03.z, d13.z, d23.z, d33.z), vec4<f32>(a[aBase + 14u]))
                + dot(vec4<f32>(d03.w, d13.w, d23.w, d33.w), vec4<f32>(a[aBase + 15u])));
  return (s0 + s1) + (s2 + s3);
}

@compute @workgroup_size(TILE_N, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let colBase = (wg.y * GRID_X + wg.x) * (TILE_N * COLS) + lid.x;

  let col0 = colBase + 0u * TILE_N;
  var acc0: f32 = 0.0;

  // BLOCK-MAJOR weights (repacked at load): block b of every column is contiguous, so the
  // TILE_N threads of this workgroup read consecutive vec4<u32>s — fully coalesced (one
  // contiguous run per column slot).
  var blk: u32 = 0u;
  loop {
    if (blk >= NUM_BLK) { break; }
    let aBase = blk * (VPV / 4u);
    if (col0 < N) {
      acc0 = acc0 + block_dot(bits_buf[blk * N + col0], aBase);
    }
    blk = blk + 1u;
  }

  let zpA = ZP * sum_a[0];
  if (col0 < N) {
    // x255 undoes the unorm 1/255 decode scale once per column.
    let v0 = f32(srq(scale[col0] * fma(acc0, 255.0, -zpA), params.outScale));
    out[col0] = v0;
  }
}
