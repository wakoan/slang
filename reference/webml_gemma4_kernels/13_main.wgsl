enable subgroups;
struct Params { seqQ: u32, keyLen: u32, qOffset: u32, qHeads: u32, kvHeads: u32, window: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

// Tiled flash prefill attention (seqQ >= 64): TILE_Q queries x LPQ-lane clusters per
// workgroup, K/V staged in workgroup memory once per tile for all TILE_Q queries.
//
// Each workgroup shares one K/V tile across TILE_Q queries and splits each query
// across an 8-lane subgroup cluster to fit f32/headDim register pressure:
//   - thread (qSub, lane8) holds q/o register slices of HEAD_DIM/8 = 32 dims (8 vec4s each)
//   - scores: per-lane partial dot + 3 subgroupShuffleXor adds (cluster-internal, no barriers)
//   - online softmax state (m, l) per thread, replicated across the cluster (identical values)
//   - V accumulate: o_slice += p_k * v_tile[k][slice] straight from workgroup memory
// K/V tiles are TILE_K=8 keys x HEAD_DIM, cooperatively loaded as vec4s.
//
// Causality/window are per-query masks; the key loop runs over the union range of the
// workgroup's queries (start at the first query's window floor, end at the last query's
// causal ceiling) — uniform trip count, masked probabilities for out-of-range (query, key)
// pairs. All bounds come from runtime uniforms, so replay can keep a stable
// dispatch shape.

const HEAD_DIM: u32 = 256u;
const TILE_Q: u32 = 16u;
const LPQ: u32 = 8u;                      // lanes per query cluster
const SLICE: u32 = HEAD_DIM / (4u * LPQ); // vec4s per lane slice
const TILE_K: u32 = 8u;
const WG: u32 = TILE_Q * LPQ;
const SCALE: f32 = 1;
const NEG_INF: f32 = -3.4028234663852886e38;

// Staged K/V tile dtype: f16 halves workgroup storage at headDim=512.
// Scores/PV still accumulate in f32 (converted on read), so only K/V carry f16
// rounding.
var<workgroup> k_tile: array<vec4<f32>, TILE_K * (HEAD_DIM / 4u)>;
var<workgroup> v_tile: array<vec4<f32>, TILE_K * (HEAD_DIM / 4u)>;

@compute @workgroup_size(WG, 1, 1)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let h = wg.y;
  let tid = lid.x;
  let qSub = tid / LPQ;
  let lane8 = tid % LPQ;
  let qIdx = wg.x * TILE_Q + qSub;
  let qValid = qIdx < params.seqQ && h < params.qHeads;

  let groupSize = params.qHeads / params.kvHeads;
  let hKv = h / groupSize;
  let qPos = params.qOffset + min(qIdx, params.seqQ - 1u);

  // Per-thread q slice (8 vec4s) and output accumulator (8 vec4s) in registers.
  let qBase4 = (min(qIdx, params.seqQ - 1u) * params.qHeads + h) * (HEAD_DIM / 4u) + lane8 * SLICE;
  var qr: array<vec4<f32>, SLICE>;
  var o: array<vec4<f32>, SLICE>;
  for (var c: u32 = 0u; c < SLICE; c = c + 1u) {
    qr[c] = vec4<f32>(q[qBase4 + c]);
    o[c] = vec4<f32>(0.0);
  }
  var m: f32 = NEG_INF;
  var l: f32 = 0.0;

  // Per-query causal/window bounds + the workgroup's union key range.
  let maxKj = min(params.keyLen, qPos + 1u);
  var minKj: u32 = 0u;
  if (params.window > 0u && qPos + 1u > params.window) {
    minKj = qPos + 1u - params.window;
  }
  let lastQPos = params.qOffset + min(wg.x * TILE_Q + TILE_Q - 1u, params.seqQ - 1u);
  let wgEnd = min(params.keyLen, lastQPos + 1u);
  let firstQPos = params.qOffset + wg.x * TILE_Q;
  var wgStart: u32 = 0u;
  if (params.window > 0u && firstQPos + 1u > params.window) {
    wgStart = firstQPos + 1u - params.window;
  }

  var kStart: u32 = wgStart;
  loop {
    if (kStart >= wgEnd) { break; }

    // --- cooperative K/V tile load (vec4-coalesced; OOB keys zero-filled) ---
    workgroupBarrier();
    for (var i: u32 = tid; i < TILE_K * (HEAD_DIM / 4u); i = i + WG) {
      let slot = i / (HEAD_DIM / 4u);
      let d4 = i % (HEAD_DIM / 4u);
      let kj = kStart + slot;
      let base4 = (kj * params.kvHeads + hKv) * (HEAD_DIM / 4u) + d4;
      if (kj < wgEnd) {
        k_tile[i] = vec4<f32>(k[base4]);
        v_tile[i] = vec4<f32>(v[base4]);
      } else {
        k_tile[i] = vec4<f32>(0.0);
        v_tile[i] = vec4<f32>(0.0);
      }
    }
    workgroupBarrier();

    // --- scores for this tile's keys (per-thread registers; cluster shuffle combine) ---
    var s0: f32 = NEG_INF;
    {
      let kj = kStart + 0u;
      var part: f32 = 0.0;
      let kb = 0u * (HEAD_DIM / 4u) + lane8 * SLICE;
      for (var c: u32 = 0u; c < SLICE; c = c + 1u) {
        part = part + dot(qr[c], vec4<f32>(k_tile[kb + c]));
      }
      part = part + subgroupShuffleXor(part, 1u);
      part = part + subgroupShuffleXor(part, 2u);
      part = part + subgroupShuffleXor(part, 4u);
      if (kj >= minKj && kj < maxKj) {
        s0 = part * SCALE;
      }
    }
    var s1: f32 = NEG_INF;
    {
      let kj = kStart + 1u;
      var part: f32 = 0.0;
      let kb = 1u * (HEAD_DIM / 4u) + lane8 * SLICE;
      for (var c: u32 = 0u; c < SLICE; c = c + 1u) {
        part = part + dot(qr[c], vec4<f32>(k_tile[kb + c]));
      }
      part = part + subgroupShuffleXor(part, 1u);
      part = part + subgroupShuffleXor(part, 2u);
      part = part + subgroupShuffleXor(part, 4u);
      if (kj >= minKj && kj < maxKj) {
        s1 = part * SCALE;
      }
    }
    var s2: f32 = NEG_INF;
    {
      let kj = kStart + 2u;
      var part: f32 = 0.0;
      let kb = 2u * (HEAD_DIM / 4u) + lane8 * SLICE;
      for (var c: u32 = 0u; c < SLICE; c = c + 1u) {
        part = part + dot(qr[c], vec4<f32>(k_tile[kb + c]));
      }
      part = part + subgroupShuffleXor(part, 1u);
      part = part + subgroupShuffleXor(part, 2u);
      part = part + subgroupShuffleXor(part, 4u);
      if (kj >= minKj && kj < maxKj) {
        s2 = part * SCALE;
      }
    }
    var s3: f32 = NEG_INF;
    {
      let kj = kStart + 3u;
      var part: f32 = 0.0;
      let kb = 3u * (HEAD_DIM / 4u) + lane8 * SLICE;
      for (var c: u32 = 0u; c < SLICE; c = c + 1u) {
        part = part + dot(qr[c], vec4<f32>(k_tile[kb + c]));
      }
      part = part + subgroupShuffleXor(part, 1u);
      part = part + subgroupShuffleXor(part, 2u);
      part = part + subgroupShuffleXor(part, 4u);
      if (kj >= minKj && kj < maxKj) {
        s3 = part * SCALE;
      }
    }
    var s4: f32 = NEG_INF;
    {
      let kj = kStart + 4u;
      var part: f32 = 0.0;
      let kb = 4u * (HEAD_DIM / 4u) + lane8 * SLICE;
      for (var c: u32 = 0u; c < SLICE; c = c + 1u) {
        part = part + dot(qr[c], vec4<f32>(k_tile[kb + c]));
      }
      part = part + subgroupShuffleXor(part, 1u);
      part = part + subgroupShuffleXor(part, 2u);
      part = part + subgroupShuffleXor(part, 4u);
      if (kj >= minKj && kj < maxKj) {
        s4 = part * SCALE;
      }
    }
    var s5: f32 = NEG_INF;
    {
      let kj = kStart + 5u;
      var part: f32 = 0.0;
      let kb = 5u * (HEAD_DIM / 4u) + lane8 * SLICE;
      for (var c: u32 = 0u; c < SLICE; c = c + 1u) {
        part = part + dot(qr[c], vec4<f32>(k_tile[kb + c]));
      }
      part = part + subgroupShuffleXor(part, 1u);
      part = part + subgroupShuffleXor(part, 2u);
      part = part + subgroupShuffleXor(part, 4u);
      if (kj >= minKj && kj < maxKj) {
        s5 = part * SCALE;
      }
    }
    var s6: f32 = NEG_INF;
    {
      let kj = kStart + 6u;
      var part: f32 = 0.0;
      let kb = 6u * (HEAD_DIM / 4u) + lane8 * SLICE;
      for (var c: u32 = 0u; c < SLICE; c = c + 1u) {
        part = part + dot(qr[c], vec4<f32>(k_tile[kb + c]));
      }
      part = part + subgroupShuffleXor(part, 1u);
      part = part + subgroupShuffleXor(part, 2u);
      part = part + subgroupShuffleXor(part, 4u);
      if (kj >= minKj && kj < maxKj) {
        s6 = part * SCALE;
      }
    }
    var s7: f32 = NEG_INF;
    {
      let kj = kStart + 7u;
      var part: f32 = 0.0;
      let kb = 7u * (HEAD_DIM / 4u) + lane8 * SLICE;
      for (var c: u32 = 0u; c < SLICE; c = c + 1u) {
        part = part + dot(qr[c], vec4<f32>(k_tile[kb + c]));
      }
      part = part + subgroupShuffleXor(part, 1u);
      part = part + subgroupShuffleXor(part, 2u);
      part = part + subgroupShuffleXor(part, 4u);
      if (kj >= minKj && kj < maxKj) {
        s7 = part * SCALE;
      }
    }

    // --- per-thread online softmax over the tile ---
    var tileMax: f32 = s0;
    tileMax = max(tileMax, s1);
    tileMax = max(tileMax, s2);
    tileMax = max(tileMax, s3);
    tileMax = max(tileMax, s4);
    tileMax = max(tileMax, s5);
    tileMax = max(tileMax, s6);
    tileMax = max(tileMax, s7);
    let newMax = max(m, tileMax);
    // All-masked tiles keep m = NEG_INF; exp(NEG_INF - NEG_INF) is NaN, so guard via select.
    let corr = select(exp(m - newMax), 0.0, m == NEG_INF);
    let p0 = select(0.0, exp(s0 - newMax), s0 != NEG_INF);
    let p1 = select(0.0, exp(s1 - newMax), s1 != NEG_INF);
    let p2 = select(0.0, exp(s2 - newMax), s2 != NEG_INF);
    let p3 = select(0.0, exp(s3 - newMax), s3 != NEG_INF);
    let p4 = select(0.0, exp(s4 - newMax), s4 != NEG_INF);
    let p5 = select(0.0, exp(s5 - newMax), s5 != NEG_INF);
    let p6 = select(0.0, exp(s6 - newMax), s6 != NEG_INF);
    let p7 = select(0.0, exp(s7 - newMax), s7 != NEG_INF);
    l = l * corr + (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7);
    for (var c: u32 = 0u; c < SLICE; c = c + 1u) {
      var acc = o[c] * corr;
      acc = acc + p0 * vec4<f32>(v_tile[0u * (HEAD_DIM / 4u) + lane8 * SLICE + c]);
      acc = acc + p1 * vec4<f32>(v_tile[1u * (HEAD_DIM / 4u) + lane8 * SLICE + c]);
      acc = acc + p2 * vec4<f32>(v_tile[2u * (HEAD_DIM / 4u) + lane8 * SLICE + c]);
      acc = acc + p3 * vec4<f32>(v_tile[3u * (HEAD_DIM / 4u) + lane8 * SLICE + c]);
      acc = acc + p4 * vec4<f32>(v_tile[4u * (HEAD_DIM / 4u) + lane8 * SLICE + c]);
      acc = acc + p5 * vec4<f32>(v_tile[5u * (HEAD_DIM / 4u) + lane8 * SLICE + c]);
      acc = acc + p6 * vec4<f32>(v_tile[6u * (HEAD_DIM / 4u) + lane8 * SLICE + c]);
      acc = acc + p7 * vec4<f32>(v_tile[7u * (HEAD_DIM / 4u) + lane8 * SLICE + c]);
      o[c] = acc;
    }
    m = newMax;

    kStart = kStart + TILE_K;
  }

  if (qValid) {
    let outBase4 = (qIdx * params.qHeads + h) * (HEAD_DIM / 4u) + lane8 * SLICE;
    let inv = 1.0 / l;
    for (var c: u32 = 0u; c < SLICE; c = c + 1u) {
      out[outBase4 + c] = vec4<f32>(o[c] * inv);
    }
  }
}