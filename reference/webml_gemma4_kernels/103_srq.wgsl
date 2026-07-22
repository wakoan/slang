enable subgroups;
struct Params { seqQ: u32, keyLen: u32, qOffset: u32, qHeads: u32, kvHeads: u32, window: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> cosTbl: array<f32>;
@group(0) @binding(3) var<storage, read> sinTbl: array<f32>;
@group(0) @binding(4) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> partials: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

// Decode (seqQ==1) fused attention, pass 1 of 2: per-(head, key-chunk) flash partials.
//
// Fuses q-head RMSNorm + split-half RoPE into the attention kernel. Each
// (head, chunk) workgroup owns a normalized/rotated query copy in shared memory,
// so chunks can run independently over their key ranges.
//
// The active key range is split into chunks. Each workgroup owns one chunk and
// writes flash partials (running max m, denominator l, unnormalized
// V-accumulator acc[hd]) to scratch; the last-arriver merge combines them in
// the same dispatch. RoPE tables carry cos=1/sin=0 beyond the partial-rotary
// cutoff, so rotating every pair is exact.
//
// The dispatch launches NCHUNK chunk workgroups for replay stability, while
// nActive = clamp(ceil(activeKeys / 64), 8, NCHUNK) chooses how many chunks
// actually participate. Surplus workgroups return before touching the
// last-arriver ticket. Sliding-window layers cap activeKeys at the window, so
// only full-attention layers fan out at long context.

const HEAD_DIM: u32 = 256u;
const HALF_DIM: u32 = 128u;
const NCHUNK: u32 = 32u;
const WG: u32 = 256u;
const EPS: f32 = 0.000001;
const SCALE: f32 = 1;
const NEG_INF: f32 = -3.4028234663852886e38;
const PP_COUNTER_BASE: u32 = 8u * NCHUNK * (HEAD_DIM + 2u);
// Pre-applies the o-projection's input SRQ at the merged output; this pass's
// uniform stays position-only.
const OUT_Q: f32 = 0.023868119344115257;

var<workgroup> lastFlag: u32;

fn srq(x: f32, s: f32) -> f32 {
  if (s == 0.0) { return x; }
  return clamp(round(x / s), -128.0, 127.0) * s;
}

var<workgroup> qn_sh: array<f32, HEAD_DIM>;
var<workgroup> out_acc: array<f32, HEAD_DIM>;
var<workgroup> probs: array<f32, WG>;
var<workgroup> sval_sh: array<f32, WG>;
var<workgroup> red: array<f32, WG>;
var<workgroup> wgt_sh: array<f32, NCHUNK>;
var<workgroup> vacc_sh: array<vec4<f32>, WG>;
var<workgroup> running_max: f32;
var<workgroup> running_denom: f32;

// Reductions over each logical 32-lane block. sgExact32 (fixed 32-wide adapter) -> hardware
// subgroup ops; otherwise a 32-lane subgroupShuffleXor butterfly that reduces each block
// independently — correct for any subgroup width >= 32 (NVIDIA D3D12 [32,128], AMD [32,64])
// where a plain subgroup op would span multiple 32-blocks.
fn sg_sum(value: f32) -> f32 {
  return subgroupAdd(value);
}
fn sg_max(value: f32) -> f32 {
  return subgroupMax(value);
}

// Hybrid 2-barrier reductions: 32-block reduce, followed by a cross-block combine.
fn reduce_max(value: f32, tid: u32) -> f32 {
  let s = sg_max(value);
  if ((tid & 31u) == 0u) { red[tid >> 5u] = s; }
  workgroupBarrier();
  var total: f32 = NEG_INF;
  for (var i: u32 = 0u; i < WG / 32u; i = i + 1u) { total = max(total, red[i]); }
  workgroupBarrier();
  return total;
}

fn reduce_sum(value: f32, tid: u32) -> f32 {
  let s = sg_sum(value);
  if ((tid & 31u) == 0u) { red[tid >> 5u] = s; }
  workgroupBarrier();
  var total: f32 = 0.0;
  for (var i: u32 = 0u; i < WG / 32u; i = i + 1u) { total = total + red[i]; }
  workgroupBarrier();
  return total;
}

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;
  let ci = wg.y;
  if (h >= params.qHeads) { return; }

  let tid = lid.x;
  let groupSize = params.qHeads / params.kvHeads;
  let hKv = h / groupSize;
  let qPos = params.qOffset;
  let qBase = h * HEAD_DIM;

  // --- runtime chunk partition (uniform per workgroup: params + builtins only) ---
  let maxKj = min(params.keyLen, qPos + 1u);
  var minKj: u32 = 0u;
  if (params.window > 0u && qPos + 1u > params.window) {
    minKj = qPos + 1u - params.window;
  }
  let activeKeys = maxKj - minKj;
  let nActive = clamp((activeKeys + 63u) / 64u, 8u, NCHUNK);
  if (ci >= nActive) { return; }

  // --- fused q RMSNorm (matches DecodeQkNormRope: f32, WG-tree, eps after /dim) ---
  var ss: f32 = 0.0;
  var d: u32 = tid;
  loop {
    if (d >= HEAD_DIM) { break; }
    let v = f32(q[qBase + d]);
    ss = ss + v * v;
    d = d + WG;
  }
  let nscale = inverseSqrt(reduce_sum(ss, tid) / f32(HEAD_DIM) + EPS);

  // --- + split-half RoPE into shared qn ---
  var p: u32 = tid;
  loop {
    if (p >= HALF_DIM) { break; }
    let n0 = f32(q[qBase + p]) * nscale * f32(w[p]);
    let n1 = f32(q[qBase + p + HALF_DIM]) * nscale * f32(w[p + HALF_DIM]);
    let c = cosTbl[p];
    let s = sinTbl[p];
    qn_sh[p] = n0 * c - n1 * s;
    qn_sh[p + HALF_DIM] = n1 * c + n0 * s;
    p = p + WG;
  }
  for (var i: u32 = tid; i < HEAD_DIM; i = i + WG) {
    out_acc[i] = 0.0;
  }
  if (tid == 0u) {
    running_max = NEG_INF;
    running_denom = 0.0;
  }
  workgroupBarrier();

  // --- this chunk's key range ---
  let chunkLen = (activeKeys + nActive - 1u) / nActive;
  let start = minKj + ci * chunkLen;
  let end = min(start + chunkLen, maxKj);

  // --- flash loop over the chunk (tiles of WG keys) ---
  var tile: u32 = start;
  loop {
    if (tile >= end) { break; }
    let kj = tile + tid;

    // Cooperative Q.K: one 32-lane subgroup per key, lanes splitting HEAD_DIM.
    // The loop has a uniform trip count, so subgroupAdd stays in subgroup-uniform
    // flow while the head-dimension dot is reduced by the subgroup.
    let tileCountS = min(WG, end - tile);
    let sgRounds = (tileCountS + (WG / 32u) - 1u) / (WG / 32u);
    for (var rr: u32 = 0u; rr < sgRounds; rr = rr + 1u) {
      let j = rr * (WG / 32u) + (tid / 32u);
      var accS: f32 = 0.0;
      if (j < tileCountS) {
        let kBase4 = ((tile + j) * params.kvHeads + hKv) * (HEAD_DIM / 4u);
        for (var d4: u32 = (tid & 31u); d4 < HEAD_DIM / 4u; d4 = d4 + 32u) {
          let kv4 = vec4<f32>(k[kBase4 + d4]);
          accS = accS + dot(vec4<f32>(qn_sh[d4 * 4u], qn_sh[d4 * 4u + 1u], qn_sh[d4 * 4u + 2u], qn_sh[d4 * 4u + 3u]), kv4);
        }
      }
      let sj = sg_sum(accS);
      if ((tid & 31u) == 0u && j < tileCountS) { sval_sh[j] = sj * SCALE; }
    }
    workgroupBarrier();
    var sval: f32 = NEG_INF;
    if (kj < end) { sval = sval_sh[tid]; }

    let tileMax = reduce_max(sval, tid);
    let newMax = max(running_max, tileMax);
    let correction = exp(running_max - newMax);

    var pr: f32 = 0.0;
    if (kj < end) { pr = exp(sval - newMax); }
    probs[tid] = pr;
    let tileDenom = reduce_sum(pr, tid);

    if (tid == 0u) {
      running_denom = running_denom * correction + tileDenom;
      running_max = newMax;
    }
    workgroupBarrier();

    // V accumulation, j-split across the whole workgroup: thread (jg, d4)
    // accumulates keys j == jg mod J_GROUPS for dim block d4 into a register,
    // then the groups combine through shared memory. This keeps all lanes active
    // during the per-key V accumulation.
    let tileCount = min(WG, end - tile);
    let jg = tid / (HEAD_DIM / 4u);
    let d4v = tid % (HEAD_DIM / 4u);
    const J_GROUPS: u32 = WG / (HEAD_DIM / 4u);
    var vacc = vec4<f32>(0.0);
    var jj: u32 = jg;
    loop {
      if (jj >= tileCount) { break; }
      let vBase4 = ((tile + jj) * params.kvHeads + hKv) * (HEAD_DIM / 4u);
      vacc = vacc + probs[jj] * vec4<f32>(v[vBase4 + d4v]);
      jj = jj + J_GROUPS;
    }
    vacc_sh[tid] = vacc;
    workgroupBarrier();
    for (var d4: u32 = tid; d4 < HEAD_DIM / 4u; d4 = d4 + WG) {
      var a4 = vec4<f32>(out_acc[d4 * 4u], out_acc[d4 * 4u + 1u], out_acc[d4 * 4u + 2u], out_acc[d4 * 4u + 3u]) * correction;
      for (var g: u32 = 0u; g < J_GROUPS; g = g + 1u) {
        a4 = a4 + vacc_sh[g * (HEAD_DIM / 4u) + d4];
      }
      out_acc[d4 * 4u] = a4.x;
      out_acc[d4 * 4u + 1u] = a4.y;
      out_acc[d4 * 4u + 2u] = a4.z;
      out_acc[d4 * 4u + 3u] = a4.w;
    }
    workgroupBarrier();

    tile = tile + WG;
  }

  // --- write the flash partial (m, l, acc[hd]) via bitcast-atomics; WGSL only
  // guarantees cross-workgroup visibility for this same-dispatch merge through atomics. ---
  let pBase = (h * NCHUNK + ci) * (HEAD_DIM + 2u);
  for (var i: u32 = tid; i < HEAD_DIM; i = i + WG) {
    atomicStore(&partials[pBase + i], bitcast<u32>(out_acc[i]));
  }
  if (tid == 0u) {
    atomicStore(&partials[pBase + HEAD_DIM], bitcast<u32>(running_max));
    atomicStore(&partials[pBase + HEAD_DIM + 1u], bitcast<u32>(running_denom));
  }
  storageBarrier();

  // --- last-arriver merge: the final active chunk workgroup for head h combines
  // all chunk partials in this dispatch. ---
  if (tid == 0u) {
    let ticket = atomicAdd(&partials[PP_COUNTER_BASE + h], 1u);
    lastFlag = select(0u, 1u, ticket == nActive - 1u);
  }
  if (workgroupUniformLoad(&lastFlag) != 1u) {
    return;
  }
  if (tid == 0u) { atomicStore(&partials[PP_COUNTER_BASE + h], 0u); }

  // Parallel weight pass: thread c < nActive owns chunk c's (m, l). The chunk
  // weights live in workgroup memory so the accumulator loop can index them
  // dynamically without private-array spilling.
  var mloc: f32 = NEG_INF;
  var lloc: f32 = 0.0;
  if (tid < nActive) {
    let pb = (h * NCHUNK + tid) * (HEAD_DIM + 2u);
    mloc = bitcast<f32>(atomicLoad(&partials[pb + HEAD_DIM]));
    lloc = bitcast<f32>(atomicLoad(&partials[pb + HEAD_DIM + 1u]));
  }
  let newM = reduce_max(mloc, tid);
  var wloc: f32 = 0.0;
  if (tid < nActive) {
    wloc = exp(mloc - newM);
    wgt_sh[tid] = wloc;
  }
  let denom = reduce_sum(lloc * wloc, tid);
  let inv = 1.0 / denom;
  for (var d: u32 = tid; d < HEAD_DIM; d = d + WG) {
    var acc: f32 = 0.0;
    for (var c: u32 = 0u; c < nActive; c = c + 1u) {
      acc = acc + bitcast<f32>(atomicLoad(&partials[(h * NCHUNK + c) * (HEAD_DIM + 2u) + d])) * wgt_sh[c];
    }
    out[h * HEAD_DIM + d] = f32(srq(acc * inv, OUT_Q));
  }
}