enable subgroups;
struct Params { rows: u32, rowStride: u32, inScale: f32, _pad0: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> sum_a: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Fused weighted RMSNorm + SRQ activation quantization + sum-of-quantized-activations.
//   n[j]    = x[j] * inverseSqrt(mean(x^2) + eps) * w[j]        (mirrors com.xenova.RMSNorm)
//   y[j]    = toY(srq(f32(toY(n[j])), inScale))                  (the value a downstream QAT
//             GEMV would otherwise recompute per workgroup; toY = output dtype rounding,
//             applied BEFORE srq so the result is bit-identical to the GEMV reading a
//             toY-typed normed buffer and srq-ing it inline)
//   sum[row] = sum_j f32(y[j])                                   (the GEMV's ZP correction term)
// Produces srq'd activations and their per-row sums once, so downstream QAT
// GEMVs can consume both the quantized values and the ZP correction term directly.

const DIM: u32 = 1536u;
const EPS: f32 = 0.000001;
const WG: u32 = 256u;

// Hybrid 2-barrier reduction: subgroupAdd per subgroup + cross-subgroup combine via shared.
var<workgroup> sgp: array<f32, WG / 32u>;

// Sum over each logical 32-lane block. sgExact32 (fixed 32-wide adapter) -> hardware
// subgroupAdd; otherwise a 32-lane subgroupShuffleXor butterfly that reduces each block
// independently, correct for any subgroup width >= 32 (NVIDIA D3D12 [32,128], AMD [32,64]).
fn sg_sum(value: f32) -> f32 {
  return subgroupAdd(value);
}

fn reduce_sum(value: f32, tid: u32) -> f32 {
  let s = sg_sum(value);
  if ((tid & 31u) == 0u) { sgp[tid >> 5u] = s; }
  workgroupBarrier();
  var total: f32 = 0.0;
  for (var i: u32 = 0u; i < WG / 32u; i = i + 1u) { total = total + sgp[i]; }
  workgroupBarrier();
  return total;
}

fn srq(x: f32, s: f32) -> f32 {
  if (s == 0.0) {
    return x;
  }
  return clamp(round(x / s), -128.0, 127.0) * s;
}

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let rowStride = select(params.rowStride, params.rows, params.rowStride == 0u);
  let row = wg.x + wg.y * rowStride;
  if (row >= params.rows) {
    return;
  }
  let tid = lid.x;
  let base = row * DIM;
  let inScale = params.inScale;

  // Sum of squares (identical reduction shape to com.xenova.RMSNorm).
  var acc: f32 = 0.0;
  var i: u32 = tid;
  loop {
    if (i >= DIM) {
      break;
    }
    let v = f32(x[base + i]);
    acc = acc + v * v;
    i = i + WG;
  }
  let scale = inverseSqrt(reduce_sum(acc, tid) / f32(DIM) + EPS);

  // Normalize + weight + quantize; accumulate the quantized sum.
  var qAcc: f32 = 0.0;
  var j: u32 = tid;
  loop {
    if (j >= DIM) {
      break;
    }
    let n = f32(x[base + j]) * scale * f32(w[j]);
    let q = f32(srq(f32(f32(n)), inScale));
    y[base + j] = q;
    qAcc = qAcc + f32(q);
    j = j + WG;
  }
  let qSum = reduce_sum(qAcc, tid);
  if (tid == 0u) {
    sum_a[row] = qSum;
  }
}