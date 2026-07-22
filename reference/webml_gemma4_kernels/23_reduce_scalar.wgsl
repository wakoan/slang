enable subgroups;
struct Params { rows: u32, rowStride: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> scale: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

// Subgroup-parallel single-pass row statistics + fused normalize/affine.
//
// One workgroup owns one contiguous normalization span ("row": a last-axis
// row, an instance plane, or a channel group). Threads stride the row once,
// accumulating (sum, sum_sq) simultaneously; partials are reduced with
// subgroupAdd plus a single shared-memory combine, then every thread applies
// the fused normalize + affine write.
//
// Numerics: for mean/variance modes the accumulation is shifted by the row's
// first element, so a large common offset cannot catastrophically cancel in
// the E[x^2] - E[x]^2 identity (variance is unchanged by shifting). Epsilon
// placement and the variance formula follow each op's reference shader:
//   layer:    inverseSqrt(variance + EPSILON)
//   group:    (x - mean) / sqrt(variance + EPSILON)
//   instance: inverseSqrt(max(variance, 0) + EPSILON)
//   rms:      inverseSqrt(sum_sq / HIDDEN + EPSILON)   (no mean, no shift)
//   lp:       x / norm with norm == 0 -> 0             (no epsilon)
const HIDDEN: u32 = 512u;
const HIDDEN_V: u32 = 128u;
const WG: u32 = 128u;
const EPSILON: f32 = 0.000001;

// One partial per subgroup (requiredSubgroupMinSize 32 bounds the count).
const MAX_SG: u32 = 4u;
var<workgroup> sg_partials: array<f32, MAX_SG>;

fn reduce_scalar(value: f32, tid: u32, sg_lane: u32, sg_size: u32) -> f32 {
  let s = subgroupAdd(value);
  if (sg_lane == 0u) {
    sg_partials[tid / sg_size] = s;
  }
  workgroupBarrier();
  let num_sg = (WG + sg_size - 1u) / sg_size;
  var total = 0.0;
  for (var i = 0u; i < num_sg; i = i + 1u) {
    total = total + sg_partials[i];
  }
  return total;
}

@compute @workgroup_size(WG, 1, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lane: u32,
  @builtin(subgroup_size) sg_size: u32
) {
  let row = wg_id.x + wg_id.y * params.rowStride;
  if (row >= params.rows) {
    return;
  }
  let tid = lid.x;
  let base = row * HIDDEN_V;


  var acc = 0.0;
  for (var i = tid; i < HIDDEN_V; i = i + WG) {
    let v = vec4<f32>(x[base + i]);
    acc = acc + dot(v, v);
  }

  let total = reduce_scalar(acc, tid, sg_lane, sg_size);

  let inv = inverseSqrt(total / f32(HIDDEN) + EPSILON);

  for (var i = tid; i < HIDDEN_V; i = i + WG) {
    let idx = base + i;
    let v = vec4<f32>(x[idx]);
    y[idx] = vec4<f32>(v * inv * vec4<f32>(scale[i]));
  }
}