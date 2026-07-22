struct Params { seq: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> bits_buf: array<u32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Gather + dequantize a QAT-packed embedding row:
//   y[t, c] = EMBED_SCALE * scale[id, g] * (q - ZP)
// where id = ids[t], q is the LSB-first unpacked code, g = c / GROUP_SIZE.
// scale is a plain per-(row, group) table [vocab, NUM_GROUPS] (scale-only; the
// symmetric zero point ZP and the sqrt(dim) embedding scale are applied here).

const HIDDEN:        u32 = 8960u;
const VOCAB:         u32 = 262144u;
const GROUP_SIZE:    u32 = 256u;
const NUM_GROUPS:    u32 = 35u;
const WORDS_PER_ROW: u32 = 1120u;
const VALS_PER_WORD: u32 = 8u;
const BITS:          u32 = 4u;
const MASK:          u32 = 15u;
const ZP:            f32 = 8.0;
const EMBED_SCALE:   f32 = 16;
const WG: u32 = 64u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wg.x;
  if (t >= params.seq) {
    return;
  }
  let id = ids[t];
  if (id >= VOCAB) {
    return;
  }

  let row_words_base:  u32 = id * WORDS_PER_ROW;
  let row_scale_base:  u32 = id * NUM_GROUPS;

  var w: u32 = lid.x;
  loop {
    if (w >= WORDS_PER_ROW) {
      break;
    }
    let packed: u32 = bits_buf[row_words_base + w];
    let colBase: u32 = w * VALS_PER_WORD;
    for (var v: u32 = 0u; v < VALS_PER_WORD; v = v + 1u) {
      let c: u32 = colBase + v;
      let g: u32 = c / GROUP_SIZE;
      let s: f32 = scale[row_scale_base + g];
      let q: f32 = f32((packed >> (v * BITS)) & MASK);
      y[t * HIDDEN + c] = f32(EMBED_SCALE * s * (q - ZP));
    }
    w = w + WG;
  }
}