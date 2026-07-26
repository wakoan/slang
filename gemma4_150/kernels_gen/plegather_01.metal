// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void plegather_01(
    device const uint* ids [[buffer(0)]],
    device const uint* bits_buf [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint4& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  // 4-bit per-layer-embedding gather. 35 scale groups of 256 per row, so the
  // scale index moves with the column — unlike embed_00's single group.
  uint t = wg.x;
  if (t < params.x) {
    uint id = ids[t];
    if (id < uint(262144)) {
      for (uint w = tid; w < 1120; w += 64) {
        uint packed = bits_buf[id * uint(1120) + w];
        for (uint v = 0; v < 8; v++) {
          uint c = w * uint(8) + uint(v);
          float s = scale[id * uint(35) + c / uint(256)];
          float q = float(((packed >> (uint(v) * uint(4))) & uint(15)));
          y[t * uint(8960) + c] = float(16.0) * s * (q - float(8.0));
        }
      }
    }
  }
}