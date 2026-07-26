// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void embed_00(
    device const uint* ids [[buffer(0)]],
    device const uint* bits_buf [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint4& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  // 2-bit embedding gather, one workgroup per token.
  uint t = wg.x;
  if (t < params.x) {
    uint id = ids[t];
    if (id < uint(262144)) {
      for (uint w = tid; w < 96; w += 64) {
        uint packed = bits_buf[id * uint(96) + w];
        float s = scale[id];
        for (uint v = 0; v < 16; v++) {
          float q = float(((packed >> (uint(v) * uint(2))) & uint(3)));
          y[t * uint(1536) + w * uint(16) + uint(v)] = float(39.191835884530846) * s * (q - float(2.0));
        }
      }
    }
  }
}