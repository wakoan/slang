// Auto-generated from probe_sg
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (32)
kernel void probe_sg(
    device uint* out [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]],
    uint sgs [[threads_per_simdgroup]]
) {
  if (gid.x == 0) {
    out[0] = sgs;
  }
}