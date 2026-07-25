// Auto-generated from step_setup
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (1)
kernel void step_setup(
    device uint* pos_buf [[buffer(0)]],
    device uint* kv_dims [[buffer(1)]],
    device uint* sc_slide [[buffer(2)]],
    device uint* sc_full [[buffer(3)]],
    device float* rope_l [[buffer(4)]],
    device float* rope_g [[buffer(5)]],
    device const uint* cfg_c [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
  if (gid.x > 0) {
    return;
  }
  uint pos = pos_buf[0];
  uint kv_len = pos + 1;
  uint window = cfg_c[0];
  kv_dims[1] = pos;
  sc_slide[2] = kv_len;
  if (kv_len > window) {
    sc_slide[3] = kv_len - window;
  } else {
    sc_slide[3] = 0;
  }
  sc_full[2] = kv_len;
  rope_l[1] = float(pos);
  rope_g[1] = float(pos);
  pos_buf[0] = pos + 1;
}