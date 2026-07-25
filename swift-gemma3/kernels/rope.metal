// Auto-generated from rope
// Generated from py_shader_lang_wgpu DSL

#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (64)
kernel void rope(
    device float* x_io [[buffer(0)]],
    device const float* fparams [[buffer(1)]],
    device const uint* dims [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
  uint idx = gid.x;
  uint n_heads = dims[0];
  uint head_dim = dims[1];
  uint half_ = head_dim / 2;
  if (idx >= n_heads * half_) {
    return;
  }
  uint h = idx / half_;
  uint i = idx % half_;
  float theta = fparams[0];
  float pos = fparams[1];
  float inv_freq = 1.0 / pow(theta, float(2 * i) / float(head_dim));
  float angle = pos * inv_freq;
  float c = cos(angle);
  float s = sin(angle);
  float a = x_io[h * head_dim + i];
  float b = x_io[h * head_dim + i + half_];
  x_io[h * head_dim + i] = a * c - b * s;
  x_io[h * head_dim + i + half_] = b * c + a * s;
}