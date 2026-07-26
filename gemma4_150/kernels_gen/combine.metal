// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

// dispatch with threadsPerThreadgroup = (256)
kernel void combine(
    device const float* ctx [[buffer(0)]],
    device const float* ple [[buffer(1)]],
    device const float* nw [[buffer(2)]],
    device float* outp [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 wg [[threadgroup_position_in_grid]]
) {
  threadgroup float red[256];
  // PLE input: per-row RMSNorm(ctx * H^-0.5) + ple, scaled by 2^-0.5.
  //
  // HINV and RS2 stay literals so this is a drop-in replacement for the
  // hand-written kernel's binding layout; promoting them to a uniform would be
  // an improvement but changes the runner's call.
  uint base = wg.x * uint(256) + tid;
  float c = ctx[base] * float(0.025515046504);
  red[tid] = c * c;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint s = uint(128);
  while (s > uint(0)) {
    if (tid < s) {
      red[tid] = red[tid] + red[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = s / uint(2);
  }
  float rms = rsqrt(red[0] / float(256.0) + float(1e-06));
  outp[base] = (c * rms * nw[tid] + ple[base]) * float(0.7071067811865476);
}