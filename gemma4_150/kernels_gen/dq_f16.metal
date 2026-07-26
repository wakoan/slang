// GENERATED from gemma4_150/kernels_dsl.py by `python -m gemma4_150.gen_msl` — do not edit.
#include <metal_stdlib>
using namespace metal;

constant uint BITS=2u;

// dispatch with threadsPerThreadgroup = (256)
kernel void dq_f16(
    device const uint* bits [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint4& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
  // Dequantize a QAT weight matrix to plain f16 so a GEMM can consume it.
  //
  // One kernel for all three widths; BITS is a const (2 / 4 / 8).
  //
  // w = row_scale[o] * (q - ZP),   ZP = 1 << (BITS-1)   -> 2 / 8 / 128
  //
  // Values are byte-major then low-bits-first within the byte, so value i sits
  // at bit 8*(i/(8/BITS)) + BITS*(i%(8/BITS)). Getting that order wrong is
  // silent — the dot products just pair the wrong operands — which is why the
  // test checks it EXACTLY against a float64 reconstruction rather than to a
  // tolerance.
  //
  // One thread per packed word.
  uint VPW = uint(32) / uint(BITS);
  uint PER_BYTE = uint(8) / uint(BITS);
  float ZP = float((1 << (BITS - 1)));
  uint MASK = uint((1 << BITS) - 1);
  uint WPR = p.x / VPW;
  if (gid.x < p.y * WPR) {
    uint o = gid.x / WPR;
    uint w = gid.x % WPR;
    uint packed = bits[gid.x];
    float s = scale[o];
    uint base = o * p.x + w * VPW;
    uint i = uint(0);
    while (i < VPW) {
      uint q = ((packed >> (uint(8) * (i / PER_BYTE) + uint(BITS) * (i % PER_BYTE))) & MASK);
      out[base + i] = half(s * (float(q) - ZP));
      i = i + uint(1);
    }
  }
}