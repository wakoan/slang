#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
// sgmm2's B-reuse structure, but SG simdgroups per threadgroup (each owning its
// own 8-wide N slice) to restore occupancy — the thing 2-D register tiling lost.
// dispatch with threadsPerThreadgroup = (32*SG); grid = N/(8*SG)
kernel void sgmm4(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]],
                  device half* C [[buffer(2)]], constant uint4& p [[buffer(3)]],
                  uint lidx [[thread_index_in_threadgroup]],
                  uint3 wg [[threadgroup_position_in_grid]])
{
    const uint MT = 8u, SG = 8u;
    uint N=p.y, K=p.z;
    uint sgId = lidx/32u;
    uint tileN = (wg.x*SG + sgId)*8u;
    simdgroup_half8x8 acc[MT];
    for (uint m=0;m<MT;m++) acc[m]=simdgroup_half8x8(0.h);
    for (uint k=0; k<K; k+=8u) {
        simdgroup_half8x8 b; simdgroup_load(b, B + k*N + tileN, N);
        for (uint m=0;m<MT;m++) {
            simdgroup_half8x8 a; simdgroup_load(a, A + (m*8u)*K + k, K);
            simdgroup_multiply_accumulate(acc[m], a, b, acc[m]);
        }
    }
    for (uint m=0;m<MT;m++) simdgroup_store(acc[m], C + (m*8u)*N + tileN, N);
}
