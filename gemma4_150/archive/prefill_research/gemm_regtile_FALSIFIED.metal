#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
// 2-D register tiling: each simdgroup computes MT x NT tiles of 8x8.
// Per k-step it issues MT + NT loads and does MT*NT MACs, so raising both
// raises MACs-per-load — the actual limiter once B reuse is fixed.
// dispatch with threadsPerThreadgroup = (32); grid = N/(8*NT)
kernel void sgmm3(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]],
                  device half* C [[buffer(2)]], constant uint4& p [[buffer(3)]],
                  uint3 wg [[threadgroup_position_in_grid]])
{
    const uint MT = 8u, NT = 4u;
    uint N=p.y, K=p.z;
    uint tileN = wg.x*8u*NT;
    simdgroup_half8x8 acc[MT][NT];
    for (uint m=0;m<MT;m++) for(uint n=0;n<NT;n++) acc[m][n]=simdgroup_half8x8(0.h);
    for (uint k=0; k<K; k+=8u) {
        simdgroup_half8x8 b[NT], a[MT];
        for (uint n=0;n<NT;n++) simdgroup_load(b[n], B + k*N + tileN + n*8u, N);
        for (uint m=0;m<MT;m++) simdgroup_load(a[m], A + (m*8u)*K + k, K);
        for (uint m=0;m<MT;m++) for(uint n=0;n<NT;n++)
            simdgroup_multiply_accumulate(acc[m][n], a[m], b[n], acc[m][n]);
    }
    for (uint m=0;m<MT;m++) for(uint n=0;n<NT;n++)
        simdgroup_store(acc[m][n], C + (m*8u)*N + tileN + n*8u, N);
}
