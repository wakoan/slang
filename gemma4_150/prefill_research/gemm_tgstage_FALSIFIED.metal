#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
// sgmm4 + A staged in threadgroup memory: all SG simdgroups in a threadgroup need
// the SAME A tiles (they differ only in N), so load A once cooperatively instead
// of SG times from device memory.
// dispatch with threadsPerThreadgroup = (32*SG); grid = N/(8*SG)
kernel void sgmm5(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]],
                  device half* C [[buffer(2)]], constant uint4& p [[buffer(3)]],
                  uint lidx [[thread_index_in_threadgroup]],
                  uint3 wg [[threadgroup_position_in_grid]])
{
    const uint MT = 8u, SG = 16u, KC = 8u;
    threadgroup half As[MT*8u*KC];              // 64 rows x 8 k = 1KB
    uint N=p.y, K=p.z;
    uint sgId = lidx/32u;
    uint tileN = (wg.x*SG + sgId)*8u;
    simdgroup_half8x8 acc[MT];
    for (uint m=0;m<MT;m++) acc[m]=simdgroup_half8x8(0.h);

    for (uint k=0; k<K; k+=KC) {
        // cooperative load of the A panel (MT*8 rows x KC) — 512 halves
        for (uint t=lidx; t<MT*8u*KC; t+=32u*SG) {
            uint r=t/KC, kk=t%KC;
            As[r*KC+kk] = A[r*K + k + kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 b; simdgroup_load(b, B + k*N + tileN, N);
        for (uint m=0;m<MT;m++) {
            simdgroup_half8x8 a;
            simdgroup_load(a, As + (m*8u)*KC, KC);      // from threadgroup memory
            simdgroup_multiply_accumulate(acc[m], a, b, acc[m]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint m=0;m<MT;m++) simdgroup_store(acc[m], C + (m*8u)*N + tileN, N);
}
