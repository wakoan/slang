#include <metal_stdlib>
using namespace metal;
// translated from reference/webml_gemma4_kernels/00_main.wgsl
// dispatch with threadsPerThreadgroup = (64) ; one threadgroup per token
struct P00 { uint seq; uint p0; uint p1; uint p2; };

kernel void embed_00(
    device const uint*  ids      [[buffer(0)]],
    device const uint*  bits_buf [[buffer(1)]],
    device const float* scale    [[buffer(2)]],
    device       float* y        [[buffer(3)]],
    constant     P00&   params   [[buffer(4)]],
    uint  tid [[thread_index_in_threadgroup]],
    uint3 wg  [[threadgroup_position_in_grid]])
{
    const uint HIDDEN = 1536u, VOCAB = 262144u, GROUP_SIZE = 1536u;
    const uint WORDS_PER_ROW = 96u, VALS_PER_WORD = 16u, BITS = 2u, MASK = 3u, WG = 64u;
    const float ZP = 2.0f, EMBED_SCALE = 39.191835884530846f;

    uint t = wg.x;
    if (t >= params.seq) return;
    uint id = ids[t];
    if (id >= VOCAB) return;
    uint row_words = id * WORDS_PER_ROW;
    uint row_scale = id;                       // NUM_GROUPS == 1
    for (uint w = tid; w < WORDS_PER_ROW; w += WG) {
        uint packed = bits_buf[row_words + w];
        uint colBase = w * VALS_PER_WORD;
        for (uint v = 0u; v < VALS_PER_WORD; v++) {
            uint c = colBase + v;
            uint g = c / GROUP_SIZE;
            float s = scale[row_scale + g];
            float q = float((packed >> (v * BITS)) & MASK);
            y[t * HIDDEN + c] = EMBED_SCALE * s * (q - ZP);
        }
    }
}
