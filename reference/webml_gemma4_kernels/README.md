# webml-community/gemma-4-webgpu-kernels — captured decode kernels (~150 tok/s, M4 Pro)

Captured live from the HF space by hooking `GPUDevice.createShaderModule` in
headless Chrome (scratchpad/cap2.mjs). These are the FINAL rendered WGSL the
space compiles — the reference for matching its ~150 tok/s decode. Kernels
authored by "Fable 5" (agentic kernel optimization).

## The recipe for ~150 tok/s decode (vs our ~90)

The gap is NOT a hardware feature (subgroup-matrix is prefill-only here; decode
uses plain subgroups). It's the matmul STRUCTURE + fusion:

1. **Thread-per-output-column GEMV, no cross-thread reduction** (33_srq, 110_srq,
   "QAT GEMV specialization M=1"): each thread computes one full output column
   with a private f32 accumulator over all K — eliminates the barrier-tree
   reduction entirely (our workgroup-per-row + 6-barrier tree is the main waste).
2. **Block-major (transposed) weight repack at load**: `bits[blk*N + col]` so the
   TILE_N=128 threads (consecutive col) read consecutive vec4<u32> — fully
   coalesced. This is the enabler for #1.
3. **vec4<u32> (128-bit) loads + unpack4xU8** to dequant 4 sub-byte weights/instr;
   `dot(vec4,vec4)` hardware dot. Per-row symmetric scale factors out; fixed ZP.
4. **SRQ int8 activations** (srq(x,s)=clamp(round(x/s),-128,127)*s), scales from
   the checkpoint. OPTIONAL: inScale==0 => pass-through f32 (weight-only, our path).
   "presrq" kernels take pre-quantized activations + row sum_a, folding the ZP
   correction into the epilogue (sum_k q*a - ZP*sum_a).
5. **Aggressive fusion — one dispatch each**: q/k/v proj (70_srq); o-proj +
   post-attn residual-norm-add + pre-FFN norm (73_sg_sum); down + post-FFN
   norm-add (75_srq); gate/up geglu (16/30/74_sg_sum). Cuts ~16 dispatches/layer
   to ~6, keeping activations in registers.
6. **Decode attention = flash with same-dispatch cross-workgroup merge via
   atomics** (101_srq): fused q-norm+RoPE, NCHUNK chunk workgroups, last-arriver
   ticket merges partials in ONE dispatch (no separate combine).

Key files: 33_srq (logits/dense GEMV), 70_srq (qkv), 73_sg_sum (o+norms),
75_srq (down+norm), 16/30/74_sg_sum (gate/up), 101_srq (decode attention),
00/01_main (2-bit embed gather). 26 kernels use subgroupMatrix = PREFILL only.
