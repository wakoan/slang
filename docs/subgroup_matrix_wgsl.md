# WebGPU subgroup-matrix (chromium_experimental_subgroup_matrix) — verified syntax

Verified working + numerically correct in headless Chrome 150 on M4 Pro
(`--enable-unsafe-webgpu`). Adapter `subgroupMatrixConfigs`: 8×8×8 tiles,
{f16 component / f16 result} and {f32 / f32}. We use f16 inputs with an f32
result accumulator (no precision loss over the K reduction).

```wgsl
enable chromium_experimental_subgroup_matrix;
enable f16;

@group(0) @binding(0) var<storage, read> a : array<f16>;   // M×K, row-major
@group(0) @binding(1) var<storage, read> b : array<f16>;   // K×N, row-major
@group(0) @binding(2) var<storage, read_write> c : array<f32>; // M×N

@compute @workgroup_size(32)   // one subgroup holds the tile cooperatively
fn main() {
  let left  = subgroupMatrixLoad<subgroup_matrix_left <f16, 8, 8>>(&a, 0u, false, 8u);
  let right = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&b, 0u, false, 8u);
  var acc = subgroup_matrix_result<f32, 8, 8>();            // zero-init
  acc = subgroupMatrixMultiplyAccumulate(left, right, acc); // acc = left@right + acc
  subgroupMatrixStore(&c, 0u, acc, false, 8u);
}
```

- Types: `subgroup_matrix_left<T, K, M>`, `subgroup_matrix_right<T, N, K>`,
  `subgroup_matrix_result<T, N, M>` (all 8 here). Semantics: result[M][N] =
  Σ_k left[M][k]·right[k][N] — plain row-major matmul (verified vs numpy).
- `subgroupMatrixLoad<TYPE>(ptr, offset_elems, colMajor: bool, stride_elems)`.
- `subgroupMatrixStore(ptr, offset_elems, mat, colMajor: bool, stride_elems)`.
- Loads/stores work from `storage`; `workgroup` (shared) needs verifying for
  the on-the-fly int→f16 dequant-tile GEMV.
- Feature is origin-trial-gated in normal Chrome; `--enable-unsafe-webgpu`
  exposes it (headless dev harness: gendemo4/bench_headless.mjs).
