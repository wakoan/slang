"""Example: Python-to-WGSL translation of a matrix multiplication kernel."""

from py_shader_lang_wgpu import kernel, u32, f32, StorageBuffer, Builtin


@kernel(workgroup_size=(8, 8))
def matmul(
    global_id: Builtin.global_invocation_id,
    matrix_a: StorageBuffer[f32, "read"],
    matrix_b: StorageBuffer[f32, "read"],
    matrix_c: StorageBuffer[f32, "read_write"],
    dims: StorageBuffer[u32, "read"],  # packed as [m, k, n]
):
    row: u32 = global_id.y
    col: u32 = global_id.x
    m: u32 = dims[0]
    k: u32 = dims[1]
    n: u32 = dims[2]

    if row >= m or col >= n:
        return

    total: f32 = 0.0
    for i in range(k):
        total += matrix_a[row * k + i] * matrix_b[i * n + col]

    matrix_c[row * n + col] = total


if __name__ == "__main__":
    print(matmul.wgsl)
