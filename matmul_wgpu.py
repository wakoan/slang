import wgpu
import numpy as np

# WGSL shader for matrix multiplication
# This shader multiplies two matrices A (MxK) and B (KxN) into C (MxN)
shader_code = """
struct Matrix {
    size: vec2<u32>,
    data: array<f32>,
};

@group(0) @binding(0) var<storage, read> matrix_a: Matrix;
@group(0) @binding(1) var<storage, read> matrix_b: Matrix;
@group(0) @binding(2) var<storage, read_write> matrix_c: Matrix;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    
    let m = matrix_a.size.x;
    let k = matrix_a.size.y;
    let n = matrix_b.size.y;
    
    if (row >= m || col >= n) {
        return;
    }
    
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < k; i = i + 1u) {
        let a_idx = row * k + i;
        let b_idx = i * n + col;
        sum = sum + matrix_a.data[a_idx] * matrix_b.data[b_idx];
    }
    
    let c_idx = row * n + col;
    matrix_c.data[c_idx] = sum;
    matrix_c.size = vec2<u32>(m, n);
}
"""

def run_matmul(m, k, n):
    # Initialize data
    data_a = np.random.rand(m, k).astype(np.float32)
    data_b = np.random.rand(k, n).astype(np.float32)
    data_c = np.zeros((m, n), dtype=np.float32)

    # Get device
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()

    # Create buffers
    # Buffer A
    buffer_a = device.create_buffer_with_data(
        data=np.concatenate([np.array([m, k], dtype=np.uint32).view(np.float32), data_a.flatten()]),
        usage=wgpu.BufferUsage.STORAGE
    )
    # Buffer B
    buffer_b = device.create_buffer_with_data(
        data=np.concatenate([np.array([k, n], dtype=np.uint32).view(np.float32), data_b.flatten()]),
        usage=wgpu.BufferUsage.STORAGE
    )
    # Buffer C
    buffer_c = device.create_buffer(
        size=8 + m * n * 4, # 2 uint32s + m*n float32s
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    # Create bind group layout and bind group
    bind_group_layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
    ])
    
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[
        {"binding": 0, "resource": {"buffer": buffer_a, "offset": 0, "size": buffer_a.size}},
        {"binding": 1, "resource": {"buffer": buffer_b, "offset": 0, "size": buffer_b.size}},
        {"binding": 2, "resource": {"buffer": buffer_c, "offset": 0, "size": buffer_c.size}},
    ])

    # Create compute pipeline
    shader_module = device.create_shader_module(code=shader_code)
    pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[bind_group_layout]),
        compute={"module": shader_module, "entry_point": "main"}
    )

    # Encode and submit commands
    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    
    # Calculate grid size (workgroup_size is 8x8)
    grid_x = (n + 7) // 8
    grid_y = (m + 7) // 8
    compute_pass.dispatch_workgroups(grid_x, grid_y)
    compute_pass.end()

    device.queue.submit([encoder.finish()])

    # Read back results
    # Use a staging buffer to map and read from
    staging_buffer = device.create_buffer(
        size=buffer_c.size,
        usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )
    
    encoder = device.create_command_encoder()
    encoder.copy_buffer_to_buffer(buffer_c, 0, staging_buffer, 0, buffer_c.size)
    device.queue.submit([encoder.finish()])
    
    # Map and get the data
    staging_buffer.map_sync("READ")
    data_c_raw = staging_buffer.read_mapped()
    # Skip the first 8 bytes (size: vec2<u32>)
    result = np.frombuffer(data_c_raw, dtype=np.float32, offset=8)
    result = result.copy().reshape((m, n))
    staging_buffer.unmap()

    # Verification
    expected = np.matmul(data_a, data_b)
    np.testing.assert_allclose(result, expected, atol=1e-5)
    print("Matrix multiplication successful and verified!")

if __name__ == "__main__":
    run_matmul(64, 64, 64)
