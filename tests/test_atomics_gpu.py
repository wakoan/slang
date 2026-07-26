"""Run the DSL's atomic last-arriver pattern on real GPUs.

Emitting plausible text is not the bar — the pattern this unlocks is a
cross-workgroup merge, and those fail in ways that only show up under real
concurrency (wrong memory order, wrong barrier scope, a counter that is not
reset). So this actually dispatches the kernel on Metal and on wgpu and checks
the merged result.

The kernel mirrors what kernels 73/75/77 do: every workgroup publishes one
partial through the atomic buffer, bumps a ticket, and the last arriver sums all
partials in the same dispatch.
"""
import numpy as np
import pytest

from py_shader_lang_wgpu import (
    kernel, translate, u32, f32, StorageBuffer, AtomicBuffer, Builtin,
)

NWG = 16
WG = 64
CTR = NWG              # ticket counter lives just past the partials


# NB the DSL resolves helper FUNCTIONS lexically but does not capture closure
# VALUES, so the shapes below have to be literals rather than the constants above.
@kernel(workgroup_size=(64, 1, 1))
def merge_sum(
    x: StorageBuffer[f32, "read"],
    pp: AtomicBuffer[u32],
    out: StorageBuffer[f32],
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    # each workgroup reduces its own slice, thread 0 publishes it
    acc: f32 = f32(0.0)
    for i in range(64):
        acc = acc + x[wg.x * u32(64) + u32(i)]
    if tid == u32(0):
        atomicStore(pp, wg.x, bitcast_u32(acc))
    storageBarrier()
    if tid == u32(0):
        ticket: u32 = atomicAdd(pp, u32(16), u32(1))
        if ticket == u32(15):
            total: f32 = f32(0.0)
            for j in range(16):
                total = total + bitcast_f32(atomicLoad(pp, u32(j)))
            out[0] = total
            atomicStore(pp, u32(16), u32(0))      # self-reset for the next call


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(NWG * WG).astype(np.float32)
    return x, float(x.astype(np.float64).sum())


def test_metal(data):
    Metal = pytest.importorskip("Metal")
    x, expect = data
    dev = Metal.MTLCreateSystemDefaultDevice()
    if dev is None:
        pytest.skip("no Metal device")
    src = translate(merge_sum, target="msl")
    lib, err = dev.newLibraryWithSource_options_error_(src, None, None)
    assert lib is not None, f"MSL compile failed: {err}\n{src}"
    pso, err = dev.newComputePipelineStateWithFunction_error_(
        lib.newFunctionWithName_("merge_sum"), None)
    assert pso is not None, f"pipeline failed: {err}"

    opt = Metal.MTLResourceStorageModeShared
    bx = dev.newBufferWithBytes_length_options_(x.tobytes(), x.nbytes, opt)
    bpp = dev.newBufferWithLength_options_((NWG + 1) * 4, opt)
    bout = dev.newBufferWithLength_options_(4, opt)
    bpp.contents().as_buffer((NWG + 1) * 4)[:] = b"\x00" * ((NWG + 1) * 4)

    q = dev.newCommandQueue()
    cb = q.commandBuffer()
    enc = cb.computeCommandEncoder()
    enc.setComputePipelineState_(pso)
    for i, b in enumerate((bx, bpp, bout)):
        enc.setBuffer_offset_atIndex_(b, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(NWG, 1, 1), Metal.MTLSizeMake(WG, 1, 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    got = np.frombuffer(bout.contents().as_buffer(4), dtype=np.float32)[0]
    assert got == pytest.approx(expect, rel=1e-5), f"{got} != {expect}"


def test_wgpu(data):
    wgpu = pytest.importorskip("wgpu")
    x, expect = data
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = adapter.request_device_sync()
    src = translate(merge_sum)
    mod = dev.create_shader_module(code=src)

    bx = dev.create_buffer_with_data(data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    bpp = dev.create_buffer_with_data(
        data=np.zeros(NWG + 1, np.uint32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
    bout = dev.create_buffer(
        size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    pipeline = dev.create_compute_pipeline(
        layout=wgpu.enums.AutoLayoutMode.auto, compute={"module": mod, "entry_point": "merge_sum"})
    bg = dev.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
        {"binding": 0, "resource": {"buffer": bx, "offset": 0, "size": bx.size}},
        {"binding": 1, "resource": {"buffer": bpp, "offset": 0, "size": bpp.size}},
        {"binding": 2, "resource": {"buffer": bout, "offset": 0, "size": bout.size}},
    ])
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(NWG, 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])

    got = np.frombuffer(dev.queue.read_buffer(bout), dtype=np.float32)[0]
    assert got == pytest.approx(expect, rel=1e-5), f"{got} != {expect}"
