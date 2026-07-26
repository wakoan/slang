"""Atomics in the DSL — the feature that gates porting gemma4_150's fused kernels.

The fast decode path publishes each workgroup's partial result through an
atomic buffer and lets whichever workgroup draws the last ticket finish the
reduction in the SAME dispatch (the "last-arriver" merge in kernels 73/75/77/101).
Without atomics those kernels cannot be expressed in the DSL at all, so every
backend has to keep a hand-maintained copy.

These tests pin the emitted spelling on both backends, because the two differ in
ways that are silently wrong rather than loudly wrong:
  * MSL needs an explicit memory order; the default is seq_cst, and the
    hand-written kernels rely on relaxed.
  * storageBarrier must map to mem_device, NOT mem_threadgroup — the latter
    compiles fine and then fails to publish partials across workgroups.
"""
import pytest

from py_shader_lang_wgpu import (
    kernel, translate, u32, f32, StorageBuffer, AtomicBuffer, Uniform,
    WorkgroupArray, Builtin, TranslationError,
)


@kernel(workgroup_size=(64, 1, 1))
def last_arriver(
    x: StorageBuffer[f32, "read"],
    pp: AtomicBuffer[u32],
    out: StorageBuffer[f32],
    tid: Builtin.local_invocation_index,
    wg: Builtin.workgroup_id,
):
    acc: f32 = x[wg.x * 64 + tid]
    atomicStore(pp, wg.x, bitcast_u32(acc))
    storageBarrier()
    ticket: u32 = atomicAdd(pp, 1024, u32(1))
    if ticket == u32(15):
        total: f32 = f32(0.0)
        for i in range(16):
            total = total + bitcast_f32(atomicLoad(pp, i))
        out[0] = total


def _wgsl():
    return translate(last_arriver)


def _msl():
    return translate(last_arriver, target="msl")


class TestWGSL:
    def test_binding_is_array_of_atomic(self):
        assert "var<storage, read_write> pp: array<atomic<u32>>;" in _wgsl()

    def test_index_form_becomes_pointer(self):
        w = _wgsl()
        assert "atomicStore(&pp[wg.x], bitcast<u32>(acc))" in w
        assert "atomicAdd(&pp[1024], u32(1))" in w
        assert "atomicLoad(&pp[i])" in w

    def test_bitcast_roundtrip(self):
        assert "bitcast<f32>(atomicLoad(&pp[i]))" in _wgsl()

    def test_storage_barrier(self):
        assert "storageBarrier();" in _wgsl()


class TestMSL:
    def test_atomic_param_type(self):
        assert "device atomic_uint* pp [[buffer(1)]]" in _msl()

    def test_relaxed_memory_order(self):
        m = _msl()
        # seq_cst is the MSL default and is NOT what the hand-written kernels use
        assert "atomic_store_explicit(&pp[wg.x], as_type<uint>(acc), memory_order_relaxed)" in m
        assert "atomic_fetch_add_explicit(&pp[1024], uint(1), memory_order_relaxed)" in m
        assert "atomic_load_explicit(&pp[i], memory_order_relaxed)" in m

    def test_storage_barrier_is_device_scope(self):
        m = _msl()
        assert "threadgroup_barrier(mem_flags::mem_device)" in m
        # the workgroup-scope barrier would compile and silently not publish
        assert m.count("threadgroup_barrier(mem_flags::mem_threadgroup)") == 0

    def test_bitcast(self):
        assert "as_type<float>(atomic_load_explicit(&pp[i], memory_order_relaxed))" in _msl()


def test_atomic_rejects_float_element():
    with pytest.raises(TypeError):
        AtomicBuffer[f32]


def test_arity_is_checked():
    # NB @kernel translates eagerly, so the error surfaces at definition time —
    # the decorator has to be inside the raises block, not the translate() call.
    with pytest.raises(TranslationError):
        @kernel(workgroup_size=(1, 1, 1))
        def bad(pp: AtomicBuffer[u32], tid: Builtin.local_invocation_index):
            atomicStore(pp)


class TestWorkgroupUniformLoad:
    @staticmethod
    @kernel(workgroup_size=(64, 1, 1))
    def wul(out: StorageBuffer[f32], flag: WorkgroupArray[u32, 1],
            tid: Builtin.local_invocation_index):
        if tid == u32(0):
            flag[0] = u32(7)
        v: u32 = workgroupUniformLoad(flag)
        out[tid] = f32(v)

    def test_wgsl_takes_pointer(self):
        assert "workgroupUniformLoad(&flag)" in translate(TestWorkgroupUniformLoad.wul)

    def test_msl_emits_barrier_helper(self):
        m = translate(TestWorkgroupUniformLoad.wul, target="msl")
        # MSL has no equivalent intrinsic; the implicit barrier must survive
        assert "_wgUniformLoad" in m
        assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in m
