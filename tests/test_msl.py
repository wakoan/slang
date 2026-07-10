"""Tests for the MSL (Metal Shading Language) translator backend."""

import os
import tempfile

import numpy as np
import pytest

from py_shader_lang_wgpu import (
    translate, kernel,
    u32, f32, f16,
    StorageBuffer, Uniform, Builtin, WorkgroupArray,
)


def msl(func, workgroup_size=(1,)):
    return translate(func, workgroup_size=workgroup_size, target="msl")


class TestSignature:
    def test_buffers_and_builtin(self):
        def fn(
            gid: Builtin.global_invocation_id,
            x_in: StorageBuffer[f32, "read"],
            y_out: StorageBuffer[f32, "read_write"],
            dims: StorageBuffer[u32, "read"],
        ):
            y_out[gid.x] = x_in[gid.x]
        src = msl(fn)
        assert "#include <metal_stdlib>" in src
        assert "kernel void fn(" in src
        assert "device const float* x_in [[buffer(0)]]" in src
        assert "device float* y_out [[buffer(1)]]" in src
        assert "device const uint* dims [[buffer(2)]]" in src
        assert "uint3 gid [[thread_position_in_grid]]" in src

    def test_uniform(self):
        def fn(gid: Builtin.global_invocation_id, n: Uniform[u32]):
            pass
        assert "constant uint& n [[buffer(0)]]" in msl(fn)

    def test_all_builtins(self):
        def fn(
            gid: Builtin.global_invocation_id,
            lid: Builtin.local_invocation_id,
            wid: Builtin.workgroup_id,
            lane: Builtin.subgroup_invocation_id,
            sgs: Builtin.subgroup_size,
        ):
            pass
        src = msl(fn)
        assert "uint3 lid [[thread_position_in_threadgroup]]" in src
        assert "uint3 wid [[threadgroup_position_in_grid]]" in src
        assert "uint lane [[thread_index_in_simdgroup]]" in src
        assert "uint sgs [[threads_per_simdgroup]]" in src

    def test_workgroup_size_comment(self):
        def fn(gid: Builtin.global_invocation_id):
            pass
        assert "threadsPerThreadgroup = (8, 8)" in msl(fn, workgroup_size=(8, 8))


class TestBody:
    def test_typed_declarations(self):
        def fn(gid: Builtin.global_invocation_id):
            i: u32 = gid.x
            v: f32 = 1.5
        src = msl(fn)
        assert "uint i = gid.x;" in src
        assert "float v = 1.5;" in src

    def test_let_and_var(self):
        def fn(gid: Builtin.global_invocation_id):
            a = 1
            b = 2
            b = 3
        src = msl(fn)
        assert "const auto a = 1;" in src
        assert "auto b = 2;" in src

    def test_for_loop(self):
        def fn(gid: Builtin.global_invocation_id):
            total: f32 = 0.0
            for j in range(2, 10, 2):
                total += 1.0
        assert "for (uint j = 2; j < 10; j += 2) {" in msl(fn)

    def test_descending_loop_int(self):
        def fn(gid: Builtin.global_invocation_id):
            for i in range(10, 0, -1):
                pass
        assert "for (int i = 10; i > 0; i -= 1) {" in msl(fn)

    def test_casts(self):
        def fn(gid: Builtin.global_invocation_id, out: StorageBuffer[f32, "read_write"]):
            out[0] = f32(gid.x)
        assert "float(gid.x)" in msl(fn)

    def test_f16_buffer_is_half_no_enable(self):
        def fn(gid: Builtin.global_invocation_id, w: StorageBuffer[f16, "read"],
               out: StorageBuffer[f32, "read_write"]):
            out[gid.x] = f32(w[gid.x])
        src = msl(fn)
        assert "device const half* w" in src
        assert "enable" not in src  # WGSL directives never leak into MSL


class TestIntrinsics:
    def test_barrier(self):
        def fn(lid: Builtin.local_invocation_id, smem: WorkgroupArray[f32, 64]):
            smem[lid.x] = 0.0
            barrier()
        src = msl(fn)
        assert "threadgroup float smem[64];" in src
        assert "threadgroup_barrier(mem_flags::mem_threadgroup);" in src

    def test_subgroup_ops(self):
        def fn(lane: Builtin.subgroup_invocation_id,
               out: StorageBuffer[f32, "read_write"]):
            total: f32 = subgroupAdd(1.0)
            m: f32 = subgroupMax(2.0)
            subgroup_barrier()
            if lane == 0:
                out[0] = total + m
        src = msl(fn)
        assert "simd_sum(1.0)" in src
        assert "simd_max(2.0)" in src
        assert "simdgroup_barrier(mem_flags::mem_device);" in src

    def test_unpack2x16float(self):
        def fn(gid: Builtin.global_invocation_id, w: StorageBuffer[u32, "read"],
               out: StorageBuffer[f32, "read_write"]):
            pair = unpack2x16float(w[gid.x])
            out[gid.x] = pair.x + pair.y
        assert "float2(as_type<half2>(w[gid.x]))" in msl(fn)


class TestReservedWords:
    def test_half_is_mangled(self):
        def fn(gid: Builtin.global_invocation_id, dims: StorageBuffer[u32, "read"]):
            half: u32 = dims[0] / 2
            x = half + 1
        src = msl(fn)
        assert "uint half_ = dims[0] / 2;" in src
        assert "half_ + 1" in src
        # WGSL is unaffected (half is not reserved there)
        assert "var half: u32 = dims[0] / 2;" in translate(fn)

    def test_loop_var_mangled(self):
        def fn(gid: Builtin.global_invocation_id):
            for thread in range(4):
                pass
        assert "for (uint thread_ = 0; thread_ < 4; thread_++)" in msl(fn)


class TestKernelDecoratorMSL:
    def test_attaches_both_targets(self):
        @kernel(workgroup_size=(64,))
        def fn(gid: Builtin.global_invocation_id):
            pass
        assert "@compute" in fn.wgsl
        assert "kernel void fn(" in fn.msl
        assert fn.workgroup_size == (64,)


# ---------------------------------------------------------------------------
# Compile validation: every gemma kernel must compile via the Metal compiler
# ---------------------------------------------------------------------------

metalgpu = pytest.importorskip("metalgpu")


def _capture_fd2(fn):
    """Capture C-level stderr (Metal compiler diagnostics print there)."""
    saved = os.dup(2)
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            fn()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        tmp.seek(0)
        return tmp.read().decode(errors="replace")


class TestMetalCompilation:
    @pytest.fixture(scope="class")
    @classmethod
    def inst(cls):
        try:
            return metalgpu.Interface()
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"Metal unavailable: {exc}")

    def test_all_gemma_kernels_compile(self, inst):
        from gemma3.kernels import KERNELS

        failures = {}
        for name, kern in KERNELS.items():
            err = _capture_fd2(lambda k=kern, n=name: (
                inst.load_shader_from_string(k.msl), inst.set_function(n)))
            if "error" in err.lower():
                failures[name] = err[:300]
        assert not failures, failures

    def test_all_metal_kernels_compile(self, inst):
        from gemma3.kernels_metal import METAL_KERNELS

        failures = {}
        for name, kern in METAL_KERNELS.items():
            err = _capture_fd2(lambda k=kern, n=name: (
                inst.load_shader_from_string(k.msl), inst.set_function(n)))
            if "error" in err.lower():
                failures[name] = err[:300]
        assert not failures, failures
