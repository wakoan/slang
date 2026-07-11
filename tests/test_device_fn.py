"""Tests for @device_fn: shader-level helper functions."""

import numpy as np
import pytest

from py_shader_lang_wgpu import (
    translate, kernel, device_fn, TranslationError,
    u32, f32, StorageBuffer, Builtin,
)


@device_fn
def square(x: f32) -> f32:
    return x * x


@device_fn
def cube(x: f32) -> f32:
    return square(x) * x  # calls another device_fn


class TestWGSL:
    def test_definition_emitted_before_entry(self):
        def fn(gid: Builtin.global_invocation_id,
               out: StorageBuffer[f32, "read_write"]):
            out[gid.x] = square(2.0)
        src = translate(fn)
        assert "fn square(x: f32) -> f32 {" in src
        assert src.index("fn square") < src.index("fn fn(")
        assert "out[gid.x] = square(2.0);" in src

    def test_transitive_dependency_callee_first(self):
        def fn(gid: Builtin.global_invocation_id,
               out: StorageBuffer[f32, "read_write"]):
            out[gid.x] = cube(2.0)
        src = translate(fn)
        assert "fn square" in src and "fn cube" in src
        assert src.index("fn square") < src.index("fn cube")

    def test_deduplicated(self):
        def fn(gid: Builtin.global_invocation_id,
               out: StorageBuffer[f32, "read_write"]):
            out[0] = square(1.0) + square(2.0) + cube(3.0)
        src = translate(fn)
        assert src.count("fn square(") == 1

    def test_no_definitions_when_unused(self):
        def fn(gid: Builtin.global_invocation_id,
               out: StorageBuffer[f32, "read_write"]):
            out[gid.x] = 1.0
        src = translate(fn)
        assert "fn square" not in src

    def test_uint_params_and_multiple_args(self):
        @device_fn
        def flat_index(row: u32, col: u32, width: u32) -> u32:
            return row * width + col

        def fn(gid: Builtin.global_invocation_id,
               out: StorageBuffer[f32, "read_write"]):
            out[flat_index(gid.y, gid.x, 8)] = 1.0
        src = translate(fn)
        assert "fn flat_index(row: u32, col: u32, width: u32) -> u32 {" in src


class TestMSL:
    def test_definition_and_types(self):
        def fn(gid: Builtin.global_invocation_id,
               out: StorageBuffer[f32, "read_write"]):
            out[gid.x] = cube(2.0)
        src = translate(fn, target="msl")
        assert "float square(float x) {" in src
        assert "float cube(float x) {" in src
        assert src.index("float square") < src.index("float cube")

    def test_reserved_name_mangled(self):
        @device_fn
        def half(x: f32) -> f32:  # `half` is an MSL type
            return x * 0.5

        def fn(gid: Builtin.global_invocation_id,
               out: StorageBuffer[f32, "read_write"]):
            out[gid.x] = half(4.0)
        src = translate(fn, target="msl")
        assert "float half_(float x) {" in src
        assert "half_(4.0)" in src
        # WGSL untouched
        assert "fn half(x: f32)" in translate(fn)


class TestErrors:
    def test_unknown_function_raises(self):
        def fn(gid: Builtin.global_invocation_id,
               out: StorageBuffer[f32, "read_write"]):
            out[gid.x] = sqtr(2.0)  # typo of sqrt
        with pytest.raises(TranslationError, match="Unknown function 'sqtr'"):
            translate(fn)

    def test_builtins_still_allowed(self):
        def fn(gid: Builtin.global_invocation_id,
               out: StorageBuffer[f32, "read_write"]):
            out[gid.x] = sqrt(abs(-4.0))
        assert "sqrt(abs(-4.0))" in translate(fn)

    def test_buffer_param_rejected(self):
        with pytest.raises(TranslationError, match="scalar or"):
            @device_fn
            def bad(buf: StorageBuffer[f32, "read"]) -> f32:
                return buf[0]

            def fn(gid: Builtin.global_invocation_id,
                   out: StorageBuffer[f32, "read_write"]):
                out[0] = bad(1.0)
            translate(fn)

    def test_recursion_rejected(self):
        @device_fn
        def rec(x: f32) -> f32:
            return rec(x - 1.0)

        def fn(gid: Builtin.global_invocation_id,
               out: StorageBuffer[f32, "read_write"]):
            out[0] = rec(3.0)
        with pytest.raises(TranslationError, match="[Rr]ecursi"):
            translate(fn)


class TestGPUExecution:
    def test_device_fn_kernel_matches_numpy(self):
        wgpu = pytest.importorskip("wgpu")

        @kernel(workgroup_size=(64,))
        def poly(
            gid: Builtin.global_invocation_id,
            x_in: StorageBuffer[f32, "read"],
            y_out: StorageBuffer[f32, "read_write"],
            dims: StorageBuffer[u32, "read"],
        ):
            i: u32 = gid.x
            if i >= dims[0]:
                return
            y_out[i] = cube(x_in[i]) + square(x_in[i])

        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            device = adapter.request_device_sync()
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"no GPU: {exc}")

        n = 256
        x = np.random.default_rng(3).standard_normal(n).astype(np.float32)
        bufs = []
        for arr, rw in [(x, False), (np.zeros(n, np.float32), True),
                        (np.array([n], np.uint32), False)]:
            bufs.append(device.create_buffer_with_data(
                data=arr.tobytes(),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC))
        layout = device.create_bind_group_layout(entries=[
            {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage if i == 1
                        else wgpu.BufferBindingType.read_only_storage}}
            for i in range(3)])
        bg = device.create_bind_group(layout=layout, entries=[
            {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
            for i, b in enumerate(bufs)])
        pipe = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[layout]),
            compute={"module": device.create_shader_module(code=poly.wgsl),
                     "entry_point": "poly"})
        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups((n + 63) // 64, 1, 1)
        cp.end()
        device.queue.submit([enc.finish()])
        got = np.frombuffer(device.queue.read_buffer(bufs[1]), dtype=np.float32)
        np.testing.assert_allclose(got, x**3 + x**2, rtol=1e-5, atol=1e-5)
