"""Tests for the WGSL type system."""

import pytest
from py_shader_lang_wgpu.types import (
    u32, f32, i32, bool_,
    vec2, vec3, vec4,
    StorageBuffer, StorageBufferType,
    Uniform, UniformType,
    Builtin, BuiltinValue,
    WGSLType,
)


class TestScalarTypes:
    def test_wgsl_names(self):
        assert u32.wgsl_name == "u32"
        assert f32.wgsl_name == "f32"
        assert i32.wgsl_name == "i32"
        assert bool_.wgsl_name == "bool"

    def test_repr(self):
        assert repr(u32) == "u32"
        assert repr(f32) == "f32"


class TestVecTypes:
    def test_vec3_u32(self):
        t = vec3[u32]
        assert t.wgsl_name == "vec3<u32>"

    def test_vec4_f32(self):
        t = vec4[f32]
        assert t.wgsl_name == "vec4<f32>"

    def test_vec2_i32(self):
        t = vec2[i32]
        assert t.wgsl_name == "vec2<i32>"

    def test_each_factory_returns_new_type(self):
        a = vec3[u32]
        b = vec3[u32]
        assert a.wgsl_name == b.wgsl_name


class TestStorageBuffer:
    def test_read_only(self):
        t = StorageBuffer[f32, "read"]
        assert isinstance(t, StorageBufferType)
        assert t.elem_type is f32
        assert t.access == "read"
        assert t.wgsl_name == "array<f32>"

    def test_read_write(self):
        t = StorageBuffer[f32, "read_write"]
        assert t.access == "read_write"

    def test_default_access_is_read_write(self):
        t = StorageBuffer[u32]
        assert t.access == "read_write"

    def test_elem_type_preserved(self):
        t = StorageBuffer[u32, "read"]
        assert t.elem_type is u32


class TestUniform:
    def test_uniform_u32(self):
        t = Uniform[u32]
        assert isinstance(t, UniformType)
        assert t.elem_type is u32
        assert t.wgsl_name == "u32"

    def test_uniform_f32(self):
        t = Uniform[f32]
        assert t.wgsl_name == "f32"


class TestBuiltin:
    def test_global_invocation_id(self):
        b = Builtin.global_invocation_id
        assert isinstance(b, BuiltinValue)
        assert b.builtin_name == "global_invocation_id"
        assert b.wgsl_name == "vec3<u32>"

    def test_local_invocation_id(self):
        b = Builtin.local_invocation_id
        assert b.wgsl_name == "vec3<u32>"

    def test_workgroup_id(self):
        b = Builtin.workgroup_id
        assert b.wgsl_name == "vec3<u32>"

    def test_local_invocation_index(self):
        b = Builtin.local_invocation_index
        assert b.wgsl_name == "u32"

    def test_num_workgroups(self):
        b = Builtin.num_workgroups
        assert b.wgsl_name == "vec3<u32>"
