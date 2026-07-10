"""Tests for the Python-to-WGSL AST translator."""

import textwrap
import pytest
from py_shader_lang_wgpu import (
    kernel, translate, TranslationError,
    u32, f32, i32,
    vec3,
    StorageBuffer, Uniform, Builtin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wgsl(func, workgroup_size=(1,)):
    return translate(func, workgroup_size=workgroup_size)


def fn_body(wgsl_src: str) -> str:
    """Extract the lines inside fn { } from generated WGSL."""
    lines = wgsl_src.splitlines()
    inside = []
    depth = 0
    for line in lines:
        if line.startswith("fn "):
            depth = 1
            continue
        if depth > 0:
            inside.append(line)
    # Strip the closing brace
    if inside and inside[-1].strip() == "}":
        inside = inside[:-1]
    return "\n".join(l.strip() for l in inside if l.strip())


def has_binding(wgsl_src: str, idx: int, access: str, name: str, elem: str) -> bool:
    expected = f"@group(0) @binding({idx}) var<storage, {access}> {name}: array<{elem}>;"
    return expected in wgsl_src


# ---------------------------------------------------------------------------
# Binding declarations
# ---------------------------------------------------------------------------

class TestBindings:
    def test_storage_read(self):
        def fn(gid: Builtin.global_invocation_id, buf: StorageBuffer[f32, "read"]):
            pass
        src = wgsl(fn)
        assert has_binding(src, 0, "read", "buf", "f32")

    def test_storage_read_write(self):
        def fn(gid: Builtin.global_invocation_id, out: StorageBuffer[f32, "read_write"]):
            pass
        src = wgsl(fn)
        assert has_binding(src, 0, "read_write", "out", "f32")

    def test_multiple_bindings_get_sequential_indices(self):
        def fn(
            gid: Builtin.global_invocation_id,
            a: StorageBuffer[f32, "read"],
            b: StorageBuffer[f32, "read"],
            c: StorageBuffer[f32, "read_write"],
        ):
            pass
        src = wgsl(fn)
        assert has_binding(src, 0, "read", "a", "f32")
        assert has_binding(src, 1, "read", "b", "f32")
        assert has_binding(src, 2, "read_write", "c", "f32")

    def test_u32_storage_buffer(self):
        def fn(gid: Builtin.global_invocation_id, buf: StorageBuffer[u32, "read"]):
            pass
        src = wgsl(fn)
        assert has_binding(src, 0, "read", "buf", "u32")

    def test_uniform_declaration(self):
        def fn(gid: Builtin.global_invocation_id, size: Uniform[u32]):
            pass
        src = wgsl(fn)
        assert "@group(0) @binding(0) var<uniform> size: u32;" in src

    def test_builtins_go_in_fn_params_not_bindings(self):
        def fn(gid: Builtin.global_invocation_id, buf: StorageBuffer[f32, "read"]):
            pass
        src = wgsl(fn)
        assert "@builtin(global_invocation_id) gid: vec3<u32>" in src
        # binding index 0 should be the buffer, not the builtin
        assert has_binding(src, 0, "read", "buf", "f32")

    def test_binding_before_uniform_mixed(self):
        def fn(
            gid: Builtin.global_invocation_id,
            buf: StorageBuffer[f32, "read"],
            n: Uniform[u32],
        ):
            pass
        src = wgsl(fn)
        assert has_binding(src, 0, "read", "buf", "f32")
        assert "@group(0) @binding(1) var<uniform> n: u32;" in src

    def test_no_bindings_when_only_builtins(self):
        def fn(gid: Builtin.global_invocation_id):
            pass
        src = wgsl(fn)
        assert "@binding" not in src

    def test_missing_annotation_raises(self):
        def fn(gid: Builtin.global_invocation_id, buf):
            pass
        with pytest.raises(TranslationError, match="no type annotation"):
            wgsl(fn)


# ---------------------------------------------------------------------------
# Workgroup size
# ---------------------------------------------------------------------------

class TestWorkgroupSize:
    def test_1d(self):
        def fn(gid: Builtin.global_invocation_id):
            pass
        assert "@compute @workgroup_size(64)" in wgsl(fn, workgroup_size=(64,))

    def test_2d(self):
        def fn(gid: Builtin.global_invocation_id):
            pass
        assert "@compute @workgroup_size(8, 8)" in wgsl(fn, workgroup_size=(8, 8))

    def test_3d(self):
        def fn(gid: Builtin.global_invocation_id):
            pass
        assert "@compute @workgroup_size(4, 4, 4)" in wgsl(fn, workgroup_size=(4, 4, 4))


# ---------------------------------------------------------------------------
# Variable declarations
# ---------------------------------------------------------------------------

class TestVarDeclarations:
    def _src(self, body_fn):
        return fn_body(wgsl(body_fn))

    def test_annotated_assignment_emits_var(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
        assert "var x: u32 = 0;" in self._src(fn)

    def test_annotated_assignment_no_value(self):
        def fn(gid: Builtin.global_invocation_id):
            x: f32
        assert "var x: f32;" in self._src(fn)

    def test_plain_assign_to_new_name_emits_let(self):
        def fn(gid: Builtin.global_invocation_id):
            x = 1
        assert "let x = 1;" in self._src(fn)

    def test_plain_assign_to_mutated_name_emits_var(self):
        def fn(gid: Builtin.global_invocation_id, buf: StorageBuffer[f32]):
            x = 0.0
            x += buf[0]
        body = self._src(fn)
        assert "var x = 0.0;" in body

    def test_subsequent_assign_to_declared_name(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            x = 5
        body = self._src(fn)
        assert "var x: u32 = 0;" in body
        assert "x = 5;" in body

    def test_vec3_annotation(self):
        def fn(gid: Builtin.global_invocation_id):
            v: vec3[u32]
        assert "var v: vec3<u32>;" in self._src(fn)


# ---------------------------------------------------------------------------
# Augmented assignment
# ---------------------------------------------------------------------------

class TestAugAssign:
    def _src(self, body_fn):
        return fn_body(wgsl(body_fn))

    def test_plus_equals(self):
        def fn(gid: Builtin.global_invocation_id):
            x: f32 = 0.0
            x += 1.0
        assert "x += 1.0;" in self._src(fn)

    def test_minus_equals(self):
        def fn(gid: Builtin.global_invocation_id):
            x: f32 = 0.0
            x -= 2.0
        assert "x -= 2.0;" in self._src(fn)

    def test_times_equals(self):
        def fn(gid: Builtin.global_invocation_id):
            x: f32 = 1.0
            x *= 3.0
        assert "x *= 3.0;" in self._src(fn)


# ---------------------------------------------------------------------------
# For loops
# ---------------------------------------------------------------------------

class TestForLoops:
    def _src(self, body_fn):
        return fn_body(wgsl(body_fn))

    def test_range_one_arg(self):
        def fn(gid: Builtin.global_invocation_id):
            for i in range(10):
                pass
        assert "for (var i: u32 = 0; i < 10; i++)" in self._src(fn)

    def test_range_two_args(self):
        def fn(gid: Builtin.global_invocation_id):
            for i in range(2, 8):
                pass
        assert "for (var i: u32 = 2; i < 8; i++)" in self._src(fn)

    def test_range_three_args(self):
        def fn(gid: Builtin.global_invocation_id):
            for i in range(0, 16, 2):
                pass
        assert "for (var i: u32 = 0; i < 16; i += 2)" in self._src(fn)

    def test_range_with_variable(self):
        def fn(gid: Builtin.global_invocation_id, buf: StorageBuffer[u32, "read"]):
            n: u32 = buf[0]
            for i in range(n):
                pass
        body = self._src(fn)
        assert "for (var i: u32 = 0; i < n; i++)" in body

    def test_loop_body_executes(self):
        def fn(gid: Builtin.global_invocation_id, buf: StorageBuffer[f32]):
            total: f32 = 0.0
            for i in range(4):
                total += buf[i]
        body = self._src(fn)
        assert "total += buf[i];" in body

    def test_non_range_raises(self):
        def fn(gid: Builtin.global_invocation_id):
            for x in [1, 2, 3]:
                pass
        with pytest.raises(TranslationError, match="range"):
            wgsl(fn)


# ---------------------------------------------------------------------------
# If / elif / else
# ---------------------------------------------------------------------------

class TestIfStatements:
    def _src(self, body_fn):
        return fn_body(wgsl(body_fn))

    def test_simple_if(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            if x > 5:
                x = 1
        body = self._src(fn)
        assert "if (x > 5)" in body
        assert "x = 1;" in body

    def test_if_else(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            if x > 5:
                x = 1
            else:
                x = 2
        body = self._src(fn)
        assert "} else {" in body
        assert "x = 2;" in body

    def test_elif(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            if x > 10:
                x = 1
            elif x > 5:
                x = 2
            else:
                x = 3
        body = self._src(fn)
        assert "} else if (" in body
        assert "x = 3;" in body

    def test_if_with_early_return(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            if x >= 100:
                return
        body = self._src(fn)
        assert "return;" in body


# ---------------------------------------------------------------------------
# Return
# ---------------------------------------------------------------------------

class TestReturn:
    def _src(self, body_fn):
        return fn_body(wgsl(body_fn))

    def test_void_return(self):
        def fn(gid: Builtin.global_invocation_id):
            return
        assert "return;" in self._src(fn)

    def test_value_return(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 1
            return x
        assert "return x;" in self._src(fn)


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

class TestExpressions:
    def _src(self, body_fn):
        return fn_body(wgsl(body_fn))

    def test_attribute_access(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = gid.x
            y: u32 = gid.y
            z: u32 = gid.z
        body = self._src(fn)
        assert "var x: u32 = gid.x;" in body
        assert "var y: u32 = gid.y;" in body
        assert "var z: u32 = gid.z;" in body

    def test_subscript(self):
        def fn(gid: Builtin.global_invocation_id, buf: StorageBuffer[f32, "read"]):
            v: f32 = buf[0]
        assert "var v: f32 = buf[0];" in self._src(fn)

    def test_addition(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 1
            y: u32 = x + 2
        assert "var y: u32 = x + 2;" in self._src(fn)

    def test_multiplication(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 3
            y: u32 = x * 4
        assert "var y: u32 = x * 4;" in self._src(fn)

    def test_binary_op_precedence_preserved_via_parens(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 1
            y: u32 = (x + 2) * 3
        body = self._src(fn)
        assert "(x + 2) * 3" in body

    def test_unary_negate(self):
        def fn(gid: Builtin.global_invocation_id):
            x: f32 = -1.0
        assert "var x: f32 = -1.0;" in self._src(fn)

    def test_unary_not(self):
        def fn(gid: Builtin.global_invocation_id):
            b: bool_
            result = not b
        assert "!(b)" in self._src(fn)

    def test_comparison_eq(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            if x == 1:
                return
        assert "x == 1" in self._src(fn)

    def test_comparison_neq(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            if x != 1:
                return
        assert "x != 1" in self._src(fn)

    def test_bool_and(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            y: u32 = 1
            if x > 0 and y > 0:
                return
        assert "&&" in self._src(fn)

    def test_bool_or(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            y: u32 = 1
            if x > 5 or y > 5:
                return
        assert "||" in self._src(fn)

    def test_chained_comparison(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 3
            if 0 < x < 10:
                return
        body = self._src(fn)
        assert "(0 < x)" in body
        assert "(x < 10)" in body
        assert "&&" in body

    def test_function_call(self):
        def fn(gid: Builtin.global_invocation_id):
            x: f32 = 1.0
            y: f32 = abs(x)
        assert "var y: f32 = abs(x);" in self._src(fn)

    def test_ternary_maps_to_select(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            y: u32 = 1 if x > 0 else 0
        body = self._src(fn)
        assert "select(" in body

    def test_float_literal(self):
        def fn(gid: Builtin.global_invocation_id):
            x: f32 = 3.14
        assert "var x: f32 = 3.14;" in self._src(fn)

    def test_integer_literal(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 42
        assert "var x: u32 = 42;" in self._src(fn)

    def test_bool_true_false(self):
        def fn(gid: Builtin.global_invocation_id):
            a = True
            b = False
        body = self._src(fn)
        assert "let a = true;" in body
        assert "let b = false;" in body


# ---------------------------------------------------------------------------
# Unsupported syntax → TranslationError
# ---------------------------------------------------------------------------

class TestErrors:
    def test_unsupported_loop_iter_raises(self):
        def fn(gid: Builtin.global_invocation_id):
            for x in [1, 2]:
                pass
        with pytest.raises(TranslationError):
            wgsl(fn)

    def test_unsupported_param_type_raises(self):
        def fn(gid: Builtin.global_invocation_id, x: int):
            pass
        with pytest.raises(TranslationError, match="Unsupported parameter type"):
            wgsl(fn)


# ---------------------------------------------------------------------------
# @kernel decorator
# ---------------------------------------------------------------------------

class TestKernelDecorator:
    def test_attaches_wgsl_attribute(self):
        @kernel(workgroup_size=(8, 8))
        def fn(gid: Builtin.global_invocation_id):
            pass
        assert hasattr(fn, "wgsl")
        assert "@compute @workgroup_size(8, 8)" in fn.wgsl

    def test_function_still_callable(self):
        @kernel(workgroup_size=(1,))
        def fn(gid: Builtin.global_invocation_id):
            pass
        fn(None)  # should not raise

    def test_used_without_args(self):
        @kernel
        def fn(gid: Builtin.global_invocation_id):
            pass
        assert hasattr(fn, "wgsl")
        assert "@compute @workgroup_size(1)" in fn.wgsl


# ---------------------------------------------------------------------------
# Full integration: matmul kernel
# ---------------------------------------------------------------------------

class TestMatmulKernel:
    @pytest.fixture(scope="class")
    @classmethod
    def matmul_wgsl(cls):
        @kernel(workgroup_size=(8, 8))
        def matmul(
            global_id: Builtin.global_invocation_id,
            matrix_a: StorageBuffer[f32, "read"],
            matrix_b: StorageBuffer[f32, "read"],
            matrix_c: StorageBuffer[f32, "read_write"],
            dims: StorageBuffer[u32, "read"],
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

        return matmul.wgsl

    def test_has_four_bindings(self, matmul_wgsl):
        for i in range(4):
            assert f"@binding({i})" in matmul_wgsl

    def test_workgroup_size(self, matmul_wgsl):
        assert "@compute @workgroup_size(8, 8)" in matmul_wgsl

    def test_builtin_param(self, matmul_wgsl):
        assert "@builtin(global_invocation_id) global_id: vec3<u32>" in matmul_wgsl

    def test_entry_point_name(self, matmul_wgsl):
        assert "fn matmul(" in matmul_wgsl

    def test_for_loop_present(self, matmul_wgsl):
        assert "for (var i: u32 = 0; i < k; i++)" in matmul_wgsl

    def test_early_return_present(self, matmul_wgsl):
        assert "return;" in matmul_wgsl

    def test_accumulator_augmented_assign(self, matmul_wgsl):
        assert "total +=" in matmul_wgsl
