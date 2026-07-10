"""Regression tests for translator correctness fixes.

Covers: let-reassignment, branch-scoped declarations, negative range steps,
floor division, workgroup_size validation, source-less functions,
while/break/continue, range() keywords, and parenthesisation rules.
"""

import pytest
from py_shader_lang_wgpu import (
    translate, TranslationError,
    u32, f32,
    StorageBuffer, Builtin, WorkgroupArray,
)


def wgsl(func, workgroup_size=(1,)):
    return translate(func, workgroup_size=workgroup_size)


# ---------------------------------------------------------------------------
# Fix 1: plain reassignment must emit `var`, not `let`
# ---------------------------------------------------------------------------

class TestReassignment:
    def test_double_assign_emits_var(self):
        def fn(gid: Builtin.global_invocation_id):
            x = 1
            x = 2
        src = wgsl(fn)
        assert "var x = 1;" in src
        assert "x = 2;" in src
        assert "let x" not in src

    def test_single_assign_still_let(self):
        def fn(gid: Builtin.global_invocation_id):
            x = 1
        assert "let x = 1;" in wgsl(fn)

    def test_reassign_inside_loop(self):
        def fn(gid: Builtin.global_invocation_id):
            x = 0
            for i in range(4):
                x = i
        src = wgsl(fn)
        assert "var x = 0;" in src
        assert "x = i;" in src


# ---------------------------------------------------------------------------
# Fix 2: branch-scoped declarations must not leak
# ---------------------------------------------------------------------------

class TestBranchScoping:
    def test_assign_in_both_branches_raises(self):
        def fn(gid: Builtin.global_invocation_id):
            if gid.x > 0:
                y = 1
            else:
                y = 2
        with pytest.raises(TranslationError, match="nested block"):
            wgsl(fn)

    def test_use_after_branch_declaration_raises(self):
        def fn(gid: Builtin.global_invocation_id):
            if gid.x > 0:
                y = 1
            z = y
        with pytest.raises(TranslationError, match="outside the block|nested block"):
            wgsl(fn)

    def test_annotated_declaration_before_branch_works(self):
        def fn(gid: Builtin.global_invocation_id):
            y: u32 = 0
            if gid.x > 0:
                y = 1
            else:
                y = 2
            z = y
        src = wgsl(fn)
        assert "var y: u32 = 0;" in src
        assert "y = 1;" in src
        assert "y = 2;" in src
        assert "let z = y;" in src

    def test_local_to_branch_is_fine(self):
        def fn(gid: Builtin.global_invocation_id):
            if gid.x > 0:
                tmp = 1
        assert "let tmp = 1;" in wgsl(fn)

    def test_loop_var_used_after_loop_raises(self):
        def fn(gid: Builtin.global_invocation_id):
            for i in range(4):
                pass
            x = i
        with pytest.raises(TranslationError, match="outside the block"):
            wgsl(fn)

    def test_redeclaration_in_same_scope_raises(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            x: u32 = 1
        with pytest.raises(TranslationError, match="already declared"):
            wgsl(fn)


# ---------------------------------------------------------------------------
# Fix 3: negative / descending range steps
# ---------------------------------------------------------------------------

class TestNegativeRange:
    def test_descending_range_uses_i32_and_gt(self):
        def fn(gid: Builtin.global_invocation_id):
            for i in range(10, 0, -1):
                pass
        assert "for (var i: i32 = 10; i > 0; i -= 1)" in wgsl(fn)

    def test_descending_range_step_two(self):
        def fn(gid: Builtin.global_invocation_id):
            for i in range(10, 0, -2):
                pass
        assert "for (var i: i32 = 10; i > 0; i -= 2)" in wgsl(fn)

    def test_negative_start_uses_i32(self):
        def fn(gid: Builtin.global_invocation_id):
            for i in range(-5, 5):
                pass
        assert "for (var i: i32 = -5; i < 5; i++)" in wgsl(fn)

    def test_ascending_range_still_u32(self):
        def fn(gid: Builtin.global_invocation_id):
            for i in range(0, 10, 2):
                pass
        assert "for (var i: u32 = 0; i < 10; i += 2)" in wgsl(fn)

    def test_zero_step_raises(self):
        def fn(gid: Builtin.global_invocation_id):
            for i in range(0, 10, 0):
                pass
        with pytest.raises(TranslationError, match="step must not be zero"):
            wgsl(fn)


# ---------------------------------------------------------------------------
# Fix 4: floor division is rejected
# ---------------------------------------------------------------------------

class TestFloorDiv:
    def test_floordiv_raises(self):
        def fn(gid: Builtin.global_invocation_id):
            a: f32 = 7.0
            b: f32 = a // 2.0
        with pytest.raises(TranslationError, match="Floor division"):
            wgsl(fn)

    def test_aug_floordiv_raises(self):
        def fn(gid: Builtin.global_invocation_id):
            a: f32 = 7.0
            a //= 2.0
        with pytest.raises(TranslationError, match="Floor division"):
            wgsl(fn)


# ---------------------------------------------------------------------------
# Fix 5: workgroup_size validation
# ---------------------------------------------------------------------------

class TestWorkgroupSizeValidation:
    def _fn(self):
        def fn(gid: Builtin.global_invocation_id):
            pass
        return fn

    def test_four_dims_raises(self):
        with pytest.raises(TranslationError, match="1–3 dimensions"):
            wgsl(self._fn(), workgroup_size=(1, 2, 3, 4))

    def test_empty_raises(self):
        with pytest.raises(TranslationError, match="1–3 dimensions"):
            wgsl(self._fn(), workgroup_size=())

    def test_zero_entry_raises(self):
        with pytest.raises(TranslationError, match="positive integers"):
            wgsl(self._fn(), workgroup_size=(0,))

    def test_negative_entry_raises(self):
        with pytest.raises(TranslationError, match="positive integers"):
            wgsl(self._fn(), workgroup_size=(8, -1))

    def test_non_int_entry_raises(self):
        with pytest.raises(TranslationError, match="positive integers"):
            wgsl(self._fn(), workgroup_size=(8.0,))


# ---------------------------------------------------------------------------
# Fix 6: source-less functions raise TranslationError, not OSError
# ---------------------------------------------------------------------------

class TestSourcelessFunction:
    def test_exec_defined_function_raises_translation_error(self):
        ns = {"Builtin": Builtin}
        exec(
            "def f(gid: Builtin.global_invocation_id):\n    pass",
            ns,
        )
        with pytest.raises(TranslationError, match="source"):
            translate(ns["f"])


# ---------------------------------------------------------------------------
# Fix 7: while / break / continue
# ---------------------------------------------------------------------------

class TestWhileBreakContinue:
    def test_while_loop(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            while x < 10:
                x += 1
        src = wgsl(fn)
        assert "while (x < 10) {" in src
        assert "x += 1;" in src

    def test_while_true(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            while True:
                x += 1
                if x > 5:
                    break
        src = wgsl(fn)
        assert "while (true) {" in src
        assert "break;" in src

    def test_continue_in_for(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            for i in range(10):
                if i == 3:
                    continue
                x += i
        src = wgsl(fn)
        assert "continue;" in src

    def test_break_in_for(self):
        def fn(gid: Builtin.global_invocation_id):
            for i in range(10):
                if i == 3:
                    break
        assert "break;" in wgsl(fn)

    def test_while_else_raises(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            while x < 10:
                x += 1
            else:
                x = 0
        with pytest.raises(TranslationError, match="while-else"):
            wgsl(fn)

    def test_for_else_raises(self):
        def fn(gid: Builtin.global_invocation_id):
            for i in range(10):
                pass
            else:
                x = 1
        with pytest.raises(TranslationError, match="for-else"):
            wgsl(fn)

    def test_while_body_scoped(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            while x < 10:
                y = 1
                x += 1
            z = y
        with pytest.raises(TranslationError, match="outside the block"):
            wgsl(fn)


# ---------------------------------------------------------------------------
# Fix 8: range() keyword arguments are rejected
# ---------------------------------------------------------------------------

class TestRangeKeywords:
    def test_range_keyword_raises(self):
        def fn(gid: Builtin.global_invocation_id):
            for i in range(0, 10, step=2):
                pass
        with pytest.raises(TranslationError, match="keyword arguments"):
            wgsl(fn)


# ---------------------------------------------------------------------------
# Fix 9: minimal parenthesisation
# ---------------------------------------------------------------------------

class TestParenthesisation:
    def test_bare_compare_in_if(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            if x > 5:
                return
        assert "if (x > 5) {" in wgsl(fn)

    def test_arith_no_redundant_parens(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 1
            y: u32 = x + 2 * 3
        assert "var y: u32 = x + 2 * 3;" in wgsl(fn)

    def test_needed_parens_kept(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 1
            y: u32 = (x + 2) * 3
        assert "var y: u32 = (x + 2) * 3;" in wgsl(fn)

    def test_subscript_index_bare(self):
        def fn(gid: Builtin.global_invocation_id, buf: StorageBuffer[f32, "read"]):
            row: u32 = gid.y
            v: f32 = buf[row * 4 + 1]
        assert "buf[row * 4 + 1]" in wgsl(fn)

    def test_bool_or_of_compares_bare(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 0
            y: u32 = 0
            if x > 5 or y > 5:
                return
        assert "if (x > 5 || y > 5) {" in wgsl(fn)

    def test_mixed_and_or_parenthesised(self):
        def fn(gid: Builtin.global_invocation_id):
            a: u32 = 0
            b: u32 = 0
            c: u32 = 0
            if a > 0 or b > 0 and c > 0:
                return
        # WGSL forbids mixing && and || without parens
        assert "if (a > 0 || (b > 0 && c > 0)) {" in wgsl(fn)

    def test_bitwise_always_parenthesised(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 6
            y: u32 = x & 3
            z: u32 = x >> 1
        src = wgsl(fn)
        assert "var y: u32 = (x & 3);" in src
        assert "var z: u32 = (x >> 1);" in src

    def test_subtraction_right_assoc_parens_kept(self):
        def fn(gid: Builtin.global_invocation_id):
            a: u32 = 9
            b: u32 = 3
            c: u32 = 1
            d: u32 = a - (b - c)
        assert "var d: u32 = a - (b - c);" in wgsl(fn)

    def test_chained_compare_wrapped_in_bool_context(self):
        def fn(gid: Builtin.global_invocation_id):
            x: u32 = 3
            b: u32 = 0
            if b > 0 or 0 < x < 10:
                return
        # The && chain must be parenthesised inside the ||
        assert "|| ((0 < x) && (x < 10))" in wgsl(fn)


# ---------------------------------------------------------------------------
# Workgroup shared memory + barrier
# ---------------------------------------------------------------------------

class TestWorkgroupMemory:
    def test_workgroup_array_declaration(self):
        def fn(
            gid: Builtin.global_invocation_id,
            buf: StorageBuffer[f32, "read"],
            partial: WorkgroupArray[f32, 64],
        ):
            pass
        src = wgsl(fn)
        assert "var<workgroup> partial: array<f32, 64>;" in src
        # not a binding
        assert "@binding(1)" not in src

    def test_workgroup_array_u32(self):
        def fn(gid: Builtin.global_invocation_id, idxs: WorkgroupArray[u32, 32]):
            pass
        assert "var<workgroup> idxs: array<u32, 32>;" in wgsl(fn)

    def test_barrier_renamed(self):
        def fn(
            lid: Builtin.local_invocation_id,
            partial: WorkgroupArray[f32, 64],
        ):
            partial[lid.x] = 0.0
            barrier()
        src = wgsl(fn)
        assert "workgroupBarrier();" in src
        assert "barrier();" not in src.replace("workgroupBarrier();", "")

    def test_reduction_pattern(self):
        def fn(
            wid: Builtin.workgroup_id,
            lid: Builtin.local_invocation_id,
            x_in: StorageBuffer[f32, "read"],
            y_out: StorageBuffer[f32, "read_write"],
            dims: StorageBuffer[u32, "read"],
            partial: WorkgroupArray[f32, 64],
        ):
            li: u32 = lid.x
            n: u32 = dims[0]
            acc: f32 = 0.0
            for j in range(li, n, 64):
                acc += x_in[j]
            partial[li] = acc
            barrier()
            s: u32 = 32
            while s > 0:
                if li < s:
                    partial[li] = partial[li] + partial[li + s]
                barrier()
                s = s / 2
            if li == 0:
                y_out[wid.x] = partial[0]
        src = wgsl(fn)
        assert "var<workgroup> partial: array<f32, 64>;" in src
        assert "for (var j: u32 = li; j < n; j += 64)" in src
        assert "while (s > 0)" in src
        assert src.count("workgroupBarrier();") == 2


# ---------------------------------------------------------------------------
# f16 support
# ---------------------------------------------------------------------------

class TestF16:
    def test_f16_buffer_emits_enable_directive(self):
        from py_shader_lang_wgpu import f16

        def fn(
            gid: Builtin.global_invocation_id,
            w: StorageBuffer[f16, "read"],
            out: StorageBuffer[f32, "read_write"],
        ):
            out[gid.x] = f32(w[gid.x])
        src = wgsl(fn)
        assert src.startswith("enable f16;")
        assert "var<storage, read> w: array<f16>;" in src

    def test_no_directive_without_f16(self):
        def fn(gid: Builtin.global_invocation_id, buf: StorageBuffer[f32, "read"]):
            pass
        assert "enable f16;" not in wgsl(fn)


# ---------------------------------------------------------------------------
# Subgroup support
# ---------------------------------------------------------------------------

class TestSubgroups:
    def test_subgroup_call_emits_directive(self):
        def fn(
            lid: Builtin.local_invocation_id,
            out: StorageBuffer[f32, "read_write"],
        ):
            acc: f32 = 1.0
            total: f32 = subgroupAdd(acc)
            if lid.x == 0:
                out[0] = total
        src = wgsl(fn)
        assert "enable subgroups;" in src
        assert "subgroupAdd(acc)" in src

    def test_subgroup_builtin_emits_directive(self):
        def fn(
            lane: Builtin.subgroup_invocation_id,
            sgs: Builtin.subgroup_size,
            out: StorageBuffer[u32, "read_write"],
        ):
            if lane == 0:
                out[0] = sgs
        src = wgsl(fn)
        assert "enable subgroups;" in src
        assert "@builtin(subgroup_invocation_id) lane: u32" in src
        assert "@builtin(subgroup_size) sgs: u32" in src

    def test_no_directive_without_subgroups(self):
        def fn(gid: Builtin.global_invocation_id):
            pass
        assert "enable subgroups;" not in wgsl(fn)

    def test_subgroup_max_detected(self):
        def fn(lid: Builtin.local_invocation_id, out: StorageBuffer[f32, "read_write"]):
            m: f32 = subgroupMax(1.0)
            out[0] = m
        assert "enable subgroups;" in wgsl(fn)
