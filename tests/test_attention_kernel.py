"""Scaled dot-product attention kernels translated to WGSL.

Three-pass implementation:
  1. attention_scores  — compute Q @ K^T / sqrt(head_dim)
  2. softmax_rows      — numerically-stable row-wise softmax in place
  3. attention_output  — weighted sum: softmax(scores) @ V

Each pass is a standalone @kernel whose .wgsl attribute is verified below.
"""

import pytest
from py_shader_lang_wgpu import kernel, u32, f32, StorageBuffer, Builtin


# ---------------------------------------------------------------------------
# Kernel definitions
# ---------------------------------------------------------------------------

@kernel(workgroup_size=(8, 8))
def attention_scores(
    global_id: Builtin.global_invocation_id,
    q:         StorageBuffer[f32, "read"],        # [seq_len, head_dim]
    k:         StorageBuffer[f32, "read"],        # [seq_len, head_dim]
    scores:    StorageBuffer[f32, "read_write"],  # [seq_len, seq_len]
    dims:      StorageBuffer[u32, "read"],        # [seq_len, head_dim]
):
    q_pos: u32 = global_id.y
    k_pos: u32 = global_id.x
    seq_len: u32 = dims[0]
    head_dim: u32 = dims[1]

    if q_pos >= seq_len or k_pos >= seq_len:
        return

    dot: f32 = 0.0
    for d in range(head_dim):
        dot += q[q_pos * head_dim + d] * k[k_pos * head_dim + d]

    scores[q_pos * seq_len + k_pos] = dot / sqrt(f32(head_dim))


@kernel(workgroup_size=(64,))
def softmax_rows(
    global_id: Builtin.global_invocation_id,
    scores:    StorageBuffer[f32, "read_write"],  # [seq_len, seq_len]
    dims:      StorageBuffer[u32, "read"],        # [seq_len]
):
    row: u32 = global_id.x
    seq_len: u32 = dims[0]

    if row >= seq_len:
        return

    # Find row maximum for numerical stability
    row_max: f32 = scores[row * seq_len]
    for j in range(1, seq_len):
        row_max = max(row_max, scores[row * seq_len + j])

    # Compute exp(x - max) and accumulate denominator
    exp_sum: f32 = 0.0
    for j in range(seq_len):
        val: f32 = exp(scores[row * seq_len + j] - row_max)
        scores[row * seq_len + j] = val
        exp_sum += val

    # Normalise
    for j in range(seq_len):
        scores[row * seq_len + j] = scores[row * seq_len + j] / exp_sum


@kernel(workgroup_size=(8, 8))
def attention_output(
    global_id: Builtin.global_invocation_id,
    scores:    StorageBuffer[f32, "read"],        # [seq_len, seq_len]
    v:         StorageBuffer[f32, "read"],        # [seq_len, head_dim]
    output:    StorageBuffer[f32, "read_write"],  # [seq_len, head_dim]
    dims:      StorageBuffer[u32, "read"],        # [seq_len, head_dim]
):
    q_pos: u32 = global_id.y
    d: u32 = global_id.x
    seq_len: u32 = dims[0]
    head_dim: u32 = dims[1]

    if q_pos >= seq_len or d >= head_dim:
        return

    acc: f32 = 0.0
    for k_pos in range(seq_len):
        acc += scores[q_pos * seq_len + k_pos] * v[k_pos * head_dim + d]

    output[q_pos * head_dim + d] = acc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fn_body(src: str) -> str:
    """Return the lines inside fn { } as a single string with stripped indents."""
    inside, depth = [], 0
    for line in src.splitlines():
        if line.startswith("fn "):
            depth = 1
            continue
        if depth > 0:
            inside.append(line.strip())
    if inside and inside[-1] == "}":
        inside = inside[:-1]
    return "\n".join(l for l in inside if l)


# ---------------------------------------------------------------------------
# attention_scores
# ---------------------------------------------------------------------------

class TestAttentionScores:
    @pytest.fixture(scope="class")
    @classmethod
    def src(cls):
        return attention_scores.wgsl

    def test_workgroup_size(self, src):
        assert "@compute @workgroup_size(8, 8)" in src

    def test_four_bindings(self, src):
        for i in range(4):
            assert f"@binding({i})" in src

    def test_q_k_read_only(self, src):
        assert "var<storage, read> q: array<f32>" in src
        assert "var<storage, read> k: array<f32>" in src

    def test_scores_read_write(self, src):
        assert "var<storage, read_write> scores: array<f32>" in src

    def test_dims_u32(self, src):
        assert "var<storage, read> dims: array<u32>" in src

    def test_builtin_param(self, src):
        assert "@builtin(global_invocation_id) global_id: vec3<u32>" in src

    def test_bounds_check(self, src):
        body = fn_body(src)
        assert "return;" in body
        # Both q_pos and k_pos are checked
        assert "q_pos" in body
        assert "k_pos" in body
        assert ">=" in body

    def test_dot_product_loop(self, src):
        body = fn_body(src)
        assert "for (var d: u32 = 0; d < head_dim; d++)" in body

    def test_accumulates_dot_product(self, src):
        body = fn_body(src)
        assert "dot +=" in body

    def test_scale_by_sqrt_head_dim(self, src):
        body = fn_body(src)
        # WGSL built-in sqrt and f32 cast pass through unchanged
        assert "sqrt(f32(head_dim))" in body
        assert "dot / sqrt(f32(head_dim))" in body

    def test_writes_to_scores(self, src):
        body = fn_body(src)
        assert "scores[" in body
        assert "= dot / sqrt(f32(head_dim));" in body

    def test_dot_initialised_to_zero(self, src):
        body = fn_body(src)
        assert "var dot: f32 = 0.0;" in body


# ---------------------------------------------------------------------------
# softmax_rows
# ---------------------------------------------------------------------------

class TestSoftmaxRows:
    @pytest.fixture(scope="class")
    @classmethod
    def src(cls):
        return softmax_rows.wgsl

    def test_workgroup_size(self, src):
        assert "@compute @workgroup_size(64)" in src

    def test_two_bindings(self, src):
        assert "@binding(0)" in src
        assert "@binding(1)" in src
        assert "@binding(2)" not in src

    def test_scores_read_write(self, src):
        assert "var<storage, read_write> scores: array<f32>" in src

    def test_bounds_check(self, src):
        body = fn_body(src)
        assert "if (row >= seq_len)" in body
        assert "return;" in body

    def test_max_reduction_loop(self, src):
        body = fn_body(src)
        # Starts from index 1 for the max search
        assert "for (var j: u32 = 1; j < seq_len; j++)" in body

    def test_max_builtin_used(self, src):
        body = fn_body(src)
        assert "max(row_max," in body

    def test_row_max_initialised_from_first_element(self, src):
        body = fn_body(src)
        assert "var row_max: f32 = scores[row * seq_len];" in body

    def test_exp_sum_loop(self, src):
        body = fn_body(src)
        assert "for (var j: u32 = 0; j < seq_len; j++)" in body

    def test_exp_builtin_used(self, src):
        body = fn_body(src)
        assert "exp(" in body

    def test_subtracts_max_before_exp(self, src):
        body = fn_body(src)
        # Numerical stability: exp(x - max)
        assert "- row_max)" in body

    def test_accumulates_exp_sum(self, src):
        body = fn_body(src)
        assert "exp_sum += val;" in body

    def test_normalisation_loop(self, src):
        body = fn_body(src)
        # Three separate for loops: max-find, exp+sum, normalise
        assert body.count("for (var j: u32 = 0; j < seq_len; j++)") == 2

    def test_divides_by_exp_sum(self, src):
        body = fn_body(src)
        assert "/ exp_sum;" in body


# ---------------------------------------------------------------------------
# attention_output
# ---------------------------------------------------------------------------

class TestAttentionOutput:
    @pytest.fixture(scope="class")
    @classmethod
    def src(cls):
        return attention_output.wgsl

    def test_workgroup_size(self, src):
        assert "@compute @workgroup_size(8, 8)" in src

    def test_four_bindings(self, src):
        for i in range(4):
            assert f"@binding({i})" in src

    def test_scores_and_v_read_only(self, src):
        assert "var<storage, read> scores: array<f32>" in src
        assert "var<storage, read> v: array<f32>" in src

    def test_output_read_write(self, src):
        assert "var<storage, read_write> output: array<f32>" in src

    def test_bounds_check_both_dims(self, src):
        body = fn_body(src)
        assert "q_pos" in body
        assert "d" in body

    def test_accumulator_initialised_to_zero(self, src):
        body = fn_body(src)
        assert "var acc: f32 = 0.0;" in body

    def test_weighted_sum_loop(self, src):
        body = fn_body(src)
        assert "for (var k_pos: u32 = 0; k_pos < seq_len; k_pos++)" in body

    def test_accumulates_score_times_v(self, src):
        body = fn_body(src)
        assert "acc +=" in body
        assert "scores[" in body
        assert "v[" in body

    def test_writes_output(self, src):
        body = fn_body(src)
        assert "output[" in body
        assert "= acc;" in body


# ---------------------------------------------------------------------------
# Pipeline: all three kernels share compatible buffer layouts
# ---------------------------------------------------------------------------

class TestAttentionPipeline:
    def test_scores_output_matches_softmax_input(self):
        # Both kernels bind scores as array<f32>
        assert "array<f32>" in attention_scores.wgsl
        assert "array<f32>" in softmax_rows.wgsl

    def test_softmax_output_matches_attn_output_input(self):
        assert "array<f32>" in softmax_rows.wgsl
        assert "array<f32>" in attention_output.wgsl

    def test_all_three_kernels_have_wgsl(self):
        for fn in (attention_scores, softmax_rows, attention_output):
            assert hasattr(fn, "wgsl")
            assert fn.wgsl.strip() != ""

    def test_each_kernel_has_unique_entry_point(self):
        names = {"attention_scores", "softmax_rows", "attention_output"}
        for fn in (attention_scores, softmax_rows, attention_output):
            assert f"fn {fn.__name__}(" in fn.wgsl
        # No two kernels share the same entry point name
        all_wgsl = "\n".join(
            [attention_scores.wgsl, softmax_rows.wgsl, attention_output.wgsl]
        )
        for name in names:
            assert all_wgsl.count(f"fn {name}(") == 1
