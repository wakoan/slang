"""Python-to-WGSL AST translator."""

from __future__ import annotations

import ast
import inspect
import textwrap
import typing
from typing import Callable

from .types import (
    WGSLType, StorageBufferType, UniformType, BuiltinValue, WorkgroupArrayType,
)

# Python-level intrinsics renamed to their WGSL equivalents
_INTRINSIC_RENAMES = {"barrier": "workgroupBarrier"}

# Calls that require `enable subgroups;`
_SUBGROUP_FNS = {
    "subgroupAdd", "subgroupMax", "subgroupMin", "subgroupMul",
    "subgroupAnd", "subgroupOr", "subgroupXor",
    "subgroupBroadcast", "subgroupBroadcastFirst", "subgroupShuffle",
    "subgroupElect", "subgroupBallot",
}


class TranslationError(Exception):
    pass


# Expression precedence levels (higher binds tighter)
_P_OR, _P_AND, _P_CMP, _P_ADD, _P_MUL, _P_UNARY, _P_ATOM = 1, 2, 3, 4, 5, 6, 7

_ARITH_OPS: dict[type, tuple[str, int]] = {
    ast.Add: ("+", _P_ADD),
    ast.Sub: ("-", _P_ADD),
    ast.Mult: ("*", _P_MUL),
    ast.Div: ("/", _P_MUL),
    ast.Mod: ("%", _P_MUL),
}

# WGSL requires explicit parentheses when mixing these with other operators,
# so they are always emitted fully parenthesised.
_BITWISE_OPS: dict[type, str] = {
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
}

_CMP_MAP: dict[type, str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}

_VEC_NAMES = {"vec2", "vec3", "vec4"}

_FLOORDIV_MSG = (
    "Floor division '//' has no direct WGSL equivalent; "
    "use '/' for unsigned integers or floor(a / b) for floats"
)


def _const_int(node: ast.expr) -> int | None:
    """Return the value of a constant integer expression, or None."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const_int(node.operand)
        return -inner if inner is not None else None
    return None


class _WGSLTranslator:
    def __init__(
        self,
        func: Callable,
        annotations: dict[str, WGSLType],
        workgroup_size: tuple[int, ...],
    ) -> None:
        self._func = func
        self._annotations = annotations
        self._workgroup_size = workgroup_size
        self._lines: list[str] = []
        self._indent = 0
        # Scope stack: scopes[0] holds params/bindings; a new scope is pushed
        # for every nested block (if/else/for/while body).
        self._scopes: list[set[str]] = [set()]
        self._ever_declared: set[str] = set()
        self._mutable: set[str] = set()

    def _emit(self, line: str = "") -> None:
        self._lines.append(("  " * self._indent + line) if line else "")

    def _visible(self, name: str) -> bool:
        return any(name in scope for scope in self._scopes)

    def _declare(self, name: str) -> None:
        self._scopes[-1].add(name)
        self._ever_declared.add(name)

    def run(self) -> str:
        try:
            source = inspect.getsource(self._func)
        except (OSError, TypeError) as exc:
            raise TranslationError(
                f"Cannot retrieve source for '{self._func.__qualname__}'; "
                "functions defined in a REPL or via exec() have no source file"
            ) from exc
        source = textwrap.dedent(source)
        tree = ast.parse(source)
        func_def = tree.body[0]
        if not isinstance(func_def, ast.FunctionDef):
            raise TranslationError("Top-level node must be a function definition")
        func_def.decorator_list = []

        # Pre-pass: a name is mutable (needs `var`, not `let`) if it is the
        # target of an AugAssign, a loop variable, or assigned more than once.
        assign_counts: dict[str, int] = {}
        for node in ast.walk(func_def):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        assign_counts[t.id] = assign_counts.get(t.id, 0) + 1
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                self._mutable.add(node.target.id)
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                self._mutable.add(node.target.id)
        self._mutable |= {n for n, c in assign_counts.items() if c >= 2}

        self._translate_fn(func_def)
        return "\n".join(self._lines)

    # ------------------------------------------------------------------ #
    # Function-level                                                       #
    # ------------------------------------------------------------------ #

    def _translate_fn(self, node: ast.FunctionDef) -> None:
        bindings: list[tuple[int, int, str, StorageBufferType]] = []
        uniforms: list[tuple[int, int, str, UniformType]] = []
        builtins: list[tuple[str, BuiltinValue]] = []
        wg_arrays: list[tuple[str, WorkgroupArrayType]] = []

        binding_idx = 0
        for arg in node.args.args:
            name = arg.arg
            ann = self._annotations.get(name)
            if ann is None:
                raise TranslationError(f"Parameter '{name}' has no type annotation")
            if isinstance(ann, BuiltinValue):
                builtins.append((name, ann))
            elif isinstance(ann, StorageBufferType):
                bindings.append((0, binding_idx, name, ann))
                binding_idx += 1
            elif isinstance(ann, UniformType):
                uniforms.append((0, binding_idx, name, ann))
                binding_idx += 1
            elif isinstance(ann, WorkgroupArrayType):
                wg_arrays.append((name, ann))
            else:
                raise TranslationError(
                    f"Unsupported parameter type for '{name}': {ann!r}\n"
                    "Use StorageBuffer, Uniform, WorkgroupArray, or a Builtin value."
                )
            self._scopes[0].add(name)

        # f16 anywhere in the interface requires the enable directive
        if any("f16" in t.wgsl_name for _, _, _, t in bindings) or \
           any("f16" in t.wgsl_name for _, _, _, t in uniforms) or \
           any("f16" in t.wgsl_name for _, t in wg_arrays):
            self._emit("enable f16;")
            self._emit()

        # subgroup builtins / intrinsic calls require `enable subgroups;`
        uses_subgroups = any(
            bv.builtin_name.startswith("subgroup") for _, bv in builtins
        ) or any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in _SUBGROUP_FNS
            for n in ast.walk(node)
        )
        if uses_subgroups:
            self._emit("enable subgroups;")
            self._emit()

        # Emit @group/@binding declarations
        for group, idx, name, typ in bindings:
            access = "read" if typ.access == "read" else "read_write"
            self._emit(
                f"@group({group}) @binding({idx}) "
                f"var<storage, {access}> {name}: array<{typ.elem_type.wgsl_name}>;"
            )
        for group, idx, name, typ in uniforms:
            self._emit(
                f"@group({group}) @binding({idx}) "
                f"var<uniform> {name}: {typ.elem_type.wgsl_name};"
            )
        for name, typ in wg_arrays:
            self._emit(f"var<workgroup> {name}: {typ.wgsl_name};")
        if bindings or uniforms or wg_arrays:
            self._emit()

        # @compute entry point
        ws = ", ".join(str(s) for s in self._workgroup_size)
        fn_params = [
            f"@builtin({bv.builtin_name}) {pname}: {bv.wgsl_name}"
            for pname, bv in builtins
        ]
        self._emit(f"@compute @workgroup_size({ws})")
        self._emit(f"fn {node.name}({', '.join(fn_params)}) {{")
        self._indent += 1
        for stmt in node.body:
            self._stmt(stmt)
        self._indent -= 1
        self._emit("}")

    # ------------------------------------------------------------------ #
    # Statements                                                           #
    # ------------------------------------------------------------------ #

    def _stmt(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign):
            self._s_assign(node)
        elif isinstance(node, ast.AnnAssign):
            self._s_ann_assign(node)
        elif isinstance(node, ast.AugAssign):
            self._s_aug_assign(node)
        elif isinstance(node, ast.For):
            self._s_for(node)
        elif isinstance(node, ast.While):
            self._s_while(node)
        elif isinstance(node, ast.If):
            self._s_if(node)
        elif isinstance(node, ast.Return):
            self._s_return(node)
        elif isinstance(node, ast.Break):
            self._emit("break;")
        elif isinstance(node, ast.Continue):
            self._emit("continue;")
        elif isinstance(node, ast.Expr):
            self._emit(self._expr(node.value) + ";")
        elif isinstance(node, ast.Pass):
            pass
        else:
            raise TranslationError(f"Unsupported statement: {type(node).__name__}")

    def _body(self, stmts: list[ast.stmt], scope: set[str] | None = None) -> None:
        """Translate a nested block inside its own scope."""
        self._scopes.append(scope if scope is not None else set())
        self._ever_declared |= self._scopes[-1]
        self._indent += 1
        for s in stmts:
            self._stmt(s)
        self._indent -= 1
        self._scopes.pop()

    def _s_assign(self, node: ast.Assign) -> None:
        value = self._expr(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                if self._visible(name):
                    self._emit(f"{name} = {value};")
                elif name in self._ever_declared:
                    raise TranslationError(
                        f"Variable '{name}' was declared in a nested block and is "
                        "not visible here; declare it before the block with a type "
                        f"annotation (e.g. \"{name}: f32 = 0.0\")"
                    )
                else:
                    self._declare(name)
                    kw = "var" if name in self._mutable else "let"
                    self._emit(f"{kw} {name} = {value};")
            else:
                self._emit(f"{self._expr(target)} = {value};")

    def _s_ann_assign(self, node: ast.AnnAssign) -> None:
        target = self._expr(node.target)
        wgsl_type = self._ann_to_wgsl(node.annotation)
        if isinstance(node.target, ast.Name):
            if node.target.id in self._scopes[-1]:
                raise TranslationError(
                    f"Variable '{node.target.id}' is already declared in this scope"
                )
            self._declare(node.target.id)
        if node.value is not None:
            self._emit(f"var {target}: {wgsl_type} = {self._expr(node.value)};")
        else:
            self._emit(f"var {target}: {wgsl_type};")

    def _s_aug_assign(self, node: ast.AugAssign) -> None:
        if isinstance(node.op, ast.FloorDiv):
            raise TranslationError(_FLOORDIV_MSG)
        op_t = type(node.op)
        if op_t in _ARITH_OPS:
            sym = _ARITH_OPS[op_t][0]
        elif op_t in _BITWISE_OPS:
            sym = _BITWISE_OPS[op_t]
        else:
            raise TranslationError(f"Unsupported operator: {op_t.__name__}")
        self._emit(f"{self._expr(node.target)} {sym}= {self._expr(node.value)};")

    def _s_for(self, node: ast.For) -> None:
        if node.orelse:
            raise TranslationError("for-else is not supported")
        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            raise TranslationError("Only 'for x in range(...)' is supported")
        if node.iter.keywords:
            raise TranslationError("range() keyword arguments are not supported")
        if not isinstance(node.target, ast.Name):
            raise TranslationError("For loop target must be a simple name")

        var = node.target.id
        args = node.iter.args
        if len(args) == 1:
            start_s, start_val = "0", 0
            stop_node, step_node = args[0], None
        elif len(args) == 2:
            start_s, start_val = self._expr(args[0]), _const_int(args[0])
            stop_node, step_node = args[1], None
        elif len(args) == 3:
            start_s, start_val = self._expr(args[0]), _const_int(args[0])
            stop_node, step_node = args[1], args[2]
        else:
            raise TranslationError("range() takes 1–3 arguments")

        step_val = 1 if step_node is None else _const_int(step_node)
        if step_val == 0:
            raise TranslationError("range() step must not be zero")
        stop_val = _const_int(stop_node)

        # A descending or negative-domain loop cannot use u32.
        descending = step_val is not None and step_val < 0
        signed = (
            descending
            or (start_val is not None and start_val < 0)
            or (stop_val is not None and stop_val < 0)
        )
        loop_ty = "i32" if signed else "u32"
        cmp = ">" if descending else "<"

        if step_val == 1:
            incr = f"{var}++"
        elif descending:
            incr = f"{var} -= {-step_val}"
        else:
            incr = f"{var} += {self._expr(step_node)}"

        stop_s = self._expr(stop_node)
        self._emit(f"for (var {var}: {loop_ty} = {start_s}; {var} {cmp} {stop_s}; {incr}) {{")
        self._body(node.body, scope={var})
        self._emit("}")

    def _s_while(self, node: ast.While) -> None:
        if node.orelse:
            raise TranslationError("while-else is not supported")
        self._emit(f"while ({self._expr(node.test)}) {{")
        self._body(node.body)
        self._emit("}")

    def _s_if(self, node: ast.If) -> None:
        self._emit(f"if ({self._expr(node.test)}) {{")
        self._body(node.body)
        self._emit_else(node.orelse)

    def _emit_else(self, orelse: list[ast.stmt]) -> None:
        if not orelse:
            self._emit("}")
        elif len(orelse) == 1 and isinstance(orelse[0], ast.If):
            inner = orelse[0]
            self._emit(f"}} else if ({self._expr(inner.test)}) {{")
            self._body(inner.body)
            self._emit_else(inner.orelse)
        else:
            self._emit("} else {")
            self._body(orelse)
            self._emit("}")

    def _s_return(self, node: ast.Return) -> None:
        if node.value:
            self._emit(f"return {self._expr(node.value)};")
        else:
            self._emit("return;")

    # ------------------------------------------------------------------ #
    # Expressions                                                          #
    # ------------------------------------------------------------------ #

    def _expr(self, node: ast.expr, prec: int = 0) -> str:
        """Render an expression, parenthesising only when required.

        ``prec`` is the minimum precedence the surrounding context demands;
        the rendered node wraps itself in parentheses when it binds looser.
        """
        if isinstance(node, ast.Name):
            if node.id in self._ever_declared and not self._visible(node.id):
                raise TranslationError(
                    f"Variable '{node.id}' is used outside the block where it was "
                    "declared; declare it before the block with a type annotation "
                    f"(e.g. \"{node.id}: f32 = 0.0\")"
                )
            return node.id
        if isinstance(node, ast.Constant):
            return self._literal(node.value)
        if isinstance(node, ast.Attribute):
            return f"{self._expr(node.value, _P_ATOM)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            return f"{self._expr(node.value, _P_ATOM)}[{self._expr(node.slice)}]"
        if isinstance(node, ast.BinOp):
            return self._binop(node, prec)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                s = f"-{self._expr(node.operand, _P_UNARY)}"
                return f"({s})" if _P_UNARY < prec else s
            if isinstance(node.op, ast.UAdd):
                return self._expr(node.operand, prec)
            if isinstance(node.op, ast.Not):
                return f"!({self._expr(node.operand)})"
            raise TranslationError(f"Unsupported unary op: {type(node.op).__name__}")
        if isinstance(node, ast.Compare):
            return self._compare(node, prec)
        if isinstance(node, ast.BoolOp):
            return self._boolop(node, prec)
        if isinstance(node, ast.Call):
            fn = self._expr(node.func, _P_ATOM)
            fn = _INTRINSIC_RENAMES.get(fn, fn)
            args = ", ".join(self._expr(a) for a in node.args)
            return f"{fn}({args})"
        if isinstance(node, ast.IfExp):
            # Python ternary → WGSL select(false_val, true_val, cond)
            return (
                f"select({self._expr(node.orelse)}, "
                f"{self._expr(node.body)}, "
                f"{self._expr(node.test)})"
            )
        raise TranslationError(
            f"Unsupported expression: {type(node).__name__}: {ast.dump(node)}"
        )

    def _binop(self, node: ast.BinOp, prec: int) -> str:
        if isinstance(node.op, ast.FloorDiv):
            raise TranslationError(_FLOORDIV_MSG)
        op_t = type(node.op)
        if op_t in _BITWISE_OPS:
            # WGSL forbids mixing bitwise/shift with other operators without
            # parentheses, so these are always self-parenthesised.
            sym = _BITWISE_OPS[op_t]
            return f"({self._expr(node.left)} {sym} {self._expr(node.right)})"
        if op_t not in _ARITH_OPS:
            raise TranslationError(f"Unsupported operator: {op_t.__name__}")
        sym, myprec = _ARITH_OPS[op_t]
        left = self._expr(node.left, myprec)
        right = self._expr(node.right, myprec + 1)
        s = f"{left} {sym} {right}"
        return f"({s})" if myprec < prec else s

    def _compare(self, node: ast.Compare, prec: int) -> str:
        if len(node.ops) == 1:
            op = _CMP_MAP.get(type(node.ops[0]))
            if op is None:
                raise TranslationError(f"Unsupported comparison: {type(node.ops[0]).__name__}")
            s = f"{self._expr(node.left, _P_CMP + 1)} {op} {self._expr(node.comparators[0], _P_CMP + 1)}"
            return f"({s})" if _P_CMP < prec else s
        # Chained comparison: a < b < c → (a < b) && (b < c)
        parts = []
        prev = self._expr(node.left, _P_CMP + 1)
        for op_node, comp in zip(node.ops, node.comparators):
            op = _CMP_MAP.get(type(op_node))
            if op is None:
                raise TranslationError(f"Unsupported comparison: {type(op_node).__name__}")
            cur = self._expr(comp, _P_CMP + 1)
            parts.append(f"({prev} {op} {cur})")
            prev = cur
        s = " && ".join(parts)
        return f"({s})" if prec > 0 else s

    def _boolop(self, node: ast.BoolOp, prec: int) -> str:
        myprec = _P_AND if isinstance(node.op, ast.And) else _P_OR
        sym = "&&" if isinstance(node.op, ast.And) else "||"
        rendered = []
        for v in node.values:
            if isinstance(v, ast.BoolOp):
                # WGSL forbids mixing && and || without parentheses.
                rendered.append(f"({self._expr(v)})")
            else:
                rendered.append(self._expr(v, myprec + 1))
        s = f" {sym} ".join(rendered)
        return f"({s})" if myprec < prec else s

    def _literal(self, value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            s = str(value)
            if "." not in s and "e" not in s.lower():
                s += ".0"
            return s
        raise TranslationError(f"Unsupported literal: {value!r}")

    def _ann_to_wgsl(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return "bool" if node.id == "bool_" else node.id
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in _VEC_NAMES:
                return f"{node.value.id}<{self._ann_to_wgsl(node.slice)}>"
            raise TranslationError(f"Unsupported generic annotation: {ast.dump(node)}")
        if isinstance(node, ast.Attribute):
            raise TranslationError(
                "Attribute-style annotations are not supported in the function body"
            )
        raise TranslationError(f"Unsupported annotation: {ast.dump(node)}")


def translate(func: Callable, workgroup_size: tuple[int, ...] = (1,)) -> str:
    """Translate a Python function to a WGSL compute shader string.

    Parameters
    ----------
    func:
        Python function whose parameters are annotated with WGSL types from
        ``py_shader_lang_wgpu.types``.
    workgroup_size:
        Workgroup dimensions, e.g. ``(8, 8)`` for a 2-D 8×8 workgroup.

    Returns
    -------
    str
        Complete WGSL source for the compute shader (bindings + entry point).
    """
    if not 1 <= len(workgroup_size) <= 3:
        raise TranslationError(
            f"workgroup_size must have 1–3 dimensions, got {len(workgroup_size)}"
        )
    for s in workgroup_size:
        if not isinstance(s, int) or isinstance(s, bool) or s < 1:
            raise TranslationError(
                f"workgroup_size entries must be positive integers, got {s!r}"
            )
    try:
        annotations = {
            k: v
            for k, v in typing.get_type_hints(func).items()
            if k != "return"
        }
    except Exception:
        annotations = {
            k: v
            for k, v in func.__annotations__.items()
            if k != "return"
        }
    return _WGSLTranslator(func, annotations, workgroup_size).run()


def kernel(
    func: Callable | None = None,
    *,
    workgroup_size: tuple[int, ...] = (1,),
) -> Callable:
    """Decorator that translates a Python kernel function to WGSL.

    Can be used with or without arguments::

        @kernel(workgroup_size=(8, 8))
        def my_shader(global_id: Builtin.global_invocation_id, ...):
            ...

        @kernel
        def my_shader(global_id: Builtin.global_invocation_id, ...):
            ...

    The original function is returned unchanged; the WGSL string is attached
    as ``func.wgsl``.
    """
    def _wrap(f: Callable) -> Callable:
        f.wgsl = translate(f, workgroup_size)
        return f

    if func is not None:
        return _wrap(func)
    return _wrap


__all__ = ["translate", "kernel", "TranslationError"]
