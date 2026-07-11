"""Metal Shading Language (MSL) backend for the kernel translator.

Emits an MSL compute kernel equivalent to the WGSL output:
- StorageBuffer  → `device [const] T*` with [[buffer(n)]]
- Uniform        → `constant T&` with [[buffer(n)]]
- Builtin        → Metal thread-position attributes
- WorkgroupArray → `threadgroup T name[N];` declared in the kernel body
- barrier()          → threadgroup_barrier(mem_flags::mem_threadgroup)
- subgroup_barrier() → simdgroup_barrier(mem_flags::mem_device)
- subgroupAdd/Max/…  → simd_sum/simd_max/…
- unpack2x16float(x) → float2(as_type<half2>(x))
- f32()/u32()/i32()/f16() casts → float()/uint()/int()/half()

The workgroup_size is dispatch-time state in Metal; it is recorded as a
comment header for the host code.
"""

from __future__ import annotations

import ast

from .translator import _WGSLTranslator, TranslationError, _P_ATOM
from .types import StorageBufferType, UniformType, BuiltinValue, WorkgroupArrayType

_MSL_SCALARS = {
    "u32": "uint",
    "f32": "float",
    "f16": "half",
    "i32": "int",
    "bool": "bool",
    "bool_": "bool",
}

_MSL_BUILTIN_ATTRS = {
    "global_invocation_id": ("uint3", "thread_position_in_grid"),
    "local_invocation_id": ("uint3", "thread_position_in_threadgroup"),
    "workgroup_id": ("uint3", "threadgroup_position_in_grid"),
    "num_workgroups": ("uint3", "threadgroups_per_grid"),
    "local_invocation_index": ("uint", "thread_index_in_threadgroup"),
    "subgroup_invocation_id": ("uint", "thread_index_in_simdgroup"),
    "subgroup_size": ("uint", "threads_per_simdgroup"),
}

_MSL_INTRINSICS = {
    "f32": "float",
    "u32": "uint",
    "i32": "int",
    "f16": "half",
    "subgroupAdd": "simd_sum",
    "subgroupMax": "simd_max",
    "subgroupMin": "simd_min",
    "subgroupMul": "simd_product",
    "subgroupBroadcastFirst": "simd_broadcast_first",
    "subgroupShuffle": "simd_shuffle",
}


# MSL/C++ words that cannot be used as identifiers; renamed with a suffix
_MSL_RESERVED = frozenset("""
half float double int uint short ushort char uchar long ulong bool void
constant device threadgroup thread kernel vertex fragment template class
struct union enum namespace using new delete this auto const static signed
unsigned sampler texture atomic operator public private protected virtual
float2 float3 float4 half2 half3 half4 uint2 uint3 uint4 int2 int3 int4
bool2 bool3 bool4 short2 short3 short4 ushort2 ushort3 ushort4
""".split())


def _msl_type(wgsl_name: str) -> str:
    """Map a WGSL type spelling to MSL (scalars and vecN<T>)."""
    if wgsl_name in _MSL_SCALARS:
        return _MSL_SCALARS[wgsl_name]
    if wgsl_name.startswith("vec") and "<" in wgsl_name:
        n = wgsl_name[3]
        inner = wgsl_name[wgsl_name.index("<") + 1 : wgsl_name.rindex(">")]
        return f"{_MSL_SCALARS.get(inner, inner)}{n}"
    raise TranslationError(f"No MSL mapping for type {wgsl_name!r}")


class _MSLTranslator(_WGSLTranslator):
    def _ident(self, name: str) -> str:
        return name + "_" if name in _MSL_RESERVED else name

    # ------------------------------------------------------------------ #
    # Function-level                                                       #
    # ------------------------------------------------------------------ #

    def _translate_fn(self, node: ast.FunctionDef) -> None:
        bindings, uniforms, builtins, wg_arrays = self._classify_params(node)

        self._emit("#include <metal_stdlib>")
        self._emit("using namespace metal;")
        self._emit()
        self._emit_device_fns()
        ws = ", ".join(str(s) for s in self._workgroup_size)
        self._emit(f"// dispatch with threadsPerThreadgroup = ({ws})")

        params: list[str] = []
        for group, idx, name, typ in bindings:
            const = "const " if typ.access == "read" else ""
            elem = _msl_type(typ.elem_type.wgsl_name)
            params.append(f"device {const}{elem}* {self._ident(name)} [[buffer({idx})]]")
        for group, idx, name, typ in uniforms:
            elem = _msl_type(typ.elem_type.wgsl_name)
            params.append(f"constant {elem}& {self._ident(name)} [[buffer({idx})]]")
        for pname, bv in builtins:
            ty, attr = _MSL_BUILTIN_ATTRS[bv.builtin_name]
            params.append(f"{ty} {self._ident(pname)} [[{attr}]]")

        self._emit(f"kernel void {node.name}(")
        self._indent += 2
        for i, p in enumerate(params):
            self._emit(p + ("," if i < len(params) - 1 else ""))
        self._indent -= 2
        self._emit(") {")
        self._indent += 1
        for name, typ in wg_arrays:
            elem = _msl_type(typ.elem_type.wgsl_name)
            self._emit(f"threadgroup {elem} {self._ident(name)}[{typ.size}];")
        for stmt in node.body:
            self._stmt(stmt)
        self._indent -= 1
        self._emit("}")

    def _translate_device_fn(self, node):
        params = self._device_params(node)
        sig = ", ".join(f"{_msl_type(t.wgsl_name)} {self._ident(n)}"
                        for n, t in params)
        ret = _msl_type(self._return_type.wgsl_name) if self._return_type else "void"
        self._emit(f"{ret} {self._ident(node.name)}({sig}) {{")
        self._indent += 1
        for stmt in node.body:
            self._stmt(stmt)
        self._indent -= 1
        self._emit("}")

    # ------------------------------------------------------------------ #
    # Backend hooks                                                        #
    # ------------------------------------------------------------------ #

    def _decl_infer(self, name: str, value: str, mutable: bool) -> str:
        kw = "auto" if mutable else "const auto"
        return f"{kw} {name} = {value};"

    def _decl_typed(self, target: str, type_str: str, value: str | None) -> str:
        if value is None:
            return f"{type_str} {target};"
        return f"{type_str} {target} = {value};"

    def _for_header(self, var: str, loop_ty: str, start: str, cmp: str,
                    stop: str, incr: str) -> str:
        ty = _MSL_SCALARS[loop_ty]
        return f"for ({ty} {var} = {start}; {var} {cmp} {stop}; {incr}) {{"

    def _call_special(self, fn: str, args: list[str]) -> str | None:
        joined = ", ".join(args)
        if fn in ("barrier", "workgroupBarrier"):
            return "threadgroup_barrier(mem_flags::mem_threadgroup)"
        if fn in ("subgroup_barrier", "subgroupBarrier"):
            return "simdgroup_barrier(mem_flags::mem_device)"
        if fn == "unpack2x16float":
            return f"float2(as_type<half2>({joined}))"
        if fn in _MSL_INTRINSICS:
            return f"{_MSL_INTRINSICS[fn]}({joined})"
        return None

    def _ann_to_wgsl(self, node: ast.expr) -> str:
        # body annotations (x: f32, v: vec3[u32]) → MSL spellings
        if isinstance(node, ast.Name):
            if node.id in _MSL_SCALARS:
                return _MSL_SCALARS[node.id]
            raise TranslationError(f"Unsupported annotation: {node.id}")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id in ("vec2", "vec3", "vec4"):
            inner = self._ann_to_wgsl(node.slice)
            return f"{inner}{node.value.id[3]}"
        raise TranslationError(f"Unsupported annotation: {ast.dump(node)}")
