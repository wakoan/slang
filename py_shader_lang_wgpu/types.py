"""WGSL type objects for use as Python annotations in kernel functions."""

from __future__ import annotations


class WGSLType:
    def __init__(self, name: str) -> None:
        self.wgsl_name = name

    def __repr__(self) -> str:
        return self.wgsl_name


class ScalarType(WGSLType):
    pass


# Scalar primitives
u32 = ScalarType("u32")
f32 = ScalarType("f32")
f16 = ScalarType("f16")  # requires the shader-f16 device feature
i32 = ScalarType("i32")
bool_ = ScalarType("bool")


class VecTypeFactory:
    """Factory for WGSL vector types. Usage: vec3[u32], vec4[f32], etc."""

    def __init__(self, size: int) -> None:
        self._size = size

    def __getitem__(self, elem_type: WGSLType) -> WGSLType:
        return WGSLType(f"vec{self._size}<{elem_type.wgsl_name}>")

    def __repr__(self) -> str:
        return f"vec{self._size}"


vec2 = VecTypeFactory(2)
vec3 = VecTypeFactory(3)
vec4 = VecTypeFactory(4)


class StorageBufferType(WGSLType):
    def __init__(self, elem_type: WGSLType, access: str = "read_write") -> None:
        self.elem_type = elem_type
        self.access = access  # "read" or "read_write"
        super().__init__(f"array<{elem_type.wgsl_name}>")

    def __repr__(self) -> str:
        return f"StorageBuffer[{self.elem_type!r}, {self.access!r}]"


class _StorageBufferFactory:
    """Usage: StorageBuffer[f32, "read"] or StorageBuffer[f32] (defaults to read_write)."""

    def __getitem__(self, args: object) -> StorageBufferType:
        if isinstance(args, tuple):
            elem_type, access = args
        else:
            elem_type, access = args, "read_write"
        return StorageBufferType(elem_type, access)

    def __repr__(self) -> str:
        return "StorageBuffer"


StorageBuffer = _StorageBufferFactory()


class UniformType(WGSLType):
    def __init__(self, elem_type: WGSLType) -> None:
        self.elem_type = elem_type
        super().__init__(elem_type.wgsl_name)

    def __repr__(self) -> str:
        return f"Uniform[{self.elem_type!r}]"


class _UniformFactory:
    """Usage: Uniform[u32], Uniform[f32], etc."""

    def __getitem__(self, elem_type: WGSLType) -> UniformType:
        return UniformType(elem_type)

    def __repr__(self) -> str:
        return "Uniform"


Uniform = _UniformFactory()


class WorkgroupArrayType(WGSLType):
    """A fixed-size array in workgroup (shared) memory."""

    def __init__(self, elem_type: WGSLType, size: int) -> None:
        self.elem_type = elem_type
        self.size = size
        super().__init__(f"array<{elem_type.wgsl_name}, {size}>")

    def __repr__(self) -> str:
        return f"WorkgroupArray[{self.elem_type!r}, {self.size}]"


class _WorkgroupArrayFactory:
    """Usage: WorkgroupArray[f32, 64] — shared memory across the workgroup."""

    def __getitem__(self, args: tuple) -> WorkgroupArrayType:
        elem_type, size = args
        if not isinstance(size, int) or size < 1:
            raise TypeError("WorkgroupArray size must be a positive int literal")
        return WorkgroupArrayType(elem_type, size)

    def __repr__(self) -> str:
        return "WorkgroupArray"


WorkgroupArray = _WorkgroupArrayFactory()


class BuiltinValue(WGSLType):
    """A WGSL @builtin parameter value."""

    def __init__(self, builtin_name: str, wgsl_type_name: str) -> None:
        self.builtin_name = builtin_name
        super().__init__(wgsl_type_name)

    def __repr__(self) -> str:
        return f"Builtin.{self.builtin_name}"


class _BuiltinNamespace:
    """Namespace of @builtin values. Use as parameter type annotations."""

    global_invocation_id = BuiltinValue("global_invocation_id", "vec3<u32>")
    local_invocation_id = BuiltinValue("local_invocation_id", "vec3<u32>")
    workgroup_id = BuiltinValue("workgroup_id", "vec3<u32>")
    num_workgroups = BuiltinValue("num_workgroups", "vec3<u32>")
    local_invocation_index = BuiltinValue("local_invocation_index", "u32")
    # require the "subgroups" device feature
    subgroup_invocation_id = BuiltinValue("subgroup_invocation_id", "u32")
    subgroup_size = BuiltinValue("subgroup_size", "u32")


Builtin = _BuiltinNamespace()
