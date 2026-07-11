"""py_shader_lang_wgpu — translate Python kernel functions to WGSL compute shaders."""

from .types import (
    u32, f32, f16, i32, bool_,
    vec2, vec3, vec4,
    StorageBuffer, Uniform, Builtin, WorkgroupArray,
    WGSLType, StorageBufferType, UniformType, BuiltinValue, WorkgroupArrayType,
)
from .translator import translate, kernel, device_fn, TranslationError

__all__ = [
    # Type primitives
    "u32", "f32", "f16", "i32", "bool_",
    "vec2", "vec3", "vec4",
    "StorageBuffer", "Uniform", "Builtin", "WorkgroupArray",
    # Type classes (for isinstance checks / custom types)
    "WGSLType", "StorageBufferType", "UniformType", "BuiltinValue",
    "WorkgroupArrayType",
    # Core API
    "translate", "kernel", "device_fn", "TranslationError",
]
