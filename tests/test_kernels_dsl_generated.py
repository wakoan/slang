"""The generated artifacts must not drift from kernels_dsl.py.

Swift and Rust cannot call the Python DSL at runtime, so they read the .metal
files `gen_msl.py` writes. Nothing else notices if those go stale: every Python
test would still pass while two backends quietly ran an older kernel. This is
the test that makes that impossible.

The WGSL side is checked the same way — every kernel must actually compile, not
just translate. Emitting plausible text is not the bar; naga is.
"""
import numpy as np
import pytest

from py_shader_lang_wgpu import translate
from gemma4_150 import gen_msl, kernels_dsl


def test_generated_msl_is_current():
    """`python -m gemma4_150.gen_msl` would be a no-op."""
    stale = []
    for name in sorted(gen_msl.SPECS):
        path = gen_msl.KDIR / f"{name}.metal"
        if not path.exists():
            stale.append(f"{name} (missing)")
        elif path.read_text() != gen_msl.render(name):
            stale.append(name)
    assert not stale, (
        "kernels_gen/ is out of date with kernels_dsl.py — Swift and Rust would "
        f"run stale kernels. Run `python -m gemma4_150.gen_msl`. Stale: {stale}")


def test_every_ported_kernel_is_generated():
    """A kernel added to kernels_dsl.py must be added to gen_msl.SPECS too.

    Otherwise it exists for the Python backends and silently does not for
    Swift/Rust. The two deliberate exceptions are named, not skipped by accident.
    """
    ported = {n for n in dir(kernels_dsl) if hasattr(getattr(kernels_dsl, n), "msl")}
    # gemm_tiled is the portable GEMM: used directly from Python (prefill_gpu),
    # never string-patched from a file, so it has no .metal artifact.
    exempt = {"gemm_tiled"}
    missing = ported - set(gen_msl.SPECS) - exempt
    assert not missing, f"ported but not generated for Swift/Rust: {sorted(missing)}"


# Shape variants each kernel is compiled at, so the WGSL check covers the
# specializations the runners actually use rather than only the defaults.
_WGSL_CASES = {
    "kvnorm": [((hd, 1, 1), {"HD": hd, "HALF": hd // 2}) for hd in (256, 512)],
    "kvnorm_b": [((hd, 1, 1), {"HD": hd, "HALF": hd // 2}) for hd in (256, 512)],
    "qprep_b": [((hd, 1, 1), {"HEAD_DIM": hd, "HALF_DIM": hd // 2}) for hd in (256, 512)],
    "oproj_73": [((256, 1, 1), {"WPR": w}) for w in (256, 512)],
    "qkv_70": [((32, 1, 1), {"Q_OUT": qd, "KV_OUT": hd, "Q_WGS": qd // 2,
                             "KV_WGS": hd // 2, "TOTAL_WGS": qd // 2 + hd,
                             "GRID_X": qd // 2 + hd})
               for qd, hd in ((2048, 256), (4096, 512))],
    "attn_101": [((256, 1, 1), {"HEAD_DIM": hd, "HALF_DIM": hd // 2, "HD4": hd // 4,
                                "J_GROUPS": 256 // (hd // 4),
                                "PP_COUNTER_BASE": 8 * 32 * (hd + 2),
                                "OUT_Q": 0.0148868})
                 for hd in (256, 512)],
    "attn_prefill": [((256, 1, 1), {"HEAD_DIM": hd, "HALF_DIM": hd // 2,
                                    "HD4": hd // 4, "J_GROUPS": 256 // (hd // 4),
                                    "OUT_Q": 0.0148868})
                     for hd in (256, 512)],
}


def _cases(name):
    if name in _WGSL_CASES:
        return _WGSL_CASES[name]
    fn = getattr(kernels_dsl, name)
    return [(getattr(fn, "workgroup_size", None), getattr(fn, "consts", None) or None)]


@pytest.fixture(scope="module")
def wgpu_device():
    wgpu = pytest.importorskip("wgpu")
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    # NB the feature is "subgroup", singular. Requesting "subgroups" silently
    # yields a device without it, and then every subgroupAdd kernel "fails" —
    # which cost a debugging cycle when this was first checked by hand.
    feats = [f for f in ("shader-f16", "subgroup", "subgroup-barrier")
             if f in adapter.features]
    return adapter.request_device_sync(required_features=feats)


@pytest.mark.parametrize("name", sorted(
    n for n in dir(kernels_dsl) if hasattr(getattr(kernels_dsl, n), "wgsl")))
def test_wgsl_compiles(wgpu_device, name):
    """Every kernel, at every shape the runners specialize it to."""
    for ws, consts in _cases(name):
        src = translate(getattr(kernels_dsl, name), workgroup_size=ws, consts=consts)
        # naga does not accept the directive yet, though it supports the ops;
        # the runners strip it the same way.
        wgpu_device.create_shader_module(code=src.replace("enable subgroups;\n", ""))


def test_wgsl_uses_workgroup_uniform_load_for_last_arriver():
    """The last-arriver kernels must gate on workgroupUniformLoad.

    A plain read leaves the branch non-uniform, which naga accepts and Dawn
    rejects ("'workgroupBarrier' must only be called from uniform control
    flow"). That combination means a regression here is invisible in wgpu-py and
    only breaks the browser, so it is asserted rather than left to chance.
    """
    for name in ("oproj_73", "down_75", "down_96", "pleproj_77", "attn_101"):
        w = translate(getattr(kernels_dsl, name))
        assert "workgroupUniformLoad(&lastFlag" in w, f"{name} reads lastFlag directly"
