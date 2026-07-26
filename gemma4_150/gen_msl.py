"""Generate kernels_msl/*.metal from kernels_dsl.py.

    python -m gemma4_150.gen_msl [--check]

The Swift and Rust backends cannot call the Python DSL at runtime, so they read
`.metal` files from disk. This makes those files BUILD ARTIFACTS rather than a
second hand-maintained source: the Python kernels stay authoritative and the
.metal is regenerated from them.

Shape constants are emitted as `constant uint NAME=512u;` — no spaces around
the `=` — because both backends specialize by substring replace and already
look for exactly that spelling. The values below are the defaults those
replaces expect to find.

`--check` regenerates into memory and reports which files would change, for CI
or a pre-commit hook; it writes nothing.
"""
import sys
from pathlib import Path

from py_shader_lang_wgpu import translate
from gemma4_150 import kernels_dsl

# Generated output goes to its OWN directory. Writing into kernels_msl/ would
# overwrite the hand-written kernels that every parity test and both gates use
# as their independent reference — which is exactly what happened the first
# time, and it silently turned the verification into a self-comparison.
KDIR = Path(__file__).resolve().parent / "kernels_gen"

# name -> (workgroup size, default consts). The consts are the values the
# hand-written kernels declared, so the backends' existing replace-pairs match.
SPECS = {
    "embed_00": ((64, 1, 1), {}),
    "plegather_01": ((64, 1, 1), {}),
    "proj_68": ((32, 1, 1), {}),
    "rmssrq_69": ((256, 1, 1), {}),
    "qkv_70": ((32, 1, 1), {"Q_OUT": 2048, "KV_OUT": 256, "Q_WGS": 1024,
                            "KV_WGS": 128, "TOTAL_WGS": 1280, "GRID_X": 1280}),
    "oproj_73": ((256, 1, 1), {"WPR": 256}),
    "gateup_74": ((64, 1, 1), {}),
    "down_75": ((256, 1, 1), {}),
    "plegate_76": ((32, 1, 1), {}),
    "pleproj_77": ((256, 1, 1), {}),
    "gateup_95": ((64, 1, 1), {}),
    "down_96": ((256, 1, 1), {}),
    "attn_101": ((256, 1, 1), {"HEAD_DIM": 512, "HALF_DIM": 256, "HD4": 128,
                               "J_GROUPS": 2, "PP_COUNTER_BASE": 8 * 32 * 514,
                               "OUT_Q": 0.014886821620166302}),
    "logits_33": ((128, 1, 1), {}),
    "argmax1_34": ((256, 1, 1), {}),
    "argmax2_35": ((256, 1, 1), {}),
    "kvnorm": ((256, 1, 1), {"HD": 256, "HALF": 128}),
    "combine": ((256, 1, 1), {}),
    # prefill
    "srqh_b": ((256, 1, 1), {}),
    "srq_b": ((256, 1, 1), {}),
    "geglu_b": ((256, 1, 1), {}),
    "rmsadd_b": ((256, 1, 1), {}),
    "rmssrqh_b": ((256, 1, 1), {}),
    "combine_b": ((256, 1, 1), {}),
    "attnout_b": ((256, 1, 1), {}),
    "plegate_b": ((256, 1, 1), {}),
    "smax_b": ((256, 1, 1), {}),
    "kvnorm_b": ((256, 1, 1), {"HD": 256, "HALF": 128}),
    "qprep_b": ((512, 1, 1), {"HEAD_DIM": 512, "HALF_DIM": 256}),
    "attn_prefill": ((256, 1, 1), {"HEAD_DIM": 512, "HALF_DIM": 256, "HD4": 128,
                                   "J_GROUPS": 2,
                                   "OUT_Q": 0.014886821620166302}),
}

BANNER = ("// GENERATED from gemma4_150/kernels_dsl.py by "
          "`python -m gemma4_150.gen_msl` — do not edit.\n")


def render(name):
    ws, consts = SPECS[name]
    src = translate(getattr(kernels_dsl, name), workgroup_size=ws, target="msl",
                    consts=consts or None, consts_as_decls=bool(consts))
    return BANNER + src


def main():
    KDIR.mkdir(exist_ok=True)
    check = "--check" in sys.argv
    changed = []
    for name in sorted(SPECS):
        path = KDIR / f"{name}.metal"
        new = render(name)
        old = path.read_text() if path.exists() else None
        if old == new:
            continue
        changed.append(name)
        if not check:
            path.write_text(new)
    verb = "would change" if check else "wrote"
    print(f"{verb} {len(changed)} of {len(SPECS)} kernels"
          + (f": {', '.join(changed)}" if changed else ""))
    # The two remaining .metal files are NOT generated: down_96_b and
    # gateup_95_b belong to the falsified token-batching benchmark and have no
    # DSL counterpart on purpose (see PORT_NOTES.md).
    return 1 if (check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
