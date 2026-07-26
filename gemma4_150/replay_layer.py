"""Replay one kernel per layer on the REAL inputs it sees during decode.

    python -m gemma4_150.replay_layer [kernel]        # default: oproj_73

Synthetic parity tests compare a kernel against its reference on random data.
That was not enough for oproj_73: it is bit-exact on random inputs and on a
single real input, yet swapping it changes generated tokens (see PORT_NOTES.md).
The gap is that a kernel feeding a quantizer fails on inputs that sit exactly on
a `round()` boundary — which real activations do constantly, because they are
already on a quantization grid, and random normals essentially never do.

So this captures each layer's actual input buffers mid-decode, then replays the
stock and DSL kernels on them layer by layer and reports the FIRST layer that
differs. That turns "somewhere in 35 layers x N tokens" into one concrete case.

Capture works by flushing the command buffer immediately before the kernel's
dispatch, so the input buffers hold their final values, then resuming — the
forward is split around each dispatch rather than reordered.
"""
import sys

import numpy as np

from py_shader_lang_wgpu import translate
from gemma4_150 import kernels_dsl
from gemma4_150.metal_runner import G4, TOKJSON, Batch, _SHARED

# kernel -> (runner key prefix, input buffers to capture, output buffers to diff)
SPECS = {
    "pleproj_77": {
        "prefix": "pleproj_77",
        "inputs": [("gate", 256, np.float32), ("hidden", 1536, np.float32),
                   ("pp77", 1537, np.uint32)],
        "outputs": [("pp77", 1536, np.float32), ("hidden", 1536, np.float32),
                    ("y2n", 1536, np.float32), ("sum2n", 1, np.float32)],
        "consts": lambda key: None,
        "threads": 256,
    },
    "oproj_73": {
        "prefix": "oproj_",
        "inputs": [("attn", 4096, np.float32), ("hidden", 1536, np.float32),
                   ("pp73", 1537, np.uint32)],
        "outputs": [("pp73", 1536, np.float32), ("hidden", 1536, np.float32),
                    ("y2", 1536, np.float16), ("sum2", 1, np.float32)],
        "consts": lambda key: {"WPR": int(key.split("_")[1]) // 8},
        "threads": 256,
    },
}


def _read(g, name, count, dtype):
    n = count * np.dtype(dtype).itemsize
    return np.frombuffer(g.pool[name].contents().as_buffer(n), dtype).copy()


def _write(g, name, arr):
    b = g.pool[name]
    b.contents().as_buffer(arr.nbytes)[:] = arr.tobytes()


def capture(g, spec, ids):
    """Run one decode step, snapshotting inputs before every matching dispatch."""
    shots = []
    orig = Batch.dg

    def dg(self, name, bufs, groups):
        if name.startswith(spec["prefix"]):
            if self.enc is not None:
                self.enc.endEncoding()
                self.enc = None
            self.cb.commit()
            self.cb.waitUntilCompleted()
            shots.append((name, list(bufs), groups,
                          {n: _read(g, n, c, d) for n, c, d in spec["inputs"]}))
            self.cb = self.r.q.commandBuffer()
        orig(self, name, bufs, groups)

    Batch.dg = dg
    try:
        pos = g.prefill(ids[:-1], 0)
        g.set_cur(ids[-1])
        g._forward(pos, 0, True, True, True)
    finally:
        Batch.dg = orig
    return shots


_RAW_DG = Batch.dg          # bound before any patching, so replay never recurses


def replay(g, spec, shot):
    """Dispatch the currently-registered kernel on a captured input set."""
    name, bufs, groups, inputs = shot
    for n, arr in inputs.items():
        _write(g, n, arr)
    b = Batch(g)
    _RAW_DG(b, name, bufs, groups)
    b.commit_wait()
    return {n: _read(g, n, c, d) for n, c, d in spec["outputs"]}


def sweep(g, spec, which, ids, n_tokens):
    """Compare stock vs DSL at EVERY dispatch across n_tokens of real decode.

    Token 0's inputs were not enough — the kernel matched on all 35 layers there.
    Divergence appears only after the trajectory has wandered, so the comparison
    has to ride along with a real multi-token decode rather than replay one step.
    """
    fn = getattr(kernels_dsl, which)
    stock = {k: v for k, v in g.kernels.items() if k.startswith(spec["prefix"])}
    dsl = {}
    for key in stock:
        g._compile_one(translate(fn, workgroup_size=(spec["threads"], 1, 1),
                                 target="msl", consts=spec["consts"](key)), which, key)
        dsl[key] = g.kernels[key]
        g.kernels[key] = stock[key]

    state = {"tok": 0, "layer": 0, "bad": None, "checked": 0}
    orig = Batch.dg

    def dg(self, name, bufs, groups):
        if name.startswith(spec["prefix"]) and state["bad"] is None:
            if self.enc is not None:
                self.enc.endEncoding(); self.enc = None
            self.cb.commit(); self.cb.waitUntilCompleted()
            saved = {n: _read(g, n, c, d) for n, c, d in spec["inputs"]}
            shot = (name, list(bufs), groups, saved)
            g.kernels[name] = stock[name]
            ref = replay(g, spec, shot)
            g.kernels[name] = dsl[name]
            got = replay(g, spec, shot)
            g.kernels[name] = stock[name]
            state["checked"] += 1
            if any(not np.array_equal(ref[n], got[n]) for n in ref):
                state["bad"] = (state["tok"], state["layer"], name, ref, got, saved)
            for n, arr in saved.items():
                _write(g, n, arr)
            self.cb = self.r.q.commandBuffer()
            state["layer"] += 1
        orig(self, name, bufs, groups)

    Batch.dg = dg
    try:
        pos = g.prefill(ids[:-1], 0)
        g.set_cur(ids[-1])
        for t in range(n_tokens):
            state["tok"], state["layer"] = t, 0
            g._forward(pos + t, t, True, True, True)
            if state["bad"] is not None:
                break
    finally:
        Batch.dg = orig
    return state


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "oproj_73"
    spec = SPECS[which]
    print(f"loading model… (replaying {which})")
    g = G4()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKJSON)
    ids = [g.bos] + tok.encode(
        "<|turn>user\nWrite a 200-word essay about the sea.<turn|>\n<|turn>model\n",
        add_special_tokens=False).ids

    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if n_tokens > 1:
        st = sweep(g, spec, which, ids, n_tokens)
        print(f"compared {st['checked']} dispatches over {n_tokens} tokens")
        if st["bad"] is None:
            print("no divergence found")
            return 0
        tok, layer, key, ref, got, saved = st["bad"]
        print(f"\nFIRST DIVERGENCE: token {tok}, layer {layer} ({key})")
        for n in ref:
            d = int(np.count_nonzero(ref[n] != got[n]))
            if d:
                a, b = ref[n].astype(np.float64), got[n].astype(np.float64)
                k = int(np.argmax(np.abs(a - b)))
                print(f"  {n:8} {d:5d}/{len(a)} differ  max|d|={np.abs(a-b).max():.6e}"
                      f"  at [{k}] ref={a[k]!r} dsl={b[k]!r}")
        np.savez("/tmp/oproj_divergence.npz", **saved,
                 **{f"ref_{n}": ref[n] for n in ref}, **{f"dsl_{n}": got[n] for n in got})
        print("  (inputs + outputs saved to /tmp/oproj_divergence.npz)")
        return 1

    shots = capture(g, spec, ids)
    print(f"captured {len(shots)} dispatches (one per layer)\n")

    stock = {key: g.kernels[key] for key in {s[0] for s in shots}}
    fn = getattr(kernels_dsl, which)

    bad = 0
    for i, shot in enumerate(shots):
        key = shot[0]
        g.kernels[key] = stock[key]
        ref = replay(g, spec, shot)
        g._compile_one(translate(fn, workgroup_size=(spec["threads"], 1, 1),
                                 target="msl", consts=spec["consts"](key)), which, key)
        got = replay(g, spec, shot)
        g.kernels[key] = stock[key]

        diffs = {n: int(np.count_nonzero(ref[n] != got[n])) for n in ref}
        if any(diffs.values()):
            bad += 1
            if bad == 1:
                print(f"FIRST DIVERGENCE at layer {i} ({key})")
                for n in ref:
                    if diffs[n]:
                        a = ref[n].astype(np.float64)
                        b = got[n].astype(np.float64)
                        k = int(np.argmax(np.abs(a - b)))
                        print(f"  {n:8} {diffs[n]:5d}/{len(a)} differ   "
                              f"max|d|={np.abs(a - b).max():.6e}  "
                              f"at [{k}] ref={a[k]!r} dsl={b[k]!r}")
    print(f"\n{bad} of {len(shots)} layers differ")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
