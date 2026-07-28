"""Gate for the wgpu batched prefill (gemma4_150/prefill_wgpu.py).

The contract of prefill is the KV caches, so that is the observable. The bar is
NOT bit-identity with the per-token path: dequant + f16 GEMM replaces the exact
int-dot arithmetic, and the difference compounds through the 15 cache-owning
layers. Demanding bit-identity here would demand the wrong thing and the module
would never pass.

Instead the bar is EQUIVALENCE TO THE METAL BATCHED PATH, which has been in
production since it beat LiteRT on prefill. Both take the same approximation, so
they should drift from their own per-token paths by the same amount; if wgpu
drifts materially more, the extra is a port defect. That framing is what caught
the real bug: k and v were dequantized with q's row scales (a missing byte
offset into the concatenated qkv_scales), which read as 1.1 relative — an order
of magnitude past any f16 story.

Marked slow: each test loads the 2.5GB model.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.slow

PROMPT = ("Marta kept a small brass key on a chain around her neck. It had belonged "
          "to her grandmother, who ran a bakery on Oleander Street for forty-one "
          "years. The bakery closed in 1998 and the building sat empty. ") * 3

# The metric is RELATIVE L2 over each cache, not max-abs-relative. Max-abs is
# dominated by a handful of elements sitting on the SRQ rounding grid, where a
# last-ulp difference flips round() and the "relative error" of a near-zero
# element is meaningless — it reads 0.87 on a path that generates identical text.
#
# Envelope: the Metal batched path (shipping, and the one that beat LiteRT) was
# measured against its OWN per-token path on this prompt at n = 16/32/64/96/128/
# 160/256 and drifts 0.159 -> 0.236, rising with length. wgpu tracks it within
# ~5% relative (0.155 -> 0.237). 0.30 leaves margin above the longest measured
# point while still being an order of magnitude below the 1.1 that the missing
# k/v scale offset produced.
TOL = 0.30


@pytest.fixture(scope="module")
def setup():
    pytest.importorskip("wgpu")
    from tokenizers import Tokenizer
    from gemma4_150.runner import G4Runner
    from gemma4_150.metal_runner import TOKJSON
    r = G4Runner()
    tok = Tokenizer.from_file(TOKJSON)
    ids = [2] + tok.encode(PROMPT, add_special_tokens=False).ids
    # The caches are allocated without COPY_SRC (nothing reads them back in
    # normal operation); this comparison does.
    base = r._tmp
    r._tmp = lambda n, src=True: base(n, src=True)
    return r, tok, ids


def _snap(r, n):
    return {(k, l): r.read(buf, n * hd * 4).view(np.float32).copy()
            for k, d in (("k", r.kc), ("v", r.vc)) for l, buf in d.items()
            for hd in [next(s["head_dim"] for s in r.man["layers"] if s["index"] == l)]}


def test_batched_matches_per_token_within_f16_drift(setup):
    from gemma4_150.prefill_wgpu import PrefillWGPU
    r, _, ids = setup
    n = len(ids)

    r.setup_caches()
    hidden = r._scratch("hidden", r.cfg["H"] * 4)
    for p, t in enumerate(ids):
        r.forward(int(t), p, hidden)
    ref = _snap(r, n)

    r.setup_caches()
    PrefillWGPU(r).prefill(ids, 0)
    got = _snap(r, n)

    worst, where = 0.0, None
    for key in ref:
        a, b = ref[key], got[key]
        d = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9))
        if d > worst:
            worst, where = d, key
    assert worst < TOL, (
        f"batched prefill drifts {worst:.3e} (relative L2) from per-token at "
        f"{where} — past the {TOL} envelope the Metal path's own f16 drift "
        f"occupies, so this is a port defect, not f16")


def test_batched_prefill_does_not_change_generation(setup):
    """The end-to-end bar: same prompt, same answer, batching on or off."""
    import os
    r, tok, ids = setup
    q = tok.encode("\n\nWho did the brass key belong to?", add_special_tokens=False).ids
    full = ids + q
    outs = []
    for mode in ("0", "1"):
        os.environ["G4_BATCHED_PREFILL"] = mode
        r._pf = None
        out, _ = r.generate_resident(full, 24)
        outs.append(tok.decode(out))
    os.environ.pop("G4_BATCHED_PREFILL", None)
    assert outs[0] == outs[1], f"per-token {outs[0]!r} != batched {outs[1]!r}"
    assert "grandmother" in outs[1].lower(), f"incoherent answer: {outs[1]!r}"
