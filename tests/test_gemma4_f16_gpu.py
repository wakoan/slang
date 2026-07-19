"""Gemma 4 E2B f16-weight path: argmax parity with the numpy reference.

Separate module from the f32 e2e tests so each holds only one GPU model
at a time. Skipped without weights or a WebGPU adapter.
"""

from pathlib import Path

import pytest

wgpu = pytest.importorskip("wgpu")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").exists(),
    reason="Gemma 4 E2B weights not downloaded",
)

PROMPT_IDS = [2, 818, 5279, 529, 7001, 563]  # BOS + "The capital of France is"


@pytest.fixture(scope="module")
def model():
    from gemma4.loader import load_model
    from gemma4.runner import Gemma4GPU

    try:
        cfg, idx = load_model(MODEL_DIR)
        gpu = Gemma4GPU(cfg, idx, max_seq=64, dtype="f16")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU unavailable: {exc}")
    if gpu.dtype != "f16":
        pytest.skip("shader-f16 not supported on this adapter")
    return cfg, idx, gpu


def test_weight_footprint(model):
    cfg, idx, gpu = model
    wbytes = sum(b.size for b in gpu._wbuf.values())
    assert 4.0e9 < wbytes < 5.0e9  # ~4.6GB: big weights f16, norms f32


@pytest.mark.slow
def test_argmax_matches_reference(model):
    from gemma4.reference import ReferenceGemma4

    cfg, idx, gpu = model
    ref = ReferenceGemma4(cfg, idx, max_seq=64)
    flips = 0
    for pos, tid in enumerate(PROMPT_IDS):
        gl = gpu.step(tid, pos)
        rl = ref.forward(tid, pos)
        flips += int(gl.argmax()) != int(rl.argmax())
    assert flips <= 1  # f16 weights: allow at most one flip over the prompt


def test_factual_completion_greedy(model):
    from tokenizers import Tokenizer

    cfg, idx, gpu = model
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    out = gpu.generate(tok.encode("The capital of France is").ids,
                       max_new_tokens=16)
    assert "Paris" in tok.decode(out)
