"""End-to-end test: QAT GPU runner vs weight-only numpy reference + coherent
chat output. Requires models/gemma-4-E2B-qat/ and a WebGPU adapter."""

from pathlib import Path

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")

QAT_DIR = Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B-qat"

pytestmark = pytest.mark.skipif(
    not (QAT_DIR / "model.safetensors").exists(),
    reason="Gemma 4 QAT checkpoint not downloaded",
)


def _chat(tok, prompt: str) -> list[int]:
    body = f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
    return [2] + tok.encode(body, add_special_tokens=False).ids


@pytest.fixture(scope="module")
def model():
    from gemma4.loader import Gemma4Config
    from gemma4.qat_loader import load_qat
    from gemma4.qat_runner import Gemma4QATGPU

    try:
        qat = load_qat(QAT_DIR)
        cfg = Gemma4Config(QAT_DIR / "config.json", qat.idx)
        gpu = Gemma4QATGPU(cfg, qat, max_seq=128)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU unavailable: {exc}")
    return cfg, qat, gpu


@pytest.mark.slow
def test_gpu_matches_weightonly_reference(model):
    from gemma4.qat_reference import QATReference

    cfg, qat, gpu = model
    ref = QATReference(cfg, qat, max_seq=32)
    ids = [2, 818, 5279, 529, 7001, 563]
    for pos, tid in enumerate(ids[:3]):
        gl = gpu.step(tid, pos, argmax=False)
        rl = ref.forward(tid, pos)
        assert int(gl.argmax()) == int(rl.argmax()), f"argmax @ pos {pos}"
        assert np.abs(gl - rl).max() < 0.1  # f32-vs-f32, softcapped logits


def test_coherent_chat_answers(model):
    from tokenizers import Tokenizer

    cfg, qat, gpu = model
    tok = Tokenizer.from_file(str(QAT_DIR / "tokenizer.json"))
    out = gpu.generate(_chat(tok, "What is the capital of France?"), max_new_tokens=16)
    assert "Paris" in tok.decode(out)


def test_packed_weight_footprint(model):
    cfg, qat, gpu = model
    wbytes = sum(r["w"].size for r in gpu.lin.values())
    wbytes += gpu.w["embed"].size + gpu.w["ple_table"].size
    assert wbytes < 2.5e9  # packed int2/4/8 << 4.6GB f16
