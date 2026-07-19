"""End-to-end test: GPU Gemma 4 E2B vs numpy reference on real weights.

Requires models/gemma-4-E2B/ (9.8GB checkpoint) and a WebGPU adapter;
skipped otherwise. The reference comparison is marked slow (it streams
9.2GB of bf16 weights per verified position).
"""

from pathlib import Path

import numpy as np
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
        gpu = Gemma4GPU(cfg, idx, max_seq=64, dtype="f32")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU unavailable: {exc}")
    return cfg, idx, gpu


class TestGpuMatchesReference:
    @pytest.mark.slow
    def test_logits_and_argmax_match_over_prompt(self, model):
        from gemma4.reference import ReferenceGemma4

        cfg, idx, gpu = model
        ref = ReferenceGemma4(cfg, idx, max_seq=64)
        for pos, tid in enumerate(PROMPT_IDS):
            gl = gpu.step(tid, pos)
            rl = ref.forward(tid, pos)
            assert int(gl.argmax()) == int(rl.argmax()), f"argmax @ pos {pos}"
            np.testing.assert_allclose(gl, rl, rtol=1e-2, atol=5e-3,
                                       err_msg=f"logits @ pos {pos}")
        # softcap is part of the compared logits
        assert np.abs(gl).max() < cfg.final_logit_softcapping


class TestGeneration:
    def test_factual_completion_greedy(self, model):
        from tokenizers import Tokenizer

        cfg, idx, gpu = model
        tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        ids = tok.encode("The capital of France is").ids
        assert ids[0] == cfg.bos_token_id
        out = gpu.generate(ids, max_new_tokens=16)
        assert "Paris" in tok.decode(out)

    def test_kv_cache_aliasing_layout(self, model):
        cfg, idx, gpu = model
        # caches exist only for layers 0-14; shared layers bind 13/14's
        assert sorted(gpu.k_cache) == list(range(15))
        assert gpu.k_cache[4].size == 64 * 512 * 4   # full-attention head_dim
        assert gpu.k_cache[0].size == 64 * 256 * 4   # sliding head_dim
