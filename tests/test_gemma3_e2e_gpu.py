"""End-to-end test: GPU Gemma 3 vs numpy reference on real weights.

Requires models/gemma-3-270m-it/ (downloaded weights) and a WebGPU adapter;
skipped otherwise.
"""

from pathlib import Path

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gemma-3-270m-it"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").exists(),
    reason="Gemma 3 270M weights not downloaded",
)


@pytest.fixture(scope="module")
def model():
    from gemma3.loader import load_model
    from gemma3.runner import GemmaGPU

    try:
        cfg, weights = load_model(MODEL_DIR)
        gpu = GemmaGPU(cfg, weights, max_seq=256, dtype="f32")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU unavailable: {exc}")
    return cfg, weights, gpu


class TestGpuMatchesReference:
    def test_logits_and_argmax_match_over_prompt(self, model):
        from gemma3.reference import ReferenceGemma

        cfg, weights, gpu = model
        ref = ReferenceGemma(cfg, weights, max_seq=256)
        # "<bos><start_of_turn>user\nWhat is..." prefix tokens
        ids = [2, 105, 2364, 107, 3689, 563]
        for pos, tid in enumerate(ids):
            ref_logits = ref.forward(tid, pos)
            gpu_logits = gpu.step(tid, pos)
            assert np.argmax(gpu_logits) == np.argmax(ref_logits), f"pos {pos}"
            np.testing.assert_allclose(gpu_logits, ref_logits, rtol=1e-2, atol=5e-3)


class TestProfiling:
    def test_profile_collects_and_reports(self):
        from gemma3.loader import load_model
        from gemma3.runner import GemmaGPU

        cfg, weights = load_model(MODEL_DIR)
        try:
            gpu = GemmaGPU(cfg, weights, max_seq=64, profile=True)
        except RuntimeError as exc:
            pytest.skip(str(exc))  # timestamp-query unsupported
        for pos, tid in enumerate([2, 105, 2364]):
            gpu.step(tid, pos, want_logits=pos == 2)
        report = gpu.profile_report()
        assert "mv_logits" in report
        assert "GPU busy" in report
        assert "bandwidth floor" in report
        # every call-site was recorded
        assert gpu._prof_kernels["mv_qkv"][0] == 3 * cfg.num_layers

    def test_profile_off_raises_on_report(self, model):
        _, _, gpu = model
        with pytest.raises(RuntimeError, match="profile=True"):
            gpu.profile_report()


class TestGeneration:
    def test_generates_paris(self, model):
        from tokenizers import Tokenizer

        cfg, _, gpu = model
        # Fresh KV state: use positions after a fresh generate call only.
        tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        text = "<start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\n"
        ids = [cfg.bos_token_id] + tok.encode(text, add_special_tokens=False).ids
        out = gpu.generate(ids, max_new_tokens=16)
        decoded = tok.decode(out)
        assert "Paris" in decoded
        assert out[-1] in cfg.eos_token_ids  # finished cleanly


class TestF16Model:
    def test_f16_generates_paris_and_tracks_f32(self):
        from tokenizers import Tokenizer
        from gemma3.loader import load_model
        from gemma3.runner import GemmaGPU
        from gemma3.reference import ReferenceGemma

        cfg, weights = load_model(MODEL_DIR)
        gpu = GemmaGPU(cfg, weights, max_seq=256, dtype="f16")
        if gpu.dtype != "f16":
            pytest.skip("shader-f16 not supported")

        # logits track the f32 reference loosely; argmax on a real prompt holds
        ref = ReferenceGemma(cfg, weights, max_seq=256)
        ids = [2, 105, 2364, 107, 3689, 563]
        agree = 0
        for pos, tid in enumerate(ids):
            rl = ref.forward(tid, pos)
            gl = gpu.step(tid, pos)
            assert np.isfinite(gl).all()
            agree += int(np.argmax(gl) == np.argmax(rl))
        assert agree >= len(ids) - 1  # allow one argmax flip from f16 rounding

        tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        text = ("<start_of_turn>user\nWhat is the capital of France?"
                "<end_of_turn>\n<start_of_turn>model\n")
        pids = [cfg.bos_token_id] + tok.encode(text, add_special_tokens=False).ids
        out = gpu.generate(pids, max_new_tokens=16)
        assert "Paris" in tok.decode(out)


class TestResidentDecode:
    def test_resident_matches_per_step_greedy(self):
        from tokenizers import Tokenizer
        from gemma3.loader import load_model
        from gemma3.runner import GemmaGPU

        cfg, weights = load_model(MODEL_DIR)
        tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        text = ("<start_of_turn>user\nWhy is the sky blue?"
                "<end_of_turn>\n<start_of_turn>model\n")
        ids = [cfg.bos_token_id] + tok.encode(text, add_special_tokens=False).ids

        gpu1 = GemmaGPU(cfg, weights, max_seq=256)
        resident = gpu1._generate_resident(ids, max_new_tokens=24, on_token=None)

        gpu2 = GemmaGPU(cfg, weights, max_seq=256)
        classic = []
        logits = None
        for pos, tid in enumerate(ids):
            logits = gpu2.step(tid, pos, want_logits=pos == len(ids) - 1)
        pos = len(ids)
        for _ in range(24):
            nid = int(np.argmax(logits))
            classic.append(nid)
            if nid in cfg.eos_token_ids:
                break
            logits = gpu2.step(nid, pos)
            pos += 1

        assert resident == classic

    def test_resident_stops_at_eos(self):
        from tokenizers import Tokenizer
        from gemma3.loader import load_model
        from gemma3.runner import GemmaGPU

        cfg, weights = load_model(MODEL_DIR)
        tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        text = ("<start_of_turn>user\nWhat is the capital of France?"
                "<end_of_turn>\n<start_of_turn>model\n")
        ids = [cfg.bos_token_id] + tok.encode(text, add_special_tokens=False).ids
        gpu = GemmaGPU(cfg, weights, max_seq=256)
        out = gpu.generate(ids, max_new_tokens=64)
        assert out[-1] in cfg.eos_token_ids
        assert len(out) < 64  # stopped early, EOS from a mid-chunk position
