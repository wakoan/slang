"""Tests for the resource download script (offline paths only)."""

import json
import struct
from pathlib import Path

import pytest

from gemma3.download import fetch_all, file_valid, safetensors_valid

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gemma-3-270m-it"


class TestValidation:
    @pytest.mark.skipif(not (MODEL_DIR / "model.safetensors").exists(),
                        reason="weights not downloaded")
    def test_real_safetensors_valid(self):
        assert safetensors_valid(MODEL_DIR / "model.safetensors")

    def test_junk_safetensors_invalid(self, tmp_path):
        p = tmp_path / "junk.safetensors"
        p.write_bytes(b"not a safetensors file at all")
        assert not safetensors_valid(p)

    def test_truncated_safetensors_invalid(self, tmp_path):
        # valid header, missing payload bytes
        header = json.dumps({"model.embed_tokens.weight":
                             {"dtype": "BF16", "shape": [4, 4],
                              "data_offsets": [0, 32]}}).encode()
        p = tmp_path / "trunc.safetensors"
        p.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 16)
        assert not safetensors_valid(p)

    def test_missing_file_invalid(self, tmp_path):
        assert not file_valid(tmp_path / "nope.json")

    def test_bad_json_invalid(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        assert not file_valid(p)


@pytest.mark.skipif(not (MODEL_DIR / "model.safetensors").exists(),
                    reason="weights not downloaded")
class TestIdempotent:
    def test_all_files_skipped_when_present(self):
        # must not touch the network when everything is already valid
        actions = fetch_all(MODEL_DIR)
        assert len(actions) == 3
        assert all(a.startswith("skip") for a in actions)
