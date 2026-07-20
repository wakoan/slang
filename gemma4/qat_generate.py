"""Generate text with the Gemma 4 E2B QAT-mobile checkpoint (int2/4/8 weights)
running entirely on GPU shaders written in py_shader_lang_wgpu.

This is the instruction-tuned checkpoint, so the prompt is wrapped in its
chat format by default. Use --raw for plain completion.

    python -m gemma4.qat_generate "What is the capital of France?"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer

from .loader import Gemma4Config
from .qat_loader import load_qat
from .qat_runner import Gemma4QATGPU

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B-qat"


def chat_ids(tok: Tokenizer, prompt: str) -> list[int]:
    body = f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
    return [2] + tok.encode(body, add_special_tokens=False).ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--raw", action="store_true", help="plain completion (no chat wrap)")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    args = ap.parse_args()

    print("loading QAT weights...", file=sys.stderr)
    qat = load_qat(args.model_dir)
    cfg = Gemma4Config(Path(args.model_dir) / "config.json", qat.idx)
    tok = Tokenizer.from_file(str(Path(args.model_dir) / "tokenizer.json"))
    gpu = Gemma4QATGPU(cfg, qat, max_seq=max(256, args.max_tokens + 64))

    ids = (tok.encode(args.prompt).ids if args.raw else chat_ids(tok, args.prompt))
    print(f"prompt: {len(ids)} tokens", file=sys.stderr)

    gen: list[int] = []
    printed = 0

    def on_token(tid: int) -> None:
        nonlocal printed
        gen.append(tid)
        text = tok.decode(gen)
        if len(text) > printed:
            print(text[printed:], end="", flush=True)
            printed = len(text)

    t0 = time.time()
    out = gpu.generate(ids, max_new_tokens=args.max_tokens, on_token=on_token)
    dt = time.time() - t0
    print()
    print(f"[{len(out)} tokens, {len(out) / dt:.1f} tok/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
