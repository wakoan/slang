"""Generate text with Gemma 4 E2B running entirely on GPU shaders
written in py_shader_lang_wgpu.

Plain completion (no chat template: the E2B checkpoint ships an empty
chat_template and `<start_of_turn>` is not a special token).

Usage:
    python -m gemma4.generate "The capital of France is" [--max-tokens 64]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer

from .loader import load_model
from .runner import Gemma4GPU

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy (deterministic); try 0.7 if output loops")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--profile", action="store_true",
                        help="collect GPU/CPU timings and print a report")
    parser.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    args = parser.parse_args()

    print("loading weights...", file=sys.stderr)
    cfg, index = load_model(args.model_dir)
    tok = Tokenizer.from_file(str(Path(args.model_dir) / "tokenizer.json"))
    gpu = Gemma4GPU(cfg, index, profile=args.profile, dtype=args.dtype)

    ids = tok.encode(args.prompt).ids  # tokenizer prepends BOS itself
    print(f"prompt: {len(ids)} tokens", file=sys.stderr)

    # Incremental detokenization: decode the whole sequence each step and
    # print only the new suffix (per-token decode drops SentencePiece
    # word-boundary spaces).
    gen_ids: list[int] = []
    printed = 0

    def on_token(tid: int) -> None:
        nonlocal printed
        gen_ids.append(tid)
        text = tok.decode(gen_ids)
        if len(text) > printed:
            print(text[printed:], end="", flush=True)
            printed = len(text)

    t0 = time.time()
    out = gpu.generate(ids, max_new_tokens=args.max_tokens, on_token=on_token,
                       temperature=args.temperature, top_k=args.top_k,
                       seed=args.seed)
    dt = time.time() - t0
    print()
    n = len(ids) + len(out)
    print(f"[{n} tokens total, {len(out)} generated, "
          f"{n / dt:.1f} tok/s]", file=sys.stderr)
    if args.profile:
        print("\n" + gpu.profile_report(), file=sys.stderr)


if __name__ == "__main__":
    main()
