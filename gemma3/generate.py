"""Generate text with Gemma 3 running entirely on GPU shaders
written in py_shader_lang_wgpu.

Usage:
    python -m gemma3.generate "Why is the sky blue?" [--max-tokens 64]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer

from .loader import load_model
from .runner import GemmaGPU

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gemma-3-270m-it"


def chat_ids(tok: Tokenizer, bos: int, prompt: str) -> list[int]:
    text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    return [bos] + tok.encode(text, add_special_tokens=False).ids


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
    parser.add_argument("--backend", choices=("wgpu", "metal"), default="wgpu",
                        help="wgpu (WGSL shaders) or metal (native MSL via metalgpu)")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    args = parser.parse_args()

    print("loading weights...", file=sys.stderr)
    cfg, weights = load_model(args.model_dir)
    tok = Tokenizer.from_file(str(Path(args.model_dir) / "tokenizer.json"))
    if args.backend == "metal":
        if args.profile or args.temperature > 0:
            parser.error("--profile/--temperature are wgpu-backend only")
        from .runner_metal import GemmaMetal
        gpu = GemmaMetal(cfg, weights)
    else:
        gpu = GemmaGPU(cfg, weights, profile=args.profile)
    if args.profile:
        print("[profile mode: per-kernel timing enabled — generation runs "
              "~3x slower than normal; the tok/s figure is not representative]",
              file=sys.stderr)

    ids = chat_ids(tok, cfg.bos_token_id, args.prompt)
    print(f"prompt: {len(ids)} tokens", file=sys.stderr)

    # Incremental detokenization: decode the whole sequence each step and
    # print only the new suffix. Decoding tokens one at a time drops
    # SentencePiece word-boundary spaces and hides whitespace-only tokens.
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
    if args.backend == "metal":
        out = gpu.generate(ids, max_new_tokens=args.max_tokens, on_token=on_token)
    else:
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
    if args.backend == "metal":
        gpu.close()


if __name__ == "__main__":
    main()
