"""Greedy generation with the gemma4_150 GPU runner (coherence + tok/s).

    python -m gemma4_150.generate "The capital of France is" 20
"""
import sys
from tokenizers import Tokenizer
from gemma4_150.runner import G4Runner
from gemma4_150.loader import MODEL

TURN_START, TURN_END, BOS, EOS = 105, 106, 2, 1


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    body = tok.encode(f"user\n{prompt}", add_special_tokens=False).ids
    ids = [BOS, TURN_START] + body + [TURN_END] \
        + tok.encode("\n", add_special_tokens=False).ids \
        + [TURN_START] + tok.encode("model\n", add_special_tokens=False).ids

    r = G4Runner()
    out, tps = r.generate(ids, n_new, eos=EOS)
    print(f"prompt: {prompt!r}  ({len(ids)} tokens)")
    print(f"=> {tok.decode(out)!r}")
    print(f"decode: {tps:.1f} tok/s")


if __name__ == "__main__":
    main()
