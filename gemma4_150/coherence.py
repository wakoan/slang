"""End-to-end coherence gate for the gemma4_150 SRQ recipe.

Greedy-decodes a few tokens with the standalone numpy SRQ reference on the -it
chat format. If the SRQ recipe is wired correctly it should answer factual
prompts sensibly (the checkpoint is QAT-trained for this int8-activation path).

    python -m gemma4_150.coherence "The capital of France is"
"""
import sys
import numpy as np
from tokenizers import Tokenizer
from gemma4_150.reference import ReferenceSRQ
from gemma4_150.loader import MODEL

TURN_START, TURN_END = 105, 106   # <|turn>, <turn|>
BOS, EOS = 2, 1


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    body = tok.encode(f"user\n{prompt}", add_special_tokens=False).ids
    ids = [BOS, TURN_START] + body + [TURN_END] + tok.encode("\n", add_special_tokens=False).ids \
        + [TURN_START] + tok.encode("model\n", add_special_tokens=False).ids

    ref = ReferenceSRQ()
    print(f"prompt ids ({len(ids)}): {ids}")
    for pos, t in enumerate(ids):          # prefill
        logits = ref.forward(t, pos)
    out = []
    for _ in range(n_new):
        nxt = int(np.argmax(logits))
        if nxt == EOS:
            break
        out.append(nxt)
        pos += 1
        logits = ref.forward(nxt, pos)
    print("continuation ids:", out)
    print("=> " + repr(tok.decode(out)))


if __name__ == "__main__":
    main()
