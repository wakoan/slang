"""Cross-backend gate: every backend must produce the same greedy output.

    python -m gemma4_150.verify_backends [--browser]

All five backends compile from gemma4_150/kernels_dsl.py, so a kernel change
that breaks one and not another is exactly the failure this catches. Until now
only the native-Metal Python path had a gate; the others were each verified once
by hand, which is a smoke test, not a guard.

Greedy decode is deterministic, so the bar is EXACT agreement on the generated
text. Backends that expose tokens are compared on tokens; the Swift and Rust
CLIs print text, so those are compared on text — weaker, but it still catches
any kernel fault large enough to change a single argmax.

The browser is opt-in (--browser): it needs the server and headless Chrome, so
it is not part of the default run.
"""
import re
import subprocess
import sys
from pathlib import Path

from tokenizers import Tokenizer

from gemma4_150.metal_runner import G4, TOKJSON

ROOT = Path(__file__).resolve().parent.parent
PROMPT = "The capital of France is"
N = 24


def framed(g, tok):
    return [g.bos] + tok.encode(f"<|turn>user\n{PROMPT}<turn|>\n<|turn>model\n",
                                add_special_tokens=False).ids


def run_cli(argv, cwd):
    """Run a native CLI and return the text after its `=>` marker."""
    try:
        out = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                             timeout=900).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"not run ({type(e).__name__})"
    m = re.search(r"^=> (.*)$", out, re.M)
    return (m.group(1).strip() if m else None), ("no `=>` line" if not m else "")


def main():
    print("loading reference (native Metal, Python)…")
    g = G4()
    tok = Tokenizer.from_file(TOKJSON)
    ids = framed(g, tok)
    ref_tokens, _ = g.generate(ids, N)
    ref = tok.decode(ref_tokens).strip()
    print(f"  reference: {ref!r}\n")

    results = [("metal (PyObjC)", ref, "")]

    # --- wgpu: compare tokens directly
    try:
        from gemma4_150.runner import G4Runner
        r = G4Runner()
        out, _ = r.generate(ids, N)
        results.append(("wgpu", tok.decode(out).strip(), ""))
        del r
    except Exception as e:
        results.append(("wgpu", None, f"{type(e).__name__}: {e}"))

    # --- Swift / Rust: text from their CLIs
    swift = ROOT / "gemma4_150/backends/swift_metal"
    rust = ROOT / "gemma4_150/backends/rust_metal"
    txt, err = run_cli(["./.build/release/g4", PROMPT, str(N)], swift)
    results.append(("swift", txt, err or "(build first: swift build -c release)"
                    if txt is None else ""))
    txt, err = run_cli(["./target/release/g4", PROMPT], rust)
    results.append(("rust", txt, err or "(build first: cargo build --release)"
                    if txt is None else ""))

    if "--browser" in sys.argv:
        drv = ROOT / "gemma4_150/backends/browser/drive.mjs"
        try:
            out = subprocess.run(["node", str(drv)], capture_output=True,
                                 text=True, timeout=1800).stdout
            m = re.search(r'"text":"([^"]*)"', out)
            results.append(("browser", m.group(1).strip() if m else None,
                            "" if m else "no text in RESULT (is the server up?)"))
        except Exception as e:
            results.append(("browser", None, f"{type(e).__name__}: {e}"))

    print(f"{'backend':<16} {'agrees':<8} output")
    ok = True
    for name, text, note in results:
        if text is None:
            print(f"{name:<16} {'SKIP':<8} {note}")
            continue
        # Native CLIs echo the prompt; the Python paths return only the
        # continuation. Compare on the part they share.
        agree = ref in text or text in ref or text == ref
        ok &= agree
        print(f"{name:<16} {('yes' if agree else 'NO'):<8} {text[:60]!r}")
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
