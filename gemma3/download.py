"""Download the resources needed to run the Gemma 3 example.

Fetches config.json, tokenizer.json, and model.safetensors (~570MB total)
from the ungated mirror `unsloth/gemma-3-270m-it` — the official
google/gemma-3-270m repo is license-gated and needs an HF token.

Idempotent: files already present and valid are skipped. Downloads go to
a .part file and are renamed only when complete.

Usage:
    python -m gemma3.download [--model-dir DIR] [--force]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/unsloth/gemma-3-270m-it/resolve/main"
FILES = ("config.json", "tokenizer.json", "model.safetensors")
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models" / "gemma-3-270m-it"


def safetensors_valid(path: Path) -> bool:
    """Cheap integrity check: parse the header and confirm the embedding
    tensor is present and the payload has the promised length."""
    try:
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        if "model.embed_tokens.weight" not in header:
            return False
        data_end = max(
            (info["data_offsets"][1] for name, info in header.items()
             if name != "__metadata__"),
            default=0,
        )
        return path.stat().st_size == 8 + header_len + data_end
    except Exception:
        return False


def file_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if path.suffix == ".safetensors":
        return safetensors_valid(path)
    if path.suffix == ".json":
        try:
            json.loads(path.read_text())
            return True
        except Exception:
            return False
    return True


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB "
                          f"({100 * done / total:.0f}%)", end="", file=sys.stderr)
        print(file=sys.stderr)
    tmp.rename(dest)  # atomic: never leave a truncated file under the real name


def fetch_all(model_dir: Path, force: bool = False) -> list[str]:
    """Ensure all files are present and valid; returns actions taken."""
    model_dir.mkdir(parents=True, exist_ok=True)
    actions = []
    for name in FILES:
        dest = model_dir / name
        if not force and file_valid(dest):
            actions.append(f"skip {name} (already valid)")
            continue
        print(f"downloading {name}…", file=sys.stderr)
        download(f"{BASE_URL}/{name}", dest)
        if not file_valid(dest):
            raise RuntimeError(f"{name} failed validation after download")
        actions.append(f"downloaded {name}")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--force", action="store_true",
                        help="re-download even if files look valid")
    args = parser.parse_args()

    for action in fetch_all(Path(args.model_dir), force=args.force):
        print(f"  {action}")
    print("done. next steps:")
    print("  python -m gemma3.generate \"Hello!\"        # CLI generation")
    print("  python -m gemma3.gendemo_server            # browser demo + tensorscope")
    print("  (the server packs web weights automatically on first run)")


if __name__ == "__main__":
    main()
