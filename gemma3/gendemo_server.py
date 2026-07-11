"""HTTP server for browser-side Gemma inference.

Serves the WebGPU page, DSL-generated WGSL kernels, prepacked weights,
and tokenizer endpoints. All inference happens in the browser.

Usage:
    python -m gemma3.gendemo_server [--port 8000] [--model-dir ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENDEMO_DIR = ROOT / "gendemo"
TENSORSCOPE_DIR = ROOT / "tensorscope"
DEFAULT_MODEL_DIR = ROOT / "models" / "gemma-3-270m-it"

# Kernels the browser uses: portable WGSL only (no f16 / subgroup features;
# packed-u32 weights use core unpack2x16float).
_BROWSER_KERNELS = [
    "embed_scale_packed",       # from kernels_metal (WGSL flavor)
    "matvec_wg_packed", "matvec_packed",
    "rmsnorm_wg", "rmsnorm_add_wg",
    "attention_fused",
    "rope", "kv_append", "geglu",
    "step_setup", "argmax_stage1", "argmax_stage2",
]


def build_kernels_json() -> str:
    from .kernels import KERNELS
    from .kernels_metal import METAL_KERNELS

    all_kernels = {**KERNELS, **METAL_KERNELS}
    out = {}
    for name in _BROWSER_KERNELS:
        kern = all_kernels[name]
        out[name] = {"wgsl": kern.wgsl, "workgroup_size": list(kern.workgroup_size)}
    return json.dumps(out)


class GemmaWebHandler(BaseHTTPRequestHandler):
    # class-level state, set by serve()
    model_dir: Path = DEFAULT_MODEL_DIR
    tokenizer = None
    kernels_json: str = ""
    bos_token_id: int = 2

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter: one line per request
        print(f"  {self.command} {self.path}")

    # ---------------- GET ---------------- #

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_file(GENDEMO_DIR / "index.html", "text/html")
        elif self.path == "/app.js":
            self._send_file(GENDEMO_DIR / "app.js", "text/javascript")
        elif self.path in ("/tensorscope", "/tensorscope/"):
            self._send_file(TENSORSCOPE_DIR / "index.html", "text/html")
        elif self.path == "/tensorscope.js":
            self._send_file(TENSORSCOPE_DIR / "tensorscope.js", "text/javascript")
        elif self.path == "/kernels.json":
            self._send_bytes(self.kernels_json.encode(), "application/json")
        elif self.path == "/manifest.json":
            # inject a weights version so the browser cache invalidates
            # when weights.bin is re-exported
            mpath = self.model_dir / "gendemo" / "manifest.json"
            if not mpath.exists():
                self.send_error(404)
                return
            manifest = json.loads(mpath.read_text())
            st = (self.model_dir / "gendemo" / "weights.bin").stat()
            manifest["weightsVersion"] = f"{st.st_size}-{int(st.st_mtime)}"
            self._send_bytes(json.dumps(manifest).encode(), "application/json")
        elif self.path == "/weights.bin":
            self._send_file(self.model_dir / "gendemo" / "weights.bin",
                            "application/octet-stream")
        else:
            self.send_error(404)

    def do_HEAD(self):
        if self.path == "/weights.bin":
            path = self.model_dir / "gendemo" / "weights.bin"
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
        else:
            self.send_error(404)

    # ---------------- POST ---------------- #

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON")
            return

        if self.path == "/tokenize":
            text = body.get("text", "")
            if body.get("chat", True):
                text = (f"<start_of_turn>user\n{text}<end_of_turn>\n"
                        f"<start_of_turn>model\n")
            ids = [self.bos_token_id] + self.tokenizer.encode(
                text, add_special_tokens=False).ids
            self._send_bytes(json.dumps({"ids": ids}).encode(), "application/json")
        elif self.path == "/detokenize":
            ids = body.get("ids", [])
            text = self.tokenizer.decode([int(i) for i in ids])
            self._send_bytes(json.dumps({"text": text}).encode(), "application/json")
        else:
            self.send_error(404)

    # ---------------- helpers ---------------- #

    def _send_bytes(self, data: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, ctype: str):
        if not path.exists():
            self.send_error(404, f"{path.name} not found")
            return
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, length=1 << 20)


def make_server(port: int = 8000,
                model_dir: Path = DEFAULT_MODEL_DIR) -> ThreadingHTTPServer:
    from tokenizers import Tokenizer

    demo_weights = model_dir / "gendemo" / "weights.bin"
    if not demo_weights.exists():
        print("weights.bin missing — exporting (one-time)...")
        from .export_gendemo import export_gendemo
        export_gendemo(model_dir)

    GemmaWebHandler.model_dir = Path(model_dir)
    GemmaWebHandler.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    GemmaWebHandler.kernels_json = build_kernels_json()
    cfg = json.loads((model_dir / "config.json").read_text())
    GemmaWebHandler.bos_token_id = cfg.get("bos_token_id", 2)
    return ThreadingHTTPServer(("127.0.0.1", port), GemmaWebHandler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()

    httpd = make_server(args.port, Path(args.model_dir))
    print(f"Gemma WebGPU server: http://localhost:{args.port}")
    print(f"tensor debugger:     http://localhost:{args.port}/tensorscope")
    print("open in Chrome/Edge (WebGPU required); Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
