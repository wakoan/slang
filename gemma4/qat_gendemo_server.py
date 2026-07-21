"""HTTP server for browser-side Gemma 4 E2B QAT inference.

Serves the WebGPU page, the DSL-generated WGSL kernels (portable subset —
all QAT kernels are integer/packed-u32, no f16/subgroup features), the
prepacked int2/4/8 weights, and tokenizer endpoints. All inference runs in
the browser (Gemma4QATGPU ported to app.js).

Usage:
    python -m gemma4.qat_gendemo_server [--port 8000] [--model-dir ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENDEMO_DIR = ROOT / "gendemo4"
DEFAULT_MODEL_DIR = ROOT / "models" / "gemma-4-E2B-qat"


def build_kernels_json() -> str:
    from .qat_kernels import BROWSER_EXTRA_KERNELS, KERNELS
    allk = {**KERNELS, **BROWSER_EXTRA_KERNELS}  # extras: int8/dot4I8Packed experiment
    out = {name: {"wgsl": k.wgsl, "workgroup_size": list(k.workgroup_size)}
           for name, k in allk.items()}
    return json.dumps(out)


class QATWebHandler(BaseHTTPRequestHandler):
    model_dir: Path = DEFAULT_MODEL_DIR
    tokenizer = None
    kernels_json: str = ""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"  {self.command} {self.path}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_file(GENDEMO_DIR / "index.html", "text/html")
        elif self.path == "/app.js":
            self._send_file(GENDEMO_DIR / "app.js", "text/javascript")
        elif self.path == "/kernels.json":
            self._send_bytes(self.kernels_json.encode(), "application/json")
        elif self.path == "/manifest.json":
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
                # QAT instruction-tuned chat format (turn tokens 105/106)
                inner = f"<|turn>user\n{text}<turn|>\n<|turn>model\n"
                ids = [2] + self.tokenizer.encode(inner, add_special_tokens=False).ids
            else:
                ids = self.tokenizer.encode(text).ids
            self._send_bytes(json.dumps({"ids": ids}).encode(), "application/json")
        elif self.path == "/detokenize":
            ids = body.get("ids", [])
            text = self.tokenizer.decode([int(i) for i in ids])
            self._send_bytes(json.dumps({"text": text}).encode(), "application/json")
        else:
            self.send_error(404)

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
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, length=1 << 20)


def make_server(port: int = 8000,
                model_dir: Path = DEFAULT_MODEL_DIR) -> ThreadingHTTPServer:
    from tokenizers import Tokenizer

    if not (model_dir / "gendemo" / "weights.bin").exists():
        print("weights.bin missing — exporting (one-time, ~2GB)...")
        from .export_qat_gendemo import export_qat_gendemo
        export_qat_gendemo(model_dir)

    QATWebHandler.model_dir = Path(model_dir)
    QATWebHandler.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    QATWebHandler.kernels_json = build_kernels_json()
    return ThreadingHTTPServer(("127.0.0.1", port), QATWebHandler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()

    httpd = make_server(args.port, Path(args.model_dir))
    print(f"Gemma 4 E2B QAT WebGPU server: http://localhost:{args.port}")
    print("open in Chrome (WebGPU + ~2GB GPU required); Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
