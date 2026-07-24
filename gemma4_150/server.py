"""Server for the gemma4_150 browser demo.

Serves the WebGPU page + app.js, the reference WGSL kernels (+ the two custom
ones from runner.py) as kernels.json, the exported manifest.json + weights.bin
(g4_150/, produced by export.py), and tokenize/detokenize endpoints. The browser
runs the SAME reference fused SRQ kernels the wgpu-py runner does — Dawn's cheap
in-process pass recording is the path past the ~97 tok/s wgpu-py ceiling toward
the reference's 150.

    python -m gemma4_150.server        # http://localhost:8000
"""
from __future__ import annotations

import json
import shutil
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from gemma4_150.runner import G4Runner, _kernel   # _kernel is lru-cached WGSL reader
from gemma4_150.loader import MODEL

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"

# reference kernels the runner dispatches + the two custom templates it defines
REF_KERNELS = ["00_main", "01_main", "68_reduce", "69_sg_sum", "70_srq", "73_sg_sum",
               "74_sg_sum", "95_sg_sum", "75_srq", "96_srq", "76_reduce", "77_sg_sum",
               "101_srq", "33_srq", "34_main", "35_main"]


def build_kernels_json() -> str:
    ks = {name: _kernel(name) for name in REF_KERNELS}
    ks["_COMBINE"] = G4Runner._COMBINE       # PLE-input norm/combine (has one %.9e slot)
    ks["_KVNORM"] = G4Runner._KVNORM         # fused k/v cache norm (has %d,%d slots)
    return json.dumps(ks)


class Handler(BaseHTTPRequestHandler):
    tokenizer = None
    kernels_json = ""
    gdir = MODEL / "g4_150"

    def log_message(self, fmt, *args):
        print(f"  {self.command} {self.path}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._file(WEB / "index.html", "text/html")
        elif self.path == "/app.js":
            self._file(WEB / "app.js", "text/javascript")
        elif self.path == "/kernels.json":
            self._bytes(self.kernels_json.encode(), "application/json")
        elif self.path == "/manifest.json":
            self._file(self.gdir / "manifest.json", "application/json")
        elif self.path == "/weights.bin":
            self._file(self.gdir / "weights.bin", "application/octet-stream")
        else:
            self.send_error(404)

    def do_HEAD(self):
        if self.path == "/weights.bin":
            p = self.gdir / "weights.bin"
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(p.stat().st_size))
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400)
            return
        if self.path == "/tokenize":
            text = body.get("text", "")
            inner = f"<|turn>user\n{text}<turn|>\n<|turn>model\n"   # -it chat format
            ids = [2] + self.tokenizer.encode(inner, add_special_tokens=False).ids
            self._bytes(json.dumps({"ids": ids}).encode(), "application/json")
        elif self.path == "/detokenize":
            text = self.tokenizer.decode([int(i) for i in body.get("ids", [])])
            self._bytes(json.dumps({"text": text}).encode(), "application/json")
        else:
            self.send_error(404)

    def _bytes(self, data: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path, ctype: str):
        if not path.exists():
            self.send_error(404, f"{path.name} not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, length=1 << 20)


def main():
    from tokenizers import Tokenizer
    if not (Handler.gdir / "weights.bin").exists():
        print("weights.bin missing — run `python -m gemma4_150.export` first")
        return
    Handler.tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    Handler.kernels_json = build_kernels_json()
    print("gemma4_150 browser demo: http://localhost:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()


if __name__ == "__main__":
    main()
