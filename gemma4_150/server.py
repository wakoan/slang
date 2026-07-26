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
import re
import os
import shutil
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from gemma4_150.runner import G4Runner, _kernel   # _kernel is lru-cached WGSL reader
from gemma4_150.loader import MODEL

HERE = Path(__file__).resolve().parent
WEB = HERE / "backends/browser"

# reference kernels the runner dispatches + the two custom templates it defines
REF_KERNELS = ["00_main", "01_main", "68_reduce", "69_sg_sum", "70_srq", "73_sg_sum",
               "74_sg_sum", "95_sg_sum", "75_srq", "96_srq", "76_reduce", "77_sg_sum",
               "101_srq", "33_srq", "34_main", "35_main"]


# DSL kernel per bundle key, with the consts app.js patches client-side. They
# are emitted as NAMED constants rather than folded (consts_as_decls) precisely
# so that patching still works — app.js rewrites `const NAME: u32 = ...;`.
DSL_KERNELS = {
    "00_main": ("embed_00", (64, 1, 1), {}),
    "01_main": ("plegather_01", (64, 1, 1), {}),
    "68_reduce": ("proj_68", (32, 1, 1), {}),
    "69_sg_sum": ("rmssrq_69", (256, 1, 1), {}),
    "70_srq": ("qkv_70", (32, 1, 1),
               {"Q_OUT": 2048, "KV_OUT": 256, "Q_WGS": 1024, "KV_WGS": 128,
                "TOTAL_WGS": 1280, "GRID_X": 1280}),
    "73_sg_sum": ("oproj_73", (256, 1, 1), {"WPR": 256}),
    "74_sg_sum": ("gateup_74", (64, 1, 1), {}),
    "95_sg_sum": ("gateup_95", (64, 1, 1), {}),
    "75_srq": ("down_75", (256, 1, 1), {}),
    "96_srq": ("down_96", (256, 1, 1), {}),
    "76_reduce": ("plegate_76", (32, 1, 1), {}),
    "77_sg_sum": ("pleproj_77", (256, 1, 1), {}),
    "101_srq": ("attn_101", (256, 1, 1),
                {"HEAD_DIM": 512, "HALF_DIM": 256, "HD4": 128, "J_GROUPS": 2,
                 "PP_COUNTER_BASE": 8 * 32 * 514, "OUT_Q": 0.014886821620166302}),
    "33_srq": ("logits_33", (128, 1, 1), {}),
    "34_main": ("argmax1_34", (256, 1, 1), {}),
    "35_main": ("argmax2_35", (256, 1, 1), {}),
}

# Serves the DSL-generated bundle: 18 kernels from gemma4_150/kernels_dsl.py,
# entry point renamed to `main`, consts emitted as named declarations so app.js
# can still patch per-layer shapes client-side. Verified in headless Chrome
# against the captured bundle. G4_KERNEL_SOURCE=ref for the captured kernels.
USE_DSL = os.environ.get("G4_KERNEL_SOURCE", "dsl") != "ref"


def _dsl_wgsl(dsl_name, threads, consts):
    """DSL kernel as WGSL, with the entry point renamed to `main`.

    app.js hardcodes entryPoint "main" and has a 569-op dispatch plan that is
    verified against the runner; renaming here keeps the bundle contract
    unchanged rather than perturbing that file.
    """
    from gemma4_150 import kernels_dsl
    from py_shader_lang_wgpu import translate
    fn = getattr(kernels_dsl, dsl_name)
    src = translate(fn, workgroup_size=threads, consts=consts or None,
                    consts_as_decls=True)
    return re.sub(rf"\bfn {dsl_name}\s*\(", "fn main(", src)


def build_kernels_json() -> str:
    if USE_DSL:
        ks = {key: _dsl_wgsl(*spec) for key, spec in DSL_KERNELS.items()}
        ks["_COMBINE"] = _dsl_wgsl("combine", (256, 1, 1), {})
        ks["_KVNORM"] = _dsl_wgsl("kvnorm", (256, 1, 1), {"HD": 256, "HALF": 128})
        return json.dumps(ks)
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
