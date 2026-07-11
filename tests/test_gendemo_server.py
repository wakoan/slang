"""Tests for the gendemo HTTP server and weight export."""

import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gemma-3-270m-it"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").exists(),
    reason="Gemma 3 270M weights not downloaded",
)


@pytest.fixture(scope="module")
def base_url():
    from gemma3.gendemo_server import make_server

    httpd = make_server(port=0, model_dir=MODEL_DIR)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.read(), dict(r.headers)


def post_json(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


class TestExport:
    def test_manifest_matches_blob(self):
        web = MODEL_DIR / "gendemo"
        manifest = json.loads((web / "manifest.json").read_text())
        blob_size = (web / "weights.bin").stat().st_size
        assert manifest["totalBytes"] == blob_size
        last = manifest["tensors"][-1]
        assert last["offset"] + last["byteLength"] == blob_size
        # offsets aligned and non-overlapping
        prev_end = 0
        for t in manifest["tensors"]:
            assert t["offset"] % 256 == 0
            assert t["offset"] >= prev_end
            prev_end = t["offset"] + t["byteLength"]

    def test_qkv_tensor_matches_loader(self):
        from gemma3.loader import load_model

        cfg, w = load_model(MODEL_DIR)
        web = MODEL_DIR / "gendemo"
        manifest = json.loads((web / "manifest.json").read_text())
        entry = next(t for t in manifest["tensors"] if t["name"] == "L0.qkv")
        with open(web / "weights.bin", "rb") as f:
            f.seek(entry["offset"])
            raw = f.read(entry["byteLength"])
        got = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        a = "model.layers.0.self_attn."
        want = np.concatenate([w[a + "q_proj.weight"], w[a + "k_proj.weight"],
                               w[a + "v_proj.weight"]]).astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(got, want.ravel())

    def test_config_in_manifest(self):
        manifest = json.loads((MODEL_DIR / "gendemo" / "manifest.json").read_text())
        cfg = manifest["config"]
        assert cfg["num_layers"] == 18
        assert cfg["vocab_size"] == 262144
        assert len(cfg["layer_types"]) == 18


class TestEndpoints:
    def test_index_html(self, base_url):
        body, headers = get(base_url + "/")
        assert b"WebGPU" in body
        assert headers["Content-Type"] == "text/html"

    def test_app_js(self, base_url):
        body, _ = get(base_url + "/app.js")
        assert b"requestAdapter" in body

    def test_kernels_json_complete(self, base_url):
        body, _ = get(base_url + "/kernels.json")
        kernels = json.loads(body)
        expected = {"embed_scale_packed", "matvec_wg_packed", "matvec_packed",
                    "rmsnorm_wg", "rmsnorm_add_wg", "attention_fused",
                    "rope", "kv_append", "geglu",
                    "step_setup", "argmax_stage1", "argmax_stage2"}
        assert set(kernels) == expected
        for name, k in kernels.items():
            assert f"fn {name}(" in k["wgsl"]
            assert isinstance(k["workgroup_size"], list)
        # browser kernels must not require optional features
        for name, k in kernels.items():
            assert "enable f16;" not in k["wgsl"], name
            assert "enable subgroups;" not in k["wgsl"], name

    def test_manifest_endpoint(self, base_url):
        body, _ = get(base_url + "/manifest.json")
        manifest = json.loads(body)
        assert manifest["config"]["hidden_size"] == 640
        # served manifest carries a cache-busting version (size-mtime)
        size = (MODEL_DIR / "gendemo" / "weights.bin").stat().st_size
        assert manifest["weightsVersion"].startswith(f"{size}-")

    def test_weights_head(self, base_url):
        req = urllib.request.Request(base_url + "/weights.bin", method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as r:
            size = int(r.headers["Content-Length"])
        assert size == (MODEL_DIR / "gendemo" / "weights.bin").stat().st_size

    def test_tokenize_roundtrip(self, base_url):
        out = post_json(base_url + "/tokenize", {"text": "Hello world", "chat": False})
        assert out["ids"][0] == 2  # BOS
        detok = post_json(base_url + "/detokenize", {"ids": out["ids"][1:]})
        assert detok["text"] == "Hello world"

    def test_tokenize_chat_template(self, base_url):
        out = post_json(base_url + "/tokenize", {"text": "Hi", "chat": True})
        # template starts with <bos><start_of_turn> = [2, 105, ...]
        assert out["ids"][:2] == [2, 105]

    def test_tensorscope_page(self, base_url):
        body, headers = get(base_url + "/tensorscope")
        assert b"tensorscope" in body
        assert headers["Content-Type"] == "text/html"

    def test_tensorscope_js(self, base_url):
        body, _ = get(base_url + "/tensorscope.js")
        assert b"debugStep" in body
        assert b"copyBufferToBuffer" in body

    def test_404(self, base_url):
        with pytest.raises(urllib.error.HTTPError):
            get(base_url + "/nope")


class TestBrowserArtifactsEndToEnd:
    """Drive the exported web artifacts (manifest + weights.bin + browser
    kernel set) through wgpu-py with the exact dispatch sequence app.js
    encodes. Validates the layout, offsets, and kernel selection the
    browser will use — everything except the JS itself."""

    def test_generate_paris_from_web_artifacts(self):
        wgpu = pytest.importorskip("wgpu")
        from gemma3.gendemo_server import build_kernels_json

        web = MODEL_DIR / "gendemo"
        manifest = json.loads((web / "manifest.json").read_text())
        cfg = manifest["config"]
        kernels = json.loads(build_kernels_json())
        blob = np.fromfile(web / "weights.bin", dtype=np.uint8)

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        embed_bytes = cfg["vocab_size"] * cfg["hidden_size"] * 2
        device = adapter.request_device_sync(required_limits={
            "max-buffer-size": max(embed_bytes, 1 << 28),
            "max-storage-buffer-binding-size": max(embed_bytes, 1 << 28),
        })
        q = device.queue

        ST, CD, CS = (wgpu.BufferUsage.STORAGE, wgpu.BufferUsage.COPY_DST,
                      wgpu.BufferUsage.COPY_SRC)
        W = {}
        for t in manifest["tensors"]:
            W[t["name"]] = device.create_buffer_with_data(
                data=blob[t["offset"]:t["offset"] + t["byteLength"]].tobytes(),
                usage=ST)

        import re
        pipes = {}
        for name, k in kernels.items():
            access = sorted(
                ((int(m.group(1)), m.group(2) == "read_write") for m in
                 re.finditer(r"@binding\((\d+)\) var<storage, (read_write|read)>",
                             k["wgsl"])))
            layout = device.create_bind_group_layout(entries=[
                {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.storage if rw
                            else wgpu.BufferBindingType.read_only_storage}}
                for i, rw in access])
            module = device.create_shader_module(code=k["wgsl"])
            pipes[name] = (device.create_compute_pipeline(
                layout=device.create_pipeline_layout(bind_group_layouts=[layout]),
                compute={"module": module, "entry_point": name}), layout)

        MAX_SEQ, N_AWG = 256, 128
        h, hd, nh = cfg["hidden_size"], cfg["head_dim"], cfg["num_heads"]
        inter, q_dim = cfg["intermediate_size"], cfg["num_heads"] * cfg["head_dim"]

        def fbuf(n):
            return device.create_buffer(size=n * 4, usage=ST | CS | CD)

        B = {n: fbuf(sz) for n, sz in {
            "x": h, "xn": h, "qkv": q_dim + 2 * hd, "qn": q_dim, "kn": hd,
            "scores": nh * MAX_SEQ, "attn": q_dim, "attn_proj": h,
            "gateup": 2 * inter, "ffh": inter, "mlp_out": h,
            "logits": cfg["vocab_size"], "token": 1, "pos": 1, "counter": 1,
            "out_tokens": MAX_SEQ, "part_val": N_AWG, "part_idx": N_AWG}.items()}
        kc = [fbuf(MAX_SEQ * hd) for _ in range(cfg["num_layers"])]
        vc = [fbuf(MAX_SEQ * hd) for _ in range(cfg["num_layers"])]

        def ubuf(*vals):
            b = device.create_buffer(size=len(vals) * 4, usage=ST | CD)
            q.write_buffer(b, 0, np.array(vals, np.uint32).tobytes())
            return b

        D = {"embed": ubuf(h), "norm_h": ubuf(1, h), "norm_q": ubuf(nh, hd),
             "norm_k": ubuf(1, hd), "mv_qkv": ubuf(q_dim + 2 * hd, h),
             "mv_o": ubuf(h, q_dim), "mv_gu": ubuf(2 * inter, h),
             "mv_down": ubuf(h, inter), "mv_logits": ubuf(cfg["vocab_size"], h),
             "rope_q": ubuf(nh, hd), "rope_k": ubuf(1, hd), "geglu": ubuf(inter),
             "kv": ubuf(hd, 0), "sc_s": ubuf(nh, hd, 1, 0, MAX_SEQ),
             "sc_f": ubuf(nh, hd, 1, 0, MAX_SEQ),
             "am1": ubuf(cfg["vocab_size"], N_AWG), "am2": ubuf(N_AWG),
             "scfg": ubuf(cfg["sliding_window"])}
        rope_l = device.create_buffer(size=8, usage=ST | CD)
        rope_g = device.create_buffer(size=8, usage=ST | CD)
        q.write_buffer(rope_l, 0, np.array([cfg["rope_theta_local"], 0], np.float32).tobytes())
        q.write_buffer(rope_g, 0, np.array([cfg["rope_theta_global"], 0], np.float32).tobytes())

        def bg(name, *bufs):
            _, layout = pipes[name]
            entries = []
            for i, b in enumerate(bufs):
                if isinstance(b, tuple):
                    entries.append({"binding": i, "resource":
                                    {"buffer": b[0], "offset": b[1], "size": b[2]}})
                else:
                    entries.append({"binding": i, "resource":
                                    {"buffer": b, "offset": 0, "size": b.size}})
            return device.create_bind_group(layout=layout, entries=entries)

        qB, kvB = q_dim * 4, hd * 4
        layers = []
        for L in range(cfg["num_layers"]):
            sliding = cfg["layer_types"][L] == "sliding_attention"
            rf = rope_l if sliding else rope_g
            sd = D["sc_s"] if sliding else D["sc_f"]
            layers.append({
                "n1": bg("rmsnorm_wg", B["x"], W[f"L{L}.norm_in"], B["xn"], D["norm_h"]),
                "qkv": bg("matvec_wg_packed", W[f"L{L}.qkv"], B["xn"], B["qkv"], D["mv_qkv"]),
                "qn": bg("rmsnorm_wg", (B["qkv"], 0, qB), W[f"L{L}.q_norm"], B["qn"], D["norm_q"]),
                "kn": bg("rmsnorm_wg", (B["qkv"], qB, kvB), W[f"L{L}.k_norm"], B["kn"], D["norm_k"]),
                "rq": bg("rope", B["qn"], rf, D["rope_q"]),
                "rk": bg("rope", B["kn"], rf, D["rope_k"]),
                "ak": bg("kv_append", B["kn"], kc[L], D["kv"]),
                "av": bg("kv_append", (B["qkv"], qB + kvB, kvB), vc[L], D["kv"]),
                "at": bg("attention_fused", B["qn"], kc[L], vc[L], B["scores"], B["attn"], sd),
                "o": bg("matvec_wg_packed", W[f"L{L}.o"], B["attn"], B["attn_proj"], D["mv_o"]),
                "pa": bg("rmsnorm_add_wg", B["attn_proj"], W[f"L{L}.norm_pa"], B["x"], D["norm_h"]),
                "pf": bg("rmsnorm_wg", B["x"], W[f"L{L}.norm_pf"], B["xn"], D["norm_h"]),
                "gu": bg("matvec_wg_packed", W[f"L{L}.gateup"], B["xn"], B["gateup"], D["mv_gu"]),
                "gg": bg("geglu", (B["gateup"], 0, inter * 4), (B["gateup"], inter * 4, inter * 4), B["ffh"], D["geglu"]),
                "dn": bg("matvec_wg_packed", W[f"L{L}.down"], B["ffh"], B["mlp_out"], D["mv_down"]),
                "pff": bg("rmsnorm_add_wg", B["mlp_out"], W[f"L{L}.norm_pff"], B["x"], D["norm_h"]),
            })
        bge = bg("embed_scale_packed", B["token"], W["embed"], B["x"], D["embed"])
        bgf = bg("rmsnorm_wg", B["x"], W["final_norm"], B["xn"], D["norm_h"])
        bgl = bg("matvec_packed", W["embed"], B["xn"], B["logits"], D["mv_logits"])
        bgs = bg("step_setup", B["pos"], D["kv"], D["sc_s"], D["sc_f"], rope_l, rope_g, D["scfg"])
        bga1 = bg("argmax_stage1", B["logits"], B["part_val"], B["part_idx"], D["am1"])
        bga2 = bg("argmax_stage2", B["part_val"], B["part_idx"], B["token"],
                  B["out_tokens"], B["counter"], D["am2"])

        def forward(cp, want_logits):
            def run(name, g, wgs):
                p, _ = pipes[name]
                cp.set_pipeline(p)
                cp.set_bind_group(0, g)
                cp.dispatch_workgroups(wgs, 1, 1)
            run("embed_scale_packed", bge, (h // 2 + 63) // 64)
            for L in layers:
                run("rmsnorm_wg", L["n1"], 1)
                run("matvec_wg_packed", L["qkv"], q_dim + 2 * hd)
                run("rmsnorm_wg", L["qn"], nh)
                run("rmsnorm_wg", L["kn"], 1)
                run("rope", L["rq"], (q_dim // 2 + 63) // 64)
                run("rope", L["rk"], (hd // 2 + 63) // 64)
                run("kv_append", L["ak"], (hd + 63) // 64)
                run("kv_append", L["av"], (hd + 63) // 64)
                run("attention_fused", L["at"], nh)
                run("matvec_wg_packed", L["o"], h)
                run("rmsnorm_add_wg", L["pa"], 1)
                run("rmsnorm_wg", L["pf"], 1)
                run("matvec_wg_packed", L["gu"], 2 * inter)
                run("geglu", L["gg"], (inter + 63) // 64)
                run("matvec_wg_packed", L["dn"], h)
                run("rmsnorm_add_wg", L["pff"], 1)
            if want_logits:
                run("rmsnorm_wg", bgf, 1)
                run("matvec_packed", bgl, (cfg["vocab_size"] + 63) // 64)

        # tokenize the France prompt via the tokenizer directly
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        text = ("<start_of_turn>user\nWhat is the capital of France?"
                "<end_of_turn>\n<start_of_turn>model\n")
        ids = [cfg["bos_token_id"]] + tok.encode(text, add_special_tokens=False).ids

        # prefill (CPU params) — mirrors app.js writeStepParams
        for pos, tid in enumerate(ids[:-1]):
            kv_len = pos + 1
            start = max(0, kv_len - cfg["sliding_window"])
            q.write_buffer(B["token"], 0, np.array([tid], np.uint32).tobytes())
            q.write_buffer(D["kv"], 4, np.array([pos], np.uint32).tobytes())
            q.write_buffer(D["sc_s"], 8, np.array([kv_len, start], np.uint32).tobytes())
            q.write_buffer(D["sc_f"], 8, np.array([kv_len], np.uint32).tobytes())
            q.write_buffer(rope_l, 4, np.array([pos], np.float32).tobytes())
            q.write_buffer(rope_g, 4, np.array([pos], np.float32).tobytes())
            enc = device.create_command_encoder()
            cp = enc.begin_compute_pass()
            forward(cp, False)
            cp.end()
            q.submit([enc.finish()])

        # resident chunk — mirrors app.js generate()
        start_pos = len(ids) - 1
        q.write_buffer(B["token"], 0, np.array([ids[-1]], np.uint32).tobytes())
        q.write_buffer(B["pos"], 0, np.array([start_pos], np.uint32).tobytes())
        q.write_buffer(B["counter"], 0, np.zeros(1, np.uint32).tobytes())
        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        for _ in range(16):
            p, _ = pipes["step_setup"]; cp.set_pipeline(p); cp.set_bind_group(0, bgs); cp.dispatch_workgroups(1, 1, 1)
            forward(cp, True)
            p, _ = pipes["argmax_stage1"]; cp.set_pipeline(p); cp.set_bind_group(0, bga1); cp.dispatch_workgroups(N_AWG, 1, 1)
            p, _ = pipes["argmax_stage2"]; cp.set_pipeline(p); cp.set_bind_group(0, bga2); cp.dispatch_workgroups(1, 1, 1)
        cp.end()
        q.submit([enc.finish()])

        toks = np.frombuffer(q.read_buffer(B["out_tokens"], size=16 * 4), dtype=np.uint32)
        out = []
        for t in toks:
            out.append(int(t))
            if int(t) in cfg["eos_token_ids"]:
                break
        decoded = tok.decode(out)
        assert "Paris" in decoded, decoded
