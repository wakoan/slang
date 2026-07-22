"""GPU runner for the gemma4_150 fast-path port (wgpu-py).

Reads the exported manifest + weights.bin, uploads every buffer, and drives the
webml reference fused SRQ kernels (templated per layer shape) to decode Gemma 4
E2B. Correctness gate: argmax == gemma4_150.reference (numpy SRQ oracle).

wgpu-native/naga compiles the reference kernels once the `enable subgroups;`
directive is stripped (the ops are supported; same shim as gemma4/runner.py).

This is built up stage by stage; each stage validates a GPU intermediate against
reference.py before the next is wired.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import wgpu

REF_DIR = Path(__file__).resolve().parent.parent / "reference" / "webml_gemma4_kernels"
_BIND = re.compile(r"@group\(0\)\s*@binding\((\d+)\)\s*var<(storage|uniform)(?:,\s*(read|read_write))?>")


def _kernel(name: str) -> str:
    return (REF_DIR / f"{name}.wgsl").read_text()


def _access(wgsl: str):
    """Ordered per-binding kind: 'r' read-only storage, 'w' read_write, 'u' uniform."""
    slots = {}
    for m in _BIND.finditer(wgsl):
        b, cls, acc = int(m.group(1)), m.group(2), m.group(3)
        slots[b] = "u" if cls == "uniform" else ("w" if acc == "read_write" else "r")
    return [slots[i] for i in range(len(slots))]


class G4Runner:
    def __init__(self, model_dir=None):
        base = Path(model_dir) if model_dir else \
            Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B-qat"
        self.dir = base / "g4_150"
        self.man = json.loads((self.dir / "manifest.json").read_text())
        self.cfg = self.man["config"]
        self.blob = np.memmap(self.dir / "weights.bin", np.uint8, "r")

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        feats = [f for f in ("subgroup", "shader-f16") if f in adapter.features]
        big = 1 << 31   # 2GB: covers the 1.17GB ple_q buffer
        self.device = adapter.request_device_sync(
            required_features=feats,
            required_limits={"max-buffer-size": big, "max-storage-buffer-binding-size": big,
                             "max-storage-buffers-per-shader-stage": 10})
        self.queue = self.device.queue
        self._pipes: dict[str, tuple] = {}
        self._bufs: dict[str, object] = {}
        self._upload_all()

    # ---- buffers ----
    def _upload_all(self):
        st = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        pending = 0
        for name, t in self.man["tensors"].items():
            data = bytes(self.blob[t["off"]:t["off"] + t["len"]])
            self._bufs[name] = self.device.create_buffer_with_data(data=data, usage=st)
            pending += t["len"]
            if pending >= (1 << 28):        # flush every 256MB (staging pitfall)
                self.queue.submit([])
                pending = 0
        self.queue.submit([])

    def _tmp(self, nbytes, src=True):
        u = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        if src:
            u |= wgpu.BufferUsage.COPY_SRC
        return self.device.create_buffer(size=max(nbytes, 4), usage=u)

    def _uniform(self, arr: np.ndarray):
        return self.device.create_buffer_with_data(
            data=arr.tobytes(), usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    # ---- pipelines (templated: patch baked consts before compile) ----
    def _shader(self, code: str):
        try:
            return self.device.create_shader_module(code=code)
        except Exception:
            return self.device.create_shader_module(code=code.replace("enable subgroups;", ""))

    def _pipe(self, key: str, code: str):
        if key not in self._pipes:
            acc = _access(code)
            entries = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": wgpu.BufferBindingType.uniform if a == "u"
                                   else wgpu.BufferBindingType.storage if a == "w"
                                   else wgpu.BufferBindingType.read_only_storage}}
                       for i, a in enumerate(acc)]
            layout = self.device.create_bind_group_layout(entries=entries)
            pipe = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[layout]),
                compute={"module": self._shader(code), "entry_point": "main"})
            self._pipes[key] = (pipe, layout)
        return self._pipes[key]

    def dispatch(self, key, code, buffers, grid):
        pipe, layout = self._pipe(key, code)
        bg = self.device.create_bind_group(layout=layout, entries=[
            {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
            for i, b in enumerate(buffers)])
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(*grid)
        cp.end()
        self.queue.submit([enc.finish()])

    def read(self, buf, nbytes) -> np.ndarray:
        rb = self.device.create_buffer(
            size=nbytes, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
        enc = self.device.create_command_encoder()
        enc.copy_buffer_to_buffer(buf, 0, rb, 0, nbytes)
        self.queue.submit([enc.finish()])
        rb.map_sync(wgpu.MapMode.READ)
        out = np.frombuffer(rb.read_mapped(), np.uint8).copy()
        rb.unmap()
        return out

    # ---- stage 1: embed gather (kernel 00) ----
    def embed(self, token_id: int) -> np.ndarray:
        H = self.cfg["H"]
        ids = self.device.create_buffer_with_data(
            data=np.array([token_id], np.uint32).tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        y = self._tmp(H * 4)
        par = self._uniform(np.array([1, 0, 0, 0], np.uint32))
        self.dispatch("embed", _kernel("00_main"),
                      [ids, self._bufs["embed_q"], self._bufs["embed_scale"], y, par], (1, 1, 1))
        return self.read(y, H * 4).view(np.float32)


if __name__ == "__main__":
    import sys
    from gemma4_150.reference import ReferenceSRQ
    r = G4Runner()
    ref = ReferenceSRQ()
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    g = r.embed(tid)
    e = ref.embed(tid)
    print(f"embed[{tid}]: gpu[:4]={g[:4].round(4)} ref[:4]={e[:4].round(4)} "
          f"maxAbsDiff={np.abs(g - e).max():.3e}")
