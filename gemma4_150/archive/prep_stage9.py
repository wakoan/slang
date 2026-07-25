"""Stage 9 data prep + numpy oracle for reference kernel 76_reduce.wgsl
(fused per-layer-input gate: int8 +128-biased GEMV 1536->256 via unpack4x8unorm,
then GELU-LUT gate x per-layer embedding).
  out[o] = gelu_grid(srq(w.a, linOutScale), linOutScale) * ple[pleOffset + o]
Real layer-3 per_layer_input_gate weights/scales; ple gathered slice synthetic.
"""
import base64, json
import numpy as np
from gemma4_150.loader import Ckpt

LAYER = 3
IN_F = 1536
OUT_F = 256
PLE_OFFSET = LAYER * OUT_F        # this layer's 256-slice of the 8960 PLE table


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def srq(x, s):
    return np.clip(np.round(x / s), -128.0, 127.0) * s if s != 0.0 else x


def gelu_tanh(v):
    return 0.5 * v * (1.0 + np.tanh(0.7978845608028654 * (v + 0.044715 * v ** 3)))


c = Ckpt()
b = f"model.language_model.layers.{LAYER}.per_layer_input_gate."
wi8 = c._raw(b + "weight").astype(np.int16)                      # [256,1536]
row_scale = c.f32(b + "weight_scale").reshape(-1).astype(np.float32)
inScale = float(c.f32(b + "input_activation_scale"))
linOut = float(c.f32(b + "output_activation_scale"))

biased = (wi8 + 128).astype(np.uint8)
codes = np.frombuffer(np.ascontiguousarray(biased).tobytes(), np.uint32).reshape(OUT_F, IN_F // 4)

# gelu LUT: host-f64 table over the srq grid, lut[i] = gelu_tanh((i-128)*linOut)
lut = gelu_tanh((np.arange(256) - 128).astype(np.float64) * linOut).astype(np.float32)

rng = np.random.default_rng(76)
a = (rng.standard_normal(IN_F).astype(np.float32) * 0.5)
ple_full = (rng.standard_normal((LAYER + 1) * OUT_F).astype(np.float32) * 1.5)   # covers pleOffset+o

# --- oracle ---
a_srq = srq(a.astype(np.float32), inScale).astype(np.float64)
u = biased.astype(np.float64) / 255.0
s = u @ a_srq                                                   # [256]
v = row_scale.astype(np.float64) * (255.0 * s - 128.0 * a_srq.sum())
idx = np.clip(np.round(v / linOut), -128, 127).astype(np.int64) + 128
g = lut[idx].astype(np.float64)
out = (g * ple_full[PLE_OFFSET:PLE_OFFSET + OUT_F].astype(np.float64)).astype(np.float32)

DST = "/private/tmp/claude-501/-Users-wako-projects-slang/ce5a80c9-3495-4ae0-8cd2-813d9c5d619f/scratchpad/stage9.json"
json.dump({
    "a": b64(a.astype(np.float32)), "codes": b64(codes), "row_scale": b64(row_scale),
    "ple": b64(ple_full), "lut": b64(lut), "ref": b64(out),
    "OUT_F": OUT_F, "inScale": inScale, "linOut": linOut, "pleOffset": PLE_OFFSET,
}, open(DST, "w"))
print(f"L{LAYER} gate out[:4]={out[:4].round(5)}  inScale={inScale:.5g} linOut={linOut:.5g} pleOff={PLE_OFFSET}")
print(f"  -> {DST}")
