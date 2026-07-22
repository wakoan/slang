"""Stage 10 data prep + numpy oracle for reference kernel 01_main.wgsl
(4-bit per-layer-embedding gather: y[t,c] = 16 * scale[id, c/256] * (code - 8),
HIDDEN=8960 = 35 groups x 256, per-(row,group) scale). Real
embed_tokens_per_layer weights; a few token ids from the first VSUB rows.
"""
import base64, json
import numpy as np
from gemma4_150.loader import Ckpt

HID = 8960
GROUP = 256
NG = 35
VSUB = 8192
ZP = 8.0
EMBED_SCALE = 16.0
IDS = [2, 105, 106, 1000, 8191]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


c = Ckpt()
packed = c._raw("model.language_model.embed_tokens_per_layer.embedding_quantized")[:VSUB]  # u8 [VSUB,4480]
scale = c.f32("model.language_model.embed_tokens_per_layer.embedding_scale")[:VSUB].astype(np.float32)  # [VSUB,35]

u32 = np.frombuffer(np.ascontiguousarray(packed).tobytes(), np.uint32).reshape(VSUB, 1120)
codes = np.zeros((VSUB, HID), np.int32)
for v in range(8):                      # 8 4-bit vals per u32, LSB-first
    codes[:, v::8] = (u32 >> (4 * v)) & 15

ids = np.array(IDS, np.uint32)
ref = np.empty((len(IDS), HID), np.float32)
for i, tid in enumerate(IDS):
    g = np.arange(HID) // GROUP                                   # group per col
    ref[i] = (EMBED_SCALE * scale[tid, g] * (codes[tid].astype(np.float64) - ZP)).astype(np.float32)

DST = "/private/tmp/claude-501/-Users-wako-projects-slang/ce5a80c9-3495-4ae0-8cd2-813d9c5d619f/scratchpad/stage10.json"
json.dump({
    "ids": b64(ids), "bits": b64(u32), "scale": b64(scale),
    "ref": b64(ref.reshape(-1)), "seq": len(IDS), "HID": HID,
}, open(DST, "w"))
print(f"PLE gather ids={IDS} y[0,:4]={ref[0,:4].round(5)} y[0,256:260]={ref[0,256:260].round(5)}  -> {DST}")
