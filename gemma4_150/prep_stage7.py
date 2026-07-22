"""Stage 7 data prep + numpy oracle for reference kernel 00_main.wgsl
(embed gather + dequant: y[t,c] = EMBED_SCALE * scale[id] * (code - ZP)).
Real embed_tokens weights; tests a handful of token ids from the first
VSUB rows (kernel indexes bits_buf[id*96], so the buffer covers ids < VSUB).
"""
import base64, json
import numpy as np
from gemma4_150.loader import Ckpt

K = 1536
VSUB = 8192
ZP = 2.0
EMBED_SCALE = 39.191835884530846
IDS = [2, 105, 106, 1000, 5000, 8191]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


c = Ckpt()
packed = c._raw("model.language_model.embed_tokens.embedding_quantized")[:VSUB]     # u8 [VSUB,384]
scale = c.f32("model.language_model.embed_tokens.embedding_scale").reshape(-1)[:VSUB].astype(np.float32)

u32 = np.frombuffer(np.ascontiguousarray(packed).tobytes(), np.uint32).reshape(VSUB, 96)
codes = np.zeros((VSUB, K), np.int32)
for v in range(16):
    codes[:, v::16] = (u32 >> (2 * v)) & 3

ids = np.array(IDS, np.uint32)
ref = np.empty((len(IDS), K), np.float32)
for i, tid in enumerate(IDS):
    ref[i] = (EMBED_SCALE * scale[tid] * (codes[tid].astype(np.float64) - ZP)).astype(np.float32)

DST = "/private/tmp/claude-501/-Users-wako-projects-slang/ce5a80c9-3495-4ae0-8cd2-813d9c5d619f/scratchpad/stage7.json"
json.dump({
    "ids": b64(ids), "bits": b64(u32), "scale": b64(scale),
    "ref": b64(ref.reshape(-1)), "seq": len(IDS), "K": K,
}, open(DST, "w"))
print(f"embed ids={IDS} y[0,:4]={ref[0,:4].round(5)} y[1,:4]={ref[1,:4].round(5)}  -> {DST}")
