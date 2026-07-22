"""Stage 6 data prep + numpy oracle for reference kernel 33_srq.wgsl
(dense QAT GEMV logits: thread-per-column, block-major 2-bit weights, presrq
activation tile). Uses REAL lm_head weights/scales; validates a 4096-column
vocab subset (the shader's N/GRID_X are patched down to 4096 in the validator —
identical code path, block-major stride blk*N+col preserved, just less data).
"""
import base64, json
import numpy as np
from gemma4_150.loader import Ckpt

K = 1536          # hidden
NSUB = 4096       # validated vocab subset
ZP = 2.0
BLK = 24          # NUM_BLK (K / 64)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def srq(x, s):
    return np.clip(np.round(x / s), -128.0, 127.0) * s if s != 0.0 else x


c = Ckpt()
packed = c._raw("lm_head.weight")[:NSUB]                       # u8 [NSUB, 384]
scale = c.f32("lm_head.weight_scale").reshape(-1)[:NSUB].astype(np.float32)   # [NSUB]
inScale = float(c.f32("lm_head.input_activation_scale"))
outScale = float(c.f32("lm_head.output_activation_scale"))

# codes: LSB-first 2-bit unpack -> [NSUB, 1536]
u32 = np.frombuffer(np.ascontiguousarray(packed).tobytes(), np.uint32).reshape(NSUB, 96)
codes = np.zeros((NSUB, K), np.int32)
for v in range(16):
    codes[:, v::16] = (u32 >> (2 * v)) & 3

# block-major repack: bits_buf[blk, col, 0:4] = u32[col, blk*4 : blk*4+4]
bits = u32.reshape(NSUB, BLK, 4).transpose(1, 0, 2).copy()     # [24, NSUB, 4] u32

rng = np.random.default_rng(33)
a = (rng.standard_normal(K).astype(np.float32) * 0.4)

# oracle: logits[col] = srq(scale[col] * sum_k (code-ZP)*srq(a[k],inScale), outScale)
a_srq = srq(a.astype(np.float32), inScale).astype(np.float64)
gemv = codes.astype(np.float64) @ a_srq - ZP * a_srq.sum()     # [NSUB]
logits = srq((scale.astype(np.float64) * gemv), outScale).astype(np.float32)

DST = "/private/tmp/claude-501/-Users-wako-projects-slang/ce5a80c9-3495-4ae0-8cd2-813d9c5d619f/scratchpad/stage6.json"
json.dump({
    "a": b64(a.astype(np.float32)), "bits": b64(bits), "scale": b64(scale),
    "ref": b64(logits), "N": NSUB, "K": K, "inScale": inScale, "outScale": outScale,
}, open(DST, "w"))
print(f"lm_head logits[:4]={logits[:4].round(4)}  inScale={inScale:.5g} outScale={outScale:.5g}")
print(f"bits {bits.shape} {bits.nbytes//1024}KB  -> {DST}")
