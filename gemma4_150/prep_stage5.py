"""Stage 5 data prep + numpy oracle for reference kernel 101_srq.wgsl
(flash-decode attention: fused q-RMSNorm + split-half RoPE + causal softmax +
V accumulation + o-proj-input SRQ, two-pass with same-dispatch atomic merge).

Kernel is baked for a FULL layer (HEAD_DIM=512, window=0, OUT_Q =
0.0148868 = layer-19 o_proj in_scale). We synthesize q / k-cache / v-cache and
use the real layer-19 q_norm weight. Oracle is plain (non-flash) causal
attention; GPU uses chunked online-softmax flash, so we expect a match modulo
rare 1-quantum (OUT_Q) SRQ boundary flips, which we flag explicitly.
"""
import base64, json
import numpy as np
from gemma4_150.loader import Ckpt

LAYER = 19
HEAD_DIM = 512
HALF = 256
QH = 8
EPS = 1e-6
OUT_Q = 0.014886821620166302
KEYLEN = 90                      # decode context; qPos = KEYLEN-1 (causal, full)
CUTOFF = 64                      # p-RoPE active freq pairs on full layers


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def srq(x, s):
    return np.clip(np.round(x / s), -128.0, 127.0) * s if s != 0.0 else x


c = Ckpt()
w = c.f32(f"model.language_model.layers.{LAYER}.self_attn.q_norm.weight").astype(np.float32)  # [512]

rng = np.random.default_rng(101)
q = (rng.standard_normal(QH * HEAD_DIM).astype(np.float32) * 0.7)
k = (rng.standard_normal(KEYLEN * HEAD_DIM).astype(np.float32) * 0.15)   # [keyLen,512] flat, kvHeads=1
v = (rng.standard_normal(KEYLEN * HEAD_DIM).astype(np.float32) * 0.5)

# p-RoPE tables: real rotation for p<CUTOFF, identity (cos=1,sin=0) beyond.
cos = np.ones(HALF, np.float32); sin = np.zeros(HALF, np.float32)
inv = 1.0 / (1_000_000.0 ** (np.arange(CUTOFF) / CUTOFF))
ang = (KEYLEN - 1) * inv                          # angle at qPos (query rotates by its own pos)
cos[:CUTOFF] = np.cos(ang).astype(np.float32); sin[:CUTOFF] = np.sin(ang).astype(np.float32)

# --- oracle: plain causal attention (float32) ---
qPos = KEYLEN - 1
out = np.zeros(QH * HEAD_DIM, np.float32)
kk = k.reshape(KEYLEN, HEAD_DIM); vv = v.reshape(KEYLEN, HEAD_DIM)
for h in range(QH):
    qh = q[h * HEAD_DIM:(h + 1) * HEAD_DIM]
    nscale = np.float32(1.0 / np.sqrt((qh * qh).mean() + EPS))
    qn = np.empty(HEAD_DIM, np.float32)
    n0 = qh[:HALF] * nscale * w[:HALF]
    n1 = qh[HALF:] * nscale * w[HALF:]
    qn[:HALF] = n0 * cos - n1 * sin
    qn[HALF:] = n1 * cos + n0 * sin
    scores = (kk[:qPos + 1] @ qn).astype(np.float32)        # [90]
    m = scores.max()
    p = np.exp(scores - m).astype(np.float32)
    denom = p.sum()
    o = (p @ vv[:qPos + 1]) / denom                          # [512]
    out[h * HEAD_DIM:(h + 1) * HEAD_DIM] = srq(o.astype(np.float32), OUT_Q)

DST = "/private/tmp/claude-501/-Users-wako-projects-slang/ce5a80c9-3495-4ae0-8cd2-813d9c5d619f/scratchpad/stage5.json"
json.dump({
    "q": b64(q), "w": b64(w), "cos": b64(cos), "sin": b64(sin),
    "k": b64(k), "v": b64(v), "ref": b64(out),
    "keyLen": KEYLEN, "qPos": qPos, "qHeads": QH, "kvHeads": 1, "window": 0,
    "OUT_Q": OUT_Q,
}, open(DST, "w"))
print(f"L{LAYER} attn: qPos={qPos} keyLen={KEYLEN} out[:4]={out[:4].round(5)}")
print(f"nonzero out={int((out!=0).sum())}/{QH*HEAD_DIM}  -> {DST}")
