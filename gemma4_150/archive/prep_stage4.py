"""Stage 4 data prep + numpy oracle for reference kernel 73_sg_sum.wgsl
(o-proj QAT GEMV + post-attn residual norm-add + pre-FFN norm + SRQ).

Dumps stage4.json (base64 buffers + params + oracle outputs) for
validate_stage4.mjs to check the GPU kernel bit-exact. Uses real layer-3
weights (sliding layer: o_proj n_in=2048, matching the kernel's baked
IN_FEATURES=2048).
"""
import base64, json
import numpy as np
from gemma4_150.loader import Ckpt

LAYER = 3
EPS = 1e-6
ZP = 8.0
OUT_F = 1536
IN_F = 2048


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def srq(x, s):
    if s == 0.0:
        return x
    return np.clip(np.round(x / s), -128.0, 127.0) * s


c = Ckpt()
o_lin = c.linear(LAYER, "self_attn.o_proj")
codes = c.codes(o_lin).astype(np.float64)               # [1536, 2048] unsigned
scale = o_lin["wscale"].astype(np.float32)              # [1536]
outScale = np.float32(o_lin["out_scale"])
o_in_scale = np.float32(o_lin["in_scale"])
inScale2 = np.float32(c.linear(LAYER, "mlp.gate_proj")["in_scale"])

w1 = c.f32(f"model.language_model.layers.{LAYER}.post_attention_layernorm.weight").astype(np.float32)  # [1536]
w2 = c.f32(f"model.language_model.layers.{LAYER}.pre_feedforward_layernorm.weight").astype(np.float32)  # [1536]
w12 = np.concatenate([w1, w2]).astype(np.float32)       # [3072]

# bits_buf: checkpoint packed uint8 [1536,1024] reinterpreted as u32 [1536,256]
# is exactly the kernel's layout (8 consecutive LSB-first 4-bit codes / word).
bits = np.frombuffer(np.ascontiguousarray(o_lin["packed"]).tobytes(), np.uint32).reshape(OUT_F, IN_F // 8)

rng = np.random.default_rng(73)
a = srq(rng.standard_normal(IN_F).astype(np.float32) * 0.5, o_in_scale).astype(np.float32)
hidden0 = (rng.standard_normal(OUT_F).astype(np.float32) * 0.1).astype(np.float32)

# --- oracle ---
a64 = a.astype(np.float64)
gemv = codes @ a64 - ZP * a64.sum()                     # dot(code-ZP, a)
o = srq((scale.astype(np.float64) * gemv), float(outScale))   # [1536]

rms1 = 1.0 / np.sqrt((o * o).mean() + EPS)
normed = o * rms1 * w1.astype(np.float64)
hp = hidden0.astype(np.float64) + normed                # hidden after add

rms2 = 1.0 / np.sqrt((hp * hp).mean() + EPS)
n2 = hp * rms2 * w2.astype(np.float64)
n2_16 = n2.astype(np.float16).astype(np.float64)
y2 = srq(n2_16, float(inScale2)).astype(np.float16)     # stored f16
sum2 = float(y2.astype(np.float32).sum())

out = {
    "a": b64(a), "bits": b64(bits), "scale": b64(scale), "w12": b64(w12),
    "hidden": b64(hidden0), "OUT_F": OUT_F,
    "outScale": float(outScale), "inScale2": float(inScale2),
    "ref_hidden": b64(hp.astype(np.float32)),
    "ref_y2": b64(y2), "ref_sum2": sum2,
}
DST = "/private/tmp/claude-501/-Users-wako-projects-slang/ce5a80c9-3495-4ae0-8cd2-813d9c5d619f/scratchpad/stage4.json"
json.dump(out, open(DST, "w"))
print(f"L{LAYER} o_proj: gemv[:3]={gemv[:3].round(3)} o[:3]={o[:3].round(4)}")
print(f"hp[:4]={hp[:4].round(5)}  sum2={sum2:.4f}  y2[:4]={y2[:4]}")
print(f"outScale={outScale:.5g} inScale2={inScale2:.5g}  -> {DST}")
