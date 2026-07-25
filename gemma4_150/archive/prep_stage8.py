"""Stage 8 data prep + numpy oracle for reference kernel 77_sg_sum.wgsl
(PLE projection 256->1536, int8 +128-biased codes via unpack4x8unorm, +
post-PLE residual norm-add*sv + next-layer norm + SRQ, single-dispatch with
atomic last-arriver tail). Real layer-3 per_layer_projection weights/scales.

The projection weight is stored signed I8; the kernel decodes u8/255 then undoes
a +128 bias, so codes upload = (i8 + 128) as u8. sv (post-PLE residual scale)
is a runner-supplied scalar; validated here with a fixed non-trivial value.
"""
import base64, json
import numpy as np
from gemma4_150.loader import Ckpt

LAYER = 3
IN_F = 256
OUT_F = 1536
EPS = 1e-6
SV = 0.9                # placeholder post-PLE residual scale (runner supplies real one)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def srq(x, s):
    return np.clip(np.round(x / s), -128.0, 127.0) * s if s != 0.0 else x


c = Ckpt()
b = f"model.language_model.layers.{LAYER}."
wi8 = c._raw(b + "per_layer_projection.weight").astype(np.int16)          # [1536,256] signed
row_scale = c.f32(b + "per_layer_projection.weight_scale").reshape(-1).astype(np.float32)
projIn = float(c.f32(b + "per_layer_projection.input_activation_scale"))
projOut = float(c.f32(b + "per_layer_projection.output_activation_scale"))
inScale = float(c.linear(4, "self_attn.q_proj")["in_scale"])              # next-layer q input srq
w1 = c.f32(b + "post_per_layer_input_norm.weight").astype(np.float32)     # [1536]
w2 = c.f32("model.language_model.layers.4.input_layernorm.weight").astype(np.float32)  # next layer

biased = (wi8 + 128).astype(np.uint8)                                     # u8 = i8 + 128
codes = np.frombuffer(np.ascontiguousarray(biased).tobytes(), np.uint32).reshape(OUT_F, IN_F // 4)
w12s = np.concatenate([w1, w2, np.array([SV], np.float32)]).astype(np.float32)  # [3073]

rng = np.random.default_rng(77)
a = (rng.standard_normal(IN_F).astype(np.float32) * 0.3)
hidden0 = (rng.standard_normal(OUT_F).astype(np.float32) * 0.2)

# --- oracle ---
a_srq = srq(a.astype(np.float32), projIn).astype(np.float64)
u = biased.astype(np.float64) / 255.0                                     # unpack4x8unorm decode
s = u @ a_srq                                                            # [1536]
aSum = a_srq.sum()
proj = srq(row_scale.astype(np.float64) * (255.0 * s - 128.0 * aSum), projOut)

rms1 = 1.0 / np.sqrt((proj * proj).mean() + EPS)
normed = proj * rms1 * w1.astype(np.float64)
hp = (hidden0.astype(np.float64) + normed) * SV
rms2 = 1.0 / np.sqrt((hp * hp).mean() + EPS)
n2 = hp * rms2 * w2.astype(np.float64)
y2 = srq(n2.astype(np.float32), inScale).astype(np.float32)
sum2 = float(y2.sum())

DST = "/private/tmp/claude-501/-Users-wako-projects-slang/ce5a80c9-3495-4ae0-8cd2-813d9c5d619f/scratchpad/stage8.json"
json.dump({
    "a": b64(a.astype(np.float32)), "codes": b64(codes), "row_scale": b64(row_scale),
    "hidden": b64(hidden0.astype(np.float32)), "w12s": b64(w12s), "OUT_F": OUT_F,
    "inScale": inScale, "projIn": projIn, "projOut": projOut,
    "ref_hidden": b64(hp.astype(np.float32)), "ref_y2": b64(y2), "ref_sum2": sum2,
}, open(DST, "w"))
print(f"L{LAYER} PLE proj[:3]={proj[:3].round(4)} hp[:4]={hp[:4].round(5)} sum2={sum2:.4f}")
print(f"projIn={projIn:.5g} projOut={projOut:.5g} inScale={inScale:.5g}  -> {DST}")
