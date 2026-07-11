// tensorscope: step-by-step Gemma inference in WebGPU with capture of
// every intermediate tensor, visualized on an interactive canvas heatmap.
// Standalone page; shares the server's kernels/weights/tokenizer endpoints.

const $ = id => document.getElementById(id);
const ui = {
  status: $("status"), prompt: $("prompt"),
  tokenizeBtn: $("tokenize"), stepBtn: $("step"),
  stepInfo: $("stepinfo"), preds: $("preds"), runPromptBtn: $("runprompt"),
  list: $("tensorlist"), canvas: $("canvas"),
  tooltip: $("tooltip"), stats: $("stats"),
  widthSel: $("widthsel"), scaleSel: $("scalesel"), zoomSel: $("zoomsel"),
  viewSel: $("viewsel"), tokens: $("tokchips"),
  headSel: $("headsel"), headCtl: $("headctl"),
  prevTok: $("prevtok"), nextTok: $("nexttok"),
  legend: $("legend"), hist: $("hist"),
  layerSel: $("layersel"), diagram: $("diagram"),
};
const ctx2d = ui.canvas.getContext("2d");
const legendCtx = ui.legend.getContext("2d");
const histCtx = ui.hist.getContext("2d");
let histView = null;

const MAX_SEQ = 1024;
let G = null;            // gpu state
let CAPS = [];           // {key, group, srcName, floats, offset, defCols, kind}
let capIndex = {};       // key -> cap
let lastCap = null;      // Float32Array of the whole arena after a step
let lastKvLen = 0;
let promptIds = [], pos = 0, nextTok = null;
// per-token capture history: {pos, tid, piece, kvLen, cap} — cap excludes
// logits (1MB/step); logits are kept for the latest step only (lastCap)
let history = [], viewStepIdx = -1;

function status(m) { ui.status.textContent = m; }

// ------------------------------------------------------------------ init

async function init() {
  if (!navigator.gpu) { status("WebGPU not available — use Chrome/Edge."); return; }
  const [kernels, manifest] = await Promise.all([
    fetch("/kernels.json").then(r => r.json()),
    fetch("/manifest.json").then(r => r.json()),
  ]);
  const cfg = manifest.config;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  const embedBytes = cfg.vocab_size * cfg.hidden_size * 2;
  const need = Math.max(embedBytes, 1 << 28);
  const device = await adapter.requestDevice({
    requiredLimits: { maxBufferSize: need, maxStorageBufferBindingSize: need },
  });

  // weights (reuse the gendemo cache when present)
  const cacheKey = `/weights.bin?v=${manifest.weightsVersion ?? "0"}`;
  let blob = null;
  try {
    const cache = await caches.open("gemma-weights");
    const hit = await cache.match(cacheKey);
    if (hit) { status("weights from browser cache…"); blob = new Uint8Array(await hit.arrayBuffer()); }
  } catch { /* no cache */ }
  if (!blob) {
    status("downloading weights…");
    blob = new Uint8Array(await (await fetch("/weights.bin")).arrayBuffer());
  }
  status("uploading weights…");
  const W = {};
  for (const t of manifest.tensors) {
    const buf = device.createBuffer({ size: t.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, blob.buffer, t.offset, t.byteLength);
    W[t.name] = buf;
  }

  // pipelines
  const pipelines = {};
  for (const [name, k] of Object.entries(kernels)) {
    const access = [...k.wgsl.matchAll(/@binding\((\d+)\) var<storage, (read_write|read)>/g)]
      .sort((a, b) => +a[1] - +b[1]).map(m => m[2] === "read_write");
    const layout = device.createBindGroupLayout({
      entries: access.map((rw, i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: rw ? "storage" : "read-only-storage" } })),
    });
    pipelines[name] = {
      layout,
      pipeline: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: device.createShaderModule({ code: k.wgsl }), entryPoint: name },
      }),
    };
  }

  // scratch buffers (COPY_SRC everywhere so any of them can be captured)
  const h = cfg.hidden_size, hd = cfg.head_dim, nh = cfg.num_heads;
  const inter = cfg.intermediate_size, qDim = nh * hd;
  const fbuf = n => device.createBuffer({ size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const B = {
    x: fbuf(h), xn: fbuf(h), qkv: fbuf(qDim + 2 * hd), qn: fbuf(qDim), kn: fbuf(hd),
    scores: fbuf(nh * MAX_SEQ), attn: fbuf(qDim), attnProj: fbuf(h),
    gateup: fbuf(2 * inter), ffh: fbuf(inter), mlpOut: fbuf(h),
    logits: fbuf(cfg.vocab_size), token: fbuf(1),
  };
  const kCache = [], vCache = [];
  for (let L = 0; L < cfg.num_layers; L++) { kCache.push(fbuf(MAX_SEQ * hd)); vCache.push(fbuf(MAX_SEQ * hd)); }

  const ubuf = (...vals) => {
    const b = device.createBuffer({ size: vals.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, new Uint32Array(vals));
    return b;
  };
  const D = {
    embed: ubuf(h), normH: ubuf(1, h), normQ: ubuf(nh, hd), normK: ubuf(1, hd),
    mvQkv: ubuf(qDim + 2 * hd, h), mvO: ubuf(h, qDim), mvGateup: ubuf(2 * inter, h),
    mvDown: ubuf(h, inter), mvLogits: ubuf(cfg.vocab_size, h),
    ropeQ: ubuf(nh, hd), ropeK: ubuf(1, hd), geglu: ubuf(inter),
    kvAppend: ubuf(hd, 0), scSlide: ubuf(nh, hd, 1, 0, MAX_SEQ), scFull: ubuf(nh, hd, 1, 0, MAX_SEQ),
  };
  const ropeLocal = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const ropeGlobal = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(ropeLocal, 0, new Float32Array([cfg.rope_theta_local, 0]));
  device.queue.writeBuffer(ropeGlobal, 0, new Float32Array([cfg.rope_theta_global, 0]));

  const bg = (name, ...bufs) => device.createBindGroup({
    layout: pipelines[name].layout,
    entries: bufs.map((b, i) => ({ binding: i,
      resource: Array.isArray(b) ? { buffer: b[0], offset: b[1], size: b[2] } : { buffer: b } })),
  });
  const qB = qDim * 4, kvB = hd * 4;
  const layers = [];
  for (let L = 0; L < cfg.num_layers; L++) {
    const sliding = cfg.layer_types[L] === "sliding_attention";
    const ropeF = sliding ? ropeLocal : ropeGlobal;
    const scD = sliding ? D.scSlide : D.scFull;
    layers.push({
      norm1: bg("rmsnorm_wg", B.x, W[`L${L}.norm_in`], B.xn, D.normH),
      mvQkv: bg("matvec_wg_packed", W[`L${L}.qkv`], B.xn, B.qkv, D.mvQkv),
      qnorm: bg("rmsnorm_wg", [B.qkv, 0, qB], W[`L${L}.q_norm`], B.qn, D.normQ),
      knorm: bg("rmsnorm_wg", [B.qkv, qB, kvB], W[`L${L}.k_norm`], B.kn, D.normK),
      ropeQ: bg("rope", B.qn, ropeF, D.ropeQ),
      ropeK: bg("rope", B.kn, ropeF, D.ropeK),
      appK: bg("kv_append", B.kn, kCache[L], D.kvAppend),
      appV: bg("kv_append", [B.qkv, qB + kvB, kvB], vCache[L], D.kvAppend),
      attn: bg("attention_fused", B.qn, kCache[L], vCache[L], B.scores, B.attn, scD),
      mvO: bg("matvec_wg_packed", W[`L${L}.o`], B.attn, B.attnProj, D.mvO),
      normPaAdd: bg("rmsnorm_add_wg", B.attnProj, W[`L${L}.norm_pa`], B.x, D.normH),
      normPf: bg("rmsnorm_wg", B.x, W[`L${L}.norm_pf`], B.xn, D.normH),
      mvGateup: bg("matvec_wg_packed", W[`L${L}.gateup`], B.xn, B.gateup, D.mvGateup),
      geglu: bg("geglu", [B.gateup, 0, inter * 4], [B.gateup, inter * 4, inter * 4], B.ffh, D.geglu),
      mvDown: bg("matvec_wg_packed", W[`L${L}.down`], B.ffh, B.mlpOut, D.mvDown),
      normPffAdd: bg("rmsnorm_add_wg", B.mlpOut, W[`L${L}.norm_pff`], B.x, D.normH),
    });
  }
  const bgEmbed = bg("embed_scale_packed", B.token, W.embed, B.x, D.embed);
  const bgFinalNorm = bg("rmsnorm_wg", B.x, W.final_norm, B.xn, D.normH);
  const bgLogits = bg("matvec_packed", W.embed, B.xn, B.logits, D.mvLogits);

  // --- capture arena layout ---
  CAPS = []; capIndex = {}; let off = 0;
  // dims: semantic axes [label, size]; 1-D tensors stay reshapeable via
  // the width control, 2-D tensors render with fixed labeled axes
  const addCap = (key, group, src, floats, dims, defCols, kind = "plain") => {
    const c = { key, group, src, floats, offset: off, dims, defCols, kind };
    CAPS.push(c); capIndex[key] = c; off += floats * 4;
  };
  addCap("embed", "global", "x", h, [["d", h]], 32);
  for (let L = 0; L < cfg.num_layers; L++) {
    const g = `layer ${L}`;
    addCap(`L${L}.norm_in`, g, "xn", h, [["d", h]], 32);
    addCap(`L${L}.qkv`, g, "qkv", qDim + 2 * hd, [["q|k|v", qDim + 2 * hd]], 64);
    addCap(`L${L}.q`, g, "qn", qDim, [["head", nh], ["d", hd]]);
    addCap(`L${L}.k`, g, "kn", hd, [["head", 1], ["d", hd]]);
    addCap(`L${L}.attn_probs`, g, "scores", nh * MAX_SEQ, null, null, "attn");
    addCap(`L${L}.attn_out`, g, "attn", qDim, [["head", nh], ["d", hd]]);
    addCap(`L${L}.o_proj`, g, "attnProj", h, [["d", h]], 32);
    addCap(`L${L}.hidden_attn`, g, "x", h, [["d", h]], 32);
    addCap(`L${L}.norm_ff`, g, "xn", h, [["d", h]], 32);
    addCap(`L${L}.gateup`, g, "gateup", 2 * inter, [["gate/up", 2], ["ff", inter]]);
    addCap(`L${L}.geglu`, g, "ffh", inter, [["ff", inter]], 64);
    addCap(`L${L}.mlp_out`, g, "mlpOut", h, [["d", h]], 32);
    addCap(`L${L}.hidden`, g, "x", h, [["d", h]], 32);
  }
  addCap("final_norm", "global", "xn", h, [["d", h]], 32);
  addCap("logits", "global", "logits", cfg.vocab_size, [["vocab", cfg.vocab_size]], 512);
  const arena = device.createBuffer({ size: off,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const staging = device.createBuffer({ size: off,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  G = { device, cfg, pipelines, W, B, D, layers, kCache, vCache,
        ropeLocal, ropeGlobal, bgEmbed, bgFinalNorm, bgLogits, arena, staging,
        arenaBytes: off };

  buildTensorList();
  ui.layerSel.innerHTML = "";
  for (let L = 0; L < cfg.num_layers; L++) {
    const o = document.createElement("option");
    o.value = o.textContent = L;
    ui.layerSel.appendChild(o);
  }
  buildDiagram();
  ui.layerSel.addEventListener("change", updateDiagram);
  for (let hI = 0; hI < cfg.num_heads; hI++) {
    const o = document.createElement("option");
    o.value = o.textContent = hI;
    ui.headSel.appendChild(o);
  }
  status(`ready — ${CAPS.length} capturable tensors, ${(off / 1e6).toFixed(1)} MB per step`);
  ui.tokenizeBtn.disabled = false;
}

// ---------------------------------------------------- layer diagram (SVG)

const SVGNS = "http://www.w3.org/2000/svg";
// [suffix, label, x, y, width?] — one transformer layer, top to bottom
const DIAG_NODES = [
  ["norm_in", "input norm", 55, 22],
  ["qkv", "qkv proj", 55, 58],
  ["q", "q", 55, 94, 62], ["k", "k", 123, 94, 62],
  ["attn_probs", "attention", 55, 130],
  ["attn_out", "attn out", 55, 166],
  ["o_proj", "o proj", 55, 202],
  ["hidden_attn", "+ residual", 55, 238],
  ["norm_ff", "ff norm", 55, 274],
  ["gateup", "gate · up", 55, 310],
  ["geglu", "geglu", 55, 346],
  ["mlp_out", "down proj", 55, 382],
  ["hidden", "+ residual", 55, 418],
];
const DIAG_GLOBALS = [["embed", 8], ["final_norm", 86], ["logits", 164]];
const NODE_W = 130, NODE_H = 22;
let diagRects = {};   // key suffix -> {rect, text}

function svgEl(tag, attrs) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function buildDiagram() {
  const svg = ui.diagram;
  svg.innerHTML = "";
  const edge = (x1, y1, x2, y2, dash) => svg.appendChild(svgEl("path", {
    d: `M ${x1} ${y1} L ${x2} ${y2}`, stroke: dash ? "#555" : "#444",
    "stroke-dasharray": dash ? "3,3" : "none", fill: "none" }));
  // main chain
  for (let i = 0; i < DIAG_NODES.length - 1; i++) {
    const [, , x1, y1, w1] = DIAG_NODES[i];
    const [, , x2, y2, w2] = DIAG_NODES[i + 1];
    const cx1 = x1 + (w1 ?? NODE_W) / 2, cx2 = x2 + (w2 ?? NODE_W) / 2;
    edge(cx1, y1 + NODE_H, cx2, y2);
  }
  // qkv fans out to q and k, both feed attention
  edge(120, 80, 86, 94); edge(120, 80, 154, 94);
  edge(86, 116, 120, 130); edge(154, 116, 120, 130);
  // residual bypasses (dashed left rail)
  svg.appendChild(svgEl("path", { d: `M 120 10 L 12 10 L 12 249 L 55 249`,
    stroke: "#555", "stroke-dasharray": "3,3", fill: "none" }));
  svg.appendChild(svgEl("path", { d: `M 55 249 L 12 249 L 12 429 L 55 429`,
    stroke: "#555", "stroke-dasharray": "3,3", fill: "none" }));

  const mkNode = (key, label, x, y, w) => {
    const g = svgEl("g", { style: "cursor:pointer" });
    const rect = svgEl("rect", { x, y, width: w, height: NODE_H, rx: 5,
      fill: "#262626", stroke: "#3a3a3a" });
    const text = svgEl("text", { x: x + w / 2, y: y + 15, "text-anchor": "middle",
      fill: "#ccc", "font-size": "10", "font-family": "ui-monospace, monospace" });
    text.textContent = label;
    g.appendChild(rect); g.appendChild(text);
    g.addEventListener("click", () => {
      const full = key.startsWith("@") ? key.slice(1) : `L${ui.layerSel.value}.${key}`;
      ui.list.value = full;
      if (history.length) render(full);
      updateDiagram();
    });
    svg.appendChild(g);
    diagRects[key] = { rect, text };
  };
  for (const [suffix, label, x, y, w] of DIAG_NODES)
    mkNode(suffix, label, x, y, w ?? NODE_W);
  const gy = 468;
  svg.appendChild(svgEl("text", { x: 8, y: gy - 8, fill: "#666", "font-size": "9",
    "font-family": "ui-monospace, monospace" })).textContent = "global:";
  for (const [key, x] of DIAG_GLOBALS) mkNode("@" + key, key, x, gy, 70);
}

function tensorStd(key) {
  const entry = history[viewStepIdx];
  if (!entry) return null;
  const c = capIndex[key];
  if (!c) return null;
  let data;
  if (key === "logits") {
    if (viewStepIdx !== history.length - 1 || !lastCap) return null;
    data = lastCap.subarray(c.offset / 4, c.offset / 4 + c.floats);
  } else {
    data = entry.cap.subarray(c.offset / 4, c.offset / 4 + c.floats);
  }
  let sm = 0, sq = 0, n = 0, bad = 0;
  for (const v of data) {
    if (!Number.isFinite(v)) { bad++; continue; }
    sm += v; sq += v * v; n++;
  }
  if (!n) return null;
  const m = sm / n;
  return { std: Math.sqrt(Math.max(0, sq / n - m * m)), bad };
}

function updateDiagram() {
  if (!G) return;
  const L = ui.layerSel.value;
  const entries = [];
  for (const [suffix] of DIAG_NODES)
    entries.push([suffix, `L${L}.${suffix}`, tensorStd(`L${L}.${suffix}`)]);
  for (const [key] of DIAG_GLOBALS)
    entries.push(["@" + key, key, tensorStd(key)]);

  const lstds = entries.filter(e => e[2]?.std > 0).map(e => Math.log(e[2].std));
  const lo = Math.min(...lstds), hi = Math.max(...lstds);
  for (const [suffix, full, st] of entries) {
    const node = diagRects[suffix];
    if (!node) continue;
    if (!st) {
      node.rect.setAttribute("fill", "#262626");
    } else if (st.bad) {
      node.rect.setAttribute("fill", "#a020a0");            // non-finite alert
    } else {
      const t = hi > lo ? (Math.log(st.std || 1e-9) - lo) / (hi - lo) : 0;
      const r = Math.round(38 + t * 186), gb = Math.round(38 + (1 - t) * 30);
      node.rect.setAttribute("fill", `rgb(${r},${gb + Math.round((1 - t) * 20)},${gb})`);
    }
    node.rect.setAttribute("stroke",
      ui.list.value === full ? "#4a9eff" : "#3a3a3a");
    node.text.parentNode.querySelector("title")?.remove();
    const title = svgEl("title", {});
    title.textContent = st ? `${full}  σ=${st.std.toPrecision(4)}` +
      (st.bad ? `  ⚠ ${st.bad} non-finite` : "") : full;
    node.text.parentNode.appendChild(title);
  }
}

function buildTensorList() {
  ui.list.innerHTML = "";
  let curGroup = null, og = null;
  for (const c of CAPS) {
    if (c.group !== curGroup) {
      og = document.createElement("optgroup");
      og.label = c.group;
      ui.list.appendChild(og);
      curGroup = c.group;
    }
    const opt = document.createElement("option");
    opt.value = c.key;
    opt.textContent = `${c.key}  [${c.floats}]`;
    og.appendChild(opt);
  }
}

// ------------------------------------------------------------ debug step

function writeStepParams(tokenId, p) {
  const { device, cfg, B, D } = G;
  const q = device.queue;
  const kvLen = p + 1;
  const start = Math.max(0, kvLen - cfg.sliding_window);
  q.writeBuffer(B.token, 0, new Uint32Array([tokenId]));
  q.writeBuffer(D.kvAppend, 4, new Uint32Array([p]));
  q.writeBuffer(D.scSlide, 8, new Uint32Array([kvLen, start]));
  q.writeBuffer(D.scFull, 8, new Uint32Array([kvLen]));
  q.writeBuffer(G.ropeLocal, 4, new Float32Array([p]));
  q.writeBuffer(G.ropeGlobal, 4, new Float32Array([p]));
}

async function debugStep(tokenId, p) {
  const { device, cfg, pipelines, B, layers } = G;
  const h = cfg.hidden_size, hd = cfg.head_dim, nh = cfg.num_heads;
  const inter = cfg.intermediate_size, qDim = nh * hd;
  writeStepParams(tokenId, p);

  const enc = device.createCommandEncoder();
  let pass = null;
  const run = (name, group, wgs) => {
    if (!pass) pass = enc.beginComputePass();
    pass.setPipeline(pipelines[name].pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(wgs);
  };
  const cap = key => {           // copies require ending the compute pass
    if (pass) { pass.end(); pass = null; }
    const c = capIndex[key];
    enc.copyBufferToBuffer(B[c.src], 0, G.arena, c.offset, c.floats * 4);
  };

  run("embed_scale_packed", G.bgEmbed, Math.ceil(h / 2 / 64));
  cap("embed");
  for (let L = 0; L < cfg.num_layers; L++) {
    const bg = layers[L];
    run("rmsnorm_wg", bg.norm1, 1);              cap(`L${L}.norm_in`);
    run("matvec_wg_packed", bg.mvQkv, qDim + 2 * hd); cap(`L${L}.qkv`);
    run("rmsnorm_wg", bg.qnorm, nh);
    run("rmsnorm_wg", bg.knorm, 1);
    run("rope", bg.ropeQ, Math.ceil(qDim / 2 / 64));
    run("rope", bg.ropeK, Math.ceil(hd / 2 / 64));
    cap(`L${L}.q`); cap(`L${L}.k`);
    run("kv_append", bg.appK, Math.ceil(hd / 64));
    run("kv_append", bg.appV, Math.ceil(hd / 64));
    run("attention_fused", bg.attn, nh);
    cap(`L${L}.attn_probs`); cap(`L${L}.attn_out`);
    run("matvec_wg_packed", bg.mvO, h);          cap(`L${L}.o_proj`);
    run("rmsnorm_add_wg", bg.normPaAdd, 1);      cap(`L${L}.hidden_attn`);
    run("rmsnorm_wg", bg.normPf, 1);             cap(`L${L}.norm_ff`);
    run("matvec_wg_packed", bg.mvGateup, 2 * inter); cap(`L${L}.gateup`);
    run("geglu", bg.geglu, Math.ceil(inter / 64)); cap(`L${L}.geglu`);
    run("matvec_wg_packed", bg.mvDown, h);       cap(`L${L}.mlp_out`);
    run("rmsnorm_add_wg", bg.normPffAdd, 1);     cap(`L${L}.hidden`);
  }
  run("rmsnorm_wg", G.bgFinalNorm, 1);           cap("final_norm");
  run("matvec_packed", G.bgLogits, Math.ceil(cfg.vocab_size / 64)); cap("logits");
  if (pass) pass.end();
  enc.copyBufferToBuffer(G.arena, 0, G.staging, 0, G.arenaBytes);
  device.queue.submit([enc.finish()]);

  await G.staging.mapAsync(GPUMapMode.READ);
  lastCap = new Float32Array(G.staging.getMappedRange().slice(0));
  G.staging.unmap();
  lastKvLen = p + 1;
}

// ------------------------------------------------------------ tokens / UI

async function post(url, payload) {
  return (await fetch(url, { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload) })).json();
}

async function piece(id) {
  return (await post("/detokenize", { ids: [id] })).text || "∅";
}

function topK(logits, k) {
  const idx = [];
  for (let i = 0; i < logits.length; i++) {
    if (idx.length < k) { idx.push(i); idx.sort((a, b) => logits[b] - logits[a]); }
    else if (logits[i] > logits[idx[k - 1]]) {
      idx[k - 1] = i; idx.sort((a, b) => logits[b] - logits[a]);
    }
  }
  const m = logits[idx[0]];
  let z = 0;
  for (let i = 0; i < logits.length; i++) z += Math.exp(logits[i] - m);
  return idx.map(i => ({ id: i, p: Math.exp(logits[i] - m) / z }));
}

ui.tokenizeBtn.addEventListener("click", async () => {
  const out = await post("/tokenize", { text: ui.prompt.value, chat: true });
  promptIds = out.ids; pos = 0; nextTok = null; lastCap = null;
  history = []; viewStepIdx = -1; ui.tokens.innerHTML = "";
  ui.stepInfo.textContent = `prompt: ${promptIds.length} tokens — position 0 (KV cache reset)`;
  ui.preds.textContent = "";
  ui.stepBtn.disabled = false;
  ui.runPromptBtn.disabled = false;
  ctx2d.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
  ui.stats.textContent = "";
});

async function doStep() {
  const inPrompt = pos < promptIds.length;
  const tid = inPrompt ? promptIds[pos] : nextTok;
  status(`stepping: pos ${pos}, token ${tid}…`);
  const t0 = performance.now();
  await debugStep(tid, pos);
  const c = capIndex["logits"];
  const logits = lastCap.subarray(c.offset / 4, c.offset / 4 + c.floats);
  const top = topK(logits, 5);
  nextTok = top[0].id;
  const tidPiece = await piece(tid);
  const promptLeft = promptIds.length - 1 - pos;
  ui.stepInfo.textContent =
    (inPrompt ? `prompt token ${pos + 1}/${promptIds.length}` : `generated (pos ${pos})`) +
    ` — processed ${tid} ${JSON.stringify(tidPiece)} ` +
    `in ${(performance.now() - t0).toFixed(0)} ms`;
  const parts = [];
  for (const t of top) parts.push(`${JSON.stringify(await piece(t.id))} ${(t.p * 100).toFixed(1)}%`);
  if (promptLeft > 0) {
    // mid-prompt: the model has only seen a prefix — label it honestly
    ui.preds.className = "partial";
    ui.preds.textContent =
      `⚠ prompt unfinished (${promptLeft} tokens left) — continuation of the ` +
      `partial prefix, NOT the answer: ` + parts.join("  ·  ");
  } else {
    ui.preds.className = "";
    ui.preds.textContent = "next: " + parts.join("  ·  ");
  }
  history.push({ pos, tid, piece: tidPiece, kvLen: lastKvLen,
                 cap: lastCap.slice(0, capIndex["logits"].offset / 4) });
  viewStepIdx = history.length - 1;
  renderTokens();
  updateDiagram();
  pos++;
  if (pos >= promptIds.length) ui.runPromptBtn.disabled = true;
  if (ui.list.value) render(ui.list.value);
}

function gotoStep(i) {
  if (!history.length) return;
  viewStepIdx = Math.max(0, Math.min(history.length - 1, i));
  ui.viewSel.value = "step";
  renderTokens();
  updateDiagram();
  if (ui.list.value) render(ui.list.value);
}

ui.prevTok.addEventListener("click", () => gotoStep(viewStepIdx - 1));
ui.nextTok.addEventListener("click", () => gotoStep(viewStepIdx + 1));
document.addEventListener("keydown", e => {
  if (e.target === ui.prompt || !history.length) return;
  if (e.key === "ArrowLeft") { gotoStep(viewStepIdx - 1); e.preventDefault(); }
  if (e.key === "ArrowRight") { gotoStep(viewStepIdx + 1); e.preventDefault(); }
});

function renderTokens() {
  ui.tokens.innerHTML = "";
  history.forEach((en, i) => {
    const b = document.createElement("span");
    b.className = "tok" + (i === viewStepIdx ? " sel" : "");
    b.textContent = en.piece === "" ? "·" : en.piece;
    b.title = `pos ${en.pos} · id ${en.tid} — click to inspect this token`;
    b.onclick = () => gotoStep(i);
    ui.tokens.appendChild(b);
  });
}

ui.stepBtn.addEventListener("click", async () => {
  ui.stepBtn.disabled = true; ui.runPromptBtn.disabled = true;
  try { await doStep(); status("step done — select a tensor"); }
  catch (e) { status(`error: ${e.message ?? e}`); console.error(e); }
  finally {
    ui.stepBtn.disabled = false;
    ui.runPromptBtn.disabled = pos >= promptIds.length;
  }
});

ui.runPromptBtn.addEventListener("click", async () => {
  ui.stepBtn.disabled = true; ui.runPromptBtn.disabled = true;
  try {
    while (pos < promptIds.length) await doStep();
    status("prompt complete — predictions below are the real continuation");
  } catch (e) { status(`error: ${e.message ?? e}`); console.error(e); }
  finally { ui.stepBtn.disabled = false; }
});

// ------------------------------------------------------------- rendering
// treescope-style: labeled semantic axes, tick indices, color legend,
// edge-truncation for huge axes, value digits when cells are large.

const M_L = 46, M_T = 20;

let view = null; // hover state

function truncMap(size, limit) {
  // display->original index map with an ellipsis band, or null if it fits
  if (size <= limit) return null;
  const edge = Math.floor((limit - 1) / 2);
  const map = new Array(2 * edge + 1);
  for (let i = 0; i < edge; i++) map[i] = i;
  map[edge] = -1;
  for (let i = 0; i < edge; i++) map[edge + 1 + i] = size - edge + i;
  return map;
}

function render(key) {
  if (!history.length) { ui.stats.textContent = "run a step first"; return; }
  const c = capIndex[key];
  const latest = viewStepIdx === history.length - 1;
  const entry = history[viewStepIdx];
  const wantSeq = ui.viewSel.value === "seq";
  const seqOk = key !== "logits";
  const seq = wantSeq && seqOk;
  let seqNote = wantSeq && !seqOk
    ? "  ·  sequence view unavailable for logits (single token shown)" : "";
  ui.headCtl.style.display = c.kind === "attn" ? "" : "none";
  const headPick = ui.headSel.value;

  // resolve data + shape
  let data, rows, cols, rowLab, colLab, rowMeta = null, colMeta = null;
  if (seq && c.kind === "attn") {
    // per-head attention matrix: query position × kv position (triangular)
    const nh = G.cfg.num_heads;
    const hSel = headPick === "all" ? 0 : +headPick;
    if (headPick === "all")
      seqNote = "  ·  showing head 0 (pick a head for others)";
    cols = Math.max(...history.map(en => en.kvLen));
    rows = history.length;
    data = new Float32Array(rows * cols);          // upper triangle stays 0
    history.forEach((en, i) => {
      const raw = en.cap.subarray(c.offset / 4, c.offset / 4 + c.floats);
      let sum = 0;
      for (let t = 0; t < en.kvLen; t++) sum += raw[hSel * MAX_SEQ + t];
      for (let t = 0; t < en.kvLen; t++)
        data[i * cols + t] = raw[hSel * MAX_SEQ + t] / (sum || 1);
    });
    rowLab = "query"; colLab = "kv";
    rowMeta = history.map(en => en.piece);
    colMeta = Array.from({ length: cols }, (_, i) => history[i]?.piece ?? "?");
    ui.widthSel.disabled = true;
  } else if (seq) {
    // [positions × features]: this tensor stacked across every processed token
    cols = c.floats; rows = history.length;
    data = new Float32Array(rows * cols);
    history.forEach((en, i) =>
      data.set(en.cap.subarray(c.offset / 4, c.offset / 4 + c.floats), i * cols));
    rowLab = "pos";
    colLab = c.dims.length === 2 ? c.dims.map(d => d[0]).join("·") : c.dims[0][0];
    rowMeta = history.map(en => en.piece);
    ui.widthSel.disabled = true;
  } else if (key === "logits") {
    if (!latest) {
      ui.stats.textContent =
        "logits are kept for the latest token only — select the last chip";
      return;
    }
    data = lastCap.subarray(c.offset / 4, c.offset / 4 + c.floats);
    colLab = c.dims[0][0]; rowLab = "";
    cols = +ui.widthSel.value || c.defCols;
    rows = Math.ceil(data.length / cols);
    ui.widthSel.disabled = false;
  } else if (c.kind === "attn") {
    const raw = entry.cap.subarray(c.offset / 4, c.offset / 4 + c.floats);
    const nh = G.cfg.num_heads, kv = entry.kvLen;
    const heads = headPick === "all"
      ? Array.from({ length: nh }, (_, i) => i) : [+headPick];
    const out = new Float32Array(heads.length * kv);
    heads.forEach((hI, r) => {
      let sum = 0;
      for (let t = 0; t < kv; t++) sum += raw[hI * MAX_SEQ + t];
      for (let t = 0; t < kv; t++) out[r * kv + t] = raw[hI * MAX_SEQ + t] / (sum || 1);
    });
    data = out; rows = heads.length; cols = kv;
    rowLab = headPick === "all" ? "head" : `head ${headPick}`; colLab = "kv";
    colMeta = Array.from({ length: kv }, (_, i) => history[i]?.piece ?? "?");
    ui.widthSel.disabled = true;
  } else {
    data = entry.cap.subarray(c.offset / 4, c.offset / 4 + c.floats);
    if (c.dims.length === 2) {
      [[rowLab, rows], [colLab, cols]] = c.dims;
      ui.widthSel.disabled = true;
    } else {
      colLab = c.dims[0][0]; rowLab = "";
      cols = +ui.widthSel.value || c.defCols;
      rows = Math.ceil(data.length / cols);
      ui.widthSel.disabled = false;
    }
  }

  // stats
  let mn = Infinity, mx = -Infinity, sum = 0, sq = 0, bad = 0, n = 0;
  for (const v of data) {
    if (!Number.isFinite(v)) { bad++; continue; }
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += v; sq += v * v; n++;
  }
  const mean = sum / (n || 1);
  const std = Math.sqrt(Math.max(0, sq / (n || 1) - mean * mean));
  const shapeStr = `${rowLab || "row"}:${rows} × ${colLab}:${cols}`;
  const stepStr = seq ? `all ${rows} tokens` : `token ${viewStepIdx} ${JSON.stringify(entry.piece)}`;

  const zoom = ui.zoomSel.value;
  const digits = zoom === "values";
  let cell, axisLimit;
  if (zoom === "fit") { cell = 0; axisLimit = 1040; }
  else if (digits) { cell = 24; axisLimit = 63; }
  else { cell = +zoom; axisLimit = Math.min(1040, Math.floor(8192 / +zoom)); }

  const colMap = truncMap(cols, axisLimit);
  const rowMap = truncMap(rows, axisLimit);
  const dCols = colMap ? colMap.length : cols;
  const dRows = rowMap ? rowMap.length : rows;
  if (!cell) cell = Math.max(1, Math.min(Math.floor(720 / dCols), Math.floor(560 / dRows), 20));

  ui.stats.textContent =
    `${key} (${stepStr})  f32  ${shapeStr}  min ${mn.toFixed(4)}  max ${mx.toFixed(4)}  ` +
    `mean ${mean.toFixed(4)}  std ${std.toFixed(4)}` +
    (bad ? `  ⚠ ${bad} non-finite` : "") + seqNote +
    (digits ? "" : "  ·  hover for values");

  const sym = ui.scaleSel.value === "symmetric";
  const lim = sym ? Math.max(Math.abs(mn), Math.abs(mx)) || 1 : null;
  const color = v => {
    if (!Number.isFinite(v)) return [255, 0, 255];
    if (sym) {
      const t = Math.max(-1, Math.min(1, v / lim));
      return t >= 0
        ? [255, Math.round(255 * (1 - t)), Math.round(255 * (1 - t))]
        : [Math.round(255 * (1 + t)), Math.round(255 * (1 + t)), 255];
    }
    const t = (v - mn) / ((mx - mn) || 1);
    const g = Math.round(255 * t);
    return [g, g, g];
  };

  const img = ctx2d.createImageData(dCols, dRows);
  for (let dr = 0; dr < dRows; dr++) {
    for (let dc = 0; dc < dCols; dc++) {
      const r = rowMap ? rowMap[dr] : dr;
      const cc = colMap ? colMap[dc] : dc;
      let rgb;
      if (r === -1 || cc === -1) rgb = [34, 34, 34];
      else {
        const i = r * cols + cc;
        rgb = i < data.length ? color(data[i]) : [17, 17, 17];
      }
      img.data.set([...rgb, 255], (dr * dCols + dc) * 4);
    }
  }

  const gw = dCols * cell, gh = dRows * cell;
  ui.canvas.width = M_L + gw;
  ui.canvas.height = M_T + gh + 4;

  createImageBitmap(img).then(bmp => {
    ctx2d.imageSmoothingEnabled = false;
    ctx2d.fillStyle = "#111";
    ctx2d.fillRect(0, 0, ui.canvas.width, ui.canvas.height);
    ctx2d.drawImage(bmp, M_L, M_T, gw, gh);
    ctx2d.fillStyle = "#888";
    ctx2d.font = "10px ui-monospace, monospace";
    ctx2d.textAlign = "left";
    ctx2d.fillText(`${colLab} →`, M_L, 10);
    for (const dc of [0, dCols >> 1, dCols - 1]) {
      const orig = colMap ? colMap[Math.min(dc, dCols - 1)] : dc;
      if (orig >= 0) ctx2d.fillText(String(orig), M_L + dc * cell, M_T - 2);
    }
    ctx2d.save();
    ctx2d.translate(10, M_T + 2);
    ctx2d.rotate(Math.PI / 2);
    ctx2d.fillText(`${rowLab || "row"} →`, 0, 0);
    ctx2d.restore();
    ctx2d.textAlign = "right";
    for (const dr of [0, dRows >> 1, dRows - 1]) {
      const orig = rowMap ? rowMap[Math.min(dr, dRows - 1)] : dr;
      if (orig >= 0) ctx2d.fillText(String(orig), M_L - 4, M_T + dr * cell + Math.max(9, cell / 2));
    }
    if (digits) {
      ctx2d.font = "9px ui-monospace, monospace";
      ctx2d.textAlign = "center";
      for (let dr = 0; dr < dRows; dr++) {
        for (let dc = 0; dc < dCols; dc++) {
          const r = rowMap ? rowMap[dr] : dr;
          const cc = colMap ? colMap[dc] : dc;
          if (r === -1 || cc === -1) continue;
          const i = r * cols + cc;
          if (i >= data.length) continue;
          const v = data[i];
          const strong = Number.isFinite(v) && sym && Math.abs(v / lim) > 0.55;
          ctx2d.fillStyle = strong || !Number.isFinite(v) ? "#fff" : "#333";
          ctx2d.fillText(Number.isFinite(v) ? v.toFixed(2) : "NaN",
                         M_L + dc * cell + cell / 2, M_T + dr * cell + cell / 2 + 3);
        }
      }
    }
  });

  legendCtx.clearRect(0, 0, 280, 26);
  for (let x = 0; x < 240; x++) {
    const t = x / 239;
    const v = sym ? (2 * t - 1) * lim : mn + t * (mx - mn);
    legendCtx.fillStyle = `rgb(${color(v).join(",")})`;
    legendCtx.fillRect(x + 20, 2, 1, 10);
  }
  legendCtx.fillStyle = "#888";
  legendCtx.font = "10px ui-monospace, monospace";
  legendCtx.textAlign = "left";
  legendCtx.fillText((sym ? -lim : mn).toPrecision(3), 0, 24);
  legendCtx.textAlign = "right";
  legendCtx.fillText((sym ? lim : mx).toPrecision(3), 260, 24);
  if (sym) { legendCtx.textAlign = "center"; legendCtx.fillText("0", 140, 24); }

  drawHistogram(data, mn, mx);
  view = { rows, cols, dRows, dCols, rowMap, colMap, data, cell, rowLab, colLab,
           rowMeta, colMeta };
}

// value distribution: 64 bins, log-scaled counts (peaked distributions
// would otherwise hide their tails), zero marker, hover for bin counts
const H_BINS = 64, H_W = 340, H_H = 96, H_PAD = 14;

function drawHistogram(data, mn, mx) {
  const bins = new Uint32Array(H_BINS);
  const span = (mx - mn) || 1;
  let finite = 0;
  for (const v of data) {
    if (!Number.isFinite(v)) continue;
    let b = Math.floor((v - mn) / span * H_BINS);
    if (b >= H_BINS) b = H_BINS - 1;
    if (b < 0) b = 0;
    bins[b]++; finite++;
  }
  let cmax = 0;
  for (const c of bins) if (c > cmax) cmax = c;

  histCtx.clearRect(0, 0, H_W, H_H);
  histCtx.fillStyle = "#1a1a1a";
  histCtx.fillRect(0, 0, H_W, H_H - H_PAD);
  const bw = H_W / H_BINS;
  const lg = c => Math.log1p(c) / Math.log1p(cmax || 1);
  histCtx.fillStyle = "#4a9eff";
  for (let b = 0; b < H_BINS; b++) {
    const hgt = Math.round(lg(bins[b]) * (H_H - H_PAD - 4));
    if (hgt > 0) histCtx.fillRect(b * bw, H_H - H_PAD - hgt, Math.max(1, bw - 1), hgt);
  }
  if (mn < 0 && mx > 0) {                       // zero marker
    const zx = (-mn) / span * H_W;
    histCtx.strokeStyle = "#e05555";
    histCtx.setLineDash([3, 3]);
    histCtx.beginPath();
    histCtx.moveTo(zx, 0); histCtx.lineTo(zx, H_H - H_PAD);
    histCtx.stroke();
    histCtx.setLineDash([]);
  }
  histCtx.fillStyle = "#888";
  histCtx.font = "9px ui-monospace, monospace";
  histCtx.textAlign = "left";
  histCtx.fillText(mn.toPrecision(3), 0, H_H - 3);
  histCtx.textAlign = "right";
  histCtx.fillText(mx.toPrecision(3), H_W, H_H - 3);
  histCtx.textAlign = "center";
  histCtx.fillText(`distribution (${finite} values, log count)`, H_W / 2, H_H - 3);
  histView = { bins, mn, span };
}

ui.hist.addEventListener("mousemove", e => {
  if (!histView) return;
  const rect = ui.hist.getBoundingClientRect();
  const b = Math.floor((e.clientX - rect.left) / (H_W / H_BINS));
  if (b < 0 || b >= H_BINS) { ui.tooltip.textContent = ""; return; }
  const lo = histView.mn + b / H_BINS * histView.span;
  const hi = histView.mn + (b + 1) / H_BINS * histView.span;
  ui.tooltip.textContent =
    `bin [${lo.toPrecision(4)}, ${hi.toPrecision(4)})  ·  ${histView.bins[b]} values`;
});
ui.hist.addEventListener("mouseleave", () => { ui.tooltip.textContent = ""; });

ui.list.addEventListener("change", () => {
  render(ui.list.value);
  // follow the selection into the diagram (switch layer if needed)
  const m = ui.list.value.match(/^L(\d+)\./);
  if (m) ui.layerSel.value = m[1];
  updateDiagram();
});
ui.widthSel.addEventListener("change", () => ui.list.value && render(ui.list.value));
ui.scaleSel.addEventListener("change", () => ui.list.value && render(ui.list.value));
ui.zoomSel.addEventListener("change", () => ui.list.value && render(ui.list.value));
ui.viewSel.addEventListener("change", () => ui.list.value && render(ui.list.value));
ui.headSel.addEventListener("change", () => ui.list.value && render(ui.list.value));

ui.canvas.addEventListener("mousemove", e => {
  if (!view) return;
  const rect = ui.canvas.getBoundingClientRect();
  const dc = Math.floor((e.clientX - rect.left - M_L) / view.cell);
  const dr = Math.floor((e.clientY - rect.top - M_T) / view.cell);
  if (dr < 0 || dc < 0 || dr >= view.dRows || dc >= view.dCols) {
    ui.tooltip.textContent = ""; return;
  }
  const r = view.rowMap ? view.rowMap[dr] : dr;
  const cc = view.colMap ? view.colMap[dc] : dc;
  if (r === -1 || cc === -1) { ui.tooltip.textContent = "⋯ (truncated region)"; return; }
  const i = r * view.cols + cc;
  if (i >= view.data.length) { ui.tooltip.textContent = ""; return; }
  const rl = view.rowLab || "row";
  let tokStr = view.rowMeta ? `  ·  ${JSON.stringify(view.rowMeta[r] ?? "?")}` : "";
  if (view.colMeta) tokStr += ` → ${JSON.stringify(view.colMeta[cc] ?? "?")}`;
  ui.tooltip.textContent =
    `${rl} ${r}, ${view.colLab} ${cc}${tokStr}  ·  flat index ${i}  ·  value ${view.data[i]}`;
});
ui.canvas.addEventListener("mouseleave", () => { ui.tooltip.textContent = ""; });

init().catch(e => { status(`init error: ${e.message ?? e}`); console.error(e); });
