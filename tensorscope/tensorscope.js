// tensorscope: step-by-step Gemma inference in WebGPU with capture of
// every intermediate tensor, visualized on an interactive canvas heatmap.
// Standalone page; shares the server's kernels/weights/tokenizer endpoints.

const $ = id => document.getElementById(id);
const ui = {
  status: $("status"), prompt: $("prompt"),
  tokenizeBtn: $("tokenize"), stepBtn: $("step"),
  stepInfo: $("stepinfo"), preds: $("preds"),
  list: $("tensorlist"), canvas: $("canvas"),
  tooltip: $("tooltip"), stats: $("stats"),
  widthSel: $("widthsel"), scaleSel: $("scalesel"),
};
const ctx2d = ui.canvas.getContext("2d");

const MAX_SEQ = 1024;
let G = null;            // gpu state
let CAPS = [];           // {key, group, srcName, floats, offset, defCols, kind}
let capIndex = {};       // key -> cap
let lastCap = null;      // Float32Array of the whole arena after a step
let lastKvLen = 0;
let promptIds = [], pos = 0, nextTok = null;

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
  const addCap = (key, group, src, floats, defCols, kind = "plain") => {
    const c = { key, group, src, floats, offset: off, defCols, kind };
    CAPS.push(c); capIndex[key] = c; off += floats * 4;
  };
  addCap("embed", "global", "x", h, 32);
  for (let L = 0; L < cfg.num_layers; L++) {
    const g = `layer ${L}`;
    addCap(`L${L}.norm_in`, g, "xn", h, 32);
    addCap(`L${L}.qkv`, g, "qkv", qDim + 2 * hd, 64);
    addCap(`L${L}.q`, g, "qn", qDim, 64);
    addCap(`L${L}.k`, g, "kn", hd, 32);
    addCap(`L${L}.attn_probs`, g, "scores", nh * MAX_SEQ, MAX_SEQ, "attn");
    addCap(`L${L}.attn_out`, g, "attn", qDim, 64);
    addCap(`L${L}.o_proj`, g, "attnProj", h, 32);
    addCap(`L${L}.hidden_attn`, g, "x", h, 32);
    addCap(`L${L}.norm_ff`, g, "xn", h, 32);
    addCap(`L${L}.gateup`, g, "gateup", 2 * inter, 64);
    addCap(`L${L}.geglu`, g, "ffh", inter, 64);
    addCap(`L${L}.mlp_out`, g, "mlpOut", h, 32);
    addCap(`L${L}.hidden`, g, "x", h, 32);
  }
  addCap("final_norm", "global", "xn", h, 32);
  addCap("logits", "global", "logits", cfg.vocab_size, 512);
  const arena = device.createBuffer({ size: off,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const staging = device.createBuffer({ size: off,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  G = { device, cfg, pipelines, W, B, D, layers, kCache, vCache,
        ropeLocal, ropeGlobal, bgEmbed, bgFinalNorm, bgLogits, arena, staging,
        arenaBytes: off };

  buildTensorList();
  status(`ready — ${CAPS.length} capturable tensors, ${(off / 1e6).toFixed(1)} MB per step`);
  ui.tokenizeBtn.disabled = false;
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
  ui.stepInfo.textContent = `prompt: ${promptIds.length} tokens — position 0 (KV cache reset)`;
  ui.preds.textContent = "";
  ui.stepBtn.disabled = false;
  ctx2d.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
  ui.stats.textContent = "";
});

ui.stepBtn.addEventListener("click", async () => {
  ui.stepBtn.disabled = true;
  try {
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
    ui.stepInfo.textContent =
      `pos ${pos} ${inPrompt ? "(prompt)" : "(generated)"} — processed ${tid} ` +
      `${JSON.stringify(tidPiece)} in ${(performance.now() - t0).toFixed(0)} ms`;
    const parts = [];
    for (const t of top) parts.push(`${JSON.stringify(await piece(t.id))} ${(t.p * 100).toFixed(1)}%`);
    ui.preds.textContent = "next: " + parts.join("  ·  ");
    pos++;
    status("step done — select a tensor");
    if (ui.list.value) render(ui.list.value);
  } catch (e) {
    status(`error: ${e.message ?? e}`); console.error(e);
  } finally {
    ui.stepBtn.disabled = false;
  }
});

// ------------------------------------------------------------- rendering

let view = null; // {rows, cols, data, cellW, cellH} for hover mapping

function render(key) {
  if (!lastCap) { ui.stats.textContent = "run a step first"; return; }
  const c = capIndex[key];
  let data = lastCap.subarray(c.offset / 4, c.offset / 4 + c.floats);
  let cols = +ui.widthSel.value || c.defCols;
  let rows;
  if (c.kind === "attn") {
    // scores hold unnormalized exp() weights; normalize per head row and
    // crop to the visible KV length
    const nh = G.cfg.num_heads;
    const out = new Float32Array(nh * lastKvLen);
    for (let hI = 0; hI < nh; hI++) {
      let sum = 0;
      for (let t = 0; t < lastKvLen; t++) sum += data[hI * MAX_SEQ + t];
      for (let t = 0; t < lastKvLen; t++)
        out[hI * lastKvLen + t] = data[hI * MAX_SEQ + t] / (sum || 1);
    }
    data = out; cols = lastKvLen; rows = nh;
  } else {
    rows = Math.ceil(data.length / cols);
  }

  // stats (NaN/Inf aware)
  let mn = Infinity, mx = -Infinity, sum = 0, sq = 0, bad = 0, n = 0;
  for (const v of data) {
    if (!Number.isFinite(v)) { bad++; continue; }
    mn = Math.min(mn, v); mx = Math.max(mx, v); sum += v; sq += v * v; n++;
  }
  const mean = sum / (n || 1);
  const std = Math.sqrt(Math.max(0, sq / (n || 1) - mean * mean));
  ui.stats.textContent =
    `${key}  shape ${rows}×${cols}  min ${mn.toFixed(4)}  max ${mx.toFixed(4)}  ` +
    `mean ${mean.toFixed(4)}  std ${std.toFixed(4)}` +
    (bad ? `  ⚠ ${bad} non-finite values` : "");

  // color scale
  const sym = ui.scaleSel.value === "symmetric";
  const lim = sym ? Math.max(Math.abs(mn), Math.abs(mx)) || 1 : null;
  const img = ctx2d.createImageData(cols, rows);
  for (let i = 0; i < rows * cols; i++) {
    const v = i < data.length ? data[i] : 0;
    let r, g, b;
    if (!Number.isFinite(v)) { r = 255; g = 0; b = 255; }         // magenta
    else if (sym) {
      const t = Math.max(-1, Math.min(1, v / lim));
      if (t >= 0) { r = 255; g = b = Math.round(255 * (1 - t)); } // white→red
      else { b = 255; r = g = Math.round(255 * (1 + t)); }        // white→blue
    } else {
      const t = (v - mn) / ((mx - mn) || 1);
      r = g = b = Math.round(255 * t);                            // grayscale
    }
    img.data.set([r, g, b, 255], i * 4);
  }

  // scale to canvas
  const maxW = 720, maxH = 560;
  const cell = Math.max(1, Math.min(Math.floor(maxW / cols), Math.floor(maxH / rows), 24));
  ui.canvas.width = cols * cell; ui.canvas.height = rows * cell;
  createImageBitmap(img).then(bmp => {
    ctx2d.imageSmoothingEnabled = false;
    ctx2d.drawImage(bmp, 0, 0, cols * cell, rows * cell);
  });
  view = { rows, cols, data, cell, key };
}

ui.list.addEventListener("change", () => render(ui.list.value));
ui.widthSel.addEventListener("change", () => ui.list.value && render(ui.list.value));
ui.scaleSel.addEventListener("change", () => ui.list.value && render(ui.list.value));

ui.canvas.addEventListener("mousemove", e => {
  if (!view) return;
  const rect = ui.canvas.getBoundingClientRect();
  const col = Math.floor((e.clientX - rect.left) / view.cell);
  const row = Math.floor((e.clientY - rect.top) / view.cell);
  const i = row * view.cols + col;
  if (row < 0 || col < 0 || row >= view.rows || i >= view.data.length) {
    ui.tooltip.textContent = ""; return;
  }
  ui.tooltip.textContent = `[${row}, ${col}]  index ${i}  value ${view.data[i]}`;
});
ui.canvas.addEventListener("mouseleave", () => { ui.tooltip.textContent = ""; });

init().catch(e => { status(`init error: ${e.message ?? e}`); console.error(e); });
