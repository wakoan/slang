// Gemma 3 in the browser: WebGPU runner for DSL-generated WGSL kernels.
// Mirrors gemma3/runner.py (fused kernel set, GPU-resident greedy decode).

const ui = {
  status: document.getElementById("status"),
  bar: document.getElementById("bar"),
  prompt: document.getElementById("prompt"),
  output: document.getElementById("output"),
  button: document.getElementById("go"),
  stats: document.getElementById("stats"),
};

const N_ARGMAX_WGS = 128;
const MAX_SEQ = 1024;
const CHUNK = 16;

let G = null; // global model state

function status(msg) { ui.status.textContent = msg; }

// ---------------------------------------------------------------- init

async function init() {
  if (!navigator.gpu) {
    status("WebGPU not available — use Chrome/Edge (or recent Safari).");
    throw new Error("no webgpu");
  }
  const [kernels, manifest] = await Promise.all([
    fetch("/kernels.json").then(r => r.json()),
    fetch("/manifest.json").then(r => r.json()),
  ]);
  const cfg = manifest.config;

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) { status("No GPU adapter."); throw new Error("no adapter"); }
  const embedBytes = cfg.vocab_size * cfg.hidden_size * 2; // packed f16
  const need = Math.max(embedBytes, 1 << 28);
  if (adapter.limits.maxStorageBufferBindingSize < embedBytes) {
    status(`GPU limit too small: needs ${embedBytes} bytes for the embedding table.`);
    throw new Error("limits");
  }
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: Math.max(need, adapter.limits.maxBufferSize >= need ? need : 0),
      maxStorageBufferBindingSize: need,
    },
  });

  // --- weights: one fetch with progress, slice into buffers ---
  status("downloading weights…");
  const resp = await fetch("/weights.bin");
  const total = Number(resp.headers.get("Content-Length")) || manifest.totalBytes;
  const blob = new Uint8Array(total);
  let got = 0;
  const reader = resp.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    blob.set(value, got);
    got += value.length;
    ui.bar.style.width = `${(100 * got / total).toFixed(1)}%`;
    status(`downloading weights… ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`);
  }

  status("uploading to GPU…");
  const W = {};
  for (const t of manifest.tensors) {
    const buf = device.createBuffer({
      size: t.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, blob.buffer, t.offset, t.byteLength);
    W[t.name] = buf;
  }

  // --- pipelines: explicit layouts derived from the WGSL binding decls ---
  const pipelines = {};
  for (const [name, k] of Object.entries(kernels)) {
    const access = [...k.wgsl.matchAll(/@binding\((\d+)\) var<storage, (read_write|read)>/g)]
      .sort((a, b) => +a[1] - +b[1])
      .map(m => m[2] === "read_write");
    const layout = device.createBindGroupLayout({
      entries: access.map((rw, i) => ({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: rw ? "storage" : "read-only-storage" },
      })),
    });
    const module = device.createShaderModule({ code: k.wgsl });
    pipelines[name] = {
      pipeline: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module, entryPoint: name },
      }),
      layout,
    };
  }

  // --- scratch buffers ---
  const h = cfg.hidden_size, hd = cfg.head_dim, nh = cfg.num_heads;
  const inter = cfg.intermediate_size, qDim = nh * hd;
  const fbuf = n => device.createBuffer({
    size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const B = {
    x: fbuf(h), xn: fbuf(h),
    qkv: fbuf(qDim + 2 * hd), qn: fbuf(qDim), kn: fbuf(hd),
    scores: fbuf(nh * MAX_SEQ), attn: fbuf(qDim), attnProj: fbuf(h),
    gateup: fbuf(2 * inter), ffh: fbuf(inter), mlpOut: fbuf(h),
    logits: fbuf(cfg.vocab_size),
    token: fbuf(1), pos: fbuf(1), counter: fbuf(1),
    outTokens: fbuf(MAX_SEQ),
    partVal: fbuf(N_ARGMAX_WGS), partIdx: fbuf(N_ARGMAX_WGS),
  };
  const kCache = [], vCache = [];
  for (let L = 0; L < cfg.num_layers; L++) {
    kCache.push(fbuf(MAX_SEQ * hd));
    vCache.push(fbuf(MAX_SEQ * hd));
  }
  const staging = device.createBuffer({
    size: MAX_SEQ * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // --- dims buffers ---
  const ubuf = (...vals) => {
    const b = device.createBuffer({
      size: vals.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(b, 0, new Uint32Array(vals));
    return b;
  };
  const D = {
    embed: ubuf(h),
    normH: ubuf(1, h), normQ: ubuf(nh, hd), normK: ubuf(1, hd),
    mvQkv: ubuf(qDim + 2 * hd, h), mvO: ubuf(h, qDim),
    mvGateup: ubuf(2 * inter, h), mvDown: ubuf(h, inter),
    mvLogits: ubuf(cfg.vocab_size, h),
    ropeQ: ubuf(nh, hd), ropeK: ubuf(1, hd),
    geglu: ubuf(inter),
    kvAppend: ubuf(hd, 0),
    scSlide: ubuf(nh, hd, 1, 0, MAX_SEQ),
    scFull: ubuf(nh, hd, 1, 0, MAX_SEQ),
    argmax1: ubuf(cfg.vocab_size, N_ARGMAX_WGS),
    argmax2: ubuf(N_ARGMAX_WGS),
    setupCfg: ubuf(cfg.sliding_window),
  };
  const ropeLocal = device.createBuffer({
    size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const ropeGlobal = device.createBuffer({
    size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(ropeLocal, 0, new Float32Array([cfg.rope_theta_local, 0]));
  device.queue.writeBuffer(ropeGlobal, 0, new Float32Array([cfg.rope_theta_global, 0]));

  // --- bind groups (mirror runner.py, incl. qkv/gateup slice offsets) ---
  const bg = (name, ...bufs) => device.createBindGroup({
    layout: pipelines[name].layout,
    entries: bufs.map((b, i) => ({
      binding: i,
      resource: Array.isArray(b)
        ? { buffer: b[0], offset: b[1], size: b[2] }
        : { buffer: b },
    })),
  });

  const qBytes = qDim * 4, kvBytes = hd * 4;
  const layers = [];
  for (let L = 0; L < cfg.num_layers; L++) {
    const sliding = cfg.layer_types[L] === "sliding_attention";
    const ropeF = sliding ? ropeLocal : ropeGlobal;
    const scD = sliding ? D.scSlide : D.scFull;
    layers.push({
      norm1: bg("rmsnorm_wg", B.x, W[`L${L}.norm_in`], B.xn, D.normH),
      mvQkv: bg("matvec_wg_packed", W[`L${L}.qkv`], B.xn, B.qkv, D.mvQkv),
      qnorm: bg("rmsnorm_wg", [B.qkv, 0, qBytes], W[`L${L}.q_norm`], B.qn, D.normQ),
      knorm: bg("rmsnorm_wg", [B.qkv, qBytes, kvBytes], W[`L${L}.k_norm`], B.kn, D.normK),
      ropeQ: bg("rope", B.qn, ropeF, D.ropeQ),
      ropeK: bg("rope", B.kn, ropeF, D.ropeK),
      appK: bg("kv_append", B.kn, kCache[L], D.kvAppend),
      appV: bg("kv_append", [B.qkv, qBytes + kvBytes, kvBytes], vCache[L], D.kvAppend),
      attn: bg("attention_fused", B.qn, kCache[L], vCache[L], B.scores, B.attn, scD),
      mvO: bg("matvec_wg_packed", W[`L${L}.o`], B.attn, B.attnProj, D.mvO),
      normPaAdd: bg("rmsnorm_add_wg", B.attnProj, W[`L${L}.norm_pa`], B.x, D.normH),
      normPf: bg("rmsnorm_wg", B.x, W[`L${L}.norm_pf`], B.xn, D.normH),
      mvGateup: bg("matvec_wg_packed", W[`L${L}.gateup`], B.xn, B.gateup, D.mvGateup),
      geglu: bg("geglu", [B.gateup, 0, inter * 4], [B.gateup, inter * 4, inter * 4],
                B.ffh, D.geglu),
      mvDown: bg("matvec_wg_packed", W[`L${L}.down`], B.ffh, B.mlpOut, D.mvDown),
      normPffAdd: bg("rmsnorm_add_wg", B.mlpOut, W[`L${L}.norm_pff`], B.x, D.normH),
    });
  }
  const bgEmbed = bg("embed_scale_packed", B.token, W.embed, B.x, D.embed);
  const bgFinalNorm = bg("rmsnorm_wg", B.x, W.final_norm, B.xn, D.normH);
  const bgLogits = bg("matvec_packed", W.embed, B.xn, B.logits, D.mvLogits);
  const bgSetup = bg("step_setup", B.pos, D.kvAppend, D.scSlide, D.scFull,
                     ropeLocal, ropeGlobal, D.setupCfg);
  const bgArgmax1 = bg("argmax_stage1", B.logits, B.partVal, B.partIdx, D.argmax1);
  const bgArgmax2 = bg("argmax_stage2", B.partVal, B.partIdx, B.token,
                       B.outTokens, B.counter, D.argmax2);

  G = { device, cfg, pipelines, B, D, W, layers, kCache, vCache, staging,
        ropeLocal, ropeGlobal,
        bgEmbed, bgFinalNorm, bgLogits, bgSetup, bgArgmax1, bgArgmax2 };
  status("ready — model loaded, all shaders compiled from the DSL");
  ui.bar.style.width = "100%";
  ui.button.disabled = false;
}

// ---------------------------------------------------------- forward pass

function encodeForward(pass, wantLogits) {
  const { cfg, pipelines, layers } = G;
  const h = cfg.hidden_size, hd = cfg.head_dim, nh = cfg.num_heads;
  const inter = cfg.intermediate_size, qDim = nh * hd;
  const run = (name, group, wgs) => {
    pass.setPipeline(pipelines[name].pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(wgs);
  };
  run("embed_scale_packed", G.bgEmbed, Math.ceil(h / 2 / 64));
  for (const L of layers) {
    run("rmsnorm_wg", L.norm1, 1);
    run("matvec_wg_packed", L.mvQkv, qDim + 2 * hd);
    run("rmsnorm_wg", L.qnorm, nh);
    run("rmsnorm_wg", L.knorm, 1);
    run("rope", L.ropeQ, Math.ceil(qDim / 2 / 64));
    run("rope", L.ropeK, Math.ceil(hd / 2 / 64));
    run("kv_append", L.appK, Math.ceil(hd / 64));
    run("kv_append", L.appV, Math.ceil(hd / 64));
    run("attention_fused", L.attn, nh);
    run("matvec_wg_packed", L.mvO, h);
    run("rmsnorm_add_wg", L.normPaAdd, 1);
    run("rmsnorm_wg", L.normPf, 1);
    run("matvec_wg_packed", L.mvGateup, 2 * inter);
    run("geglu", L.geglu, Math.ceil(inter / 64));
    run("matvec_wg_packed", L.mvDown, h);
    run("rmsnorm_add_wg", L.normPffAdd, 1);
  }
  if (wantLogits) {
    run("rmsnorm_wg", G.bgFinalNorm, 1);
    run("matvec_packed", G.bgLogits, Math.ceil(cfg.vocab_size / 64));
  }
}

function writeStepParams(tokenId, pos) {
  const { device, cfg, B, D } = G;
  const q = device.queue;
  const kvLen = pos + 1;
  const start = Math.max(0, kvLen - cfg.sliding_window);
  q.writeBuffer(B.token, 0, new Uint32Array([tokenId]));
  q.writeBuffer(D.kvAppend, 4, new Uint32Array([pos]));
  q.writeBuffer(D.scSlide, 8, new Uint32Array([kvLen, start]));
  q.writeBuffer(D.scFull, 8, new Uint32Array([kvLen]));
  q.writeBuffer(G.ropeLocal, 4, new Float32Array([pos]));
  q.writeBuffer(G.ropeGlobal, 4, new Float32Array([pos]));
}

// ------------------------------------------------------------ generation

async function generate(promptIds, maxNew, onText) {
  const { device, cfg, B } = G;
  const q = device.queue;

  // CPU-driven prefill for all but the last prompt token
  for (let pos = 0; pos < promptIds.length - 1; pos++) {
    writeStepParams(promptIds[pos], pos);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    encodeForward(pass, false);
    pass.end();
    q.submit([enc.finish()]);
  }

  // seed resident state: last prompt token produces generation[0]
  const startPos = promptIds.length - 1;
  q.writeBuffer(B.token, 0, new Uint32Array([promptIds[startPos]]));
  q.writeBuffer(B.pos, 0, new Uint32Array([startPos]));
  q.writeBuffer(B.counter, 0, new Uint32Array([0]));

  const budget = Math.min(maxNew, MAX_SEQ - startPos);
  const genIds = [];
  let produced = 0;
  const t0 = performance.now();

  while (produced < budget) {
    const k = Math.min(CHUNK, budget - produced);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    const run = (name, group, wgs) => {
      pass.setPipeline(G.pipelines[name].pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(wgs);
    };
    for (let i = 0; i < k; i++) {
      run("step_setup", G.bgSetup, 1);
      encodeForward(pass, true);
      run("argmax_stage1", G.bgArgmax1, N_ARGMAX_WGS);
      run("argmax_stage2", G.bgArgmax2, 1);
    }
    pass.end();
    produced += k;
    enc.copyBufferToBuffer(B.outTokens, 0, G.staging, 0, produced * 4);
    q.submit([enc.finish()]);

    await G.staging.mapAsync(GPUMapMode.READ, 0, produced * 4);
    const toks = new Uint32Array(G.staging.getMappedRange(0, produced * 4)).slice();
    G.staging.unmap();

    let stop = false;
    for (let i = genIds.length; i < toks.length; i++) {
      genIds.push(toks[i]);
      if (cfg.eos_token_ids.includes(toks[i])) { stop = true; break; }
    }
    const dt = (performance.now() - t0) / 1000;
    const detok = await fetch("/detokenize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: genIds }),
    }).then(r => r.json());
    onText(detok.text, genIds.length, (promptIds.length + genIds.length) / dt);
    if (stop) break;
  }
  return genIds;
}

// ------------------------------------------------------------------- UI

ui.button.addEventListener("click", async () => {
  ui.button.disabled = true;
  ui.output.textContent = "";
  ui.stats.textContent = "";
  try {
    status("tokenizing…");
    const tok = await fetch("/tokenize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: ui.prompt.value, chat: true }),
    }).then(r => r.json());
    status(`generating (prompt: ${tok.ids.length} tokens)…`);
    await generate(tok.ids, 256, (text, n, tps) => {
      ui.output.textContent = text;
      ui.stats.textContent = `${n} tokens · ${tps.toFixed(1)} tok/s`;
    });
    status("done");
  } catch (e) {
    status(`error: ${e.message ?? e}`);
    console.error(e);
  } finally {
    ui.button.disabled = false;
  }
});

init().catch(e => console.error(e));
