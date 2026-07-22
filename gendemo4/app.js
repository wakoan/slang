// Gemma 4 E2B QAT in the browser: WebGPU runner for the DSL-generated WGSL
// kernels. Mirrors gemma4/qat_runner.py (int2/4/8 weights, PLE, KV-share
// aliasing, dual head dims, output-blocked matmuls, GPU-resident decode).

const ui = {
  status: document.getElementById("status"),
  bar: document.getElementById("bar"),
  prompt: document.getElementById("prompt"),
  output: document.getElementById("output"),
  button: document.getElementById("go"),
  stats: document.getElementById("stats"),
  cacheMsg: document.getElementById("cachemsg"),
  clearCache: document.getElementById("clearcache"),
};

const N_ARGMAX_WGS = 128;
const MAX_SEQ = 512;
const CHUNK = 16;
// FALSIFIED experiment (kept as toggle): int8 activations + dot4I8Packed for the
// wide (2-bit) down_proj. Neutral end-to-end in the browser (67.4 → 67.4) — the
// dq matmuls are load-issue-bound (why output-blocking won), and dot4I8Packed only
// speeds the dot compute we're not bound on, and can't shrink weight traffic.
const USE_INT8_DOWN = false;
// Experiment: 256-thread norms (grid=1 single-workgroup norms are ~24% of the
// browser profile; 4x threads hide their memory latency). A/B via this toggle.
const USE_T256_NORMS = true;
const K_RMS = USE_T256_NORMS ? "rmsnorm_wg_t256" : "rmsnorm_wg";
const K_RMS_ADD_NORM = USE_T256_NORMS ? "rmsnorm_add_norm_wg_t256" : "rmsnorm_add_norm_wg";
const K_RMS_ADD = USE_T256_NORMS ? "rmsnorm_add_wg_t256" : "rmsnorm_add_wg";
const K_RMS_ADD_SCALE = USE_T256_NORMS ? "rmsnorm_add_scale_wg_t256" : "rmsnorm_add_scale_wg";
// Experiment: flash-decoding attention (attention_fused_g4 runs grid=nh=8,
// occupancy-starved; split KV across nh*S workgroups + online-softmax combine).
const USE_FLASH_ATTN = true;
const FLASH_S = 8;
// A/B: threads/workgroup for the wide (2-bit) down_proj (long n_in=12288 reduction)
const DOWN_DQ2_KERNEL = "matvec_dq2_blk2_t128"; // matvec_dq2_blk2 (64) | _t128 | _t256

let G = null;
const status = (m) => { ui.status.textContent = m; };
const ceilDiv = (n, d) => Math.ceil(n / d);

// kernel/grid selection — mirrors qat_runner helpers
const mvKernel = (kind) =>
  ({ dq4: "matvec_dq4_blk2", dq2: "matvec_dq2", f16: "matvec_wg_packed_v4" }[kind]);
// wide (2-bit) gate/up: blk8 (subgroupAdd sg4 was slower, 59 vs 85 — falsified).
// narrow (4-bit) gate/up: blk2 (was unblocked).
const gateupKernel = (kind) =>
  kind === "dq2" ? "mv_gateup_geglu_dq2_blk8" : "mv_gateup_geglu_dq4_blk2";
const gateupGrid = (rec) => rec.n_out / (rec.kind === "dq2" ? 8 : 2);

// ---------------------------------------------------------------- init

async function init() {
  if (!navigator.gpu) {
    status("WebGPU not available — use Chrome/Edge.");
    throw new Error("no webgpu");
  }
  const [kernels, manifest] = await Promise.all([
    fetch("/kernels.json").then((r) => r.json()),
    fetch("/manifest.json").then((r) => r.json()),
  ]);
  const cfg = manifest.config;
  const lins = manifest.linears;
  const specs = manifest.layers;

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) { status("No GPU adapter."); throw new Error("no adapter"); }
  // --- feature probe: what does this browser expose? (gates the subgroup-matrix path) ---
  const feats = [...adapter.features];
  const hasSgMat = feats.includes("chromium-experimental-subgroup-matrix");
  const wgslFeats = navigator.gpu.wgslLanguageFeatures
    ? [...navigator.gpu.wgslLanguageFeatures] : [];
  console.log("[gpu] adapter features:", feats.sort());
  console.log("[gpu] wgslLanguageFeatures:", wgslFeats.sort());
  console.log("[gpu] subgroup-matrix:", hasSgMat,
              "| subgroups:", feats.includes("subgroups"),
              "| shader-f16:", feats.includes("shader-f16"));
  console.log("[gpu] subgroupMatrixConfigs:", adapter.info?.subgroupMatrixConfigs ?? "(none)");
  window.__gpuFeatures = feats; // inspect in console: __gpuFeatures
  const maxTensor = manifest.tensors.reduce((m, t) => Math.max(m, t.byteLength), 0);
  const need = Math.max(maxTensor, 1 << 28); // ~1.17GB PLE table drives this
  if (adapter.limits.maxStorageBufferBindingSize < maxTensor ||
      adapter.limits.maxBufferSize < maxTensor) {
    status(`GPU limit too small: needs a ${(maxTensor / 1e9).toFixed(2)} GB buffer ` +
           `(PLE table). This machine's GPU can't hold it.`);
    throw new Error("limits");
  }
  const canProfile = feats.includes("timestamp-query");
  const wantFeatures = [];
  if (canProfile) wantFeatures.push("timestamp-query");
  if (feats.includes("subgroups")) wantFeatures.push("subgroups");   // subgroupAdd kernels
  if (feats.includes("shader-f16")) wantFeatures.push("shader-f16"); // native f16 kernels
  const device = await adapter.requestDevice({
    requiredFeatures: wantFeatures,
    requiredLimits: { maxBufferSize: need, maxStorageBufferBindingSize: need },
  });
  device.lost.then((i) => status(`GPU device lost: ${i.message}`));

  // --- weights: Cache Storage first, network once ---
  const cacheKey = `/weights.bin?v=${manifest.weightsVersion ?? "0"}`;
  let cache = null;
  try { cache = await caches.open("gemma4-qat-weights"); } catch { /* unavailable */ }
  let blob = null, fromCache = false;
  if (cache) {
    const hit = await cache.match(cacheKey);
    if (hit) {
      status("loading weights from browser cache…");
      blob = new Uint8Array(await hit.arrayBuffer());
      ui.bar.style.width = "100%";
      fromCache = true;
    }
  }
  if (!blob) {
    status("downloading weights (~2 GB, one-time)…");
    const resp = await fetch("/weights.bin");
    const total = Number(resp.headers.get("Content-Length")) || manifest.totalBytes;
    blob = new Uint8Array(total);
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
    if (cache) {
      try {
        for (const k of await cache.keys()) {
          if (new URL(k.url).pathname === "/weights.bin") await cache.delete(k);
        }
        await cache.put(cacheKey, new Response(blob.slice().buffer, {
          headers: { "Content-Type": "application/octet-stream" },
        }));
        navigator.storage?.persist?.();
      } catch (e) {
        console.warn("weight caching failed (quota?):", e);
        cache = null;
      }
    }
  }
  const gb = (blob.length / 1e9).toFixed(2);
  ui.cacheMsg.textContent = fromCache
    ? `weights: ${gb} GB loaded from browser cache ✓`
    : cache ? `weights: ${gb} GB downloaded — cached, next load will be instant`
            : `weights: ${gb} GB downloaded (browser cache unavailable)`;
  if (cache) {
    ui.clearCache.hidden = false;
    ui.clearCache.onclick = async () => {
      await caches.delete("gemma4-qat-weights");
      ui.cacheMsg.textContent = "cache cleared — next load will re-download";
      ui.clearCache.hidden = true;
    };
  }

  status("uploading to GPU (~2 GB)…");
  const W = {};
  for (const t of manifest.tensors) {
    const buf = device.createBuffer({
      size: t.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, blob.buffer, t.offset, t.byteLength);
    W[t.name] = buf;
  }

  // --- pipelines from WGSL binding decls ---
  const pipelines = {};
  for (const [name, k] of Object.entries(kernels)) {
    const access = [...k.wgsl.matchAll(/@binding\((\d+)\) var<storage, (read_write|read)>/g)]
      .sort((a, b) => +a[1] - +b[1]).map((m) => m[2] === "read_write");
    const layout = device.createBindGroupLayout({
      entries: access.map((rw, i) => ({
        binding: i, visibility: GPUShaderStage.COMPUTE,
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

  // --- dims ---
  const h = cfg.hidden_size, nh = cfg.num_heads, pleH = cfg.ple_hidden;
  const nLayers = cfg.num_layers, vocab = cfg.vocab_size;
  const pleN = nLayers * pleH;
  const hdS = specs.find((s) => s.sliding).head_dim;
  const hdF = specs.find((s) => !s.sliding).head_dim;
  const cutS = specs.find((s) => s.sliding).rope_cutoff;
  const cutF = specs.find((s) => !s.sliding).rope_cutoff;
  const qMax = Math.max(...specs.map((s) => s.q_dim));
  const hdMax = Math.max(hdS, hdF);
  const interMax = Math.max(...specs.map((s) => s.intermediate));

  const ubuf = (...v) => {
    const b = device.createBuffer({
      size: v.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, new Uint32Array(v));
    return b;
  };
  const fpar = (...v) => {
    const b = device.createBuffer({
      size: v.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, new Float32Array(v));
    return b;
  };
  const dyn = (n) => device.createBuffer({
    size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  const D = {
    embed: ubuf(h), normH: ubuf(1, h),
    normPleRows: ubuf(nLayers, pleH), combine: ubuf(pleN),
    pleGather: ubuf(pleN, pleH, nLayers), mvPleCtx: ubuf(cfg.ple_model_proj_nout, h),
    argmax1: ubuf(vocab, N_ARGMAX_WGS), argmax2: ubuf(N_ARGMAX_WGS),
    setupCfg: ubuf(cfg.sliding_window),
    kvAppendS: dyn(2), kvAppendF: dyn(2),
    scSlide: dyn(5), scFull: dyn(5),
    ropeLocal: dyn(2), ropeGlobal: dyn(2),
    mvVocab: ubuf(vocab, h),
    normQ_s: ubuf(nh, hdS), normQ_f: ubuf(nh, hdF),
    normK_s: ubuf(1, hdS), normK_f: ubuf(1, hdF),
    ropeQ_s: ubuf(nh, hdS, cutS), ropeQ_f: ubuf(nh, hdF, cutF),
    ropeK_s: ubuf(1, hdS, cutS), ropeK_f: ubuf(1, hdF, cutF),
    sbuf: ubuf(FLASH_S),
    flCombine_s: ubuf(nh, hdS, FLASH_S), flCombine_f: ubuf(nh, hdF, FLASH_S),
  };
  // kv_append bind groups read the same buffers step_setup_g4 / writeStepParams write
  D.kvAppend_s = D.kvAppendS; D.kvAppend_f = D.kvAppendF;
  const FP = {
    embed_scale: fpar(cfg.embed_scale), combine: fpar(Math.pow(2, -0.5)),
    ple_scale: fpar(cfg.ple_scale), softcap: fpar(cfg.softcap),
  };
  const mvDims = {};
  const dimsFor = (rec) => {
    const key = `${rec.n_out}x${rec.n_in}`;
    return (mvDims[key] ||= ubuf(rec.n_out, rec.n_in));
  };

  // --- scratch buffers ---
  const fbuf = (n) => device.createBuffer({
    size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const B = {
    x: fbuf(h), xn: fbuf(h), q: fbuf(qMax), qn: fbuf(qMax),
    k: fbuf(hdMax), kn: fbuf(hdMax), v: fbuf(hdMax), vn: fbuf(hdMax),
    scores: fbuf(nh * MAX_SEQ), attn: fbuf(qMax), attnProj: fbuf(h),
    ffh: fbuf(interMax), mlpOut: fbuf(h), logits: fbuf(vocab),
    pleCtx: fbuf(pleN), pleCtxN: fbuf(pleN), pleIn: fbuf(pleN),
    pleH: fbuf(pleH), pleProj: fbuf(h),
    token: fbuf(1), pos: fbuf(1), counter: fbuf(1),
    outTokens: fbuf(MAX_SEQ), pleTok: fbuf(pleN),
    partVal: fbuf(N_ARGMAX_WGS), partIdx: fbuf(N_ARGMAX_WGS),
    ffhQ: device.createBuffer({   // int8-packed ffh (interMax bytes), + scale
      size: interMax, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
    actScale: fbuf(1),
    flM: fbuf(nh * FLASH_S), flL: fbuf(nh * FLASH_S), flO: fbuf(nh * FLASH_S * hdMax),
  };
  const kCache = {}, vCache = {};
  for (const s of specs) if (!s.kv_shared) {
    kCache[s.index] = fbuf(MAX_SEQ * s.head_dim);
    vCache[s.index] = fbuf(MAX_SEQ * s.head_dim);
  }
  const staging = device.createBuffer({
    size: MAX_SEQ * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  // static rope thetas
  device.queue.writeBuffer(D.ropeLocal, 0, new Float32Array([specs.find(s => s.sliding).rope_theta, 0]));
  device.queue.writeBuffer(D.ropeGlobal, 0, new Float32Array([specs.find(s => !s.sliding).rope_theta, 0]));

  // --- bind groups ---
  const bg = (name, ...bufs) => device.createBindGroup({
    layout: pipelines[name].layout,
    entries: bufs.map((b, i) => ({
      binding: i,
      resource: Array.isArray(b) ? { buffer: b[0], offset: b[1], size: b[2] } : { buffer: b },
    })),
  });
  const mvBg = (rec, xBuf, yBuf) => rec.kind === "f16"
    ? bg(mvKernel(rec.kind), W[rec.w], xBuf, yBuf, dimsFor(rec))
    : bg(mvKernel(rec.kind), W[rec.w], xBuf, W[rec.scale], yBuf, dimsFor(rec));

  const bgEmbed = bg("qat_embed_2bit", B.token, W.embed, W.embed_scale, B.x, FP.embed_scale, D.embed);
  const bgFinalNorm = bg(K_RMS, B.x, W.norm, B.xn, D.normH);
  const bgLogits = bg("matvec_dq2_blk16", W.embed, B.xn, W.embed_scale, B.logits, D.mvVocab);
  const bgArgmax1 = bg("argmax_stage1", B.logits, B.partVal, B.partIdx, D.argmax1);
  const bgArgmax2 = bg("argmax_stage2", B.partVal, B.partIdx, B.token, B.outTokens, B.counter, D.argmax2);
  const bgSetup = bg("step_setup_g4", B.pos, D.kvAppendS, D.kvAppendF, D.scSlide, D.scFull,
                     D.ropeLocal, D.ropeGlobal, D.setupCfg);
  const bgPleGather = bg("qat_ple_gather_4bit", B.token, W.ple_table, W.ple_table_scale,
                         B.pleTok, FP.ple_scale, D.pleGather);
  const bgPleCtx = bg("matvec_wg_packed_v4", W.ple_model_proj, B.x, B.pleCtx, D.mvPleCtx);
  const bgPleCtxNorm = bg(K_RMS, B.pleCtx, W.ple_proj_norm, B.pleCtxN, D.normPleRows);
  const bgPleCombine = bg("combine_scaled", B.pleCtxN, B.pleTok, B.pleIn, FP.combine, D.combine);
  const pleBytes = pleH * 4;

  const layers = specs.map((s) => {
    const L = s.index, tag = s.sliding ? "s" : "f";
    const ropeF = s.sliding ? D.ropeLocal : D.ropeGlobal;
    const scD = s.sliding ? D.scSlide : D.scFull;
    const qR = lins[`L${L}.q`], oR = lins[`L${L}.o`], gR = lins[`L${L}.gate`];
    const uR = lins[`L${L}.up`], dR = lins[`L${L}.down`];
    const pgR = lins[`L${L}.ple_gate`], ppR = lins[`L${L}.ple_proj`];
    const o = {
      spec: s, qR, oR, gR, dR, ppR, pgR,
      norm1: bg(K_RMS, B.x, W[`L${L}.norm_in`], B.xn, D.normH),
      q: mvBg(qR, B.xn, B.q),
      qnorm: bg(K_RMS, B.q, W[`L${L}.q_norm`], B.qn, D[`normQ_${tag}`]),
      ropeQ: bg("rope_pl", B.qn, ropeF, D[`ropeQ_${tag}`]),
      attn: bg("attention_fused_g4", B.qn, kCache[s.kv_source], vCache[s.kv_source],
               B.scores, B.attn, scD),
      attnP: bg("attn_flash_partial", B.qn, kCache[s.kv_source], vCache[s.kv_source],
                B.flM, B.flL, B.flO, scD, D.sbuf),
      attnC: bg("attn_flash_combine", B.flM, B.flL, B.flO, B.attn, D[`flCombine_${tag}`]),
      o: mvBg(oR, B.attn, B.attnProj),
      normPaPf: bg(K_RMS_ADD_NORM, B.attnProj, W[`L${L}.norm_pa`], W[`L${L}.norm_pf`],
                   B.x, B.xn, D.normH),
      gateup: bg(gateupKernel(gR.kind), W[gR.w], W[uR.w], B.xn, W[gR.scale], W[uR.scale],
                 B.ffh, dimsFor(gR)),
      down: dR.kind === "dq2"
        ? bg(DOWN_DQ2_KERNEL, W[dR.w], B.ffh, W[dR.scale], B.mlpOut, dimsFor(dR))
        : mvBg(dR, B.ffh, B.mlpOut),
      // int8 experiment: quantize ffh then dot4I8Packed down (dq2 layers only)
      quantDown: dR.kind === "dq2"
        ? bg("quant_i8", B.ffh, B.ffhQ, B.actScale, (mvDims[`n${dR.n_in}`] ||= ubuf(dR.n_in)))
        : null,
      downI8: dR.kind === "dq2"
        ? bg("matvec_dq2_i8", W[dR.w], B.ffhQ, W[dR.scale], B.actScale, B.mlpOut, dimsFor(dR))
        : null,
      normPffAdd: bg(K_RMS_ADD, B.mlpOut, W[`L${L}.norm_pff`], B.x, D.normH),
      pleGateup: bg("mv_geglu_f16", W[pgR.w], B.x, [B.pleIn, L * pleBytes, pleBytes],
                    B.pleH, dimsFor(pgR)),
      pleProj: mvBg(ppR, B.pleH, B.pleProj),
      pleNormAdd: bg(K_RMS_ADD_SCALE, B.pleProj, W[`L${L}.ple_norm`], B.x,
                     W[`L${L}.layer_scalar`], D.normH),
    };
    if (!s.kv_shared) {
      const kR = lins[`L${L}.k`], vR = lins[`L${L}.v`];
      o.kR = kR; o.vR = vR;
      o.k = mvBg(kR, B.xn, B.k);
      o.knorm = bg(K_RMS, B.k, W[`L${L}.k_norm`], B.kn, D[`normK_${tag}`]);
      o.ropeK = bg("rope_pl", B.kn, ropeF, D[`ropeK_${tag}`]);
      o.v = mvBg(vR, B.xn, B.v);
      o.vnorm = bg("rmsnorm_ns_wg", B.v, B.vn, D[`normK_${tag}`]);
      o.appK = bg("kv_append", B.kn, kCache[L], D[`kvAppend_${tag}`]);
      o.appV = bg("kv_append", B.vn, vCache[L], D[`kvAppend_${tag}`]);
    }
    return o;
  });

  G = { device, cfg, pipelines, B, D, W, layers, staging, canProfile, nh, h, pleH, pleN, nLayers, vocab,
        bgEmbed, bgFinalNorm, bgLogits, bgArgmax1, bgArgmax2, bgSetup,
        bgPleGather, bgPleCtx, bgPleCtxNorm, bgPleCombine };
  status("ready — Gemma 4 E2B QAT loaded, all shaders compiled from the DSL");
  ui.bar.style.width = "100%";
  ui.button.disabled = false;
  window.profileDecode = profileDecode;   // run in console: profileDecode()
  window.bench = benchDecode;             // headless/console benchmark: returns {tps,...}
  window.__ready = true;
  if (canProfile) console.log("[profile] call profileDecode() in the console for a per-kernel GPU breakdown");
}

// Headless-friendly benchmark: run a fixed decode, return the tok/s the UI shows.
async function benchDecode(prompt = "Write a detailed story about a dragon who learns to code.", nTokens = 128, runs = 2) {
  if (!G) return { error: "not loaded" };
  let best = 0, text = "";
  for (let r = 0; r < runs; r++) {
    let tps = 0;
    const out = await generate((await fetch("/tokenize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: prompt, chat: true }),
    }).then((x) => x.json())).ids, nTokens, (t, n, rate) => { tps = rate; text = t; });
    best = Math.max(best, tps);
  }
  return { tps: +best.toFixed(1), chars: text.length, sample: text.slice(0, 90) };
}

// ------------------------------------------------------- browser GPU profiler

async function profileDecode(prompt = "Write a detailed story about a dragon.", nSteps = 40) {
  if (!G) { console.warn("model not loaded yet"); return; }
  if (!G.canProfile) { console.warn("timestamp-query not available"); return; }
  const { device, B } = G;
  const q = device.queue;
  const tok = await fetch("/tokenize", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: prompt, chat: true }),
  }).then((r) => r.json());
  const promptIds = tok.ids;

  // prefill (build KV) with the normal fused path
  for (let pos = 0; pos < promptIds.length - 1; pos++) {
    writeStepParams(promptIds[pos], pos);
    const e = device.createCommandEncoder(); const p = e.beginComputePass();
    encodeForward(p, false); p.end(); q.submit([e.finish()]);
  }
  const startPos = promptIds.length - 1;
  writeStepParams(promptIds[startPos], startPos);
  q.writeBuffer(B.pos, 0, new Uint32Array([startPos]));
  q.writeBuffer(B.counter, 0, new Uint32Array([0]));

  const MAXQ = 4096;
  const qs = device.createQuerySet({ type: "timestamp", count: MAXQ });
  const resolveBuf = device.createBuffer({
    size: MAXQ * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({
    size: MAXQ * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const agg = {};
  console.log(`[profile] warming + timing ${nSteps} decode steps…`);
  for (let step = 0; step < nSteps + 2; step++) {
    const enc = device.createCommandEncoder();
    const labels = [];
    const runP = (name, group, x, y = 1) => {
      const i = labels.length * 2;
      const pass = enc.beginComputePass({ timestampWrites: {
        querySet: qs, beginningOfPassWriteIndex: i, endOfPassWriteIndex: i + 1 } });
      pass.setPipeline(G.pipelines[name].pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(x, y);
      pass.end();
      labels.push(name);
    };
    runP("step_setup_g4", G.bgSetup, 1);
    encodeForwardWith(runP, true);
    runP("argmax_stage1", G.bgArgmax1, N_ARGMAX_WGS);
    runP("argmax_stage2", G.bgArgmax2, 1);
    const nPairs = labels.length;
    enc.resolveQuerySet(qs, 0, nPairs * 2, resolveBuf, 0);
    enc.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, nPairs * 2 * 8);
    q.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ, 0, nPairs * 2 * 8);
    const ts = new BigInt64Array(readBuf.getMappedRange(0, nPairs * 2 * 8)).slice();
    readBuf.unmap();
    if (step < 2) continue; // warmup
    for (let k = 0; k < nPairs; k++) {
      const dt = Number(ts[2 * k + 1] - ts[2 * k]);
      (agg[labels[k]] ||= { ns: 0, count: 0 });
      agg[labels[k]].ns += dt; agg[labels[k]].count++;
    }
  }
  const rows = Object.entries(agg)
    .map(([l, v]) => ({ l, ms: v.ns / 1e6, count: v.count })).sort((a, b) => b.ms - a.ms);
  const tot = rows.reduce((s, r) => s + r.ms, 0);
  console.log(`=== browser GPU profile (${nSteps} steps) — ${(tot / nSteps).toFixed(3)} ms/token summed-passes ===`);
  for (const r of rows) {
    console.log(`${r.l.padEnd(22)} ${r.ms.toFixed(2).padStart(9)} ms  ${(100 * r.ms / tot).toFixed(1).padStart(5)}%  x${r.count / nSteps}`);
  }
  qs.destroy(); resolveBuf.destroy(); readBuf.destroy();
  return rows;
}

// ---------------------------------------------------------- forward pass

function encodeForward(pass, wantLogits) {
  const { pipelines } = G;
  encodeForwardWith((name, group, x, y = 1) => {
    pass.setPipeline(pipelines[name].pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(x, y);
  }, wantLogits);
}

function encodeForwardWith(run, wantLogits) {
  const { cfg, layers, nh, h, pleH, pleN, nLayers, vocab } = G;
  const mv = (rec, group) =>
    run(mvKernel(rec.kind), group, mvKernel(rec.kind).endsWith("_blk2") ? rec.n_out / 2 : rec.n_out);

  run("qat_embed_2bit", G.bgEmbed, ceilDiv(h, 64));
  run("matvec_wg_packed_v4", G.bgPleCtx, cfg.ple_model_proj_nout);
  run(K_RMS, G.bgPleCtxNorm, nLayers);
  run("qat_ple_gather_4bit", G.bgPleGather, ceilDiv(pleN, 64));
  run("combine_scaled", G.bgPleCombine, ceilDiv(pleN, 64));

  for (const L of layers) {
    const s = L.spec, hd = s.head_dim;
    run(K_RMS, L.norm1, 1);
    mv(L.qR, L.q);
    run(K_RMS, L.qnorm, nh);
    run("rope_pl", L.ropeQ, ceilDiv(nh * hd / 2, 64));
    if (!s.kv_shared) {
      mv(L.kR, L.k);
      run(K_RMS, L.knorm, 1);
      run("rope_pl", L.ropeK, ceilDiv(hd / 2, 64));
      mv(L.vR, L.v);
      run("rmsnorm_ns_wg", L.vnorm, 1);
      run("kv_append", L.appK, ceilDiv(hd, 64));
      run("kv_append", L.appV, ceilDiv(hd, 64));
    }
    if (USE_FLASH_ATTN) {
      run("attn_flash_partial", L.attnP, nh, FLASH_S);
      run("attn_flash_combine", L.attnC, nh);
    } else {
      run("attention_fused_g4", L.attn, nh);
    }
    mv(L.oR, L.o);
    run(K_RMS_ADD_NORM, L.normPaPf, 1);
    run(gateupKernel(L.gR.kind), L.gateup, gateupGrid(L.gR));
    if (USE_INT8_DOWN && L.dR.kind === "dq2") {
      run("quant_i8", L.quantDown, 1);            // ffh -> int8 + scale
      run("matvec_dq2_i8", L.downI8, L.dR.n_out); // 1 row/wg, hardware int dot
    } else {
      run(L.dR.kind === "dq2" ? DOWN_DQ2_KERNEL : "matvec_dq4_blk2", L.down, L.dR.n_out / 2);
    }
    run(K_RMS_ADD, L.normPffAdd, 1);
    run("mv_geglu_f16", L.pleGateup, pleH);
    mv(L.ppR, L.pleProj);
    run(K_RMS_ADD_SCALE, L.pleNormAdd, 1);
  }
  if (wantLogits) {
    run(K_RMS, G.bgFinalNorm, 1);
    run("matvec_dq2_blk16", G.bgLogits, vocab / 16);
  }
}

function writeStepParams(tokenId, pos) {
  const { device, cfg, B, D, nh } = G;
  const q = device.queue;
  const kvLen = pos + 1;
  const start = Math.max(0, kvLen - cfg.sliding_window);
  const hdS = G.layers.find((L) => L.spec.sliding).spec.head_dim;
  const hdF = G.layers.find((L) => !L.spec.sliding).spec.head_dim;
  q.writeBuffer(B.token, 0, new Uint32Array([tokenId]));
  q.writeBuffer(D.kvAppendS, 0, new Uint32Array([hdS, pos]));
  q.writeBuffer(D.kvAppendF, 0, new Uint32Array([hdF, pos]));
  q.writeBuffer(D.scSlide, 0, new Uint32Array([nh, hdS, kvLen, start, MAX_SEQ]));
  q.writeBuffer(D.scFull, 0, new Uint32Array([nh, hdF, kvLen, 0, MAX_SEQ]));
  q.writeBuffer(D.ropeLocal, 4, new Float32Array([pos]));  // theta at [0] stays; pos is f32
  q.writeBuffer(D.ropeGlobal, 4, new Float32Array([pos]));
}

// ------------------------------------------------------------ generation

async function generate(promptIds, maxNew, onText) {
  const { device, cfg, B } = G;
  const q = device.queue;
  const t0 = performance.now();

  // CPU-driven prefill for all but the last prompt token
  for (let pos = 0; pos < promptIds.length - 1; pos++) {
    writeStepParams(promptIds[pos], pos);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    encodeForward(pass, false);
    pass.end();
    q.submit([enc.finish()]);
  }

  const startPos = promptIds.length - 1;
  writeStepParams(promptIds[startPos], startPos); // seeds statics + token
  q.writeBuffer(B.pos, 0, new Uint32Array([startPos]));
  q.writeBuffer(B.counter, 0, new Uint32Array([0]));

  const budget = Math.min(maxNew, MAX_SEQ - startPos);
  const genIds = [];
  let produced = 0;

  while (produced < budget) {
    const k = Math.min(CHUNK, budget - produced);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    const run = (name, group, x, y = 1) => {
      pass.setPipeline(G.pipelines[name].pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(x, y);
    };
    for (let i = 0; i < k; i++) {
      run("step_setup_g4", G.bgSetup, 1);
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
    }).then((r) => r.json());
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
    }).then((r) => r.json());
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

init().catch((e) => { console.error(e); });
