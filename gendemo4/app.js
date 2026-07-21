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

let G = null;
const status = (m) => { ui.status.textContent = m; };
const ceilDiv = (n, d) => Math.ceil(n / d);

// kernel/grid selection — mirrors qat_runner helpers
const mvKernel = (kind) =>
  ({ dq4: "matvec_dq4_blk2", dq2: "matvec_dq2", f16: "matvec_wg_packed_v4" }[kind]);
const gateupKernel = (kind) =>
  kind === "dq2" ? "mv_gateup_geglu_dq2_blk8" : "mv_gateup_geglu_dq4";
const gateupGrid = (rec) => (rec.kind === "dq2" ? rec.n_out / 8 : rec.n_out);

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
  const maxTensor = manifest.tensors.reduce((m, t) => Math.max(m, t.byteLength), 0);
  const need = Math.max(maxTensor, 1 << 28); // ~1.17GB PLE table drives this
  if (adapter.limits.maxStorageBufferBindingSize < maxTensor ||
      adapter.limits.maxBufferSize < maxTensor) {
    status(`GPU limit too small: needs a ${(maxTensor / 1e9).toFixed(2)} GB buffer ` +
           `(PLE table). This machine's GPU can't hold it.`);
    throw new Error("limits");
  }
  const device = await adapter.requestDevice({
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
  const bgFinalNorm = bg("rmsnorm_wg", B.x, W.norm, B.xn, D.normH);
  const bgLogits = bg("matvec_dq2_blk16", W.embed, B.xn, W.embed_scale, B.logits, D.mvVocab);
  const bgArgmax1 = bg("argmax_stage1", B.logits, B.partVal, B.partIdx, D.argmax1);
  const bgArgmax2 = bg("argmax_stage2", B.partVal, B.partIdx, B.token, B.outTokens, B.counter, D.argmax2);
  const bgSetup = bg("step_setup_g4", B.pos, D.kvAppendS, D.kvAppendF, D.scSlide, D.scFull,
                     D.ropeLocal, D.ropeGlobal, D.setupCfg);
  const bgPleGather = bg("qat_ple_gather_4bit", B.token, W.ple_table, W.ple_table_scale,
                         B.pleTok, FP.ple_scale, D.pleGather);
  const bgPleCtx = bg("matvec_wg_packed_v4", W.ple_model_proj, B.x, B.pleCtx, D.mvPleCtx);
  const bgPleCtxNorm = bg("rmsnorm_wg", B.pleCtx, W.ple_proj_norm, B.pleCtxN, D.normPleRows);
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
      norm1: bg("rmsnorm_wg", B.x, W[`L${L}.norm_in`], B.xn, D.normH),
      q: mvBg(qR, B.xn, B.q),
      qnorm: bg("rmsnorm_wg", B.q, W[`L${L}.q_norm`], B.qn, D[`normQ_${tag}`]),
      ropeQ: bg("rope_pl", B.qn, ropeF, D[`ropeQ_${tag}`]),
      attn: bg("attention_fused_g4", B.qn, kCache[s.kv_source], vCache[s.kv_source],
               B.scores, B.attn, scD),
      o: mvBg(oR, B.attn, B.attnProj),
      normPaPf: bg("rmsnorm_add_norm_wg", B.attnProj, W[`L${L}.norm_pa`], W[`L${L}.norm_pf`],
                   B.x, B.xn, D.normH),
      gateup: bg(gateupKernel(gR.kind), W[gR.w], W[uR.w], B.xn, W[gR.scale], W[uR.scale],
                 B.ffh, dimsFor(gR)),
      down: dR.kind === "dq2"
        ? bg("matvec_dq2_blk2", W[dR.w], B.ffh, W[dR.scale], B.mlpOut, dimsFor(dR))
        : mvBg(dR, B.ffh, B.mlpOut),
      normPffAdd: bg("rmsnorm_add_wg", B.mlpOut, W[`L${L}.norm_pff`], B.x, D.normH),
      pleGateup: bg("mv_geglu_f16", W[pgR.w], B.x, [B.pleIn, L * pleBytes, pleBytes],
                    B.pleH, dimsFor(pgR)),
      pleProj: mvBg(ppR, B.pleH, B.pleProj),
      pleNormAdd: bg("rmsnorm_add_scale_wg", B.pleProj, W[`L${L}.ple_norm`], B.x,
                     W[`L${L}.layer_scalar`], D.normH),
    };
    if (!s.kv_shared) {
      const kR = lins[`L${L}.k`], vR = lins[`L${L}.v`];
      o.kR = kR; o.vR = vR;
      o.k = mvBg(kR, B.xn, B.k);
      o.knorm = bg("rmsnorm_wg", B.k, W[`L${L}.k_norm`], B.kn, D[`normK_${tag}`]);
      o.ropeK = bg("rope_pl", B.kn, ropeF, D[`ropeK_${tag}`]);
      o.v = mvBg(vR, B.xn, B.v);
      o.vnorm = bg("rmsnorm_ns_wg", B.v, B.vn, D[`normK_${tag}`]);
      o.appK = bg("kv_append", B.kn, kCache[L], D[`kvAppend_${tag}`]);
      o.appV = bg("kv_append", B.vn, vCache[L], D[`kvAppend_${tag}`]);
    }
    return o;
  });

  G = { device, cfg, pipelines, B, D, W, layers, staging, nh, h, pleH, pleN, nLayers, vocab,
        bgEmbed, bgFinalNorm, bgLogits, bgArgmax1, bgArgmax2, bgSetup,
        bgPleGather, bgPleCtx, bgPleCtxNorm, bgPleCombine };
  status("ready — Gemma 4 E2B QAT loaded, all shaders compiled from the DSL");
  ui.bar.style.width = "100%";
  ui.button.disabled = false;
}

// ---------------------------------------------------------- forward pass

function encodeForward(pass, wantLogits) {
  const { cfg, pipelines, layers, nh, h, pleH, pleN, nLayers, vocab } = G;
  const run = (name, group, wgs) => {
    pass.setPipeline(pipelines[name].pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(wgs);
  };
  const mv = (rec, group) =>
    run(mvKernel(rec.kind), group, mvKernel(rec.kind).endsWith("_blk2") ? rec.n_out / 2 : rec.n_out);

  run("qat_embed_2bit", G.bgEmbed, ceilDiv(h, 64));
  run("matvec_wg_packed_v4", G.bgPleCtx, cfg.ple_model_proj_nout);
  run("rmsnorm_wg", G.bgPleCtxNorm, nLayers);
  run("qat_ple_gather_4bit", G.bgPleGather, ceilDiv(pleN, 64));
  run("combine_scaled", G.bgPleCombine, ceilDiv(pleN, 64));

  for (const L of layers) {
    const s = L.spec, hd = s.head_dim;
    run("rmsnorm_wg", L.norm1, 1);
    mv(L.qR, L.q);
    run("rmsnorm_wg", L.qnorm, nh);
    run("rope_pl", L.ropeQ, ceilDiv(nh * hd / 2, 64));
    if (!s.kv_shared) {
      mv(L.kR, L.k);
      run("rmsnorm_wg", L.knorm, 1);
      run("rope_pl", L.ropeK, ceilDiv(hd / 2, 64));
      mv(L.vR, L.v);
      run("rmsnorm_ns_wg", L.vnorm, 1);
      run("kv_append", L.appK, ceilDiv(hd, 64));
      run("kv_append", L.appV, ceilDiv(hd, 64));
    }
    run("attention_fused_g4", L.attn, nh);
    mv(L.oR, L.o);
    run("rmsnorm_add_norm_wg", L.normPaPf, 1);
    run(gateupKernel(L.gR.kind), L.gateup, gateupGrid(L.gR));
    run(L.dR.kind === "dq2" ? "matvec_dq2_blk2" : "matvec_dq4_blk2", L.down, L.dR.n_out / 2);
    run("rmsnorm_add_wg", L.normPffAdd, 1);
    run("mv_geglu_f16", L.pleGateup, pleH);
    mv(L.ppR, L.pleProj);
    run("rmsnorm_add_scale_wg", L.pleNormAdd, 1);
  }
  if (wantLogits) {
    run("rmsnorm_wg", G.bgFinalNorm, 1);
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
    const run = (name, group, wgs) => {
      pass.setPipeline(G.pipelines[name].pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(wgs);
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
