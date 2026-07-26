// gemma4_150 browser runner — runs the reference fused SRQ kernels in WebGPU.
// Mirrors gemma4_150/runner.py (persistent pooled buffers + cached bind groups +
// GPU-resident token feedback); Dawn's cheap in-process pass recording is the lever
// past the ~97 tok/s wgpu-py ceiling toward the reference's 150.

const status = document.getElementById("status");
const outEl = document.getElementById("out");
const tpsEl = document.getElementById("tps");
const runBtn = document.getElementById("run");

let device, queue, MAN, CFG, KERN;
const BUF = {};          // weight buffers by tensor name
const POOL = {};         // persistent scratch, by role
const UNIS = {};         // persistent uniform buffers, by name
const BGC = new Map();   // bind-group cache
const PIPES = new Map(); // pipeline cache

const U32 = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const UNI = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

// ---------- kernel text + shape patching ----------
function patch(code, consts) {
  for (const [k, v] of Object.entries(consts)) {
    code = code.replace(new RegExp(`const ${k}: (u32|f32) = [^;]+;`),
      (_m, ty) => `const ${k}: ${ty} = ${v}${ty === "u32" ? "u" : ""};`);
  }
  return code;
}
function accessKinds(wgsl) {
  const slots = {};
  const re = /@group\(0\)\s*@binding\((\d+)\)\s*var<(storage|uniform)(?:,\s*(read|read_write))?>/g;
  let m;
  while ((m = re.exec(wgsl))) slots[+m[1]] = m[2] === "uniform" ? "u" : (m[3] === "read_write" ? "w" : "r");
  return Object.keys(slots).sort((a, b) => a - b).map((i) => slots[i]);
}
function pipe(key, code) {
  let p = PIPES.get(key);
  if (!p) {
    const acc = accessKinds(code);
    const layout = device.createBindGroupLayout({
      entries: acc.map((a, i) => ({
        binding: i, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: a === "u" ? "uniform" : a === "w" ? "storage" : "read-only-storage" },
      })),
    });
    const mod = device.createShaderModule({ code });
    p = {
      pipe: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: mod, entryPoint: "main" },
      }), layout,
    };
    PIPES.set(key, p);
  }
  return p;
}

// ---------- buffers ----------
function scratch(name, nbytes) {
  let b = POOL[name];
  if (!b || b.size < nbytes) { b = device.createBuffer({ size: Math.max(nbytes, 4), usage: U32 }); POOL[name] = b; }
  return b;
}
function uni(name, arr) {
  const data = arr.buffer ? new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength) : arr;
  let b = UNIS[name];
  if (!b || b.size < data.byteLength) { b = device.createBuffer({ size: Math.max(data.byteLength, 16), usage: UNI }); UNIS[name] = b; }
  queue.writeBuffer(b, 0, data);
  return b;
}
function uniStatic(name, arr) { return UNIS[name] || uni(name, arr); }

let ENC = null;
function dispatch(key, code, buffers, grid, bgkey) {
  const p = pipe(key, code);
  let bg = bgkey != null ? BGC.get(bgkey) : null;
  if (!bg) {
    bg = device.createBindGroup({ layout: p.layout, entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    if (bgkey != null) BGC.set(bgkey, bg);
  }
  const cp = ENC.beginComputePass();
  cp.setPipeline(p.pipe);
  cp.setBindGroup(0, bg);
  cp.dispatchWorkgroups(grid[0], grid[1] || 1, grid[2] || 1);
  cp.end();
}

// ---------- init ----------
async function init() {
  if (!navigator.gpu) throw new Error("WebGPU not available");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  const feats = ["subgroups", "shader-f16"].filter((f) => adapter.features.has(f));
  const big = Math.min(adapter.limits.maxStorageBufferBindingSize, 2 ** 31);
  device = await adapter.requestDevice({
    requiredFeatures: feats,
    requiredLimits: {
      maxStorageBufferBindingSize: big, maxBufferSize: big,
      maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
      maxComputeWorkgroupSizeX: Math.min(512, adapter.limits.maxComputeWorkgroupSizeX),
      maxComputeInvocationsPerWorkgroup: Math.min(512, adapter.limits.maxComputeInvocationsPerWorkgroup),
    },
  });
  queue = device.queue;
  window.__errs = [];
  device.addEventListener("uncapturederror", (e) => { window.__errs.push(e.error.message); console.error(e.error); });

  status.textContent = "fetching manifest + kernels…";
  [MAN, KERN] = await Promise.all([
    fetch("/manifest.json").then((r) => r.json()),
    fetch("/kernels.json").then((r) => r.json()),
  ]);
  CFG = MAN.config;

  status.textContent = "downloading weights (2.1 GB)…";
  const ab = await fetch("/weights.bin").then((r) => r.arrayBuffer());
  status.textContent = "uploading weights to GPU…";
  const u8 = new Uint8Array(ab);
  for (const [name, t] of Object.entries(MAN.tensors)) {
    const b = device.createBuffer({ size: Math.max(t.len, 4), usage: U32, mappedAtCreation: true });
    new Uint8Array(b.getMappedRange()).set(u8.subarray(t.off, t.off + t.len));
    b.unmap();
    BUF[name] = b;
  }
  setupScratch();
  status.textContent = "ready.";
  runBtn.disabled = false;
  window.__ready = true;
}

function K(name) { return KERN[name]; }

function setupScratch() {
  const H = CFG.H, d = CFG.ple_d, nL = CFG.nL, V = CFG.vocab;
  // KV caches for non-shared layers
  window.KC = {}; window.VC = {};
  for (const s of MAN.layers) if (!s.shared) {
    KC[s.index] = device.createBuffer({ size: 2048 * s.head_dim * 4, usage: U32 });
    VC[s.index] = device.createBuffer({ size: 2048 * s.head_dim * 4, usage: U32 });
  }
  const sizes = [["hidden", H * 4], ["a", H * 4], ["outq", 4096 * 4], ["attn", 4096 * 4],
    ["outk", 512 * 4], ["outv", 512 * 4], ["dummy", 512 * 4], ["y2", H * 4], ["y2n", H * 4],
    ["geglu", 12288 * 2], ["gate", d * 4], ["ctx", nL * d * 4], ["plegath", nL * d * 4],
    ["ple", nL * d * 4], ["normed", H * 4], ["logits", V * 4], ["cv", 256 * 4], ["ci", 256 * 4],
    ["suma", 4], ["sum2", 4], ["sum2n", 4], ["sa", 4], ["ids", 4], ["cur", 4], ["gen", 256 * 4]];
  for (const [n, s] of sizes) scratch(n, s);
  for (const [n, s] of [["pp73", (H + 1) * 4], ["pp75", (H + 1) * 4], ["pp77", (H + 1) * 4],
    ["partials256", (8 * 32 * 258 + 8) * 4], ["partials512", (8 * 32 * 514 + 8) * 4]])
    queue.writeBuffer(scratch(n, s), 0, new Uint32Array(s / 4));
}

// ---------- per-token dynamic uniforms (once per head-dim type) ----------
let ropeCfgs = null;
function writeStepUniforms(pos) {
  const nH = CFG.nH, win = CFG.window;
  if (!ropeCfgs) {
    ropeCfgs = {};
    for (const s of MAN.layers) ropeCfgs[s.head_dim] = [s.rope_theta, s.rope_cutoff, s.sliding];
  }
  for (const [hdS, [theta, cutoff, sliding]] of Object.entries(ropeCfgs)) {
    const hd = +hdS, half = hd / 2;
    const cos = new Float32Array(half), sin = new Float32Array(half);
    for (let i = 0; i < half; i++) {
      const inv = i < cutoff ? 1.0 / Math.pow(theta, i / half) : 0.0;
      const ang = pos * inv;
      cos[i] = Math.cos(ang); sin[i] = Math.sin(ang);
    }
    queue.writeBuffer(scratch(`rcos${half}`, half * 4), 0, cos);
    queue.writeBuffer(scratch(`rsin${half}`, half * 4), 0, sin);
    uni(`parA_${hd}`, new Uint32Array([1, pos + 1, pos, nH, 1, sliding ? win : 0, 0, 0]));
    uni(`pkv_${hd}`, new Uint32Array([pos * hd, 0, 0, 0]));
  }
}

// ---------- embed + PLE input ----------
function idsBuf(tok) { const b = scratch("ids", 4); queue.writeBuffer(b, 0, new Uint32Array([tok])); return b; }

function embed(tok, out, ids) {
  const H = CFG.H;
  const y = out || scratch("hidden", H * 4);
  const idb = ids || idsBuf(tok);
  const par = uniStatic("embed_par", new Uint32Array([1, 0, 0, 0]));
  dispatch("embed", K("00_main"), [idb, BUF.embed_q, BUF.embed_scale, y, par], [1, 1, 1], `embed|${bid(y)}|${bid(idb)}`);
  return y;
}
function pleInput(tok, embedBuf, ids) {
  const nL = CFG.nL, d = CFG.ple_d, H = CFG.H;
  const ctx = scratch("ctx", nL * d * 4), ple = scratch("plegath", nL * d * 4);
  const idb = ids || idsBuf(tok);
  const par0 = uniStatic("ple_par0", new Float32Array([0, 0, 0, 0]));
  const seq1 = uniStatic("ple_seq1", new Uint32Array([1, 0, 0, 0]));
  dispatch("proj68", K("68_reduce"), [embedBuf, BUF.pl_model_proj, ctx, par0], [nL * d / 8, 1, 1], `proj68|${bid(embedBuf)}`);
  dispatch("plegather", K("01_main"), [idb, BUF.ple_q, BUF.ple_scale, ple, seq1], [1, 1, 1], `plegather|${bid(idb)}`);
  const out = scratch("ple", nL * d * 4);
  const code = K("_COMBINE").replace("%.9ef", Math.pow(H, -0.5).toExponential(12) + "f");
  dispatch("combine", code, [ctx, ple, BUF.pl_proj_norm, out], [nL, 1, 1], "combine");
  return out;
}

// bind-group identity: assign each buffer a stable small id
let _bidN = 0; const _bids = new WeakMap();
function bid(b) { let i = _bids.get(b); if (i == null) { i = _bidN++; _bids.set(b, i); } return i; }

// ---------- one decoder layer ----------
function f32bits(x) { const f = new Float32Array([x]); return new Uint32Array(f.buffer)[0]; }

function layer(L, pos, hidden, pleBuf, pleOff) {
  const s = MAN.layers[L], sc = s.scales, b = BUF;
  const H = CFG.H, nH = CFG.nH, hd = s.head_dim, qd = s.q_dim, inter = s.intermediate;
  const half = hd / 2, shared = s.shared;
  const kc = KC[s.kv_source], vc = VC[s.kv_source];
  const cb = scratch(`rcos${half}`, half * 4), sb = scratch(`rsin${half}`, half * 4);
  const hk = `${L}|${bid(hidden)}|${bid(pleBuf)}`;

  const outq = scratch("outq", qd * 4), dummy = scratch("dummy", hd * 4), dummy2 = scratch("dummy2", hd * 4);
  const outk = scratch("outk", hd * 4), outv = scratch("outv", hd * 4);
  const attn = scratch("attn", qd * 4), y2 = scratch("y2", H * 4), sum2 = scratch("sum2", 4);
  const geglu = scratch("geglu", inter * 2), gate = scratch("gate", CFG.ple_d * 4);
  const y2n = scratch("y2n", H * 4), sum2n = scratch("sum2n", 4);
  const pp73 = scratch("pp73", (H + 1) * 4), pp75 = scratch("pp75", (H + 1) * 4), pp77 = scratch("pp77", (H + 1) * 4);
  const partials = scratch(`partials${hd}`, (8 * 32 * (hd + 2) + 8) * 4);

  // 69 only on layer 0; layers 1+ reuse prev k77's y2n/sum2n as qkv input
  let a, suma;
  if (L === 0) {
    a = scratch("a", H * 4); suma = scratch("suma", 4);
    dispatch("k69", K("69_sg_sum"), [hidden, b[`L${L}.in_norm`], a, suma,
      uniStatic(`p69_${L}`, new Uint32Array([1, 0, f32bits(sc.qkv_in), 0]))], [1, 1, 1], `k69|${hk}`);
  } else { a = scratch("y2n", H * 4); suma = scratch("sum2n", 4); }

  // 70 qkv (shared: q-only)
  const total = qd / 2 + hd;
  const k70 = patch(K("70_srq"), { Q_OUT: qd, Q_WGS: qd / 2, KV_OUT: hd, KV_WGS: hd / 2, TOTAL_WGS: total, GRID_X: total });
  const par70 = uniStatic(`p70_${L}`, new Float32Array([sc.q_out, sc.k_out || 0, sc.v_out || 0, 0]));
  if (shared) {
    dispatch(`k70_${qd}_${hd}`, k70, [a, b[`L${L}.q_bits`], b[`L${L}.q_bits`], b[`L${L}.q_bits`],
      b[`L${L}.q_scale`], suma, outq, dummy, dummy2, par70], [qd / 2, 1, 1], `k70|${hk}`);
  } else {
    dispatch(`k70_${qd}_${hd}`, k70, [a, b[`L${L}.q_bits`], b[`L${L}.k_bits`], b[`L${L}.v_bits`],
      b[`L${L}.qkv_scales`], suma, outq, outk, outv, par70], [total, 1, 1], `k70|${hk}`);
    // The captured template uses %du placeholders; the DSL-generated one
    // declares named consts instead. Without this branch the DSL version would
    // silently keep HD=256 and quietly corrupt every 512-wide layer.
    const kvnSrc = K("_KVNORM");
    const kvn = kvnSrc.includes("%du")
      ? kvnSrc.replace("%du", hd + "u").replace("%du", half + "u")
      : patch(kvnSrc, { HD: hd, HALF: half });
    dispatch(`kvnorm_${hd}`, kvn, [outk, outv, b[`L${L}.k_norm`], cb, sb, kc, vc, UNIS[`pkv_${hd}`]], [1, 1, 1], `kvn|${hk}`);
  }
  // attention
  let att = patch(K("101_srq"), { HEAD_DIM: hd, HALF_DIM: half });
  att = att.replace("const OUT_Q: f32 = 0.014886821620166302;", `const OUT_Q: f32 = ${sc.o_in};`);
  dispatch(`att_${hd}_${sc.o_in}`, att, [outq, b[`L${L}.q_norm`], cb, sb, kc, vc, partials, attn, UNIS[`parA_${hd}`]], [nH, 32, 1], `att|${hk}`);
  // 73 o-proj + norms
  const k73 = patch(K("73_sg_sum"), { IN_FEATURES: qd, WORDS_PER_ROW: qd / 8 });
  const par73 = uniStatic(`p73_${L}`, new Float32Array([sc.o_out, sc.gate_in, 0, 0]));
  dispatch(`k73_${qd}`, k73, [attn, b[`L${L}.o_bits`], b[`L${L}.o_scale`], pp73, hidden, b[`L${L}.o_w12`], y2, sum2, par73], [192, 1, 1], `k73|${hk}`);
  // 74/95 gate-up ; 75/96 down
  const guK = inter === 6144 ? "74_sg_sum" : "95_sg_sum", guGrid = inter === 6144 ? 768 : 3072;
  const par74 = uniStatic(`p74_${L}`, new Float32Array([sc.gate_out, sc.up_out, sc.down_in, 0]));
  dispatch(`gu_${inter}`, K(guK), [y2, b[`L${L}.gate_bits`], b[`L${L}.gate_scale`], b[`L${L}.up_bits`], b[`L${L}.up_scale`], sum2, geglu, b[`L${L}.gelu_gate`], par74], [guGrid, 1, 1], `gu|${hk}`);
  const downK = inter === 6144 ? "75_srq" : "96_srq";
  const par75 = uniStatic(`p75_${L}`, new Float32Array([sc.down_in, sc.down_out, 0, 0]));
  dispatch(`down_${inter}`, K(downK), [geglu, b[`L${L}.down_bits`], pp75, b[`L${L}.down_scale`], hidden, b[`L${L}.down_nw`], par75], [H / 4, 1, 1], `down|${hk}`);
  // 76 PLE gate
  const parP = new ArrayBuffer(16);
  new Float32Array(parP, 0, 2).set([sc.plegate_in, sc.plegate_out]);
  new Uint32Array(parP, 8, 1)[0] = pleOff;
  dispatch("k76", K("76_reduce"), [hidden, b[`L${L}.plegate_codes`], b[`L${L}.plegate_rowscale`], pleBuf, gate, b[`L${L}.gelu_plegate`], uniStatic(`p76_${L}`, new Uint8Array(parP))], [CFG.ple_d, 1, 1], `k76|${hk}`);
  // 77 PLE proj
  const nxtIn = L + 1 < CFG.nL ? MAN.layers[L + 1].scales.qkv_in : 0.0;
  const par77 = uniStatic(`p77_${L}`, new Float32Array([nxtIn, sc.pleproj_in, sc.pleproj_out, 0]));
  dispatch("k77", K("77_sg_sum"), [gate, b[`L${L}.pleproj_codes`], b[`L${L}.pleproj_rowscale`], pp77, hidden, b[`L${L}.pleproj_w12s`], y2n, sum2n, par77], [96, 1, 1], `k77|${hk}`);
}

// ---------- forward ----------
function forward(tok, pos, hidden, argmax, idsBufArg, genIds, step) {
  const H = CFG.H, nL = CFG.nL, d = CFG.ple_d, V = CFG.vocab;
  hidden = hidden || scratch("hidden", H * 4);
  const logits = scratch("logits", V * 4);
  writeStepUniforms(pos);
  ENC = device.createCommandEncoder();
  embed(tok, hidden, idsBufArg);
  const ple = pleInput(tok, hidden, idsBufArg);
  for (let L = 0; L < nL; L++) layer(L, pos, hidden, ple, L * d);
  const normed = scratch("normed", H * 4), sa = scratch("sa", 4);
  dispatch("finalnorm", K("69_sg_sum"), [hidden, BUF.final_norm, normed, sa, uniStatic("pfn", new Uint32Array([1, 0, 0, 0]))], [1, 1, 1], `finalnorm|${bid(hidden)}`);
  dispatch("logits", K("33_srq"), [normed, BUF.lmhead_blk, BUF.lmhead_scale, logits, uniStatic("plog", new Float32Array([0, 0, 0, 0]))], [V / 128, 1, 1], "logits");
  if (argmax) {
    const cv = scratch("cv", 256 * 4), ci = scratch("ci", 256 * 4);
    dispatch("amax1", K("34_main"), [logits, cv, ci], [256, 1, 1], "amax1");
    dispatch("amax2", K("35_main"), [cv, ci, argmax], [1, 1, 1], `amax2|${bid(argmax)}`);
    if (genIds) ENC.copyBufferToBuffer(argmax, 0, genIds, step * 4, 4);
  }
  queue.submit([ENC.finish()]);
  ENC = null;
  return logits;
}

async function readU32(buf, nbytes) {
  const rb = device.createBuffer({ size: nbytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(buf, 0, rb, 0, nbytes);
  queue.submit([e.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(rb.getMappedRange().slice(0));
  rb.unmap(); rb.destroy();
  return out;
}

// ---------- resident decode ----------
async function generateResident(ids, nNew, eos = 1, chunk = 64) {
  const H = CFG.H;
  const hidden = scratch("hidden", H * 4), cur = scratch("cur", 4), gen = scratch("gen", nNew * 4);
  let pos = 0;
  for (let i = 0; i < ids.length - 1; i++) { queue.writeBuffer(cur, 0, new Uint32Array([ids[i]])); forward(0, pos, hidden, null, cur); pos++; }
  queue.writeBuffer(cur, 0, new Uint32Array([ids[ids.length - 1]]));
  const out = [];
  const t0 = performance.now();
  let step = 0;
  while (step < nNew) {
    const end = Math.min(step + chunk, nNew);
    for (; step < end; step++) { forward(0, pos, hidden, cur, cur, gen, step); pos++; }
    const got = await readU32(gen, step * 4);
    for (let i = out.length; i < step; i++) {
      if (got[i] === eos) { return { out, tps: out.length / ((performance.now() - t0) / 1000) }; }
      out.push(got[i]);
    }
  }
  return { out, tps: out.length / ((performance.now() - t0) / 1000) };
}

// ---------- UI ----------
runBtn.onclick = async () => {
  runBtn.disabled = true; outEl.textContent = ""; tpsEl.textContent = "";
  const prompt = document.getElementById("prompt").value;
  const nNew = Math.max(1, Math.min(256, +document.getElementById("n").value || 48));
  status.textContent = "tokenizing…";
  const { ids } = await fetch("/tokenize", { method: "POST", body: JSON.stringify({ text: prompt }) }).then((r) => r.json());
  status.textContent = "generating…";
  const { out, tps } = await generateResident(ids, nNew, 1, Math.min(nNew, 64));
  const { text } = await fetch("/detokenize", { method: "POST", body: JSON.stringify({ ids: out }) }).then((r) => r.json());
  outEl.textContent = text;
  tpsEl.textContent = `${tps.toFixed(1)} tok/s`;
  status.textContent = "done.";
  runBtn.disabled = false;
};
window.bench = async (n = 64) => {
  const H = CFG.H; const hidden = scratch("hidden", H * 4), cur = scratch("cur", 4), gen = scratch("gen", 256 * 4);
  queue.writeBuffer(cur, 0, new Uint32Array([2]));
  for (let p = 0; p < 8; p++) forward(0, p, hidden, cur, cur, gen, p);
  await readU32(gen, 32);
  const t0 = performance.now();
  for (let i = 0; i < n; i++) forward(0, 8 + i, hidden, cur, cur, gen, i % 64);
  await readU32(gen, 4);
  return { tps: n / ((performance.now() - t0) / 1000) };
};

// expose for headless driving / debugging
async function readF32(buf, nbytes) { const u = await readU32(buf, nbytes); return new Float32Array(u.buffer); }
window.testEmbed = async (tok = 2) => {
  ENC = device.createCommandEncoder();
  const y = scratch("hidden", CFG.H * 4);
  embed(tok, y, null);
  queue.submit([ENC.finish()]); ENC = null;
  const v = await readF32(y, 16);
  return { first4: [...v].map((x) => +x.toFixed(4)), errs: window.__errs.slice() };
};
window.testForward = async (tok = 2, pos = 0) => {
  for (const s of MAN.layers) if (!s.shared) { queue.writeBuffer(KC[s.index], 0, new Uint32Array(4)); }
  const tb = scratch("tb", 4);
  window.__errs = [];
  forward(tok, pos, scratch("hidden", CFG.H * 4), tb);
  const t = (await readU32(tb, 4))[0];
  // also read top logit region
  const lg = await readF32(scratch("logits", CFG.vocab * 4), 40);
  return { argmax: t, logits5: [...lg.slice(0, 5)].map((x) => +x.toFixed(3)), errs: window.__errs.slice() };
};
window.generateResident = generateResident;
window.gen = async (prompt, n = 20) => {
  const { ids } = await fetch("/tokenize", { method: "POST", body: JSON.stringify({ text: prompt }) }).then((r) => r.json());
  const { out, tps } = await generateResident(ids, n, 1, Math.min(n, 64));
  const { text } = await fetch("/detokenize", { method: "POST", body: JSON.stringify({ ids: out }) }).then((r) => r.json());
  return { text, tps: +tps.toFixed(1) };
};

init().catch((e) => { status.textContent = "ERROR: " + e.message; console.error(e); });
