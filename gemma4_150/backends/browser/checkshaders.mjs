// Compile every served kernel under Dawn and report the actual errors.
//   node gemma4_150/backends/browser/checkshaders.mjs
// Dawn is stricter than naga, so a bundle that compiles in wgpu-py can still
// fail here — and app.js only surfaces "GPUValidationError" with no message.
import { spawn } from "node:child_process"; import { setTimeout as sleep } from "node:timers/promises";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", PORT = 9261;
const proc = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, "--user-data-dir=/tmp/cdp-g4chk",
  "--enable-unsafe-webgpu", "--use-angle=metal", "--enable-features=WebGPU", "--no-first-run",
  "--disable-dev-shm-usage", "http://localhost:8000/"], { stdio: ["ignore", "ignore", "ignore"] });
let ws; const done = (o) => { console.log(o); try { ws?.close(); } catch {} proc.kill("SIGKILL"); process.exit(0); };
const EXPR = `(async () => {
  const ks = await (await fetch('/kernels.json')).json();
  const ad = await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  const want = ['shader-f16','subgroups'].filter(f => ad.features.has(f));
  const dev = await ad.requestDevice({requiredFeatures: want});
  const out = ['features: ' + want.join(',')];
  for (const [k, code] of Object.entries(ks)) {
    const m = dev.createShaderModule({code});
    const info = await m.getCompilationInfo();
    const errs = info.messages.filter(x => x.type === 'error');
    if (errs.length) out.push(k + ' -> ' + errs.slice(0,2).map(e => 'line ' + e.lineNum + ': ' + e.message).join(' || '));
  }
  return out.join('\\n');
})()`;
try {
  let t = null;
  for (let i = 0; i < 100; i++) { try { const l = await fetch(`http://localhost:${PORT}/json`).then(r => r.json()); t = l.find(x => x.type === "page" && x.url.includes("localhost:8000") && x.webSocketDebuggerUrl); if (t) break; } catch {} await sleep(200); }
  if (!t) throw new Error("no page target");
  ws = new WebSocket(t.webSocketDebuggerUrl); await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0; const pend = new Map();
  ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } };
  const ev = (expr) => new Promise(res => { const i = ++id; pend.set(i, res); ws.send(JSON.stringify({ id: i, method: "Runtime.evaluate", params: { expression: expr, awaitPromise: true, returnByValue: true, timeout: 120000 } })); });
  await ev("1");
  const r = await ev(EXPR);
  done(r.result?.result?.value ?? JSON.stringify(r.result).slice(0, 500));
} catch (e) { done("ERR " + e.message); }
