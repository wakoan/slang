// Autonomous headless-Chrome benchmark for the Gemma 4 QAT WebGPU demo.
// No deps (node's global fetch/WebSocket + the Chrome DevTools Protocol).
//
// Launches headless Chrome with WebGPU (and --enable-unsafe-webgpu, which also
// exposes chromium-experimental-subgroup-matrix), loads http://localhost:8000/,
// waits for the model, runs window.bench(), and prints tok/s. Lets us A/B
// kernel changes without a human reloading the page.
//
//   (start the server first: python -m gemma4.qat_gendemo_server)
//   node gendemo4/bench_headless.mjs [nTokens]
//
// Reports: BENCH: {"tps":.., "chars":.., "sample":".."}

import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = Number(process.env.CDP_PORT || 9223);
const N_TOK = Number(process.argv[2] || 128);
const APP = "http://localhost:8000/";

const proc = spawn(CHROME, [
  "--headless=new", `--remote-debugging-port=${PORT}`,
  `--user-data-dir=/tmp/cdp-bench-${PORT}`,
  "--enable-unsafe-webgpu", "--use-angle=metal", "--enable-features=WebGPU",
  "--no-first-run", "--no-default-browser-check", "--disable-dev-shm-usage",
  APP,
], { stdio: ["ignore", "ignore", "ignore"] });

let ws;
const finish = (obj) => {
  console.log("BENCH:", JSON.stringify(obj));
  try { ws?.close(); } catch {}
  proc.kill("SIGKILL");
  process.exit(obj.error ? 1 : 0);
};

try {
  let target = null;
  for (let i = 0; i < 100; i++) {
    try {
      const list = await fetch(`http://localhost:${PORT}/json`).then((r) => r.json());
      target = list.find((t) => t.type === "page" && t.url.includes("localhost:8000") && t.webSocketDebuggerUrl);
      if (target) break;
    } catch {}
    await sleep(200);
  }
  if (!target) throw new Error("no page target (is the server on :8000?)");
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

  let id = 0; const pending = new Map();
  ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
  const evalJS = (expr, t = 300000) => new Promise((res) => {
    const i = ++id; pending.set(i, res);
    ws.send(JSON.stringify({ id: i, method: "Runtime.evaluate",
      params: { expression: expr, awaitPromise: true, returnByValue: true, timeout: t } }));
  });
  await new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method: "Runtime.enable" })); });

  // wait for the model to finish loading (2GB download + upload + shader compile)
  let ready = false;
  for (let i = 0; i < 600; i++) {
    const r = await evalJS("window.__ready === true", 5000);
    if (r.result?.result?.value === true) { ready = true; break; }
    await sleep(500);
  }
  if (!ready) {
    const st = await evalJS("document.getElementById('status')?.textContent");
    throw new Error("model not ready: " + JSON.stringify(st.result?.result?.value));
  }
  const r = await evalJS(`bench(undefined, ${N_TOK})`, 300000);
  if (r.result?.exceptionDetails) finish({ error: r.result.exceptionDetails.text || "eval exception" });
  finish(r.result?.result?.value ?? { error: "no result" });
} catch (e) {
  finish({ error: e.message });
}
