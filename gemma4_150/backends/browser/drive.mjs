// Headless-Chrome driver for the gemma4_150 browser demo (CDP, no deps).
//   node gemma4_150/web/drive.mjs
import { spawn } from "node:child_process"; import { setTimeout as sleep } from "node:timers/promises";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", PORT = 9260;
const proc = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, "--user-data-dir=/tmp/cdp-g4web",
  "--enable-unsafe-webgpu", "--use-angle=metal", "--enable-features=WebGPU", "--no-first-run",
  "--disable-dev-shm-usage", "http://localhost:8000/"], { stdio: ["ignore", "ignore", "ignore"] });
let ws; const done = (o) => { console.log("RESULT:", JSON.stringify(o)); try { ws?.close(); } catch {} proc.kill("SIGKILL"); process.exit(o.error ? 1 : 0); };
try {
  let t = null;
  for (let i = 0; i < 100; i++) { try { const l = await fetch(`http://localhost:${PORT}/json`).then((r) => r.json()); t = l.find((x) => x.type === "page" && x.url.includes("localhost:8000") && x.webSocketDebuggerUrl); if (t) break; } catch {} await sleep(200); }
  if (!t) throw new Error("no page target");
  ws = new WebSocket(t.webSocketDebuggerUrl); await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0; const pend = new Map();
  // Page-side diagnostics are surfaced, not swallowed: without this a failing
  // shader looks like a successful run with empty output.
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); return; }
    if (m.method === "Runtime.consoleAPICalled") {
      const t = (m.params.args || []).map(function (a) {
        if (a.value !== undefined) return String(a.value);
        var pr = (a.preview && a.preview.properties) || [];
        var kv = pr.map(function (q) { return q.name + "=" + q.value; }).join(" ");
        return kv || a.description || "";
      }).join(" ");
      if (t) console.log(`[console.${m.params.type}] ${t.slice(0, 500)}`);
    } else if (m.method === "Runtime.exceptionThrown") {
      const d = m.params.exceptionDetails;
      console.log(`[exception] ${(d.exception?.description || d.text || "").slice(0, 700)}`);
    }
  };
  const ev = (expr, tmo = 300000) => new Promise((res) => { const i = ++id; pend.set(i, res); ws.send(JSON.stringify({ id: i, method: "Runtime.evaluate", params: { expression: expr, awaitPromise: true, returnByValue: true, timeout: tmo } })); });
  await new Promise((res) => { const i = ++id; pend.set(i, res); ws.send(JSON.stringify({ id: i, method: "Runtime.enable" })); });
  let ready = false;
  for (let i = 0; i < 1200; i++) { const r = await ev("window.__ready===true", 5000); if (r.result?.result?.value === true) { ready = true; break; } await sleep(500); }
  if (!ready) { const st = await ev("document.getElementById('status').textContent"); throw new Error("not ready: " + JSON.stringify(st.result?.result?.value)); }
  // coherence: tokenize + generate + detokenize
  const gen = await ev(`window.gen('The capital of France is', 20)`);
  if (gen.result?.exceptionDetails) { done({ error: gen.result.exceptionDetails.text || "gen exception" }); }
  else { const b = await ev("window.bench(64)"); done({ gen: gen.result?.result?.value, sustained_tps: +(b.result?.result?.value?.tps || 0).toFixed(1) }); }
} catch (e) { done({ error: e.message }); }
