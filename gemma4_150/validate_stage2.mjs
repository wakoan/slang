// Stage 2: run reference kernel 74 (fused gate/up geglu, presrq) in headless,
// compare to numpy oracle (scratchpad/stage2.json).
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";
import { readFileSync } from "node:fs";
const CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", PORT=9251;
const REF="/Users/wako/projects/slang/reference/webml_gemma4_kernels";
const DATA="/private/tmp/claude-501/-Users-wako-projects-slang/ce5a80c9-3495-4ae0-8cd2-813d9c5d619f/scratchpad/stage2.json";
const k74=readFileSync(`${REF}/74_sg_sum.wgsl`,"utf8");
const D=JSON.parse(readFileSync(DATA,"utf8"));
const proc=spawn(CHROME,["--headless=new",`--remote-debugging-port=${PORT}`,"--user-data-dir=/tmp/cdp-s2",
 "--enable-unsafe-webgpu","--use-angle=metal","--enable-features=WebGPU","--no-first-run","--disable-dev-shm-usage",
 "http://localhost:8000/manifest.json"],{stdio:["ignore","ignore","ignore"]});
let ws; const done=o=>{console.log("STAGE2:",JSON.stringify(o));try{ws?.close();}catch{}proc.kill("SIGKILL");process.exit(0);};
try{
 let t=null;for(let i=0;i<100;i++){try{const l=await fetch(`http://localhost:${PORT}/json`).then(r=>r.json());t=l.find(x=>x.type==="page"&&x.webSocketDebuggerUrl);if(t)break;}catch{}await sleep(200);}
 ws=new WebSocket(t.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
 let id=0;const p=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id&&p.has(m.id)){p.get(m.id)(m);p.delete(m.id);}};
 const cmd=(mth,pr={})=>new Promise(res=>{const i=++id;p.set(i,res);ws.send(JSON.stringify({id:i,method:mth,params:pr}));});
 await cmd("Runtime.enable");
 const payload=JSON.stringify({k74,D});
 const expr=`(async()=>{const {k74,D}=${payload};
  const b64=s=>{const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u.buffer;};
  const f32=s=>new Float32Array(b64(s)); const u32=s=>new Uint32Array(b64(s)); const raw=s=>new Uint8Array(b64(s));
  const f16d=h=>{const s=(h&0x8000)?-1:1,e=(h>>10)&0x1f,m=h&0x3ff; if(e===0)return s*m*Math.pow(2,-24); if(e===31)return m?NaN:s*Infinity; return s*(1+m/1024)*Math.pow(2,e-15);};
  const adp=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  const dev=await adp.requestDevice({requiredFeatures:['subgroups','shader-f16'], requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(10,adp.limits.maxStorageBuffersPerShaderStage)}});
  const m=dev.createShaderModule({code:k74}); const ci=await m.getCompilationInfo();
  const ce=ci.messages.filter(x=>x.type==='error'); if(ce.length) return {compileErrors:ce.map(e=>e.message)};
  dev.pushErrorScope('validation');
  const ST=GPUBufferUsage.STORAGE,DST=GPUBufferUsage.COPY_DST,SRC=GPUBufferUsage.COPY_SRC,UNI=GPUBufferUsage.UNIFORM;
  const mk=(ab,extra=0)=>{const b=dev.createBuffer({size:Math.max(ab.byteLength,4),usage:ST|DST|extra});dev.queue.writeBuffer(b,0,ab);return b;};
  const hidden=mk(b64(D.hidden)), gb=mk(u32(D.gate_bits).buffer), ub=mk(u32(D.up_bits).buffer);
  const gs=mk(f32(D.gate_scale).buffer), us=mk(f32(D.up_scale).buffer), lut=mk(f32(D.lut).buffer);
  const sa=mk(new Float32Array([D.sum_a]).buffer);
  const out=dev.createBuffer({size:D.INTER*2,usage:ST|SRC});
  const par=dev.createBuffer({size:16,usage:UNI|DST}); dev.queue.writeBuffer(par,0,new Float32Array([D.gOut,D.uOut,D.outQ,0]));
  const pipe=dev.createComputePipeline({layout:'auto',compute:{module:m,entryPoint:'main'}});
  const bg=dev.createBindGroup({layout:pipe.getBindGroupLayout(0),entries:[hidden,gb,gs,ub,us,sa,out,lut,par].map((b,i)=>({binding:i,resource:{buffer:b}}))});
  const enc=dev.createCommandEncoder();const ps=enc.beginComputePass();ps.setPipeline(pipe);ps.setBindGroup(0,bg);ps.dispatchWorkgroups(768,1,1);ps.end();dev.queue.submit([enc.finish()]);
  const verr=await dev.popErrorScope(); if(verr) return {validationError:verr.message};
  const rb=dev.createBuffer({size:D.INTER*2,usage:GPUBufferUsage.MAP_READ|DST});const e2=dev.createCommandEncoder();e2.copyBufferToBuffer(out,0,rb,0,D.INTER*2);dev.queue.submit([e2.finish()]);
  await rb.mapAsync(GPUMapMode.READ); const h16=new Uint16Array(rb.getMappedRange()).slice(); rb.unmap();
  const g=new Float32Array(D.INTER); for(let i=0;i<D.INTER;i++)g[i]=f16d(h16[i]);
  const ref=f32(D.ref_out); let mism=0,maxAbs=0,mi=0;
  for(let i=0;i<D.INTER;i++){const e=Math.abs(g[i]-ref[i]); if(e>0.5)mism++; if(e>maxAbs){maxAbs=e;mi=i;}}
  return {mismatches:mism, of:D.INTER, maxAbsDiff:+maxAbs.toFixed(3), at:mi, gpu:+g[mi].toFixed(2), ref:+ref[mi].toFixed(2), gpu0:[...g].slice(0,6).map(v=>+v.toFixed(0))};
 })()`;
 const r=await cmd("Runtime.evaluate",{expression:expr,awaitPromise:true,returnByValue:true,timeout:30000});
 if(r.result?.exceptionDetails)done({exception:r.result.exceptionDetails.text||r.result.exceptionDetails.exception?.description});
 done(r.result?.result?.value);
}catch(e){done({error:e.message});}
