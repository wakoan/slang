// Stage 1: run reference kernels 69 (rms+srq) -> 70 (qkv) in headless Chrome,
// compare to the numpy oracle (scratchpad/stage1.json). No project deps.
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";
import { readFileSync } from "node:fs";
const CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", PORT=9250;
const REF="/Users/wako/projects/slang/reference/webml_gemma4_kernels";
const DATA="/private/tmp/claude-501/-Users-wako-projects-slang/ce5a80c9-3495-4ae0-8cd2-813d9c5d619f/scratchpad/stage1.json";
const k69=readFileSync(`${REF}/69_sg_sum.wgsl`,"utf8"), k70=readFileSync(`${REF}/70_srq.wgsl`,"utf8");
const D=JSON.parse(readFileSync(DATA,"utf8"));
const proc=spawn(CHROME,["--headless=new",`--remote-debugging-port=${PORT}`,"--user-data-dir=/tmp/cdp-s1",
 "--enable-unsafe-webgpu","--use-angle=metal","--enable-features=WebGPU","--no-first-run","--disable-dev-shm-usage",
 "http://localhost:8000/manifest.json"],{stdio:["ignore","ignore","ignore"]});
let ws; const done=o=>{console.log("STAGE1:",JSON.stringify(o));try{ws?.close();}catch{}proc.kill("SIGKILL");process.exit(0);};
try{
 let t=null;for(let i=0;i<100;i++){try{const l=await fetch(`http://localhost:${PORT}/json`).then(r=>r.json());t=l.find(x=>x.type==="page"&&x.webSocketDebuggerUrl);if(t)break;}catch{}await sleep(200);}
 ws=new WebSocket(t.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
 let id=0;const p=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id&&p.has(m.id)){p.get(m.id)(m);p.delete(m.id);}};
 const cmd=(mth,pr={})=>new Promise(res=>{const i=++id;p.set(i,res);ws.send(JSON.stringify({id:i,method:mth,params:pr}));});
 await cmd("Runtime.enable");
 const payload=JSON.stringify({k69,k70,D});
 const expr=`(async()=>{const {k69,k70,D}=${payload};
  const b64=s=>{const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u.buffer;};
  const f32=s=>new Float32Array(b64(s)); const u32=s=>new Uint32Array(b64(s));
  const adp=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  const dev=await adp.requestDevice({requiredFeatures:['subgroups'], requiredLimits:{maxStorageBuffersPerShaderStage: Math.min(10, adp.limits.maxStorageBuffersPerShaderStage)}}); dev.pushErrorScope('validation');
  const errs=[];
  for(const [nm,code] of [['69',k69],['70',k70]]){const m=dev.createShaderModule({code});const ci=await m.getCompilationInfo();ci.messages.filter(x=>x.type==='error').forEach(e=>errs.push(nm+': '+e.message));}
  if(errs.length) return {compileErrors:errs};
  const ST=GPUBufferUsage.STORAGE, DST=GPUBufferUsage.COPY_DST, SRC=GPUBufferUsage.COPY_SRC, UNI=GPUBufferUsage.UNIFORM;
  const mkf=(arr,extra=0)=>{const b=dev.createBuffer({size:Math.max(arr.byteLength,4),usage:ST|DST|extra});dev.queue.writeBuffer(b,0,arr);return b;};
  const x=mkf(f32(D.x)), w=mkf(f32(D.w));
  const y=dev.createBuffer({size:D.DIM*4,usage:ST|SRC|DST}); const sum_a=dev.createBuffer({size:4,usage:ST|SRC|DST});
  const qb=mkf(u32(D.q_bits)), kb=mkf(u32(D.k_bits)), vb=mkf(u32(D.v_bits)), sc=mkf(f32(D.scales));
  const oq=dev.createBuffer({size:D.Q_OUT*4,usage:ST|SRC}), ok=dev.createBuffer({size:D.KV_OUT*4,usage:ST|SRC}), ov=dev.createBuffer({size:D.KV_OUT*4,usage:ST|SRC});
  const p69=dev.createBuffer({size:16,usage:UNI|DST}); dev.queue.writeBuffer(p69,0,new Uint32Array([1,0,0,0])); dev.queue.writeBuffer(p69,8,new Float32Array([D.inScale]));
  const p70=dev.createBuffer({size:16,usage:UNI|DST}); dev.queue.writeBuffer(p70,0,new Float32Array([D.qOut,D.kOut,D.vOut,0]));
  const pipe=(code,ent='main')=>dev.createComputePipeline({layout:'auto',compute:{module:dev.createShaderModule({code}),entryPoint:ent}});
  const P69=pipe(k69), P70=pipe(k70);
  const bg=(pl,arr)=>dev.createBindGroup({layout:pl.getBindGroupLayout(0),entries:arr.map((b,i)=>({binding:i,resource:{buffer:b}}))});
  const bg69=bg(P69,[x,w,y,sum_a,p69]);
  const bg70=bg(P70,[y,qb,kb,vb,sc,sum_a,oq,ok,ov,p70]);
  const enc=dev.createCommandEncoder();
  let pass=enc.beginComputePass(); pass.setPipeline(P69); pass.setBindGroup(0,bg69); pass.dispatchWorkgroups(1,1,1); pass.end();
  pass=enc.beginComputePass(); pass.setPipeline(P70); pass.setBindGroup(0,bg70); pass.dispatchWorkgroups(1280,1,1); pass.end();
  dev.queue.submit([enc.finish()]);
  const verr=await dev.popErrorScope(); if(verr) return {validationError:verr.message};
  const read=async(buf,n)=>{const rb=dev.createBuffer({size:n*4,usage:GPUBufferUsage.MAP_READ|DST});const e=dev.createCommandEncoder();e.copyBufferToBuffer(buf,0,rb,0,n*4);dev.queue.submit([e.finish()]);await rb.mapAsync(GPUMapMode.READ);const a=new Float32Array(rb.getMappedRange()).slice();rb.unmap();return a;};
  const gy=await read(y,8); const gs=await read(sum_a,1);
  const gq=await read(oq,D.Q_OUT), gk=await read(ok,D.KV_OUT), gv=await read(ov,D.KV_OUT);
  const rq=f32(D.ref_q), rk=f32(D.ref_k), rv=f32(D.ref_v);
  const err=(g,r)=>{let m=0,mi=0;for(let i=0;i<r.length;i++){const e=Math.abs(g[i]-r[i])/(Math.abs(r[i])+1e-2);if(e>m){m=e;mi=i;}}return {maxRel:+m.toFixed(5), at:mi, g:+g[mi].toFixed(3), r:+r[mi].toFixed(3)};};
  return {y0:[...gy].slice(0,4).map(v=>+v.toFixed(3)), sum_a_gpu:+gs[0].toFixed(3), q:err(gq,rq), k:err(gk,rk), v:err(gv,rv)};
 })()`;
 const r=await cmd("Runtime.evaluate",{expression:expr,awaitPromise:true,returnByValue:true,timeout:30000});
 if(r.result?.exceptionDetails)done({exception:r.result.exceptionDetails.text||r.result.exceptionDetails.exception?.description});
 done(r.result?.result?.value);
}catch(e){done({error:e.message});}
