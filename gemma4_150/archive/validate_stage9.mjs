import { spawn } from "node:child_process"; import { setTimeout as sleep } from "node:timers/promises"; import { readFileSync } from "node:fs";
const CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", PORT=9258;
const REF="/Users/wako/projects/slang/reference/webml_gemma4_kernels";
const DATA="/private/tmp/claude-501/-Users-wako-projects-slang/ce5a80c9-3495-4ae0-8cd2-813d9c5d619f/scratchpad/stage9.json";
const k76=readFileSync(`${REF}/76_reduce.wgsl`,"utf8"); const D=JSON.parse(readFileSync(DATA,"utf8"));
const proc=spawn(CHROME,["--headless=new",`--remote-debugging-port=${PORT}`,"--user-data-dir=/tmp/cdp-s9","--enable-unsafe-webgpu","--use-angle=metal","--enable-features=WebGPU","--no-first-run","--disable-dev-shm-usage","http://localhost:8000/manifest.json"],{stdio:["ignore","ignore","ignore"]});
let ws; const done=o=>{console.log("STAGE9:",JSON.stringify(o));try{ws?.close();}catch{}proc.kill("SIGKILL");process.exit(0);};
try{
 let t=null;for(let i=0;i<100;i++){try{const l=await fetch(`http://localhost:${PORT}/json`).then(r=>r.json());t=l.find(x=>x.type==="page"&&x.webSocketDebuggerUrl);if(t)break;}catch{}await sleep(200);}
 ws=new WebSocket(t.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
 let id=0;const p=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id&&p.has(m.id)){p.get(m.id)(m);p.delete(m.id);}};
 const cmd=(mth,pr={})=>new Promise(res=>{const i=++id;p.set(i,res);ws.send(JSON.stringify({id:i,method:mth,params:pr}));});
 await cmd("Runtime.enable");
 const expr=`(async()=>{const {k76,D}=${JSON.stringify({k76,D})};
  const b64=s=>{const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u.buffer;};
  const adp=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  const dev=await adp.requestDevice({requiredFeatures:['subgroups']});
  const m=dev.createShaderModule({code:k76}); const ci=await m.getCompilationInfo(); const ce=ci.messages.filter(x=>x.type==='error'); if(ce.length)return{compileErrors:ce.map(e=>e.message)};
  dev.pushErrorScope('validation');
  const ST=GPUBufferUsage.STORAGE,DST=GPUBufferUsage.COPY_DST,SRC=GPUBufferUsage.COPY_SRC,UNI=GPUBufferUsage.UNIFORM;
  const mk=(ab,ex=0)=>{const b=dev.createBuffer({size:Math.max(ab.byteLength,4),usage:ST|DST|ex});dev.queue.writeBuffer(b,0,ab);return b;};
  const OUT_F=D.OUT_F;
  const a=mk(b64(D.a)), codes=mk(b64(D.codes)), rscale=mk(b64(D.row_scale)), ple=mk(b64(D.ple)), lut=mk(b64(D.lut));
  const out=dev.createBuffer({size:OUT_F*4,usage:ST|DST|SRC}); dev.queue.writeBuffer(out,0,new Float32Array(OUT_F));
  const par=dev.createBuffer({size:16,usage:UNI|DST});
  const pv=new ArrayBuffer(16); new Float32Array(pv,0,2).set([D.inScale,D.linOut]); new Uint32Array(pv,8,1)[0]=D.pleOffset;
  dev.queue.writeBuffer(par,0,pv);
  const pipe=dev.createComputePipeline({layout:'auto',compute:{module:m,entryPoint:'main'}});
  const bg=dev.createBindGroup({layout:pipe.getBindGroupLayout(0),entries:[a,codes,rscale,ple,out,lut,par].map((b,i)=>({binding:i,resource:{buffer:b}}))});
  const enc=dev.createCommandEncoder();const ps=enc.beginComputePass();ps.setPipeline(pipe);ps.setBindGroup(0,bg);ps.dispatchWorkgroups(OUT_F,1,1);ps.end();dev.queue.submit([enc.finish()]);
  const verr=await dev.popErrorScope(); if(verr)return{validationError:verr.message};
  const rb=dev.createBuffer({size:OUT_F*4,usage:GPUBufferUsage.MAP_READ|DST});
  const e2=dev.createCommandEncoder();e2.copyBufferToBuffer(out,0,rb,0,OUT_F*4);dev.queue.submit([e2.finish()]);
  await rb.mapAsync(1); const g=new Float32Array(rb.getMappedRange()).slice(); rb.unmap();
  const ref=new Float32Array(b64(D.ref));
  let mx=0,mi=0,ne=0; for(let i=0;i<OUT_F;i++){const e=Math.abs(g[i]-ref[i]);if(e>0)ne++;if(e>mx){mx=e;mi=i;}}
  return {maxAbsDiff:+mx.toFixed(6), nonexact:ne, total:OUT_F, at:mi, gpu:+g[mi].toFixed(6), ref:+ref[mi].toFixed(6)};
 })()`;
 const r=await cmd("Runtime.evaluate",{expression:expr,awaitPromise:true,returnByValue:true,timeout:30000});
 if(r.result?.exceptionDetails)done({exception:r.result.exceptionDetails.text});
 done(r.result?.result?.value);
}catch(e){done({error:e.message});}
