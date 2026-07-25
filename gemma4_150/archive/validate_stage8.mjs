import { spawn } from "node:child_process"; import { setTimeout as sleep } from "node:timers/promises"; import { readFileSync } from "node:fs";
const CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", PORT=9257;
const REF="/Users/wako/projects/slang/reference/webml_gemma4_kernels";
const DATA="/private/tmp/claude-501/-Users-wako-projects-slang/ce5a80c9-3495-4ae0-8cd2-813d9c5d619f/scratchpad/stage8.json";
const k77=readFileSync(`${REF}/77_sg_sum.wgsl`,"utf8"); const D=JSON.parse(readFileSync(DATA,"utf8"));
const proc=spawn(CHROME,["--headless=new",`--remote-debugging-port=${PORT}`,"--user-data-dir=/tmp/cdp-s8","--enable-unsafe-webgpu","--use-angle=metal","--enable-features=WebGPU","--no-first-run","--disable-dev-shm-usage","http://localhost:8000/manifest.json"],{stdio:["ignore","ignore","ignore"]});
let ws; const done=o=>{console.log("STAGE8:",JSON.stringify(o));try{ws?.close();}catch{}proc.kill("SIGKILL");process.exit(0);};
try{
 let t=null;for(let i=0;i<100;i++){try{const l=await fetch(`http://localhost:${PORT}/json`).then(r=>r.json());t=l.find(x=>x.type==="page"&&x.webSocketDebuggerUrl);if(t)break;}catch{}await sleep(200);}
 ws=new WebSocket(t.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
 let id=0;const p=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id&&p.has(m.id)){p.get(m.id)(m);p.delete(m.id);}};
 const cmd=(mth,pr={})=>new Promise(res=>{const i=++id;p.set(i,res);ws.send(JSON.stringify({id:i,method:mth,params:pr}));});
 await cmd("Runtime.enable");
 const expr=`(async()=>{const {k77,D}=${JSON.stringify({k77,D})};
  const b64=s=>{const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u.buffer;};
  const adp=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  const dev=await adp.requestDevice({requiredFeatures:['subgroups'],requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(10,adp.limits.maxStorageBuffersPerShaderStage)}});
  const m=dev.createShaderModule({code:k77}); const ci=await m.getCompilationInfo(); const ce=ci.messages.filter(x=>x.type==='error'); if(ce.length)return{compileErrors:ce.map(e=>e.message)};
  dev.pushErrorScope('validation');
  const ST=GPUBufferUsage.STORAGE,DST=GPUBufferUsage.COPY_DST,SRC=GPUBufferUsage.COPY_SRC,UNI=GPUBufferUsage.UNIFORM;
  const mk=(ab,ex=0)=>{const b=dev.createBuffer({size:Math.max(ab.byteLength,4),usage:ST|DST|ex});dev.queue.writeBuffer(b,0,ab);return b;};
  const OUT_F=D.OUT_F;
  const a=mk(b64(D.a)), codes=mk(b64(D.codes)), rscale=mk(b64(D.row_scale)), w12s=mk(b64(D.w12s));
  const hidden=mk(b64(D.hidden), SRC);
  const pp=dev.createBuffer({size:(OUT_F+1)*4,usage:ST|DST}); dev.queue.writeBuffer(pp,0,new Uint32Array(OUT_F+1));
  const y2=dev.createBuffer({size:OUT_F*4,usage:ST|DST|SRC}); dev.queue.writeBuffer(y2,0,new Float32Array(OUT_F));
  const sum2=dev.createBuffer({size:4,usage:ST|DST|SRC}); dev.queue.writeBuffer(sum2,0,new Float32Array(1));
  const par=dev.createBuffer({size:16,usage:UNI|DST}); dev.queue.writeBuffer(par,0,new Float32Array([D.inScale,D.projIn,D.projOut,0]));
  const pipe=dev.createComputePipeline({layout:'auto',compute:{module:m,entryPoint:'main'}});
  const bg=dev.createBindGroup({layout:pipe.getBindGroupLayout(0),entries:[a,codes,rscale,pp,hidden,w12s,y2,sum2,par].map((b,i)=>({binding:i,resource:{buffer:b}}))});
  const enc=dev.createCommandEncoder();const ps=enc.beginComputePass();ps.setPipeline(pipe);ps.setBindGroup(0,bg);ps.dispatchWorkgroups(96,1,1);ps.end();dev.queue.submit([enc.finish()]);
  const verr=await dev.popErrorScope(); if(verr)return{validationError:verr.message};
  const rbH=dev.createBuffer({size:OUT_F*4,usage:GPUBufferUsage.MAP_READ|DST});
  const rbY=dev.createBuffer({size:OUT_F*4,usage:GPUBufferUsage.MAP_READ|DST});
  const rbS=dev.createBuffer({size:4,usage:GPUBufferUsage.MAP_READ|DST});
  const e2=dev.createCommandEncoder();e2.copyBufferToBuffer(hidden,0,rbH,0,OUT_F*4);e2.copyBufferToBuffer(y2,0,rbY,0,OUT_F*4);e2.copyBufferToBuffer(sum2,0,rbS,0,4);dev.queue.submit([e2.finish()]);
  await Promise.all([rbH.mapAsync(1),rbY.mapAsync(1),rbS.mapAsync(1)]);
  const gH=new Float32Array(rbH.getMappedRange()).slice();
  const gY=new Float32Array(rbY.getMappedRange()).slice();
  const gS=new Float32Array(rbS.getMappedRange())[0];
  const refH=new Float32Array(b64(D.ref_hidden)); const refY=new Float32Array(b64(D.ref_y2));
  let mxH=0,miH=0; for(let i=0;i<OUT_F;i++){const e=Math.abs(gH[i]-refH[i]);if(e>mxH){mxH=e;miH=i;}}
  let ybad=0,mxY=0; for(let i=0;i<OUT_F;i++){const e=Math.abs(gY[i]-refY[i]);if(e>0)ybad++;if(e>mxY)mxY=e;}
  return {hidden_maxAbsDiff:+mxH.toFixed(6), at:miH, gpuH:+gH[miH].toFixed(5), refH:+refH[miH].toFixed(5),
          y2_nonexact:ybad, y2_maxAbsDiff:+mxY.toFixed(6), total:OUT_F, sum2_gpu:+gS.toFixed(3), sum2_ref:+D.ref_sum2.toFixed(3)};
 })()`;
 const r=await cmd("Runtime.evaluate",{expression:expr,awaitPromise:true,returnByValue:true,timeout:30000});
 if(r.result?.exceptionDetails)done({exception:r.result.exceptionDetails.text});
 done(r.result?.result?.value);
}catch(e){done({error:e.message});}
