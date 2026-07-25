// gemma4_150 native runner (Rust / wgpu) — a port of gemma4_150/runner.py.
// Same reference fused SRQ kernels, persistent pooled buffers, cached bind groups,
// GPU-resident token feedback. Native recording (no Python FFI) targets the
// browser's ~156 tok/s. Correctness: coherent greedy decode ("... **Paris**.").

use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use serde_json::Value;
use wgpu::util::DeviceExt;

type RBuf = Rc<wgpu::Buffer>;

const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

const COMBINE: &str = r#"
const D:u32=256u; const HINV:f32=%HINV%; const EPS:f32=1e-6; const RS2:f32=0.7071067811865476;
@group(0) @binding(0) var<storage, read> ctx: array<f32>;
@group(0) @binding(1) var<storage, read> ple: array<f32>;
@group(0) @binding(2) var<storage, read> nw: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
var<workgroup> red: array<f32, D>;
@compute @workgroup_size(D,1,1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x; let tid = lid.x; let base = row*D + tid;
  let c = ctx[base]*HINV;
  red[tid] = c*c; workgroupBarrier();
  var s:u32 = D/2u; loop { if (s==0u){break;} if (tid<s){red[tid]=red[tid]+red[tid+s];} s=s/2u; workgroupBarrier(); }
  let rms = inverseSqrt(red[0]/f32(D)+EPS);
  outp[base] = (c*rms*nw[tid] + ple[base])*RS2;
}"#;

const KVNORM: &str = r#"
const HD:u32=%HD%; const HALF:u32=%HALF%; const EPS:f32=1e-6;
@group(0) @binding(0) var<storage, read> ink: array<f32>;
@group(0) @binding(1) var<storage, read> inv: array<f32>;
@group(0) @binding(2) var<storage, read> knorm: array<f32>;
@group(0) @binding(3) var<storage, read> cosT: array<f32>;
@group(0) @binding(4) var<storage, read> sinT: array<f32>;
@group(0) @binding(5) var<storage, read_write> kcache: array<f32>;
@group(0) @binding(6) var<storage, read_write> vcache: array<f32>;
@group(0) @binding(7) var<uniform> p: vec4<u32>;
var<workgroup> rk: array<f32, HD>;
var<workgroup> rv: array<f32, HD>;
@compute @workgroup_size(HD,1,1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let ko = ink[tid]; let vo = inv[tid];
  rk[tid] = ko*ko; rv[tid] = vo*vo; workgroupBarrier();
  var s:u32 = HD/2u; loop { if(s==0u){break;} if(tid<s){rk[tid]=rk[tid]+rk[tid+s]; rv[tid]=rv[tid]+rv[tid+s];} s=s/2u; workgroupBarrier(); }
  let rmsk = inverseSqrt(rk[0]/f32(HD)+EPS);
  let rmsv = inverseSqrt(rv[0]/f32(HD)+EPS);
  vcache[p.x + tid] = vo * rmsv;
  if (tid < HALF) {
    let n0 = ink[tid]*rmsk*knorm[tid];
    let n1 = ink[tid+HALF]*rmsk*knorm[tid+HALF];
    let c = cosT[tid]; let sn = sinT[tid];
    kcache[p.x + tid] = n0*c - n1*sn;
    kcache[p.x + tid + HALF] = n1*c + n0*sn;
  }
}"#;

fn kernel(name: &str) -> String {
    std::fs::read_to_string(format!("{}/reference/webml_gemma4_kernels/{}.wgsl", REPO, name)).unwrap()
}
fn patch(mut code: String, consts: &[(&str, i64)]) -> String {
    for (k, v) in consts {
        let suffix = if code.contains(&format!("const {}: u32", k)) { "u" } else { "" };
        let re = regex::Regex::new(&format!(r"const {}: (u32|f32) = [^;]+;", k)).unwrap();
        code = re.replace(&code, format!("const {}: $1 = {}{};", k, v, suffix).as_str()).to_string();
    }
    code
}

const ST: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC).union(wgpu::BufferUsages::COPY_DST);
const UNIF: wgpu::BufferUsages = wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST);

// precomputed per-layer data (avoids JSON clone + format!/HashMap lookups in the hot loop)
struct L {
    hd: u32, qd: u32, inter: u32, half: u32, shared: bool, src: usize, o_in: f64, nxt: f32,
    sc: HashMap<String, f32>,
    w: HashMap<&'static str, RBuf>,
}

const ROLES: [&str; 25] = ["in_norm", "q_norm", "k_norm", "q_bits", "q_scale", "k_bits", "v_bits",
    "qkv_scales", "o_bits", "o_scale", "o_w12", "gate_bits", "gate_scale", "up_bits", "up_scale",
    "gelu_gate", "down_bits", "down_scale", "down_nw", "plegate_codes", "plegate_rowscale",
    "gelu_plegate", "pleproj_codes", "pleproj_rowscale", "pleproj_w12s"];

struct Runner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    man: Value,
    bufs: HashMap<String, RBuf>,
    pool: HashMap<String, RBuf>,
    unis: HashMap<String, RBuf>,
    pipes: HashMap<String, Rc<wgpu::ComputePipeline>>,
    bgc: HashMap<String, Rc<wgpu::BindGroup>>,
    kc: HashMap<usize, RBuf>,
    vc: HashMap<usize, RBuf>,
    plan: Vec<(Rc<wgpu::ComputePipeline>, Rc<wgpu::BindGroup>, [u32; 3])>,
    rope_cfgs: Vec<(u32, f64, u64, bool)>,   // (head_dim, theta, cutoff, sliding), unique per type
}

impl Runner {
    fn cfg_u(&self, k: &str) -> u64 { self.man["config"][k].as_u64().unwrap() }

    fn build_layers(&self) -> Vec<L> {
        let arr = self.man["layers"].as_array().unwrap();
        arr.iter().enumerate().map(|(l, s)| {
            let mut w = HashMap::new();
            for r in ROLES {
                if let Some(b) = self.bufs.get(&format!("L{}.{}", l, r)) { w.insert(r, b.clone()); }
            }
            let mut sc = HashMap::new();
            for (k, v) in s["scales"].as_object().unwrap() { sc.insert(k.clone(), v.as_f64().unwrap() as f32); }
            let nxt = arr.get(l + 1).map(|n| n["scales"]["qkv_in"].as_f64().unwrap() as f32).unwrap_or(0.0);
            L {
                hd: s["head_dim"].as_u64().unwrap() as u32, qd: s["q_dim"].as_u64().unwrap() as u32,
                inter: s["intermediate"].as_u64().unwrap() as u32, half: s["head_dim"].as_u64().unwrap() as u32 / 2,
                shared: s["shared"].as_bool().unwrap(), src: s["kv_source"].as_u64().unwrap() as usize,
                o_in: s["scales"]["o_in"].as_f64().unwrap(), nxt, sc, w,
            }
        }).collect()
    }

    fn scratch(&mut self, name: &str, nbytes: u64) -> RBuf {
        let n = nbytes.max(4);
        if self.pool.get(name).map_or(true, |b| b.size() < n) {
            let b = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: n, usage: ST, mapped_at_creation: false });
            self.pool.insert(name.into(), Rc::new(b));
        }
        self.pool[name].clone()
    }
    fn uni(&mut self, name: &str, data: &[u8]) -> RBuf {
        let n = (data.len() as u64).max(16);
        if self.unis.get(name).map_or(true, |b| b.size() < n) {
            let b = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: n, usage: UNIF, mapped_at_creation: false });
            self.unis.insert(name.into(), Rc::new(b));
        }
        let b = self.unis[name].clone();
        self.queue.write_buffer(&b, 0, data);
        b
    }
    fn uni_static(&mut self, name: &str, data: &[u8]) -> RBuf {
        if !self.unis.contains_key(name) { return self.uni(name, data); }
        self.unis[name].clone()
    }
    // `build` (disk read + regex patch) runs only on a pipeline-cache MISS.
    fn dispatch(&mut self, key: &str, build: impl FnOnce() -> String, buffers: &[&RBuf], grid: [u32; 3], bgkey: &str) {
        let pipe = if let Some(p) = self.pipes.get(key) {
            p.clone()
        } else {
            // naga accepts `enable f16;` but not `enable subgroups;` (ops still work).
            let code = build().replace("enable subgroups;", "");
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.as_str().into()) });
            let p = Rc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None, layout: None, module: &module, entry_point: Some("main"),
                compilation_options: Default::default(), cache: None }));
            self.pipes.insert(key.into(), p.clone());
            p
        };
        let bg = if let Some(bg) = self.bgc.get(bgkey) {
            bg.clone()
        } else {
            let bgl = pipe.get_bind_group_layout(0);
            let entries: Vec<wgpu::BindGroupEntry> = buffers.iter().enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
            let bg = Rc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &entries }));
            self.bgc.insert(bgkey.into(), bg.clone());
            bg
        };
        self.plan.push((pipe, bg, grid));
    }
    fn flush(&mut self, copy: Option<(&wgpu::Buffer, &wgpu::Buffer, u64)>) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        for (pipe, bg, grid) in self.plan.drain(..) {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&pipe);
            cp.set_bind_group(0, Some(bg.as_ref()), &[]);
            cp.dispatch_workgroups(grid[0], grid[1], grid[2]);
        }
        if let Some((src, dst, off)) = copy { enc.copy_buffer_to_buffer(src, 0, dst, off, 4); }
        self.queue.submit([enc.finish()]);
    }
    fn read_u32(&self, buf: &wgpu::Buffer, nbytes: u64) -> Vec<u32> {
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: nbytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &rb, 0, nbytes);
        self.queue.submit([enc.finish()]);
        rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let out = bytemuck::cast_slice::<u8, u32>(&rb.slice(..).get_mapped_range()).to_vec();
        rb.unmap();
        out
    }

    fn write_step_uniforms(&mut self, pos: u32) {
        let n_h = self.cfg_u("nH") as u32;
        let win = self.cfg_u("window") as u32;
        if self.rope_cfgs.is_empty() {
            let mut seen = std::collections::HashSet::new();
            for s in self.man["layers"].as_array().unwrap() {
                let hd = s["head_dim"].as_u64().unwrap();
                if seen.insert(hd) {
                    self.rope_cfgs.push((hd as u32, s["rope_theta"].as_f64().unwrap(),
                        s["rope_cutoff"].as_u64().unwrap(), s["sliding"].as_bool().unwrap()));
                }
            }
        }
        for &(hd, theta, cutoff, sliding) in self.rope_cfgs.clone().iter() {
            let half = hd / 2;
            let mut cos = vec![0f32; half as usize];
            let mut sin = vec![0f32; half as usize];
            for i in 0..half {
                let inv = if (i as u64) < cutoff { 1.0 / theta.powf(i as f64 / half as f64) } else { 0.0 };
                let ang = pos as f64 * inv;
                cos[i as usize] = ang.cos() as f32; sin[i as usize] = ang.sin() as f32;
            }
            let cb = self.scratch(&format!("rcos{}", half), (half * 4) as u64);
            let sb = self.scratch(&format!("rsin{}", half), (half * 4) as u64);
            self.queue.write_buffer(&cb, 0, bytemuck::cast_slice(&cos));
            self.queue.write_buffer(&sb, 0, bytemuck::cast_slice(&sin));
            let par_a: [u32; 8] = [1, pos + 1, pos, n_h, 1, if sliding { win } else { 0 }, 0, 0];
            self.uni(&format!("parA_{}", hd), bytemuck::cast_slice(&par_a));
            let pkv: [u32; 4] = [pos * hd, 0, 0, 0];
            self.uni(&format!("pkv_{}", hd), bytemuck::cast_slice(&pkv));
        }
    }

    fn ids_buf(&mut self, tok: u32) -> RBuf {
        let b = self.scratch("ids", 4);
        self.queue.write_buffer(&b, 0, bytemuck::cast_slice(&[tok]));
        b
    }

    fn embed(&mut self, tok: u32, out: &RBuf, ids: Option<&RBuf>) {
        let par = self.uni_static("embed_par", bytemuck::cast_slice(&[1u32, 0, 0, 0]));
        let idb = match ids { Some(b) => b.clone(), None => self.ids_buf(tok) };
        let (eq, es) = (self.bufs["embed_q"].clone(), self.bufs["embed_scale"].clone());
        self.dispatch("embed", || kernel("00_main"), &[&idb, &eq, &es, out, &par], [1, 1, 1], "embed");
    }

    fn ple_input(&mut self, tok: u32, embed_buf: &RBuf, ids: Option<&RBuf>) -> RBuf {
        let (nl, d, h) = (self.cfg_u("nL"), self.cfg_u("ple_d"), self.cfg_u("H"));
        let ctx = self.scratch("ctx", nl * d * 4);
        let ple = self.scratch("plegath", nl * d * 4);
        let out = self.scratch("ple", nl * d * 4);
        let idb = match ids { Some(b) => b.clone(), None => self.ids_buf(tok) };
        let par0 = self.uni_static("ple_par0", bytemuck::cast_slice(&[0f32, 0.0, 0.0, 0.0]));
        let seq1 = self.uni_static("ple_seq1", bytemuck::cast_slice(&[1u32, 0, 0, 0]));
        let (pmp, pq, ps, ppn) = (self.bufs["pl_model_proj"].clone(), self.bufs["ple_q"].clone(),
            self.bufs["ple_scale"].clone(), self.bufs["pl_proj_norm"].clone());
        self.dispatch("proj68", || kernel("68_reduce"), &[embed_buf, &pmp, &ctx, &par0], [(nl * d / 8) as u32, 1, 1], "proj68");
        self.dispatch("plegather", || kernel("01_main"), &[&idb, &pq, &ps, &ple, &seq1], [1, 1, 1], "plegather");
        let hinv = (h as f64).powf(-0.5);
        self.dispatch("combine", || COMBINE.replace("%HINV%", &format!("{:.10e}", hinv)), &[&ctx, &ple, &ppn, &out], [nl as u32, 1, 1], "combine");
        out
    }

    fn layer(&mut self, ld: &L, l: usize, hidden: &RBuf, ple_buf: &RBuf, ple_off: u32) {
        let sc = |k: &str| ld.sc[k];
        let sc_opt = |k: &str| *ld.sc.get(k).unwrap_or(&0.0);
        let h = self.cfg_u("H");
        let n_h = self.cfg_u("nH") as u32;
        let (hd, qd, inter, half, shared, src) = (ld.hd, ld.qd, ld.inter, ld.half, ld.shared, ld.src);

        let cb = self.scratch(&format!("rcos{}", half), (half * 4) as u64);
        let sb = self.scratch(&format!("rsin{}", half), (half * 4) as u64);
        let kc = self.kc[&src].clone();
        let vc = self.vc[&src].clone();
        let outq = self.scratch("outq", (qd * 4) as u64);
        let dummy = self.scratch("dummy", (hd * 4) as u64);
        let dummy2 = self.scratch("dummy2", (hd * 4) as u64);
        let outk = self.scratch("outk", (hd * 4) as u64);
        let outv = self.scratch("outv", (hd * 4) as u64);
        let attn = self.scratch("attn", (qd * 4) as u64);
        let y2 = self.scratch("y2", h * 4);
        let sum2 = self.scratch("sum2", 4);
        let geglu = self.scratch("geglu", (inter * 2) as u64);
        let gate = self.scratch("gate", self.cfg_u("ple_d") * 4);
        let y2n = self.scratch("y2n", h * 4);
        let sum2n = self.scratch("sum2n", 4);
        let pp73 = self.scratch("pp73", (h + 1) * 4);
        let pp75 = self.scratch("pp75", (h + 1) * 4);
        let pp77 = self.scratch("pp77", (h + 1) * 4);
        let partials = self.scratch(&format!("partials{}", hd), (8 * 32 * (hd + 2) + 8) as u64 * 4);

        macro_rules! b { ($n:expr) => { ld.w[$n].clone() } }

        let (a, suma) = if l == 0 {
            let a = self.scratch("a", h * 4);
            let suma = self.scratch("suma", 4);
            let par = self.uni_static(&format!("p69_{}", l), bytemuck::cast_slice(&[1u32, 0, sc("qkv_in").to_bits(), 0]));
            let inw = b!("in_norm");
            self.dispatch("k69", || kernel("69_sg_sum"), &[hidden, &inw, &a, &suma, &par], [1, 1, 1], &format!("k69|{}", l));
            (a, suma)
        } else {
            (self.scratch("y2n", h * 4), self.scratch("sum2n", 4))
        };

        let total = qd / 2 + hd;
        let k70 = || patch(kernel("70_srq"), &[("Q_OUT", qd as i64), ("Q_WGS", (qd / 2) as i64),
            ("KV_OUT", hd as i64), ("KV_WGS", (hd / 2) as i64), ("TOTAL_WGS", total as i64), ("GRID_X", total as i64)]);
        let par70 = self.uni_static(&format!("p70_{}", l), bytemuck::cast_slice(&[sc("q_out"), sc_opt("k_out"), sc_opt("v_out"), 0.0]));
        let k70key = format!("k70_{}_{}", qd, hd);
        if shared {
            let qb = b!("q_bits"); let qs = b!("q_scale");
            self.dispatch(&k70key, k70, &[&a, &qb, &qb, &qb, &qs, &suma, &outq, &dummy, &dummy2, &par70], [qd / 2, 1, 1], &format!("k70|{}", l));
        } else {
            let (qb, kb, vb, qkvs) = (b!("q_bits"), b!("k_bits"), b!("v_bits"), b!("qkv_scales"));
            self.dispatch(&k70key, k70, &[&a, &qb, &kb, &vb, &qkvs, &suma, &outq, &outk, &outv, &par70], [total, 1, 1], &format!("k70|{}", l));
            let kn = b!("k_norm");
            let pkv = self.unis[&format!("pkv_{}", hd)].clone();
            self.dispatch(&format!("kvnorm_{}", hd),
                || KVNORM.replacen("%HD%", &format!("{}u", hd), 1).replacen("%HALF%", &format!("{}u", half), 1),
                &[&outk, &outv, &kn, &cb, &sb, &kc, &vc, &pkv], [1, 1, 1], &format!("kvn|{}", l));
        }
        let o_in = ld.o_in;
        let qn = b!("q_norm");
        let par_a = self.unis[&format!("parA_{}", hd)].clone();
        self.dispatch(&format!("att_{}_{}", hd, sc("o_in")),
            || patch(kernel("101_srq"), &[("HEAD_DIM", hd as i64), ("HALF_DIM", half as i64)])
                .replace("const OUT_Q: f32 = 0.014886821620166302;", &format!("const OUT_Q: f32 = {};", o_in)),
            &[&outq, &qn, &cb, &sb, &kc, &vc, &partials, &attn, &par_a], [n_h, 32, 1], &format!("att|{}", l));

        let par73 = self.uni_static(&format!("p73_{}", l), bytemuck::cast_slice(&[sc("o_out"), sc("gate_in"), 0.0, 0.0]));
        let (ob, os, ow) = (b!("o_bits"), b!("o_scale"), b!("o_w12"));
        self.dispatch(&format!("k73_{}", qd),
            || patch(kernel("73_sg_sum"), &[("IN_FEATURES", qd as i64), ("WORDS_PER_ROW", (qd / 8) as i64)]),
            &[&attn, &ob, &os, &pp73, hidden, &ow, &y2, &sum2, &par73], [192, 1, 1], &format!("k73|{}", l));

        let (guk, gug) = if inter == 6144 { ("74_sg_sum", 768) } else { ("95_sg_sum", 3072) };
        let par74 = self.uni_static(&format!("p74_{}", l), bytemuck::cast_slice(&[sc("gate_out"), sc("up_out"), sc("down_in"), 0.0]));
        let (gb, gs, ub, us, gl) = (b!("gate_bits"), b!("gate_scale"), b!("up_bits"), b!("up_scale"), b!("gelu_gate"));
        self.dispatch(&format!("gu_{}", inter), || kernel(guk), &[&y2, &gb, &gs, &ub, &us, &sum2, &geglu, &gl, &par74], [gug, 1, 1], &format!("gu|{}", l));

        let downk = if inter == 6144 { "75_srq" } else { "96_srq" };
        let par75 = self.uni_static(&format!("p75_{}", l), bytemuck::cast_slice(&[sc("down_in"), sc("down_out"), 0.0, 0.0]));
        let (db, ds, dn) = (b!("down_bits"), b!("down_scale"), b!("down_nw"));
        self.dispatch(&format!("down_{}", inter), || kernel(downk), &[&geglu, &db, &pp75, &ds, hidden, &dn, &par75], [(h / 4) as u32, 1, 1], &format!("down|{}", l));

        let mut parp = [0u8; 16];
        parp[0..4].copy_from_slice(&sc("plegate_in").to_le_bytes());
        parp[4..8].copy_from_slice(&sc("plegate_out").to_le_bytes());
        parp[8..12].copy_from_slice(&ple_off.to_le_bytes());
        let p76 = self.uni_static(&format!("p76_{}", l), &parp);
        let (pc, prs, pgl) = (b!("plegate_codes"), b!("plegate_rowscale"), b!("gelu_plegate"));
        self.dispatch("k76", || kernel("76_reduce"), &[hidden, &pc, &prs, ple_buf, &gate, &pgl, &p76], [self.cfg_u("ple_d") as u32, 1, 1], &format!("k76|{}", l));

        let par77 = self.uni_static(&format!("p77_{}", l), bytemuck::cast_slice(&[ld.nxt, sc("pleproj_in"), sc("pleproj_out"), 0.0]));
        let (ppc, pprs, ppw) = (b!("pleproj_codes"), b!("pleproj_rowscale"), b!("pleproj_w12s"));
        self.dispatch("k77", || kernel("77_sg_sum"), &[&gate, &ppc, &pprs, &pp77, hidden, &ppw, &y2n, &sum2n, &par77], [96, 1, 1], &format!("k77|{}", l));
    }

    fn forward(&mut self, layers: &[L], tok: u32, pos: u32, hidden: &RBuf, argmax: Option<&RBuf>, ids: Option<&RBuf>, gen: Option<&RBuf>, step: u32) {
        let (h, nl, d, v) = (self.cfg_u("H"), self.cfg_u("nL"), self.cfg_u("ple_d"), self.cfg_u("vocab"));
        self.write_step_uniforms(pos);
        self.embed(tok, hidden, ids);
        let ple = self.ple_input(tok, hidden, ids);
        for l in 0..nl as usize { self.layer(&layers[l], l, hidden, &ple, l as u32 * d as u32); }
        let normed = self.scratch("normed", h * 4);
        let sa = self.scratch("sa", 4);
        let pfn = self.uni_static("pfn", bytemuck::cast_slice(&[1u32, 0, 0, 0]));
        let fnw = self.bufs["final_norm"].clone();
        self.dispatch("finalnorm", || kernel("69_sg_sum"), &[hidden, &fnw, &normed, &sa, &pfn], [1, 1, 1], "finalnorm");
        let logits = self.scratch("logits", v * 4);
        let plog = self.uni_static("plog", bytemuck::cast_slice(&[0f32, 0.0, 0.0, 0.0]));
        let (lb, ls) = (self.bufs["lmhead_blk"].clone(), self.bufs["lmhead_scale"].clone());
        self.dispatch("logits", || kernel("33_srq"), &[&normed, &lb, &ls, &logits, &plog], [(v / 128) as u32, 1, 1], "logits");
        if let Some(am) = argmax {
            let cv = self.scratch("cv", 256 * 4);
            let ci = self.scratch("ci", 256 * 4);
            self.dispatch("amax1", || kernel("34_main"), &[&logits, &cv, &ci], [256, 1, 1], "amax1");
            self.dispatch("amax2", || kernel("35_main"), &[&cv, &ci, am], [1, 1, 1], "amax2");
            let copy = gen.map(|g| (am.as_ref(), g.as_ref(), step as u64 * 4));
            self.flush(copy);   // gen-copy folded into the same submit
        } else {
            self.flush(None);
        }
    }

    fn setup(&mut self) {
        let (h, d, nl, v) = (self.cfg_u("H"), self.cfg_u("ple_d"), self.cfg_u("nL"), self.cfg_u("vocab"));
        for s in self.man["layers"].as_array().unwrap().clone() {
            if !s["shared"].as_bool().unwrap() {
                let idx = s["index"].as_u64().unwrap() as usize;
                let hd = s["head_dim"].as_u64().unwrap();
                self.kc.insert(idx, Rc::new(self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 2048 * hd * 4, usage: ST, mapped_at_creation: false })));
                self.vc.insert(idx, Rc::new(self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 2048 * hd * 4, usage: ST, mapped_at_creation: false })));
            }
        }
        for (n, s) in [("hidden", h*4), ("a", h*4), ("outq", 4096*4), ("attn", 4096*4), ("outk", 512*4),
            ("outv", 512*4), ("dummy", 512*4), ("dummy2", 512*4), ("y2", h*4), ("y2n", h*4), ("geglu", 12288*2),
            ("gate", d*4), ("ctx", nl*d*4), ("plegath", nl*d*4), ("ple", nl*d*4), ("normed", h*4), ("logits", v*4),
            ("cv", 256*4), ("ci", 256*4), ("suma", 4), ("sum2", 4), ("sum2n", 4), ("sa", 4), ("ids", 4), ("cur", 4)] {
            self.scratch(n, s);
        }
        for (n, s) in [("pp73", (h+1)*4), ("pp75", (h+1)*4), ("pp77", (h+1)*4),
            ("partials256", (8*32*258+8)*4), ("partials512", (8*32*514+8)*4)] {
            let b = self.scratch(n, s);
            self.queue.write_buffer(&b, 0, &vec![0u8; s as usize]);
        }
    }

    fn bench(&mut self, n: usize) -> f64 {
        self.setup();
        let layers = self.build_layers();
        let hidden = self.scratch("hidden", self.cfg_u("H") * 4);
        let cur = self.scratch("cur", 4);
        let gen = self.scratch("gen", 256 * 4);
        self.queue.write_buffer(&cur, 0, bytemuck::cast_slice(&[2u32]));
        for p in 0..8 { self.forward(&layers, 0, p, &hidden, Some(&cur), Some(&cur), Some(&gen), p); }  // warm
        self.read_u32(&gen, 32);
        let t0 = std::time::Instant::now();
        for i in 0..n as u32 { self.forward(&layers, 0, 8 + i, &hidden, Some(&cur), Some(&cur), Some(&gen), i % 64); }
        self.read_u32(&gen, 4);
        n as f64 / t0.elapsed().as_secs_f64()
    }

    fn generate(&mut self, ids: &[u32], n_new: usize, eos: u32, chunk: usize) -> (Vec<u32>, f64) {
        self.setup();
        let layers = self.build_layers();
        let hidden = self.scratch("hidden", self.cfg_u("H") * 4);
        let cur = self.scratch("cur", 4);
        let gen = self.scratch("gen", n_new as u64 * 4);
        let mut pos = 0u32;
        for &t in &ids[..ids.len() - 1] {
            self.queue.write_buffer(&cur, 0, bytemuck::cast_slice(&[t]));
            self.forward(&layers, 0, pos, &hidden, None, Some(&cur), None, 0);
            pos += 1;
        }
        self.queue.write_buffer(&cur, 0, bytemuck::cast_slice(&[*ids.last().unwrap()]));
        let mut out: Vec<u32> = vec![];
        let t0 = std::time::Instant::now();
        let mut step = 0;
        while step < n_new {
            let end = (step + chunk).min(n_new);
            while step < end {
                self.forward(&layers, 0, pos, &hidden, Some(&cur), Some(&cur), Some(&gen), step as u32);
                pos += 1; step += 1;
            }
            let got = self.read_u32(&gen, step as u64 * 4);
            while out.len() < step {
                let tk = got[out.len()];
                if tk == eos { let n = out.len() as f64; return (out, n / t0.elapsed().as_secs_f64()); }
                out.push(tk);
            }
        }
        let n = out.len() as f64;
        (out, n / t0.elapsed().as_secs_f64())
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prompt = args.get(1).cloned().unwrap_or_else(|| "The capital of France is".into());
    let n_new: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(48);

    let model = PathBuf::from(REPO).join("models/gemma-4-E2B-qat");
    let gdir = model.join("g4_150");
    let man: Value = serde_json::from_slice(&std::fs::read(gdir.join("manifest.json")).unwrap()).unwrap();

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).unwrap();
    let al = adapter.limits();
    let big = al.max_storage_buffer_binding_size.min(1u32 << 31);
    let mut lim = wgpu::Limits::default();
    lim.max_buffer_size = big as u64;
    lim.max_storage_buffer_binding_size = big;
    lim.max_storage_buffers_per_shader_stage = 10;
    lim.max_compute_workgroup_size_x = 512.min(al.max_compute_workgroup_size_x);
    lim.max_compute_invocations_per_workgroup = 512.min(al.max_compute_invocations_per_workgroup);
    let feats = (wgpu::Features::SUBGROUP | wgpu::Features::SHADER_F16) & adapter.features();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None, required_features: feats, required_limits: lim, memory_hints: Default::default(), experimental_features: Default::default(), trace: wgpu::Trace::Off })).unwrap();

    println!("loading weights…");
    let mm = unsafe { memmap2::Mmap::map(&std::fs::File::open(gdir.join("weights.bin")).unwrap()).unwrap() };
    let mut bufs: HashMap<String, RBuf> = HashMap::new();
    for (name, t) in man["tensors"].as_object().unwrap() {
        let (off, len) = (t["off"].as_u64().unwrap() as usize, t["len"].as_u64().unwrap() as usize);
        let b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &mm[off..off + len], usage: ST });
        bufs.insert(name.clone(), Rc::new(b));
    }
    queue.submit([]);

    let mut r = Runner { device, queue, man, bufs, pool: HashMap::new(), unis: HashMap::new(),
        pipes: HashMap::new(), bgc: HashMap::new(), kc: HashMap::new(), vc: HashMap::new(), plan: vec![], rope_cfgs: vec![] };

    let tok = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).unwrap();
    let inner = format!("<|turn>user\n{}<turn|>\n<|turn>model\n", prompt);
    let mut ids = vec![2u32];
    ids.extend_from_slice(tok.encode(inner, false).unwrap().get_ids());

    let (out, _tps) = r.generate(&ids, n_new, 1, n_new.min(64));
    let text = tok.decode(&out, true).unwrap();
    println!("prompt: {:?}", prompt);
    println!("=> {:?}", text);
    let sustained = r.bench(64);
    println!("decode: {:.1} tok/s (sustained, warm)", sustained);
}
