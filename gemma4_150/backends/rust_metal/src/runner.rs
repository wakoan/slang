// Full gemma4_150 decode runner in Rust + native Metal — a port of
// swift-gemma4/Sources/g4/Runner.swift. Same manifest, same weights.bin, same
// .metal kernels (compiled from gemma4_150/kernels_msl), same ring-buffered
// resident decode and multi-turn chat.
use crate::metal::MetalRunner;
use memmap2::Mmap;
use metal::{Buffer, BufferRef, CommandBuffer};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Instant;

const RING: usize = 32; // per-step uniform ring; must be >= max chunk

struct LayerInfo {
    sliding: bool,
    head_dim: usize,
    q_dim: usize,
    intermediate: usize,
    shared: bool,
    kv_source: usize,
    rope_theta: f64,
    rope_cutoff: usize,
    scales: HashMap<String, f32>,
}

pub struct G4 {
    r: MetalRunner,
    // config
    h: usize,
    nl: usize,
    nh: usize,
    vocab: usize,
    window: u32,
    ple_d: usize,
    pub bos: u32,
    pub eos: u32,
    layers: Vec<LayerInfo>,
    hd_types: Vec<(usize, bool)>, // distinct head dims -> sliding
    // resources
    w: HashMap<String, Buffer>,
    pool: HashMap<String, Buffer>,
    uni: HashMap<String, Buffer>,
    kc: HashMap<usize, Buffer>,
    vc: HashMap<usize, Buffer>,
    _mmap: Mmap,
    // chat state
    pub chat_pos: usize,
    pub chat_carry: Option<u32>,
}

fn f2u(x: f32) -> u32 {
    x.to_bits()
}

fn compile_all(r: &mut MetalRunner, layers: &[LayerInfo], kdir: &std::path::Path) -> Result<(), String> {
    for entry in std::fs::read_dir(kdir).map_err(|e| e.to_string())? {
        let p = entry.map_err(|e| e.to_string())?.path();
        if p.extension().map_or(false, |e| e == "metal") {
            r.compile_kernel(p.to_str().unwrap())?;
        }
    }
    let file = |n: &str| kdir.join(format!("{n}.metal")).to_string_lossy().into_owned();
    let mut seen_hd = std::collections::HashSet::new();
    for (l, li) in layers.iter().enumerate() {
        let (hd, qd) = (li.head_dim, li.q_dim);
        let total = qd / 2 + hd;
        r.compile_variant(&file("qkv_70"), &format!("qkv_{qd}_{hd}"), &[
            ("Q_OUT=2048u", &format!("Q_OUT={qd}u")),
            ("KV_OUT=256u", &format!("KV_OUT={hd}u")),
            ("Q_WGS=1024u", &format!("Q_WGS={}u", qd / 2)),
            ("KV_WGS=128u", &format!("KV_WGS={}u", hd / 2)),
            ("TOTAL_WGS=1280u", &format!("TOTAL_WGS={total}u")),
            ("GRID_X=1280u", &format!("GRID_X={total}u")),
        ])?;
        r.compile_variant(&file("oproj_73"), &format!("oproj_{qd}"), &[
            ("IN_FEATURES=2048u", &format!("IN_FEATURES={qd}u")),
            ("WPR=256u", &format!("WPR={}u", qd / 8)),
        ])?;
        let oin = li.scales["o_in"];
        r.compile_variant(&file("attn_101"), &format!("attn_{hd}_{l}"), &[
            ("HEAD_DIM=512u", &format!("HEAD_DIM={hd}u")),
            ("HALF_DIM=256u", &format!("HALF_DIM={}u", hd / 2)),
            ("OUT_Q=0.014886821620166302f", &format!("OUT_Q={oin}f")),
        ])?;
        if seen_hd.insert(hd) {
            r.compile_variant(&file("kvnorm"), &format!("kvnorm_{hd}"), &[
                ("HD=256u", &format!("HD={hd}u")),
                ("HALF=128u", &format!("HALF={}u", hd / 2)),
                ("threadsPerThreadgroup = (256)", &format!("threadsPerThreadgroup = ({hd})")),
            ])?;
        }
    }
    Ok(())
}

impl G4 {
    fn wb(&self, n: &str) -> &BufferRef {
        &self.w[n]
    }
    fn pb(&self, n: &str) -> &BufferRef {
        &self.pool[n]
    }
    fn ub(&self, n: &str) -> &BufferRef {
        &self.uni[n]
    }
    fn sc(&self, l: usize, k: &str) -> f32 {
        self.layers[l].scales[k]
    }
    fn sc_opt(&self, l: usize, k: &str) -> f32 {
        self.layers[l].scales.get(k).copied().unwrap_or(0.0)
    }

    pub fn new() -> Result<Self, String> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().parent().unwrap().to_path_buf();
        let gdir = root.join("models/gemma-4-E2B-qat/g4_150");
        let kdir = root.join("gemma4_150/kernels_msl");
        let man: Value = serde_json::from_slice(
            &std::fs::read(gdir.join("manifest.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let cfg = &man["config"];
        let ci = |k: &str| cfg[k].as_i64().unwrap() as usize;

        let mut layers = Vec::new();
        for lv in man["layers"].as_array().unwrap() {
            let mut scales = HashMap::new();
            for (k, v) in lv["scales"].as_object().unwrap() {
                scales.insert(k.clone(), v.as_f64().unwrap() as f32);
            }
            layers.push(LayerInfo {
                sliding: lv["sliding"].as_bool().unwrap(),
                head_dim: lv["head_dim"].as_i64().unwrap() as usize,
                q_dim: lv["q_dim"].as_i64().unwrap() as usize,
                intermediate: lv["intermediate"].as_i64().unwrap() as usize,
                shared: lv["shared"].as_bool().unwrap(),
                kv_source: lv["kv_source"].as_i64().unwrap() as usize,
                rope_theta: lv["rope_theta"].as_f64().unwrap(),
                rope_cutoff: lv["rope_cutoff"].as_i64().unwrap() as usize,
                scales,
            });
        }

        // weights.bin mmap + tensor buffers
        let file = std::fs::File::open(gdir.join("weights.bin")).map_err(|e| e.to_string())?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| e.to_string())? };
        let mut r = MetalRunner::new()?;
        let mut w = HashMap::new();
        for (name, t) in man["tensors"].as_object().unwrap() {
            let off = t["off"].as_u64().unwrap() as usize;
            let len = t["len"].as_u64().unwrap() as usize;
            w.insert(name.clone(), r.buffer_bytes(mmap[off..off + len].as_ptr(), len));
        }
        compile_all(&mut r, &layers, &kdir)?;

        let mut g = G4 {
            r,
            h: ci("H"),
            nl: ci("nL"),
            nh: ci("nH"),
            vocab: ci("vocab"),
            window: ci("window") as u32,
            ple_d: ci("ple_d"),
            bos: ci("bos") as u32,
            eos: ci("eos") as u32,
            layers,
            hd_types: Vec::new(),
            w,
            pool: HashMap::new(),
            uni: HashMap::new(),
            kc: HashMap::new(),
            vc: HashMap::new(),
            _mmap: mmap,
            chat_pos: 0,
            chat_carry: None,
        };
        g.setup();
        g.build_uniforms();
        Ok(g)
    }

    fn setup(&mut self) {
        let max_seq = 2048usize;
        // distinct head-dim types (last sliding flag wins, as in Swift)
        let mut hd_map: HashMap<usize, bool> = HashMap::new();
        for l in 0..self.nl {
            hd_map.insert(self.layers[l].head_dim, self.layers[l].sliding);
        }
        self.hd_types = hd_map.into_iter().collect();

        for l in 0..self.nl {
            if !self.layers[l].shared {
                let hd = self.layers[l].head_dim;
                self.kc.insert(l, self.r.new_buffer(max_seq * hd * 4));
                self.vc.insert(l, self.r.new_buffer(max_seq * hd * 4));
            }
        }
        let alloc = |g: &mut G4, n: &str, bytes: usize| {
            g.pool.insert(n.to_string(), g.r.new_buffer(bytes));
        };
        alloc(self, "hidden", self.h * 4);
        alloc(self, "cur", 4);
        alloc(self, "gen", 256 * 4);
        for (n, sz) in [
            ("pp73", (self.h + 1) * 4),
            ("pp75", (self.h + 1) * 4),
            ("pp77", (self.h + 1) * 4),
            ("partials256", (8 * 32 * 258 + 8) * 4),
            ("partials512", (8 * 32 * 514 + 8) * 4),
        ] {
            let b = self.r.new_buffer(sz);
            unsafe { std::ptr::write_bytes(b.contents() as *mut u8, 0, sz); }
            self.pool.insert(n.to_string(), b);
        }
        // remaining scratch sized at maxima
        let h = self.h;
        for (n, sz) in [
            ("normed", h * 4), ("logits", self.vocab * 4), ("cv", 256 * 4), ("ci", 256 * 4),
            ("sa", 4), ("a", h * 4), ("suma", 4),
            ("outq", 4096 * 4), ("outk", 512 * 4), ("outv", 512 * 4),
            ("dummy", 512 * 4), ("dummy2", 512 * 4), ("attn", 4096 * 4),
            ("y2", h * 4), ("sum2", 4), ("geglu", 12288 * 2), ("gate", self.ple_d * 4),
            ("y2n", h * 4), ("sum2n", 4),
            ("ctx", self.nl * self.ple_d * 4), ("plegath", self.nl * self.ple_d * 4),
            ("ple", self.nl * self.ple_d * 4),
        ] {
            alloc(self, n, sz);
        }
        // ring of per-step position uniforms (parA/pkv) per head-dim type
        let hds: Vec<usize> = self.hd_types.iter().map(|&(hd, _)| hd).collect();
        for hd in hds {
            for slot in 0..RING {
                self.pool.insert(format!("parA_{hd}_{slot}"), self.r.new_buffer(32));
                self.pool.insert(format!("pkv_{hd}_{slot}"), self.r.new_buffer(16));
            }
        }
        // rope cos/sin precompute per head-dim type
        for &(hd, _) in &self.hd_types.clone() {
            let (theta, cutoff) = {
                let mut t = (0.0, 0);
                for l in 0..self.nl {
                    if self.layers[l].head_dim == hd {
                        t = (self.layers[l].rope_theta, self.layers[l].rope_cutoff);
                    }
                }
                t
            };
            let half = hd / 2;
            let mut cosv = vec![0f32; max_seq * half];
            let mut sinv = vec![0f32; max_seq * half];
            for pos in 0..max_seq {
                for i in 0..half {
                    let inv = if i < cutoff { 1.0 / theta.powf(i as f64 / half as f64) } else { 0.0 };
                    let ang = pos as f64 * inv;
                    cosv[pos * half + i] = ang.cos() as f32;
                    sinv[pos * half + i] = ang.sin() as f32;
                }
            }
            self.pool.insert(format!("rcosT_{hd}"), self.r.buffer_f32(&cosv));
            self.pool.insert(format!("rsinT_{hd}"), self.r.buffer_f32(&sinv));
        }
    }

    fn build_uniforms(&mut self) {
        let uf = |g: &mut G4, k: &str, a: &[f32]| {
            g.uni.insert(k.to_string(), g.r.buffer_f32(a));
        };
        uf(self, "z4", &[0.0; 4]);
        uf(self, "plog", &[0.0; 4]);
        let uu = |g: &mut G4, k: &str, a: &[u32]| {
            g.uni.insert(k.to_string(), g.r.buffer_u32(a));
        };
        uu(self, "ones1", &[1, 0, 0, 0]); // embed / plegather seq / final-norm
        let d = self.ple_d as u32;
        for l in 0..self.nl {
            if l == 0 {
                let p = [1u32, 0, f2u(self.sc(l, "qkv_in")), 0];
                self.uni.insert(format!("p69_{l}"), self.r.buffer_u32(&p));
            }
            let p70 = [self.sc(l, "q_out"), self.sc_opt(l, "k_out"), self.sc_opt(l, "v_out"), 0.0];
            self.uni.insert(format!("p70_{l}"), self.r.buffer_f32(&p70));
            let p73 = [self.sc(l, "o_out"), self.sc(l, "gate_in"), 0.0, 0.0];
            self.uni.insert(format!("p73_{l}"), self.r.buffer_f32(&p73));
            let p74 = [self.sc(l, "gate_out"), self.sc(l, "up_out"), self.sc(l, "down_in"), 0.0];
            self.uni.insert(format!("p74_{l}"), self.r.buffer_f32(&p74));
            let p75 = [self.sc(l, "down_in"), self.sc(l, "down_out"), 0.0, 0.0];
            self.uni.insert(format!("p75_{l}"), self.r.buffer_f32(&p75));
            let p76 = [f2u(self.sc(l, "plegate_in")), f2u(self.sc(l, "plegate_out")), l as u32 * d, 0];
            self.uni.insert(format!("p76_{l}"), self.r.buffer_u32(&p76));
            let nxt = if l + 1 < self.nl { self.sc(l + 1, "qkv_in") } else { 0.0 };
            let p77 = [nxt, self.sc(l, "pleproj_in"), self.sc(l, "pleproj_out"), 0.0];
            self.uni.insert(format!("p77_{l}"), self.r.buffer_f32(&p77));
        }
    }

    fn write_step_uniforms(&self, pos: usize, slot: usize) {
        let (nh, win) = (self.nh as u32, self.window);
        for &(hd, sliding) in &self.hd_types {
            let pa = self.pb(&format!("parA_{hd}_{slot}")).contents() as *mut u32;
            let vals = [1u32, pos as u32 + 1, pos as u32, nh, 1, if sliding { win } else { 0 }, 0, 0];
            unsafe { std::ptr::copy_nonoverlapping(vals.as_ptr(), pa, 8); }
            let pk = self.pb(&format!("pkv_{hd}_{slot}")).contents() as *mut u32;
            let kv = [(pos * hd) as u32, 0, 0, 0];
            unsafe { std::ptr::copy_nonoverlapping(kv.as_ptr(), pk, 4); }
        }
    }

    // ---- small shared-memory helpers ----
    pub fn set_cur(&self, tok: u32) {
        unsafe { *(self.pb("cur").contents() as *mut u32) = tok; }
    }
    fn read_gen(&self, i: usize) -> u32 {
        unsafe { *((self.pb("gen").contents() as *const u32).add(i)) }
    }
    fn read_cur(&self) -> u32 {
        unsafe { *(self.pb("cur").contents() as *const u32) }
    }

    /// One decode/prefill step. Reads token from `cur`; on `argmax` writes the
    /// next token back into `cur` and (if `gen`) blits it to gen[step].
    fn forward(&self, pos: usize, step: usize, argmax: bool, gen: bool, wait: bool) -> Option<CommandBuffer> {
        let slot = step % RING;
        self.write_step_uniforms(pos, slot);
        let mut b = self.r.batch();

        // embed
        b.dg("embed_00", &[
            (self.pb("cur"), 0), (self.wb("embed_q"), 0), (self.wb("embed_scale"), 0),
            (self.pb("hidden"), 0), (self.ub("ones1"), 0)], 1);
        // PLE input (proj_68 + plegather_01 + combine)
        b.dg("proj_68", &[
            (self.pb("hidden"), 0), (self.wb("pl_model_proj"), 0), (self.pb("ctx"), 0), (self.ub("z4"), 0)], 1120);
        b.dg("plegather_01", &[
            (self.pb("cur"), 0), (self.wb("ple_q"), 0), (self.wb("ple_scale"), 0),
            (self.pb("plegath"), 0), (self.ub("ones1"), 0)], 1);
        b.dg("combine", &[
            (self.pb("ctx"), 0), (self.pb("plegath"), 0), (self.wb("pl_proj_norm"), 0), (self.pb("ple"), 0)],
            self.nl as u64);

        for l in 0..self.nl {
            self.layer(&mut b, l, pos, slot);
        }

        // final norm + logits
        b.dg("rmssrq_69", &[
            (self.pb("hidden"), 0), (self.wb("final_norm"), 0), (self.pb("normed"), 0),
            (self.pb("sa"), 0), (self.ub("ones1"), 0)], 1);
        b.dg("logits_33", &[
            (self.pb("normed"), 0), (self.wb("lmhead_blk"), 0), (self.wb("lmhead_scale"), 0),
            (self.pb("logits"), 0), (self.ub("plog"), 0)], (self.vocab / 128) as u64);

        if argmax {
            b.dg("argmax1_34", &[(self.pb("logits"), 0), (self.pb("cv"), 0), (self.pb("ci"), 0)], 256);
            b.dg("argmax2_35", &[(self.pb("cv"), 0), (self.pb("ci"), 0), (self.pb("cur"), 0)], 1);
            if gen {
                b.blit_copy(self.pb("cur"), 0, self.pb("gen"), (step * 4) as u64, 4);
            }
        }
        if wait {
            b.commit_and_wait();
            None
        } else {
            Some(b.commit())
        }
    }

    fn layer(&self, b: &mut crate::metal::CommandBatch, l: usize, pos: usize, slot: usize) {
        let li = &self.layers[l];
        let (hd, qd, inter) = (li.head_dim, li.q_dim, li.intermediate);
        let half = hd / 2;
        let src = li.kv_source;
        let roff = (pos * half * 4) as u64;
        let cb = self.pb(&format!("rcosT_{hd}"));
        let sb = self.pb(&format!("rsinT_{hd}"));
        let kcb = &self.kc[&src];
        let vcb = &self.vc[&src];
        let lw = |n: &str| self.wb(&format!("L{l}.{n}"));
        let par_a = self.pb(&format!("parA_{hd}_{slot}"));
        let pkv = self.pb(&format!("pkv_{hd}_{slot}"));
        let partials = self.pb(&format!("partials{hd}"));

        // 69 fused RMSNorm+SRQ only on layer 0; later layers reuse prev k77's y2n/sum2n
        let (a, suma): (&BufferRef, &BufferRef) = if l == 0 {
            b.dg("rmssrq_69", &[
                (self.pb("hidden"), 0), (lw("in_norm"), 0), (self.pb("a"), 0), (self.pb("suma"), 0),
                (self.ub(&format!("p69_{l}")), 0)], 1);
            (self.pb("a"), self.pb("suma"))
        } else {
            (self.pb("y2n"), self.pb("sum2n"))
        };

        // 70 qkv
        let par70 = self.ub(&format!("p70_{l}"));
        if li.shared {
            b.dg(&format!("qkv_{qd}_{hd}"), &[
                (a, 0), (lw("q_bits"), 0), (lw("q_bits"), 0), (lw("q_bits"), 0), (lw("q_scale"), 0),
                (suma, 0), (self.pb("outq"), 0), (self.pb("dummy"), 0), (self.pb("dummy2"), 0), (par70, 0)],
                (qd / 2) as u64);
        } else {
            b.dg(&format!("qkv_{qd}_{hd}"), &[
                (a, 0), (lw("q_bits"), 0), (lw("k_bits"), 0), (lw("v_bits"), 0), (lw("qkv_scales"), 0),
                (suma, 0), (self.pb("outq"), 0), (self.pb("outk"), 0), (self.pb("outv"), 0), (par70, 0)],
                (qd / 2 + hd) as u64);
            b.dg(&format!("kvnorm_{hd}"), &[
                (self.pb("outk"), 0), (self.pb("outv"), 0), (lw("k_norm"), 0), (cb, roff), (sb, roff),
                (kcb, 0), (vcb, 0), (pkv, 0)], 1);
        }
        // attention (2-D: heads x chunks)
        b.dg2d(&format!("attn_{hd}_{l}"), &[
            (self.pb("outq"), 0), (lw("q_norm"), 0), (cb, roff), (sb, roff), (kcb, 0), (vcb, 0),
            (partials, 0), (self.pb("attn"), 0), (par_a, 0)], self.nh as u64, 32);
        // 73 o-proj + norms
        b.dg(&format!("oproj_{qd}"), &[
            (self.pb("attn"), 0), (lw("o_bits"), 0), (lw("o_scale"), 0), (self.pb("pp73"), 0),
            (self.pb("hidden"), 0), (lw("o_w12"), 0), (self.pb("y2"), 0), (self.pb("sum2"), 0),
            (self.ub(&format!("p73_{l}")), 0)], 192);
        // 74/95 gate+up, 75/96 down
        let par74 = self.ub(&format!("p74_{l}"));
        let par75 = self.ub(&format!("p75_{l}"));
        if inter == 6144 {
            b.dg("gateup_74", &[
                (self.pb("y2"), 0), (lw("gate_bits"), 0), (lw("gate_scale"), 0), (lw("up_bits"), 0),
                (lw("up_scale"), 0), (self.pb("sum2"), 0), (self.pb("geglu"), 0), (lw("gelu_gate"), 0), (par74, 0)],
                768);
            b.dg("down_75", &[
                (self.pb("geglu"), 0), (lw("down_bits"), 0), (self.pb("pp75"), 0), (lw("down_scale"), 0),
                (self.pb("hidden"), 0), (lw("down_nw"), 0), (par75, 0)], 384);
        } else {
            b.dg("gateup_95", &[
                (self.pb("y2"), 0), (lw("gate_bits"), 0), (lw("gate_scale"), 0), (lw("up_bits"), 0),
                (lw("up_scale"), 0), (self.pb("sum2"), 0), (self.pb("geglu"), 0), (lw("gelu_gate"), 0), (par74, 0)],
                3072);
            b.dg("down_96", &[
                (self.pb("geglu"), 0), (lw("down_bits"), 0), (self.pb("pp75"), 0), (lw("down_scale"), 0),
                (self.pb("hidden"), 0), (lw("down_nw"), 0), (par75, 0)], 384);
        }
        // 76 PLE gate
        b.dg("plegate_76", &[
            (self.pb("hidden"), 0), (lw("plegate_codes"), 0), (lw("plegate_rowscale"), 0), (self.pb("ple"), 0),
            (self.pb("gate"), 0), (lw("gelu_plegate"), 0), (self.ub(&format!("p76_{l}")), 0)], 256);
        // 77 PLE proj (feeds next layer's pre-norm via y2n/sum2n)
        b.dg("pleproj_77", &[
            (self.pb("gate"), 0), (lw("pleproj_codes"), 0), (lw("pleproj_rowscale"), 0), (self.pb("pp77"), 0),
            (self.pb("hidden"), 0), (lw("pleproj_w12s"), 0), (self.pb("y2n"), 0), (self.pb("sum2n"), 0),
            (self.ub(&format!("p77_{l}")), 0)], 96);
    }

    // ---- single-shot greedy generate (mirrors Swift generate) ----
    pub fn generate(&mut self, ids: &[u32], n_new: usize) -> (Vec<u32>, f64) {
        let chunk = 32usize;
        let mut pos = 0usize;
        for &t in &ids[..ids.len() - 1] {
            self.set_cur(t);
            self.forward(pos, 0, false, false, true);
            pos += 1;
        }
        self.set_cur(*ids.last().unwrap());
        let mut out = Vec::new();
        let t0 = Instant::now();
        let mut step = 0usize;
        let cap = n_new.min(240);
        while step < cap {
            let end = (step + chunk).min(cap);
            let mut last: Option<CommandBuffer> = None;
            while step < end {
                last = self.forward(pos, step, true, true, false);
                pos += 1;
                step += 1;
            }
            if let Some(cb) = last {
                cb.wait_until_completed();
            }
            while out.len() < step {
                let tk = self.read_gen(out.len());
                if tk == self.eos {
                    let dt = t0.elapsed().as_secs_f64();
                    return (out.clone(), if out.is_empty() { 0.0 } else { out.len() as f64 / dt });
                }
                out.push(tk);
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        let tps = if out.is_empty() { 0.0 } else { out.len() as f64 / dt };
        (out, tps)
    }

    pub fn chat_reset(&mut self) {
        self.chat_pos = 0;
        self.chat_carry = None;
    }

    /// One chat turn: prefill `frame` after the carried token, then chunked
    /// resident decode until a `stop` token or `max_new`. Rewinds chat_pos/carry
    /// past the speculative tail so the KV cache stays contiguous across turns.
    pub fn chat_turn<F: FnMut(&[u32])>(
        &mut self, frame: &[u32], max_new: usize, stop: &[u32], mut on_chunk: F,
    ) -> (Vec<u32>, f64) {
        let chunk = 32usize;
        let mut pre: Vec<u32> = Vec::new();
        if let Some(c) = self.chat_carry {
            pre.push(c);
        }
        pre.extend_from_slice(frame);
        for &t in &pre[..pre.len() - 1] {
            self.set_cur(t);
            self.forward(self.chat_pos, 0, false, false, true);
            self.chat_pos += 1;
        }
        self.set_cur(*pre.last().unwrap());
        let prefill_end = self.chat_pos; // g_i lands at prefill_end + i + 1

        let mut out: Vec<u32> = Vec::new();
        let cap = max_new.min(240);
        let mut step = 0usize;
        let mut stopped = false;
        let t0 = Instant::now();
        while step < cap && !stopped && self.chat_pos < 2040 {
            let end = (step + chunk).min(cap);
            let mut last: Option<CommandBuffer> = None;
            while step < end {
                last = self.forward(self.chat_pos, step, true, true, false);
                self.chat_pos += 1;
                step += 1;
            }
            if let Some(cb) = last {
                cb.wait_until_completed();
            }
            let mut i = out.len();
            while i < step {
                let tk = self.read_gen(i);
                if stop.contains(&tk) {
                    stopped = true;
                    self.chat_carry = Some(tk);
                    self.chat_pos = prefill_end + i + 1;
                    break;
                }
                out.push(tk);
                i += 1;
            }
            on_chunk(&out);
        }
        if !stopped {
            self.chat_carry = Some(self.read_cur());
        }
        let dt = t0.elapsed().as_secs_f64();
        (out, if dt > 0.0 { step as f64 / dt } else { 0.0 })
    }
}
