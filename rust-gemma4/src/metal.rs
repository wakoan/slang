// Thin native-Metal wrapper (metal-rs), mirroring swift-gemma4's MetalRunner:
// runtime MSL compile, shape-variant string patching, and a CommandBatch that
// records many dispatches into one command buffer. No wgpu in the path.
use metal::*;
use std::collections::HashMap;
use std::ffi::c_void;

pub struct MetalRunner {
    pub device: Device,
    pub queue: CommandQueue,
    // name -> (pipeline, threadsPerThreadgroup)
    pub kernels: HashMap<String, (ComputePipelineState, u64)>,
}

fn parse_threads(src: &str) -> Option<u64> {
    let anchor = "threadsPerThreadgroup = (";
    let i = src.find(anchor)? + anchor.len();
    let digits: String = src[i..].chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

impl MetalRunner {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("Metal device not available")?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue, kernels: HashMap::new() })
    }

    /// Compile one .metal file; the kernel function is the file's basename.
    pub fn compile_kernel(&mut self, path: &str) -> Result<(), String> {
        let src = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let name = stem(path);
        self.register(&src, &name, &name)
    }

    /// Compile a shape variant: apply `replace` substitutions, register under `name`
    /// (the MSL function keeps its base/file name).
    pub fn compile_variant(&mut self, path: &str, name: &str, replace: &[(&str, &str)]) -> Result<(), String> {
        if self.kernels.contains_key(name) {
            return Ok(());
        }
        let mut src = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        for (a, b) in replace {
            src = src.replace(a, b);
        }
        let base = stem(path);
        self.register(&src, &base, name)
    }

    fn register(&mut self, src: &str, func_name: &str, key: &str) -> Result<(), String> {
        let opts = CompileOptions::new();
        let lib = self.device.new_library_with_source(src, &opts)
            .map_err(|e| format!("MSL compile failed for {key}: {e}"))?;
        let func = lib.get_function(func_name, None)
            .map_err(|e| format!("function '{func_name}' not found for {key}: {e}"))?;
        let pso = self.device.new_compute_pipeline_state_with_function(&func)
            .map_err(|e| format!("pipeline failed for {key}: {e}"))?;
        self.kernels.insert(key.to_string(), (pso, parse_threads(src).unwrap_or(64)));
        Ok(())
    }

    pub fn new_buffer(&self, bytes: usize) -> Buffer {
        self.device.new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared)
    }
    pub fn buffer_u32(&self, data: &[u32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void, (data.len() * 4) as u64, MTLResourceOptions::StorageModeShared)
    }
    pub fn buffer_f32(&self, data: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void, (data.len() * 4) as u64, MTLResourceOptions::StorageModeShared)
    }
    pub fn buffer_bytes(&self, ptr: *const u8, len: usize) -> Buffer {
        self.device.new_buffer_with_data(ptr as *const c_void, len as u64, MTLResourceOptions::StorageModeShared)
    }

    pub fn batch(&self) -> CommandBatch<'_> {
        CommandBatch { runner: self, cmd: self.queue.new_command_buffer().to_owned(), encoder: None }
    }
}

fn stem(path: &str) -> String {
    std::path::Path::new(path).file_stem().unwrap().to_string_lossy().into_owned()
}

/// One command buffer, one reusable compute encoder (split around blits).
pub struct CommandBatch<'a> {
    runner: &'a MetalRunner,
    cmd: CommandBuffer,
    encoder: Option<ComputeCommandEncoder>,
}

impl<'a> CommandBatch<'a> {
    fn enc(&mut self) -> &ComputeCommandEncoderRef {
        if self.encoder.is_none() {
            self.encoder = Some(self.cmd.new_compute_command_encoder().to_owned());
        }
        self.encoder.as_ref().unwrap()
    }

    fn encode(&mut self, name: &str, bufs: &[(&BufferRef, u64)], grid: MTLSize) {
        let (pso, tpg) = self.runner.kernels.get(name)
            .unwrap_or_else(|| panic!("kernel '{name}' not compiled"));
        let (pso, tpg) = (pso.clone(), *tpg);
        let enc = self.enc();
        enc.set_compute_pipeline_state(&pso);
        for (i, (buf, off)) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(buf), *off);
        }
        enc.dispatch_thread_groups(grid, MTLSize { width: tpg, height: 1, depth: 1 });
    }

    /// Explicit threadgroup count (one workgroup per row, etc.).
    pub fn dg(&mut self, name: &str, bufs: &[(&BufferRef, u64)], groups: u64) {
        self.encode(name, bufs, MTLSize { width: groups, height: 1, depth: 1 });
    }
    /// 2-D threadgroup grid (attention: heads x chunks).
    pub fn dg2d(&mut self, name: &str, bufs: &[(&BufferRef, u64)], width: u64, height: u64) {
        self.encode(name, bufs, MTLSize { width, height, depth: 1 });
    }

    pub fn blit_copy(&mut self, src: &BufferRef, src_off: u64, dst: &BufferRef, dst_off: u64, len: u64) {
        if let Some(e) = self.encoder.take() {
            e.end_encoding();
        }
        let blit = self.cmd.new_blit_command_encoder();
        blit.copy_from_buffer(src, src_off, dst, dst_off, len);
        blit.end_encoding();
    }

    pub fn commit_and_wait(mut self) {
        if let Some(e) = self.encoder.take() {
            e.end_encoding();
        }
        self.cmd.commit();
        self.cmd.wait_until_completed();
    }

    /// Commit without waiting (resident decode); returns the command buffer to wait on.
    pub fn commit(mut self) -> CommandBuffer {
        if let Some(e) = self.encoder.take() {
            e.end_encoding();
        }
        self.cmd.commit();
        self.cmd
    }
}
