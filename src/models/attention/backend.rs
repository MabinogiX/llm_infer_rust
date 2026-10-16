//! Attention backend dispatch matching mini-sglang's `pt` / `fa` seam.

#[cfg(has_flashinfer)]
use std::{
    cell::RefCell,
    collections::HashMap,
    rc::{Rc, Weak},
};
use tch::{Device, Kind, Tensor};

use crate::engine::{
    AttentionMetadata, BatchPhase, DecodeGraphState, MetadataRequirements, ModelRunnerError,
};

use super::BaseAttention;

type Result<T> = std::result::Result<T, ModelRunnerError>;

/// Backend identifiers are private to the attention module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttentionBackendKind {
    /// Eager libtorch operations, corresponding to Python's PyTorch/SDPA path.
    Pt,
    /// LibTorch SDPA, which selects the fused FlashAttention kernel on supported CUDA inputs.
    FlashAttention,
    /// FlashInfer paged decode and paged/ragged prefill.
    FlashInfer,
}

impl AttentionBackendKind {
    pub fn parse(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "pt" | "pytorch" => Ok(Self::Pt),
            "fa" | "flashattention" | "flash-attention" => Ok(Self::FlashAttention),
            "flashinfer" => Ok(Self::FlashInfer),
            _ => Err(model_error(&format!(
                "unknown attention backend {name:?}; expected \"pt\", \"fa\", or \"flashinfer\""
            ))),
        }
    }
}

/// Model geometry needed by every attention implementation.
#[derive(Debug, Clone, Copy)]
pub struct AttentionSpec {
    pub num_heads: i64,
    pub num_kv_heads: i64,
    pub head_dim: i64,
    pub kind: Kind,
    pub device: Device,
}

/// One selected implementation for all layers of a model.
pub struct Attention {
    backend: Box<dyn AttentionBackend>,
    spec: AttentionSpec,
    page_size: Option<i64>,
}

/// Prepared once per model forward and reused by every decoder layer.
pub struct AttentionBatch<'a> {
    backend: &'a dyn AttentionBackend,
    spec: AttentionSpec,
    metadata: Option<&'a AttentionMetadata>,
    prefill: PrefillMetadata,
    plan: BackendPlan,
}

/// Host values are read once per invocation, never in layer forward.
struct PrefillMetadata {
    boundaries: Vec<i64>,
    prefixes: Vec<i64>,
}
impl PrefillMetadata {
    fn prepare(metadata: Option<&AttentionMetadata>, total_tokens: usize) -> Result<Self> {
        let boundaries = batch_boundaries(metadata, total_tokens)?;
        let prefixes =
            if let Some(meta) = metadata.filter(|meta| meta.forward_mode == BatchPhase::Prefill) {
                tensor_i32(meta.prefix_lens.as_ref(), "prefix_lens")?
                    .into_iter()
                    .map(i64::from)
                    .collect()
            } else {
                vec![0; boundaries.len() - 1]
            };
        if prefixes.len() + 1 != boundaries.len() || prefixes.iter().any(|&prefix| prefix < 0) {
            return Err(model_error("invalid prefill prefix lengths"));
        }
        Ok(Self {
            boundaries,
            prefixes,
        })
    }
    fn has_prefix(&self) -> bool {
        self.prefixes.iter().any(|&prefix| prefix > 0)
    }
}
struct CacheRow {
    indices: Tensor,
    max_location: i64,
    blocked: Tensor,
}

impl Attention {
    pub fn new(name: &str, spec: AttentionSpec) -> Result<Self> {
        Self::validate(name, spec)?;
        let kind = AttentionBackendKind::parse(name)?;
        Ok(Self {
            backend: create_attention_backend(kind),
            spec,
            page_size: None,
        })
    }

    /// Validate a backend combination without allocating model tensors.
    pub fn validate(name: &str, spec: AttentionSpec) -> Result<()> {
        let kind = AttentionBackendKind::parse(name)?;
        if kind == AttentionBackendKind::FlashInfer {
            if !cfg!(has_flashinfer) {
                return Err(model_error(
                    "FlashInfer is unavailable; install its CUDA headers in the build venv and rebuild on Linux CUDA",
                ));
            }
            if !matches!(spec.device, Device::Cuda(_))
                || !matches!(spec.kind, Kind::BFloat16 | Kind::Half)
            {
                return Err(model_error(
                    "FlashInfer requires CUDA and bfloat16 or float16",
                ));
            }
            if spec.head_dim != 128 {
                return Err(model_error(
                    "native FlashInfer decode currently requires head_dim=128",
                ));
            }
        }
        Ok(())
    }

    pub fn bind_cache_layout(&mut self, page_size: i64) {
        self.page_size = Some(page_size);
    }

    pub fn supports_cuda_graph(&self) -> bool {
        self.backend.supports_cuda_graph()
    }

    pub fn metadata_requirements(&self, phase: BatchPhase) -> MetadataRequirements {
        self.backend.metadata_requirements(phase)
    }

    pub fn prepare_decode_graph(
        &self,
        metadata: &AttentionMetadata,
    ) -> Result<Option<Box<dyn DecodeGraphState>>> {
        self.backend.prepare_decode_graph(
            metadata,
            self.spec,
            self.page_size
                .ok_or_else(|| model_error("graph requires a bound KV cache"))?,
        )
    }

    pub fn prepare<'a>(
        &'a self,
        metadata: Option<&'a AttentionMetadata>,
        total_tokens: usize,
    ) -> Result<AttentionBatch<'a>> {
        let prefill = PrefillMetadata::prepare(metadata, total_tokens)?;
        let plan = self
            .backend
            .prepare(metadata, &prefill, self.spec, self.page_size)?;
        Ok(AttentionBatch {
            backend: self.backend.as_ref(),
            spec: self.spec,
            metadata,
            prefill,
            plan,
        })
    }
}

impl AttentionBatch<'_> {
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cache: &BaseAttention,
    ) -> Result<Tensor> {
        self.backend.forward(
            q,
            k,
            v,
            cache,
            &self.prefill,
            self.metadata,
            self.spec.num_heads,
            self.spec.num_kv_heads,
            self.spec.head_dim,
            &self.plan,
        )
    }

    /// Write live tokens into a caller-owned static output, preserving its address.
    pub fn forward_into(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cache: &BaseAttention,
        output: &Tensor,
    ) -> Result<()> {
        if q.dim() != 3
            || output.size() != [q.size()[0], self.spec.num_heads * self.spec.head_dim]
            || output.kind() != self.spec.kind
            || output.device() != self.spec.device
            || !output.is_contiguous()
        {
            return Err(model_error("invalid static attention output"));
        }
        #[cfg(has_flashinfer)]
        if matches!(self.plan, BackendPlan::FlashInferPrefill(_, _)) {
            let _ =
                flashinfer_prefill(q, k, v, cache, &self.plan, self.spec.head_dim, Some(output))?;
            return Ok(());
        }
        if let BackendPlan::SdpaPrefill(prepared) = &self.plan {
            let result = prepared.forward(
                q,
                k,
                v,
                cache,
                self.spec.num_heads,
                self.spec.num_kv_heads,
                self.spec.head_dim,
                Some(output),
            )?;
            if result.data_ptr() != output.data_ptr() {
                output.shallow_clone().copy_(&result);
            }
            return Ok(());
        }
        // Backends without an out kernel still honor the fixed-address contract.
        output.shallow_clone().copy_(&self.forward(q, k, v, cache)?);
        Ok(())
    }
}

enum BackendPlan {
    None,
    CachedPrefill(Vec<CacheRow>),
    PtPrefill(Vec<Tensor>),
    SdpaPrefill(SdpaPrefillPlan),
    #[cfg(has_flashinfer)]
    FlashInfer(Rc<FlashInferPlan>),
    #[cfg(has_flashinfer)]
    FlashInferPrefill(Rc<FlashInferPlan>, bool),
}

/// Internal seam for the three existing attention implementations.
trait AttentionBackend {
    fn metadata_requirements(&self, _phase: BatchPhase) -> MetadataRequirements {
        MetadataRequirements {
            block_table: false,
            req_to_token: true,
        }
    }
    fn supports_cuda_graph(&self) -> bool {
        false
    }

    fn prepare_decode_graph(
        &self,
        _metadata: &AttentionMetadata,
        _spec: AttentionSpec,
        _page_size: i64,
    ) -> Result<Option<Box<dyn DecodeGraphState>>> {
        Ok(None)
    }

    fn prepare(
        &self,
        _metadata: Option<&AttentionMetadata>,
        _prefill: &PrefillMetadata,
        _spec: AttentionSpec,
        _page_size: Option<i64>,
    ) -> Result<BackendPlan> {
        Ok(BackendPlan::None)
    }

    /// Computes attention after model-specific QKV projection, RoPE, and KV write.
    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cache: &BaseAttention,
        prefill: &PrefillMetadata,
        metadata: Option<&AttentionMetadata>,
        num_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
        plan: &BackendPlan,
    ) -> Result<Tensor>;
}

fn create_attention_backend(kind: AttentionBackendKind) -> Box<dyn AttentionBackend> {
    match kind {
        AttentionBackendKind::Pt => Box::new(PyTorchAttentionBackend),
        AttentionBackendKind::FlashAttention => Box::new(FlashAttentionBackend),
        AttentionBackendKind::FlashInfer => Box::new(FlashInferAttentionBackend::default()),
    }
}

#[derive(Default)]
struct FlashInferAttentionBackend {
    #[cfg(has_flashinfer)]
    plan: RefCell<Option<Rc<FlashInferPlan>>>,
    #[cfg(has_flashinfer)]
    prefill_plan: RefCell<Option<Rc<FlashInferPlan>>>,
    #[cfg(has_flashinfer)]
    graph_plans: RefCell<HashMap<usize, Weak<FlashInferPlan>>>,
}

impl AttentionBackend for FlashInferAttentionBackend {
    fn metadata_requirements(&self, _phase: BatchPhase) -> MetadataRequirements {
        MetadataRequirements {
            block_table: true,
            req_to_token: false,
        }
    }
    fn supports_cuda_graph(&self) -> bool {
        cfg!(has_flashinfer)
    }

    fn prepare_decode_graph(
        &self,
        metadata: &AttentionMetadata,
        spec: AttentionSpec,
        page_size: i64,
    ) -> Result<Option<Box<dyn DecodeGraphState>>> {
        #[cfg(has_flashinfer)]
        {
            let table = metadata
                .block_table
                .as_ref()
                .ok_or_else(|| model_error("missing graph block_table"))?;
            let lengths = metadata
                .cache_seqlens
                .as_ref()
                .ok_or_else(|| model_error("missing graph cache_seqlens"))?;
            let plan = Rc::new(FlashInferPlan(prepare_flashinfer_decode(
                metadata,
                spec,
                page_size,
                std::ptr::null_mut(),
                true,
            )?));
            let mut plans = self.graph_plans.borrow_mut();
            plans.retain(|_, plan| plan.strong_count() > 0);
            let key = table.data_ptr() as usize;
            if plans.get(&key).and_then(Weak::upgrade).is_some() {
                return Err(model_error("decode graph buffers are already registered"));
            }
            plans.insert(key, Rc::downgrade(&plan));
            return Ok(Some(Box::new(FlashInferGraphState {
                plan,
                table: table.shallow_clone(),
                lengths: lengths.shallow_clone(),
                spec,
                page_size,
            })));
        }
        #[cfg(not(has_flashinfer))]
        {
            let _ = (metadata, spec, page_size);
            Ok(None)
        }
    }

    fn prepare(
        &self,
        metadata: Option<&AttentionMetadata>,
        prefill: &PrefillMetadata,
        spec: AttentionSpec,
        page_size: Option<i64>,
    ) -> Result<BackendPlan> {
        if let Some(metadata) = metadata.filter(|meta| meta.forward_mode == BatchPhase::Decode) {
            #[cfg(has_flashinfer)]
            if let Some(plan) = metadata.block_table.as_ref().and_then(|table| {
                self.graph_plans
                    .borrow()
                    .get(&(table.data_ptr() as usize))
                    .and_then(Weak::upgrade)
            }) {
                // Already planned outside capture/replay. No GPU read or allocation.
                return Ok(BackendPlan::FlashInfer(plan));
            }
            let page_size =
                page_size.ok_or_else(|| model_error("FlashInfer requires a bound KV cache"))?;
            #[cfg(has_flashinfer)]
            {
                let mut cached = self.plan.borrow_mut();
                if cached
                    .as_ref()
                    .is_some_and(|plan| Rc::strong_count(plan) != 1)
                {
                    return Err(model_error("previous FlashInfer batch is still in use"));
                }
                let prepared = prepare_flashinfer_decode(
                    metadata,
                    spec,
                    page_size,
                    cached.as_ref().map_or(std::ptr::null_mut(), |plan| plan.0),
                    false,
                )?;
                let plan = cached.get_or_insert_with(|| Rc::new(FlashInferPlan(prepared)));
                return Ok(BackendPlan::FlashInfer(Rc::clone(plan)));
            }
            #[cfg(not(has_flashinfer))]
            return prepare_flashinfer_decode(metadata, spec, page_size);
        }
        #[cfg(has_flashinfer)]
        if let Some(metadata) = metadata {
            let mut cached = self.prefill_plan.borrow_mut();
            if cached
                .as_ref()
                .is_some_and(|plan| Rc::strong_count(plan) != 1)
            {
                return Err(model_error(
                    "previous FlashInfer prefill batch is still in use",
                ));
            }
            let prepared = prepare_flashinfer_prefill(
                metadata,
                prefill,
                spec,
                page_size.ok_or_else(|| model_error("FlashInfer requires a bound KV cache"))?,
                cached.as_ref().map_or(std::ptr::null_mut(), |plan| plan.0),
            )?;
            let plan = cached.get_or_insert_with(|| Rc::new(FlashInferPlan(prepared)));
            return Ok(BackendPlan::FlashInferPrefill(
                Rc::clone(plan),
                prefill.has_prefix(),
            ));
        }
        FlashAttentionBackend.prepare(metadata, prefill, spec, page_size)
    }

    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cache: &BaseAttention,
        prefill: &PrefillMetadata,
        metadata: Option<&AttentionMetadata>,
        num_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
        plan: &BackendPlan,
    ) -> Result<Tensor> {
        if metadata.is_some_and(|meta| meta.forward_mode == BatchPhase::Decode) {
            return flashinfer_decode(q, cache, plan, head_dim);
        }
        #[cfg(has_flashinfer)]
        if metadata.is_some() {
            return flashinfer_prefill(q, k, v, cache, plan, head_dim, None);
        }
        FlashAttentionBackend.forward(
            q,
            k,
            v,
            cache,
            prefill,
            metadata,
            num_heads,
            num_kv_heads,
            head_dim,
            plan,
        )
    }
}

#[cfg(has_flashinfer)]
struct FlashInferPlan(*mut std::ffi::c_void);

#[cfg(has_flashinfer)]
struct FlashInferGraphState {
    plan: Rc<FlashInferPlan>,
    table: Tensor,
    lengths: Tensor,
    spec: AttentionSpec,
    page_size: i64,
}

#[cfg(has_flashinfer)]
impl DecodeGraphState for FlashInferGraphState {
    fn prepare_replay(&self, host: Option<&crate::engine::DecodeHostMetadata>) -> Result<()> {
        let host = host.map(|host| {
            let mut cache_seqlens = host.cache_seqlens.clone();
            cache_seqlens.resize(self.table.size()[0] as usize, 1);
            crate::engine::DecodeHostMetadata {
                cache_seqlens,
                block_table_versions: None,
            }
        });
        let metadata = AttentionMetadata {
            forward_mode: BatchPhase::Decode,
            decode_host: host,
            write_loc: None,
            cu_seqlens_q: None,
            prefix_lens: None,
            block_table: Some(self.table.shallow_clone()),
            cache_seqlens: Some(self.lengths.shallow_clone()),
            req_to_token: None,
            max_seqlen: None,
        };
        let pointer =
            prepare_flashinfer_decode(&metadata, self.spec, self.page_size, self.plan.0, true)?;
        if pointer != self.plan.0 {
            return Err(model_error("graph plan address changed"));
        }
        Ok(())
    }
    fn needs_req_to_token(&self) -> bool {
        false
    }
}

#[cfg(has_flashinfer)]
impl Drop for FlashInferPlan {
    fn drop(&mut self) {
        unsafe extern "C" {
            fn sglang_flashinfer_plan_drop(plan: *mut std::ffi::c_void);
        }
        unsafe { sglang_flashinfer_plan_drop(self.0) };
    }
}

#[cfg(has_flashinfer)]
fn prepare_flashinfer_decode(
    metadata: &AttentionMetadata,
    spec: AttentionSpec,
    page_size: i64,
    existing: *mut std::ffi::c_void,
    graph: bool,
) -> Result<*mut std::ffi::c_void> {
    use std::ffi::{CStr, c_void};

    unsafe extern "C" {
        fn sglang_flashinfer_prepare(
            existing: *mut c_void,
            block_table: *const c_void,
            sequence_lengths: *const c_void,
            num_q_heads: i64,
            num_kv_heads: i64,
            head_dim: i64,
            page_size: i64,
            dtype_code: i64,
            graph: bool,
            host_lengths: *const i32,
            host_rows: i64,
        ) -> *mut c_void;
        fn sglang_flashinfer_error() -> *const std::ffi::c_char;
    }
    let table = metadata
        .block_table
        .as_ref()
        .ok_or_else(|| model_error("FlashInfer decode requires block_table"))?;
    let lengths = metadata
        .cache_seqlens
        .as_ref()
        .ok_or_else(|| model_error("FlashInfer decode requires cache_seqlens"))?;
    let dtype_code = match spec.kind {
        Kind::BFloat16 => 0,
        Kind::Half => 1,
        _ => return Err(model_error("FlashInfer requires bfloat16 or float16")),
    };
    let prepared = unsafe {
        sglang_flashinfer_prepare(
            existing,
            table.as_ptr().cast(),
            lengths.as_ptr().cast(),
            spec.num_heads,
            spec.num_kv_heads,
            spec.head_dim,
            page_size,
            dtype_code,
            graph,
            metadata
                .decode_host
                .as_ref()
                .map_or(std::ptr::null(), |host| host.cache_seqlens.as_ptr()),
            metadata
                .decode_host
                .as_ref()
                .map_or(0, |host| host.cache_seqlens.len() as i64),
        )
    };
    if prepared.is_null() {
        let error = unsafe { CStr::from_ptr(sglang_flashinfer_error()) };
        return Err(model_error(&format!(
            "FlashInfer plan failed: {}",
            error.to_string_lossy()
        )));
    }
    Ok(prepared)
}

#[cfg(not(has_flashinfer))]
fn prepare_flashinfer_decode(
    _metadata: &AttentionMetadata,
    _spec: AttentionSpec,
    _page_size: i64,
) -> Result<BackendPlan> {
    Err(model_error("FlashInfer is unavailable"))
}

#[cfg(has_flashinfer)]
fn flashinfer_decode(
    q: &Tensor,
    cache: &BaseAttention,
    plan: &BackendPlan,
    head_dim: i64,
) -> Result<Tensor> {
    use std::{
        ffi::{CStr, c_void},
        ptr,
    };

    unsafe extern "C" {
        fn sglang_flashinfer_decode(
            prepared: *const c_void,
            query: *const c_void,
            key_cache: *const c_void,
            value_cache: *const c_void,
        ) -> *mut c_void;
        fn sglang_flashinfer_error() -> *const std::ffi::c_char;
    }
    if !matches!(q.device(), Device::Cuda(_)) {
        return Err(model_error("FlashInfer decode requires a CUDA device"));
    }
    let BackendPlan::FlashInfer(plan) = plan else {
        return Err(model_error("FlashInfer decode requires a prepared batch"));
    };
    let (k_cache, v_cache) = cache.cache_tensors()?;
    // Q is a strided view of packed QKV for multi-request batches. The native
    // decode kernel uses packed query strides; batch-size one needs no copy.
    let q = q.contiguous();
    let output = unsafe {
        sglang_flashinfer_decode(
            plan.0,
            q.as_ptr().cast(),
            k_cache.as_ptr().cast(),
            v_cache.as_ptr().cast(),
        )
    };
    if output == ptr::null_mut() {
        let error = unsafe { CStr::from_ptr(sglang_flashinfer_error()) };
        return Err(model_error(&format!(
            "FlashInfer decode failed: {}",
            error.to_string_lossy()
        )));
    }
    let output = unsafe { Tensor::from_ptr(output.cast()) };
    Ok(output.reshape([q.size()[0], q.size()[1] * head_dim]))
}

#[cfg(not(has_flashinfer))]
fn flashinfer_decode(
    _q: &Tensor,
    _cache: &BaseAttention,
    _plan: &BackendPlan,
    _head_dim: i64,
) -> Result<Tensor> {
    Err(model_error(
        "FlashInfer is unavailable; install its CUDA headers in the build venv and rebuild on Linux CUDA",
    ))
}

#[cfg(has_flashinfer)]
fn prepare_flashinfer_prefill(
    metadata: &AttentionMetadata,
    prefill: &PrefillMetadata,
    spec: AttentionSpec,
    page_size: i64,
    existing: *mut std::ffi::c_void,
) -> Result<*mut std::ffi::c_void> {
    use std::ffi::{CStr, c_void};
    unsafe extern "C" {
        fn sglang_flashinfer_prepare_prefill(
            existing: *mut c_void,
            q_indptr: *const c_void,
            prefixes: *const c_void,
            table: *const c_void,
            boundaries_host: *mut i32,
            prefixes_host: *mut i32,
            q_heads: i64,
            kv_heads: i64,
            page_size: i64,
            dtype: i64,
        ) -> *mut c_void;
        fn sglang_flashinfer_error() -> *const std::ffi::c_char;
    }
    let cumulative = metadata
        .cu_seqlens_q
        .as_ref()
        .ok_or_else(|| model_error("missing cu_seqlens_q"))?;
    let prefixes = metadata
        .prefix_lens
        .as_ref()
        .ok_or_else(|| model_error("missing prefix_lens"))?;
    let table = metadata
        .block_table
        .as_ref()
        .ok_or_else(|| model_error("missing block_table"))?;
    // Reuse the host snapshot already read at Attention::prepare, including its
    // synchronization of previous work before the native plan reuses workspaces.
    let mut boundaries_host: Vec<i32> = prefill
        .boundaries
        .iter()
        .map(|&value| value as i32)
        .collect();
    let mut prefixes_host: Vec<i32> = prefill.prefixes.iter().map(|&value| value as i32).collect();
    let prepared = unsafe {
        sglang_flashinfer_prepare_prefill(
            existing,
            cumulative.as_ptr().cast(),
            prefixes.as_ptr().cast(),
            table.as_ptr().cast(),
            boundaries_host.as_mut_ptr(),
            prefixes_host.as_mut_ptr(),
            spec.num_heads,
            spec.num_kv_heads,
            page_size,
            if spec.kind == Kind::BFloat16 { 0 } else { 1 },
        )
    };
    if prepared.is_null() {
        let error = unsafe { CStr::from_ptr(sglang_flashinfer_error()) };
        return Err(model_error(&format!(
            "FlashInfer prefill plan failed: {}",
            error.to_string_lossy()
        )));
    }
    Ok(prepared)
}

#[cfg(has_flashinfer)]
fn flashinfer_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cache: &BaseAttention,
    plan: &BackendPlan,
    head_dim: i64,
    output: Option<&Tensor>,
) -> Result<Tensor> {
    use std::ffi::{CStr, c_void};
    unsafe extern "C" {
        fn sglang_flashinfer_prefill(
            plan: *const c_void,
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            output: *const c_void,
        ) -> *mut c_void;
        fn sglang_flashinfer_error() -> *const std::ffi::c_char;
    }
    let BackendPlan::FlashInferPrefill(plan, paged) = plan else {
        return Err(model_error("missing FlashInfer prefill plan"));
    };
    // Prefix mode is selected once at preparation, not separately in each layer.
    // The native bridge chooses whether these tensors are ragged K/V or paged cache.
    let cached;
    let (k, v) = if *paged {
        cached = cache.cache_tensors()?;
        (cached.0, cached.1)
    } else {
        (k, v)
    };
    let output = output.map(|out| out.view(q.size().as_slice()));
    let out = unsafe {
        sglang_flashinfer_prefill(
            plan.0,
            q.as_ptr().cast(),
            k.as_ptr().cast(),
            v.as_ptr().cast(),
            output
                .as_ref()
                .map_or(std::ptr::null(), |out| out.as_ptr().cast()),
        )
    };
    if out.is_null() {
        let error = unsafe { CStr::from_ptr(sglang_flashinfer_error()) };
        return Err(model_error(&format!(
            "FlashInfer prefill failed: {}",
            error.to_string_lossy()
        )));
    }
    Ok(unsafe { Tensor::from_ptr(out.cast()) }.reshape([q.size()[0], q.size()[1] * head_dim]))
}

/// Eager libtorch implementation of mini-sglang's Python `PyTorchBackend`.
struct PyTorchAttentionBackend;

impl AttentionBackend for PyTorchAttentionBackend {
    fn prepare(
        &self,
        metadata: Option<&AttentionMetadata>,
        prefill: &PrefillMetadata,
        spec: AttentionSpec,
        _page_size: Option<i64>,
    ) -> Result<BackendPlan> {
        if metadata.is_some_and(|meta| meta.forward_mode == BatchPhase::Decode) {
            return Ok(BackendPlan::None);
        }
        if !prefill.has_prefix() {
            return Ok(BackendPlan::PtPrefill(
                prefill
                    .boundaries
                    .windows(2)
                    .map(|bounds| {
                        blocked_attention_mask(
                            bounds[1] - bounds[0],
                            bounds[1] - bounds[0],
                            0,
                            spec.device,
                        )
                    })
                    .collect(),
            ));
        }
        let table = metadata
            .and_then(|meta| meta.req_to_token.as_ref())
            .ok_or_else(|| model_error("cached prefill requires req_to_token"))?;
        if table.dim() != 2 || table.size()[0] != prefill.prefixes.len() as i64 {
            return Err(model_error("invalid cached prefill table shape"));
        }
        let mut rows = Vec::with_capacity(prefill.prefixes.len());
        for (index, (bounds, &prefix)) in prefill
            .boundaries
            .windows(2)
            .zip(&prefill.prefixes)
            .enumerate()
        {
            let length = prefix + bounds[1] - bounds[0];
            if length > table.size()[1] {
                return Err(model_error("invalid cache sequence length"));
            }
            let locations = tensor_i32(
                Some(&table.get(index as i64).narrow(0, 0, length)),
                "req_to_token",
            )?;
            if locations.iter().any(|&loc| loc < 0) {
                return Err(model_error(
                    "req_to_token contains an invalid cache location",
                ));
            }
            let max_location = i64::from(*locations.iter().max().unwrap());
            let indices = Tensor::from_slice(&locations)
                .to_device(spec.device)
                .to_kind(Kind::Int64);
            rows.push(CacheRow {
                indices,
                max_location,
                blocked: blocked_attention_mask(bounds[1] - bounds[0], length, prefix, spec.device),
            });
        }
        Ok(BackendPlan::CachedPrefill(rows))
    }

    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cache: &BaseAttention,
        prefill: &PrefillMetadata,
        metadata: Option<&AttentionMetadata>,
        num_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
        _plan: &BackendPlan,
    ) -> Result<Tensor> {
        if let Some(metadata) = metadata {
            if metadata.forward_mode == BatchPhase::Decode {
                return decode_with_cache(q, cache, metadata, num_heads, num_kv_heads, head_dim);
            }
            if prefill.has_prefix() {
                return prefill_with_cache(
                    q,
                    cache,
                    prefill,
                    _plan,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );
            }
        }

        let BackendPlan::PtPrefill(masks) = _plan else {
            return Err(model_error("missing PyTorch prefill plan"));
        };
        let mut outputs = Vec::with_capacity(prefill.boundaries.len().saturating_sub(1));
        for (boundaries, blocked) in prefill.boundaries.windows(2).zip(masks) {
            let start = boundaries[0];
            let length = boundaries[1] - start;
            outputs.push(attention_against_cache(
                &q.narrow(0, start, length),
                &k.narrow(0, start, length),
                &v.narrow(0, start, length),
                0,
                num_heads,
                num_kv_heads,
                head_dim,
                Some(blocked),
            ));
        }
        Ok(Tensor::cat(&outputs, 0))
    }
}

/// Uses LibTorch SDPA so supported CUDA BF16/FP16 inputs take its FlashAttention path.
struct FlashAttentionBackend;

impl AttentionBackend for FlashAttentionBackend {
    fn prepare(
        &self,
        metadata: Option<&AttentionMetadata>,
        prefill: &PrefillMetadata,
        spec: AttentionSpec,
        _page_size: Option<i64>,
    ) -> Result<BackendPlan> {
        if metadata.is_some_and(|meta| meta.forward_mode == BatchPhase::Decode) {
            return Ok(BackendPlan::None);
        }
        Ok(BackendPlan::SdpaPrefill(SdpaPrefillPlan::prepare(
            prefill,
            metadata,
            spec.device,
        )?))
    }

    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cache: &BaseAttention,
        _prefill: &PrefillMetadata,
        metadata: Option<&AttentionMetadata>,
        num_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
        _plan: &BackendPlan,
    ) -> Result<Tensor> {
        if let Some(metadata) = metadata {
            if metadata.forward_mode == BatchPhase::Decode {
                return sdpa_decode_with_cache(
                    q,
                    cache,
                    metadata,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );
            }
        }
        let BackendPlan::SdpaPrefill(prepared) = _plan else {
            return Err(model_error("missing SDPA prefill plan"));
        };
        prepared.forward(q, k, v, cache, num_heads, num_kv_heads, head_dim, None)
    }
}

/// Padded SDPA layout and masks are independent of layer K/V values.
struct SdpaPrefillPlan {
    batch_size: i64,
    max_query: i64,
    indices: Option<Tensor>,
    output_indices: Option<Tensor>,
    cache_indices: Option<Tensor>,
    mask: Option<Tensor>,
}
impl SdpaPrefillPlan {
    fn prepare(
        prefill: &PrefillMetadata,
        metadata: Option<&AttentionMetadata>,
        device: Device,
    ) -> Result<Self> {
        let boundaries = &prefill.boundaries;
        let prefixes = &prefill.prefixes;
        let batch_size = prefixes.len();
        if batch_size == 0 {
            return Err(model_error("empty prefill batch"));
        }
        let lengths: Vec<_> = boundaries
            .windows(2)
            .map(|bounds| bounds[1] - bounds[0])
            .collect();
        if lengths.iter().any(|&length| length <= 0) {
            return Err(model_error("prefill requests must contain uncached tokens"));
        }
        if batch_size == 1 && prefixes[0] == 0 {
            return Ok(Self {
                batch_size: 1,
                max_query: lengths[0],
                indices: None,
                output_indices: None,
                cache_indices: None,
                mask: None,
            });
        }
        let max_query = *lengths.iter().max().unwrap();
        let max_key = lengths
            .iter()
            .zip(prefixes)
            .map(|(&len, &prefix)| len + prefix)
            .max()
            .unwrap();
        let mut padded_indices = Vec::with_capacity(batch_size * max_query as usize);
        let mut output_indices = Vec::with_capacity(*boundaries.last().unwrap() as usize);
        for (row, bounds) in boundaries.windows(2).enumerate() {
            for position in 0..max_query {
                padded_indices.push(bounds[0] + position.min(lengths[row] - 1));
            }
            for position in 0..lengths[row] {
                output_indices.push(row as i64 * max_query + position);
            }
        }
        let indices = Some(Tensor::from_slice(&padded_indices).to_device(device));
        let output_indices = Some(Tensor::from_slice(&output_indices).to_device(device));
        let (cache_indices, mask) = if prefill.has_prefix() {
            let table = metadata
                .and_then(|meta| meta.req_to_token.as_ref())
                .ok_or_else(|| model_error("cached prefill requires req_to_token"))?;
            if table.dim() != 2 || table.size()[0] != batch_size as i64 || table.size()[1] < max_key
            {
                return Err(model_error("cached prefill page table is too small"));
            }
            let cache_indices = table
                .narrow(1, 0, max_key)
                .clamp_min(0)
                .to_kind(Kind::Int64);
            let query_positions = Tensor::arange(max_query, (Kind::Int64, device)).unsqueeze(0)
                + Tensor::from_slice(prefixes).to_device(device).unsqueeze(1);
            let key_positions = Tensor::arange(max_key, (Kind::Int64, device));
            let key_lengths: Vec<_> = lengths
                .iter()
                .zip(prefixes)
                .map(|(&len, &prefix)| len + prefix)
                .collect();
            let mask = key_positions
                .unsqueeze(0)
                .unsqueeze(0)
                .le_tensor(&query_positions.unsqueeze(2))
                .logical_and(
                    &key_positions
                        .unsqueeze(0)
                        .lt_tensor(
                            &Tensor::from_slice(&key_lengths)
                                .to_device(device)
                                .unsqueeze(1),
                        )
                        .unsqueeze(1),
                )
                .unsqueeze(1);
            (Some(cache_indices), Some(mask))
        } else {
            (None, None)
        };
        Ok(Self {
            batch_size: batch_size as i64,
            max_query,
            indices,
            output_indices,
            cache_indices,
            mask,
        })
    }
    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cache: &BaseAttention,
        num_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
        target: Option<&Tensor>,
    ) -> Result<Tensor> {
        let Some(indices) = &self.indices else {
            return sdpa_attention(q, k, v, 0, num_heads, num_kv_heads, head_dim);
        };
        let batch_size = self.batch_size;
        let max_query = self.max_query;
        let query = q
            .index_select(0, indices)
            .view([batch_size, max_query, num_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let (keys, values) = if let Some(locations) = &self.cache_indices {
            cache.read_kv_padded(locations)?
        } else {
            (
                k.index_select(0, indices)
                    .view([batch_size, max_query, num_kv_heads, head_dim]),
                v.index_select(0, indices)
                    .view([batch_size, max_query, num_kv_heads, head_dim]),
            )
        };
        let repeats = num_heads / num_kv_heads;
        let keys = keys
            .repeat_interleave_self_int(repeats, 2, Some(num_heads))
            .permute([0, 2, 1, 3]);
        let values = values
            .repeat_interleave_self_int(repeats, 2, Some(num_heads))
            .permute([0, 2, 1, 3]);
        let output = Tensor::f_scaled_dot_product_attention(
            &query,
            &keys,
            &values,
            self.mask.as_ref(),
            0.0,
            self.mask.is_none(),
            None,
            false,
        )?;
        let output = output
            .permute([0, 2, 1, 3])
            .reshape([batch_size * max_query, num_heads * head_dim]);
        match target {
            Some(target) => {
                Ok(output.f_index_select_out(target, 0, self.output_indices.as_ref().unwrap())?)
            }
            None => Ok(output.index_select(0, self.output_indices.as_ref().unwrap())),
        }
    }
}

#[cfg(test)]
fn batched_prefill_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cache: &BaseAttention,
    boundaries: &[i64],
    prefixes: &[i64],
    metadata: Option<&AttentionMetadata>,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Result<Tensor> {
    if prefixes.is_empty()
        || boundaries.len() != prefixes.len() + 1
        || boundaries[0] != 0
        || *boundaries.last().unwrap() != q.size()[0]
        || prefixes.iter().any(|&prefix| prefix < 0)
    {
        return Err(model_error(
            "invalid prefill batch boundaries or prefix lengths",
        ));
    }
    let prefill = PrefillMetadata {
        boundaries: boundaries.to_vec(),
        prefixes: prefixes.to_vec(),
    };
    SdpaPrefillPlan::prepare(&prefill, metadata, q.device())?.forward(
        q,
        k,
        v,
        cache,
        num_heads,
        num_kv_heads,
        head_dim,
        None,
    )
}

fn sdpa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prefix: i64,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Result<Tensor> {
    let query_len = q.size()[0];
    let key_len = k.size()[0];
    let repeats = num_heads / num_kv_heads;
    let q = q.transpose(0, 1).unsqueeze(0);
    let k = k
        .transpose(0, 1)
        .repeat_interleave_self_int(repeats, 0, Some(num_heads))
        .unsqueeze(0);
    let v = v
        .transpose(0, 1)
        .repeat_interleave_self_int(repeats, 0, Some(num_heads))
        .unsqueeze(0);
    let mask = if prefix > 0 {
        let queries = Tensor::arange(query_len, (Kind::Int64, q.device())) + prefix;
        let keys = Tensor::arange(key_len, (Kind::Int64, q.device()));
        Some(
            keys.unsqueeze(0)
                .le_tensor(&queries.unsqueeze(1))
                .unsqueeze(0)
                .unsqueeze(0),
        )
    } else {
        None
    };
    let output = Tensor::f_scaled_dot_product_attention(
        &q,
        &k,
        &v,
        mask.as_ref(),
        0.0,
        prefix == 0,
        None,
        false,
    )?;
    Ok(output
        .squeeze_dim(0)
        .transpose(0, 1)
        .reshape([query_len, num_heads * head_dim]))
}

fn sdpa_decode_with_cache(
    q: &Tensor,
    cache: &BaseAttention,
    metadata: &AttentionMetadata,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Result<Tensor> {
    let table = metadata
        .req_to_token
        .as_ref()
        .ok_or_else(|| model_error("paged-KV decode requires req_to_token"))?;
    let lengths = metadata
        .cache_seqlens
        .as_ref()
        .ok_or_else(|| model_error("paged-KV decode requires cache_seqlens"))?;
    let max_len = metadata.max_seqlen.unwrap_or(table.size()[1] as usize) as i64;
    let batch_size = q.size()[0];
    if table.size()[0] != batch_size || lengths.size()[0] != batch_size {
        return Err(model_error("decode metadata row count must match queries"));
    }
    let indices = table.narrow(1, 0, max_len);
    let valid = indices.ge(0).logical_and(
        &Tensor::arange(max_len, (Kind::Int64, q.device()))
            .unsqueeze(0)
            .lt_tensor(&lengths.to_kind(Kind::Int64).unsqueeze(1)),
    );
    let (k, v) = cache.read_kv_padded(&indices.clamp_min(0).to_kind(Kind::Int64))?;
    let repeats = num_heads / num_kv_heads;
    let k = k
        .repeat_interleave_self_int(repeats, 2, Some(num_heads))
        .permute([0, 2, 1, 3]);
    let v = v
        .repeat_interleave_self_int(repeats, 2, Some(num_heads))
        .permute([0, 2, 1, 3]);
    let query = q.unsqueeze(2);
    let mask = valid.unsqueeze(1).unsqueeze(2);
    let output = Tensor::f_scaled_dot_product_attention(
        &query,
        &k,
        &v,
        Some(&mask),
        0.0,
        false,
        None,
        false,
    )?;
    Ok(output
        .squeeze_dim(2)
        .reshape([batch_size, num_heads * head_dim]))
}

fn prefill_with_cache(
    q: &Tensor,
    cache: &BaseAttention,
    prefill: &PrefillMetadata,
    plan: &BackendPlan,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Result<Tensor> {
    let BackendPlan::CachedPrefill(rows) = plan else {
        return Err(model_error("missing cached prefill plan"));
    };
    let (k_cache, v_cache) = cache.cache_tensors()?;
    let shape = k_cache.size();
    let capacity = shape[0] * shape[1];
    let mut outputs = Vec::with_capacity(rows.len());
    for ((bounds, &prefix), row) in prefill
        .boundaries
        .windows(2)
        .zip(&prefill.prefixes)
        .zip(rows)
    {
        if row.max_location >= capacity {
            return Err(model_error(
                "req_to_token points outside the bound KV cache",
            ));
        }
        let cached_k = k_cache
            .view([-1, shape[2], shape[3]])
            .index_select(0, &row.indices);
        let cached_v = v_cache
            .view([-1, shape[2], shape[3]])
            .index_select(0, &row.indices);
        outputs.push(attention_against_cache(
            &q.narrow(0, bounds[0], bounds[1] - bounds[0]),
            &cached_k,
            &cached_v,
            prefix,
            num_heads,
            num_kv_heads,
            head_dim,
            Some(&row.blocked),
        ));
    }
    Ok(Tensor::cat(&outputs, 0))
}

fn decode_with_cache(
    q: &Tensor,
    cache: &BaseAttention,
    metadata: &AttentionMetadata,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Result<Tensor> {
    let table = metadata
        .req_to_token
        .as_ref()
        .ok_or_else(|| model_error("paged-KV decode requires req_to_token"))?;
    if matches!(q.device(), Device::Cuda(_)) {
        return decode_with_cache_padded(q, cache, metadata, num_heads, num_kv_heads, head_dim);
    }
    let cache_seqlens = tensor_i32(metadata.cache_seqlens.as_ref(), "cache_seqlens")?;
    if q.size()[0] != cache_seqlens.len() as i64 || table.size()[0] != q.size()[0] {
        return Err(model_error(
            "decode K/V metadata must contain one row per query",
        ));
    }
    let mut outputs = Vec::with_capacity(cache_seqlens.len());
    for (request_index, &cache_len) in cache_seqlens.iter().enumerate() {
        if cache_len <= 0 {
            return Err(model_error("cache_seqlens must be positive"));
        }
        let (cached_k, cached_v) =
            cache.read_kv(table, request_index as i64, i64::from(cache_len))?;
        outputs.push(attention_against_cache(
            &q.narrow(0, request_index as i64, 1),
            &cached_k,
            &cached_v,
            i64::from(cache_len - 1),
            num_heads,
            num_kv_heads,
            head_dim,
            None,
        ));
    }
    Ok(Tensor::cat(&outputs, 0))
}

/// Fixed-shape decode used by CUDA Graph capture. Sequence lengths stay on
/// device; invalid columns are masked before softmax.
fn decode_with_cache_padded(
    q: &Tensor,
    cache: &BaseAttention,
    metadata: &AttentionMetadata,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Result<Tensor> {
    let table = metadata
        .req_to_token
        .as_ref()
        .ok_or_else(|| model_error("paged-KV decode requires req_to_token"))?;
    let lengths = metadata
        .cache_seqlens
        .as_ref()
        .ok_or_else(|| model_error("paged-KV decode requires cache_seqlens"))?;
    let max_len = metadata.max_seqlen.unwrap_or(table.size()[1] as usize) as i64;
    let batch_size = q.size()[0];
    if table.size()[0] != batch_size || lengths.size()[0] != batch_size {
        return Err(model_error("decode metadata row count must match queries"));
    }
    let indices = table.narrow(1, 0, max_len);
    let valid = indices.ge(0).logical_and(
        &Tensor::arange(max_len, (Kind::Int64, q.device()))
            .unsqueeze(0)
            .lt_tensor(&lengths.to_kind(Kind::Int64).unsqueeze(1)),
    );
    let (k, v) = cache.read_kv_padded(&indices.clamp_min(0).to_kind(Kind::Int64))?;
    let k = k
        .repeat_interleave_self_int(num_heads / num_kv_heads, 2, Some(num_heads))
        .permute([0, 2, 1, 3]);
    let v = v
        .repeat_interleave_self_int(num_heads / num_kv_heads, 2, Some(num_heads))
        .permute([0, 2, 1, 3]);
    let query = q.unsqueeze(2);
    let scores = (query.matmul(&k.transpose(2, 3)) * (head_dim as f64).sqrt().recip()).masked_fill(
        &valid.logical_not().unsqueeze(1).unsqueeze(2),
        f64::NEG_INFINITY,
    );
    Ok(scores
        .softmax(-1, Kind::Float)
        .to_kind(v.kind())
        .matmul(&v)
        .squeeze_dim(2)
        .reshape([batch_size, num_heads * head_dim]))
}

#[cfg(test)]
fn causal_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Tensor {
    let q = q.transpose(0, 1);
    let repeats = num_heads / num_kv_heads;
    let k = k
        .transpose(0, 1)
        .repeat_interleave_self_int(repeats, 0, Some(num_heads));
    let v = v
        .transpose(0, 1)
        .repeat_interleave_self_int(repeats, 0, Some(num_heads));
    let sequence_length = q.size()[1];
    let scale = (head_dim as f64).sqrt().recip();
    let scores = (q.bmm(&k.transpose(1, 2)) * scale).masked_fill(
        &Tensor::ones([sequence_length, sequence_length], (Kind::Bool, q.device()))
            .triu(1)
            .unsqueeze(0),
        f64::NEG_INFINITY,
    );
    scores
        .softmax(-1, Kind::Float)
        .to_kind(v.kind())
        .bmm(&v)
        .transpose(0, 1)
        .reshape([sequence_length, num_heads * head_dim])
}

fn blocked_attention_mask(query_len: i64, key_len: i64, prefix: i64, device: Device) -> Tensor {
    let query_positions = Tensor::arange(query_len, (Kind::Int64, device)) + prefix;
    let key_positions = Tensor::arange(key_len, (Kind::Int64, device));
    key_positions
        .unsqueeze(0)
        .gt_tensor(&query_positions.unsqueeze(1))
}

fn attention_against_cache(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prefix_len: i64,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    blocked: Option<&Tensor>,
) -> Tensor {
    let query_len = q.size()[0];
    let q = q.transpose(0, 1);
    let repeats = num_heads / num_kv_heads;
    let k = k
        .transpose(0, 1)
        .repeat_interleave_self_int(repeats, 0, Some(num_heads));
    let v = v
        .transpose(0, 1)
        .repeat_interleave_self_int(repeats, 0, Some(num_heads));
    let owned_mask;
    let blocked = match blocked {
        Some(mask) => mask,
        None => {
            owned_mask = blocked_attention_mask(q.size()[1], k.size()[1], prefix_len, q.device());
            &owned_mask
        }
    };
    let scale = (head_dim as f64).sqrt().recip();
    (q.bmm(&k.transpose(1, 2)) * scale)
        .masked_fill(&blocked.unsqueeze(0), f64::NEG_INFINITY)
        .softmax(-1, Kind::Float)
        .to_kind(v.kind())
        .bmm(&v)
        .transpose(0, 1)
        .reshape([query_len, num_heads * head_dim])
}

fn batch_boundaries(metadata: Option<&AttentionMetadata>, total_tokens: usize) -> Result<Vec<i64>> {
    let total = i64::try_from(total_tokens).map_err(|_| model_error("token count exceeds i64"))?;
    let Some(metadata) = metadata else {
        return Ok(vec![0, total]);
    };
    if metadata.forward_mode == BatchPhase::Decode || metadata.cu_seqlens_q.is_none() {
        return Ok(vec![0, total]);
    }
    prefill_boundaries(metadata, total_tokens)
}

fn prefill_boundaries(metadata: &AttentionMetadata, total_tokens: usize) -> Result<Vec<i64>> {
    let total_tokens =
        i64::try_from(total_tokens).map_err(|_| model_error("token count exceeds i64"))?;
    let cumulative = metadata
        .cu_seqlens_q
        .as_ref()
        .ok_or_else(|| model_error("cached prefill requires cu_seqlens_q"))?;
    let boundaries = tensor_i32(Some(cumulative), "cu_seqlens_q")?;
    if boundaries.len() < 2
        || boundaries.first().copied() != Some(0)
        || boundaries.last().copied().map(i64::from) != Some(total_tokens)
        || boundaries.windows(2).any(|window| window[0] >= window[1])
    {
        return Err(model_error("invalid prefill cu_seqlens_q"));
    }
    Ok(boundaries.into_iter().map(i64::from).collect())
}

fn tensor_i32(tensor: Option<&Tensor>, field: &str) -> Result<Vec<i32>> {
    let tensor = tensor.ok_or_else(|| model_error(&format!("missing {field}")))?;
    Vec::<i32>::try_from(&tensor.to_device(Device::Cpu)).map_err(ModelRunnerError::Torch)
}

fn model_error(message: &str) -> ModelRunnerError {
    ModelRunnerError::Model(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_prefill_reuses_metadata_and_writes_static_outputs() {
        let device = Device::Cpu;
        let spec = AttentionSpec {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            kind: Kind::Float,
            device,
        };
        let mut cache = BaseAttention::default();
        let keys = Tensor::from_slice(&[1f32, 0., 0., 1., 1., 1., 2., 1., 1., 2., 0., 2.])
            .view([1, 6, 1, 2]);
        let values = &keys * 2.;
        cache
            .bind_kv_cache(keys.shallow_clone(), values.shallow_clone())
            .unwrap();
        let q = Tensor::from_slice(&[1f32, 0., 0., 1., 1., 1.]).view([3, 1, 2]);
        for backend in ["pt", "fa"] {
            for prefixes in [[0i32, 0], [1, 0]] {
                let metadata = AttentionMetadata {
                    forward_mode: BatchPhase::Prefill,
                    decode_host: None,
                    cu_seqlens_q: Some(Tensor::from_slice(&[0i32, 2, 3])),
                    prefix_lens: Some(Tensor::from_slice(&prefixes)),
                    req_to_token: Some(Tensor::from_slice(&[0i32, 1, 2, 3, 4, 5]).view([2, 3])),
                    write_loc: None,
                    block_table: None,
                    cache_seqlens: None,
                    max_seqlen: None,
                };
                let attention = Attention::new(backend, spec).unwrap();
                let batch = attention.prepare(Some(&metadata), 3).unwrap();
                let expected = if prefixes[0] == 0 {
                    Tensor::cat(
                        &[
                            sdpa_attention(
                                &q.narrow(0, 0, 2),
                                &q.narrow(0, 0, 2),
                                &q.narrow(0, 0, 2),
                                0,
                                1,
                                1,
                                2,
                            )
                            .unwrap(),
                            sdpa_attention(
                                &q.narrow(0, 2, 1),
                                &q.narrow(0, 2, 1),
                                &q.narrow(0, 2, 1),
                                0,
                                1,
                                1,
                                2,
                            )
                            .unwrap(),
                        ],
                        0,
                    )
                } else {
                    Tensor::cat(
                        &[
                            sdpa_attention(
                                &q.narrow(0, 0, 2),
                                &keys.view([6, 1, 2]).narrow(0, 0, 3),
                                &values.view([6, 1, 2]).narrow(0, 0, 3),
                                1,
                                1,
                                1,
                                2,
                            )
                            .unwrap(),
                            sdpa_attention(
                                &q.narrow(0, 2, 1),
                                &keys.view([6, 1, 2]).narrow(0, 3, 1),
                                &values.view([6, 1, 2]).narrow(0, 3, 1),
                                0,
                                1,
                                1,
                                2,
                            )
                            .unwrap(),
                        ],
                        0,
                    )
                };
                // A layer forward must use the invocation snapshot, not re-read
                // cu_seqlens, prefixes or cached locations from metadata tensors.
                let _ = metadata
                    .cu_seqlens_q
                    .as_ref()
                    .unwrap()
                    .shallow_clone()
                    .fill_(-1);
                let _ = metadata
                    .prefix_lens
                    .as_ref()
                    .unwrap()
                    .shallow_clone()
                    .fill_(-1);
                let _ = metadata
                    .req_to_token
                    .as_ref()
                    .unwrap()
                    .shallow_clone()
                    .fill_(-1);
                for _ in 0..2 {
                    let actual = batch.forward(&q, &q, &q, &cache).unwrap();
                    assert!(
                        (&actual - &expected).abs().max().double_value(&[]) < 1e-5,
                        "{backend}"
                    );
                    let storage = Tensor::full([5, 2], 99., (Kind::Float, device));
                    let output = storage.narrow(0, 0, 3);
                    let ptr = output.data_ptr();
                    batch.forward_into(&q, &q, &q, &cache, &output).unwrap();
                    assert_eq!(output.data_ptr(), ptr);
                    assert!(
                        (&output - &expected).abs().max().double_value(&[]) < 1e-5,
                        "{backend}"
                    );
                    assert_eq!(storage.narrow(0, 3, 2).min().double_value(&[]), 99.);
                    assert!(batch.forward_into(&q, &q, &q, &cache, &storage).is_err());
                }
                assert!(attention.prepare(Some(&metadata), 3).is_err());
            }
        }
    }

    #[cfg(has_flashinfer)]
    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn flashinfer_reuses_workspace_and_updates_decode_plans() {
        assert!(
            tch::Cuda::is_available(),
            "CUDA test requires an available GPU"
        );
        unsafe extern "C" {
            fn sglang_flashinfer_workspace_allocations(plan: *const std::ffi::c_void) -> u64;
        }
        let device = Device::Cuda(0);
        let spec = AttentionSpec {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 128,
            kind: Kind::BFloat16,
            device,
        };
        let mut attention = Attention::new("flashinfer", spec).unwrap();
        attention.bind_cache_layout(16);
        let mut cache = BaseAttention::default();
        cache
            .bind_kv_cache(
                Tensor::randn([32, 16, 8, 128], (Kind::BFloat16, device)),
                Tensor::randn([32, 16, 8, 128], (Kind::BFloat16, device)),
            )
            .unwrap();
        let table = Tensor::arange(32, (Kind::Int, device)).view([2, 16]);
        let tokens = Tensor::arange(512, (Kind::Int, device)).view([2, 256]);
        let packed = Tensor::randn([2, 4096], (Kind::BFloat16, device));
        let q = packed.narrow(1, 0, 2048).view([2, 16, 128]);
        assert!(!q.is_contiguous());
        let mut previous_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut warmed_allocations = 0;
        for (rows, len) in [
            (2i64, 256i32),
            (1, 1),
            (2, 16),
            (1, 17),
            (2, 128),
            (2, 255),
            (2, 256),
        ] {
            let metadata = AttentionMetadata {
                forward_mode: BatchPhase::Decode,
                decode_host: None,
                write_loc: None,
                cu_seqlens_q: None,
                prefix_lens: None,
                block_table: Some(table.narrow(0, 0, rows)),
                req_to_token: Some(tokens.narrow(0, 0, rows)),
                cache_seqlens: Some(
                    Tensor::from_slice(&[len, len - len / 2][..rows as usize]).to_device(device),
                ),
                max_seqlen: Some(len as usize),
            };
            let batch = attention.prepare(Some(&metadata), rows as usize).unwrap();
            assert!(attention.prepare(Some(&metadata), rows as usize).is_err());
            let BackendPlan::FlashInfer(plan) = &batch.plan else {
                panic!("missing plan");
            };
            let allocations = unsafe { sglang_flashinfer_workspace_allocations(plan.0) };
            if previous_ptr.is_null() {
                previous_ptr = plan.0;
                warmed_allocations = allocations;
            } else {
                assert_eq!(plan.0, previous_ptr);
                assert_eq!(allocations, warmed_allocations);
            }
            let query = q.narrow(0, 0, rows);
            let actual = batch.forward(&query, &query, &query, &cache).unwrap();
            let expected = sdpa_decode_with_cache(&query, &cache, &metadata, 16, 8, 128).unwrap();
            let error = (actual.to_kind(Kind::Float) - expected.to_kind(Kind::Float))
                .abs()
                .max()
                .double_value(&[]);
            assert!(error <= 0.03125, "len={len}, error={error}");
        }
    }

    #[cfg(has_flashinfer)]
    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn flashinfer_prefill_matches_reference_for_ragged_and_cached_batches() {
        assert!(
            tch::Cuda::is_available(),
            "CUDA test requires an available GPU"
        );
        let device = Device::Cuda(0);
        for kind in [Kind::BFloat16, Kind::Half] {
            let spec = AttentionSpec {
                num_heads: 16,
                num_kv_heads: 8,
                head_dim: 128,
                kind,
                device,
            };
            let mut attention = Attention::new("flashinfer", spec).unwrap();
            attention.bind_cache_layout(16);
            let mut cache = BaseAttention::default();
            let keys = Tensor::randn([64, 16, 8, 128], (kind, device));
            let values = Tensor::randn_like(&keys);
            cache
                .bind_kv_cache(keys.shallow_clone(), values.shallow_clone())
                .unwrap();
            // Unequal lengths, page boundaries, long contexts and mixed prefixes.
            for (lengths, prefixes) in [
                ([17i32, 1], [0i32, 0]),
                ([129, 33], [0, 0]),
                ([1, 17], [255, 16]),
                ([17, 1], [0, 31]),
                ([257, 65], [240, 0]),
            ] {
                let bounds = [0i32, lengths[0], lengths[0] + lengths[1]];
                let table = Tensor::arange(64, (Kind::Int, device)).view([2, 32]);
                let tokens = Tensor::arange(1024, (Kind::Int, device)).view([2, 512]);
                let metadata = AttentionMetadata {
                    forward_mode: BatchPhase::Prefill,
                    decode_host: None,
                    write_loc: None,
                    cu_seqlens_q: Some(Tensor::from_slice(&bounds).to_device(device)),
                    prefix_lens: Some(Tensor::from_slice(&prefixes).to_device(device)),
                    block_table: Some(table),
                    req_to_token: Some(tokens),
                    cache_seqlens: None,
                    max_seqlen: None,
                };
                let packed = Tensor::randn([i64::from(bounds[2]), 4096], (kind, device));
                let q = packed.narrow(1, 0, 2048).view([-1, 16, 128]);
                let k = packed.narrow(1, 2048, 1024).view([-1, 8, 128]);
                let v = packed.narrow(1, 3072, 1024).view([-1, 8, 128]);
                let batch = attention
                    .prepare(Some(&metadata), bounds[2] as usize)
                    .unwrap();
                assert!(matches!(batch.plan, BackendPlan::FlashInferPrefill(_, _)));
                let actual = batch.forward(&q, &k, &v, &cache).unwrap();
                let storage = Tensor::full([i64::from(bounds[2]) + 3, 2048], 99., (kind, device));
                let target = storage.narrow(0, 0, i64::from(bounds[2]));
                let ptr = target.data_ptr();
                batch.forward_into(&q, &k, &v, &cache, &target).unwrap();
                assert_eq!(target.data_ptr(), ptr);
                assert!(Tensor::equal(&actual, &target));
                assert_eq!(
                    storage
                        .narrow(0, i64::from(bounds[2]), 3)
                        .min()
                        .double_value(&[]),
                    99.
                );
                let direct =
                    flashinfer_prefill(&q, &k, &v, &cache, &batch.plan, 128, Some(&target))
                        .unwrap();
                assert_eq!(
                    direct.data_ptr(),
                    ptr,
                    "native prefill must return the supplied storage"
                );
                let reference = batched_prefill_sdpa(
                    &q,
                    &k,
                    &v,
                    &cache,
                    &bounds.map(i64::from),
                    &prefixes.map(i64::from),
                    Some(&metadata),
                    16,
                    8,
                    128,
                )
                .unwrap();
                let error = (actual.to_kind(Kind::Float) - reference.to_kind(Kind::Float))
                    .abs()
                    .max()
                    .double_value(&[]);
                assert!(
                    error <= 0.03125,
                    "kind={kind:?},lengths={lengths:?},prefixes={prefixes:?},error={error}"
                );
            }
        }
    }

    #[test]
    fn padded_decode_matches_eager_for_different_sequence_lengths() {
        let mut cache = BaseAttention::default();
        cache
            .bind_kv_cache(
                Tensor::zeros([1, 4, 1, 2], (Kind::Float, Device::Cpu)),
                Tensor::zeros([1, 4, 1, 2], (Kind::Float, Device::Cpu)),
            )
            .unwrap();
        let k = Tensor::from_slice(&[1f32, 0., 0., 1., 1., 1.]).view([3, 1, 2]);
        let v = Tensor::from_slice(&[2f32, 0., 0., 4., 6., 6.]).view([3, 1, 2]);
        cache
            .write_kv(
                &k,
                &v,
                Some(&Tensor::from_slice(&[0i32, 1, 2])),
                BatchPhase::Prefill,
            )
            .unwrap();
        let metadata = AttentionMetadata {
            forward_mode: BatchPhase::Decode,
            decode_host: None,
            write_loc: None,
            cu_seqlens_q: None,
            prefix_lens: None,
            block_table: None,
            req_to_token: Some(Tensor::from_slice(&[0i32, 1, -1, -1, 2, -1, -1, -1]).view([2, 4])),
            cache_seqlens: Some(Tensor::from_slice(&[2i32, 1])),
            max_seqlen: Some(4),
        };
        let q = Tensor::from_slice(&[1f32, 0., 0., 1.]).view([2, 1, 2]);
        let eager = decode_with_cache(&q, &cache, &metadata, 1, 1, 2).unwrap();
        let padded = decode_with_cache_padded(&q, &cache, &metadata, 1, 1, 2).unwrap();
        let difference = (eager - padded).abs().max().double_value(&[]);
        assert!(difference < 1e-5, "decode difference: {difference}");
        let sdpa = sdpa_decode_with_cache(&q, &cache, &metadata, 1, 1, 2).unwrap();
        let reference = decode_with_cache_padded(&q, &cache, &metadata, 1, 1, 2).unwrap();
        let difference = (reference - sdpa).abs().max().double_value(&[]);
        assert!(difference < 1e-5, "SDPA decode difference: {difference}");
    }

    #[test]
    fn sdpa_causal_prefill_matches_reference() {
        let q = Tensor::from_slice(&[1f32, 0., 0., 1., 1., 1.]).view([3, 1, 2]);
        let k = q.shallow_clone();
        let v = Tensor::from_slice(&[2f32, 1., 3., 4., 5., 6.]).view([3, 1, 2]);
        let reference = causal_attention(&q, &k, &v, 1, 1, 2);
        let sdpa = sdpa_attention(&q, &k, &v, 0, 1, 1, 2).unwrap();
        let difference = (reference - sdpa).abs().max().double_value(&[]);
        assert!(difference < 1e-5, "SDPA prefill difference: {difference}");
    }

    #[test]
    fn batched_prefill_matches_per_request_attention_for_unequal_lengths() {
        let q = Tensor::from_slice(&[1f32, 0., 0., 1., 1., 1., 2., 1., 1., 2.]).view([5, 1, 2]);
        let k = q.shallow_clone();
        let v = Tensor::from_slice(&[2f32, 1., 3., 4., 5., 6., 7., 8., 9., 10.]).view([5, 1, 2]);
        let boundaries = [0, 2, 3, 5];
        let actual = batched_prefill_sdpa(
            &q,
            &k,
            &v,
            &BaseAttention::default(),
            &boundaries,
            &[0, 0, 0],
            None,
            1,
            1,
            2,
        )
        .unwrap();
        let expected = Tensor::cat(
            &boundaries
                .windows(2)
                .map(|bounds| {
                    let start = bounds[0];
                    let len = bounds[1] - start;
                    sdpa_attention(
                        &q.narrow(0, start, len),
                        &k.narrow(0, start, len),
                        &v.narrow(0, start, len),
                        0,
                        1,
                        1,
                        2,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>(),
            0,
        );
        let difference = (actual - expected).abs().max().double_value(&[]);
        assert!(
            difference < 1e-5,
            "batched prefill difference: {difference}"
        );
    }

    #[test]
    fn batched_cached_prefill_matches_per_request_attention() {
        let mut cache = BaseAttention::default();
        cache
            .bind_kv_cache(
                Tensor::zeros([1, 8, 1, 2], (Kind::Float, Device::Cpu)),
                Tensor::zeros([1, 8, 1, 2], (Kind::Float, Device::Cpu)),
            )
            .unwrap();
        let all_k = Tensor::from_slice(&[
            1f32, 0., 0., 1., 1., 1., 2., 0., 0., 2., 1., 2., 2., 1., 1., 3.,
        ])
        .view([8, 1, 2]);
        let all_v = Tensor::from_slice(&[
            2f32, 1., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ])
        .view([8, 1, 2]);
        cache
            .write_kv(
                &all_k,
                &all_v,
                Some(&Tensor::from_slice(&[0i32, 1, 2, 3, 4, 5, 6, 7])),
                BatchPhase::Prefill,
            )
            .unwrap();
        let q = Tensor::from_slice(&[1f32, 0., 0., 1., 1., 1., 2., 1., 1., 2.]).view([5, 1, 2]);
        let table = Tensor::from_slice(&[0i32, 1, 2, -1, 3, -1, -1, -1, 4, 5, 6, 7]).view([3, 4]);
        let metadata = AttentionMetadata {
            forward_mode: BatchPhase::Prefill,
            decode_host: None,
            write_loc: None,
            cu_seqlens_q: None,
            prefix_lens: None,
            block_table: None,
            req_to_token: Some(table.shallow_clone()),
            cache_seqlens: None,
            max_seqlen: Some(2),
        };
        let boundaries = [0, 2, 3, 5];
        let prefixes = [1, 0, 2];
        let actual = batched_prefill_sdpa(
            &q,
            &q,
            &q,
            &cache,
            &boundaries,
            &prefixes,
            Some(&metadata),
            1,
            1,
            2,
        )
        .unwrap();
        let expected = Tensor::cat(
            &boundaries
                .windows(2)
                .enumerate()
                .map(|(row, bounds)| {
                    let len = bounds[1] - bounds[0];
                    let (k, v) = cache
                        .read_kv(&table, row as i64, prefixes[row] + len)
                        .unwrap();
                    sdpa_attention(&q.narrow(0, bounds[0], len), &k, &v, prefixes[row], 1, 1, 2)
                        .unwrap()
                })
                .collect::<Vec<_>>(),
            0,
        );
        let difference = (actual - expected).abs().max().double_value(&[]);
        assert!(
            difference < 1e-5,
            "batched cached prefill difference: {difference}"
        );
    }
}
