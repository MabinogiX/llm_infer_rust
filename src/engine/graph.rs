//! CUDA decode graph capture and replay with fixed input buffers.

use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use tch::{Cuda, Device, Kind, Tensor};

use super::{
    AttentionMetadata, BatchPhase, DecodeGraphState, ModelRunner, ModelRunnerError, ServerArgs,
};
use super::{ForwardBatch, ForwardMode, ForwardOutput, GraphLimits, GraphPadding, GraphSupport};
use crate::engine::kvcache::KVCachePool;

type Result<T> = std::result::Result<T, ModelRunnerError>;

struct CapturedGraph {
    native: NativeCudaGraph,
    backend: Option<Box<dyn DecodeGraphState>>,
    input_ids: Tensor,
    positions: Tensor,
    metadata: AttentionMetadata,
    output: Tensor,
}

/// Holds one graph per batch size; padding reads the permanent reserved page 0.
pub(crate) struct DecodeGraphRunner {
    graphs: BTreeMap<usize, CapturedGraph>,
    pad_pages: Tensor,
    pad_locations: Tensor,
    admission: GraphLimits,
}

impl DecodeGraphRunner {
    pub fn capture(
        runner: &ModelRunner,
        args: &ServerArgs,
        _pool: Rc<RefCell<KVCachePool>>,
        context: Option<&Rc<NativeGraphCaptureContext>>,
    ) -> Result<Option<Self>> {
        if crate::logging::step_timing_enabled() {
            tracing::info!("CUDA Graph disabled for synchronized SGLANG_PROFILE_STEPS profiling");
            return Ok(None);
        }
        let admission = match runner.graph_capabilities().decode {
            GraphSupport::Unsupported(reason) => {
                tracing::info!(reason, "decode graphs unavailable; using eager");
                return Ok(None);
            }
            GraphSupport::Supported(limits) => limits,
        };
        if admission.padding != GraphPadding::ReservedKvPageZero {
            return Err(ModelRunnerError::Model(
                "decode graphs require reserved KV page zero padding".into(),
            ));
        }
        if args.max_seq_len > admission.max_context_len {
            tracing::info!("decode graph context exceeds model capability; using eager");
            return Ok(None);
        }
        let Device::Cuda(device_index) = runner.device() else {
            if args.cuda_graph_bs.is_some_and(|size| size > 0) {
                tracing::warn!(device = ?runner.device(), "CUDA Graph requested on a non-CUDA device; using eager decode");
            } else {
                tracing::info!(device = ?runner.device(), "CUDA Graph capture skipped; using eager decode");
            }
            return Ok(None);
        };
        let limit = args
            .cuda_graph_bs
            .unwrap_or(args.max_running_req)
            .min(args.max_running_req)
            .min(admission.max_batch_size)
            .min(admission.max_tokens);
        if limit == 0 {
            tracing::info!("CUDA Graph capture disabled; using eager decode");
            return Ok(None);
        }
        if !NativeCudaGraph::available() {
            tracing::warn!("CUDA Graph bridge unavailable; using eager decode");
            return Ok(None);
        }
        let context = context.ok_or_else(|| {
            ModelRunnerError::Model("CUDA graph capture context is missing".into())
        })?;
        let pad_pages = Tensor::zeros([limit as i64], (Kind::Int, runner.device()));
        let pad_locations = Tensor::zeros_like(&pad_pages);
        let mut graph_runner = Self {
            graphs: BTreeMap::new(),
            pad_pages,
            pad_locations,
            admission,
        };
        let mut sizes = vec![1, 2, 4, 8, 16, 32, 64, 128, 256];
        sizes.retain(|&size| size <= limit);
        if sizes.last().copied() != Some(limit) {
            sizes.push(limit);
        }
        for size in sizes.into_iter().rev() {
            tracing::info!(size, "capturing CUDA decode graph");
            match graph_runner.capture_one(runner, args, context, device_index, size) {
                Ok(graph) => {
                    graph_runner.graphs.insert(size, graph);
                }
                Err(error) => {
                    context.check_execution_stream()?;
                    tracing::warn!(size, %error, "CUDA Graph capture failed; this batch size uses eager decode");
                }
            }
        }
        if graph_runner.graphs.is_empty() {
            tracing::warn!("no CUDA decode graph was captured; using eager decode");
            return Ok(None);
        }
        tracing::info!(sizes = ?graph_runner.graphs.keys().collect::<Vec<_>>(), "CUDA Graph capture complete");
        Ok(Some(graph_runner))
    }

    fn capture_one(
        &self,
        runner: &ModelRunner,
        args: &ServerArgs,
        context: &Rc<NativeGraphCaptureContext>,
        device_index: usize,
        size: usize,
    ) -> Result<CapturedGraph> {
        let rows = size as i64;
        let device = runner.device();
        let input_ids = Tensor::ones([rows, 1], (Kind::Int64, device));
        let positions = Tensor::zeros([rows, 1], (Kind::Int64, device));
        let mut metadata = AttentionMetadata {
            forward_mode: BatchPhase::Decode,
            write_loc: Some(self.pad_locations.narrow(0, 0, rows).copy()),
            cu_seqlens_q: None,
            prefix_lens: None,
            block_table: Some(
                self.pad_pages
                    .narrow(0, 0, rows)
                    .unsqueeze(1)
                    .repeat([1, args.max_seq_len.div_ceil(args.page_size) as i64]),
            ),
            req_to_token: Some(
                self.pad_locations
                    .narrow(0, 0, rows)
                    .unsqueeze(1)
                    .repeat([1, args.max_seq_len as i64]),
            ),
            cache_seqlens: Some(Tensor::ones([rows], (Kind::Int, device))),
            max_seqlen: Some(args.max_seq_len),
        };
        let backend = runner.prepare_decode_graph(&metadata)?;
        if backend
            .as_ref()
            .is_some_and(|state| !state.needs_req_to_token())
        {
            metadata.req_to_token = None;
        }
        Cuda::synchronize(device_index as i64);
        let native = NativeCudaGraph::create(context)?;
        for _ in 0..3 {
            let _ = runner
                .run_model(&ForwardBatch {
                    input_ids: &input_ids,
                    positions: &positions,
                    attention: Some(&metadata),
                    mode: ForwardMode::Decode,
                })?
                .logits;
        }
        Cuda::synchronize(device_index as i64);
        native.begin()?;
        let output = match runner.run_model(&ForwardBatch {
            input_ids: &input_ids,
            positions: &positions,
            attention: Some(&metadata),
            mode: ForwardMode::Decode,
        }) {
            Ok(output) => output.logits,
            Err(error) => {
                let _ = native.end();
                return Err(error);
            }
        };
        native.end()?;
        Ok(CapturedGraph {
            native,
            backend,
            input_ids,
            positions,
            metadata,
            output,
        })
    }

    fn admission_admits(&self, rows: usize, context: usize) -> bool {
        GraphSupport::Supported(self.admission).admits(rows, rows, context)
    }

    /// Replay a batch already validated by ModelRunner, or request eager fallback.
    pub(crate) fn replay(&self, batch: ForwardBatch<'_>) -> Result<Option<ForwardOutput>> {
        if batch.phase() != BatchPhase::Decode {
            return Ok(None);
        }
        let size = batch.input_ids.numel();
        let context = batch
            .attention
            .as_ref()
            .and_then(|m| m.max_seqlen)
            .unwrap_or(0);
        if !self.admission_admits(size, context) {
            tracing::debug!(
                size,
                context,
                "decode shape exceeds graph capability; using eager"
            );
            return Ok(None);
        }
        let Some((_, graph)) = self.graphs.range(size..).next() else {
            tracing::debug!(size, "no captured decode bucket; using eager");
            return Ok(None);
        };
        let rows = size as i64;
        let source = batch.attention.ok_or_else(|| {
            ModelRunnerError::Model("decode batch requires attention metadata".to_owned())
        })?;
        // A valid eager batch can have a different cache-table width. Admit the
        // graph only when every static shape matches, before copying anything.
        for (target, value) in [
            (&graph.metadata.block_table, &source.block_table),
            (&graph.metadata.req_to_token, &source.req_to_token),
        ] {
            if let Some(target) = target {
                let Some(value) = value else { return Ok(None) };
                if value.size()[1..] != target.size()[1..] {
                    tracing::debug!("decode cache table differs from captured shape; using eager");
                    return Ok(None);
                }
            }
        }
        graph
            .input_ids
            .narrow(0, 0, rows)
            .copy_(&batch.input_ids.view([rows, 1]));
        graph
            .positions
            .narrow(0, 0, rows)
            .copy_(&batch.positions.view([rows, 1]));
        copy_field(&graph.metadata.write_loc, &source.write_loc, rows)?;
        copy_field(&graph.metadata.cache_seqlens, &source.cache_seqlens, rows)?;
        copy_field(&graph.metadata.block_table, &source.block_table, rows)?;
        if graph.metadata.req_to_token.is_some() {
            copy_field(&graph.metadata.req_to_token, &source.req_to_token, rows)?;
        }
        if size < graph.input_ids.size()[0] as usize {
            let start = rows;
            let count = graph.input_ids.size()[0] - rows;
            let _ = graph.input_ids.narrow(0, start, count).fill_(1);
            let _ = graph.positions.narrow(0, start, count).zero_();
            graph
                .metadata
                .write_loc
                .as_ref()
                .unwrap()
                .narrow(0, start, count)
                .copy_(&self.pad_locations.narrow(0, start, count));
            let _ = graph
                .metadata
                .cache_seqlens
                .as_ref()
                .unwrap()
                .narrow(0, start, count)
                .fill_(1);
            let table = graph.metadata.block_table.as_ref().unwrap();
            table.narrow(0, start, count).copy_(
                &self
                    .pad_pages
                    .narrow(0, start, count)
                    .unsqueeze(1)
                    .repeat([1, table.size()[1]]),
            );
            if let Some(tokens) = graph.metadata.req_to_token.as_ref() {
                tokens.narrow(0, start, count).copy_(
                    &self
                        .pad_locations
                        .narrow(0, start, count)
                        .unsqueeze(1)
                        .repeat([1, tokens.size()[1]]),
                );
            }
        }
        if let Some(backend) = graph.backend.as_ref() {
            backend.prepare_replay()?;
        }
        graph.native.replay()?;
        Ok(Some(ForwardOutput::new(
            graph.output.narrow(0, 0, rows).copy(),
        )))
    }
}

impl Drop for DecodeGraphRunner {
    fn drop(&mut self) {
        // Native graph pools and planning buffers can still be in flight.
        if let Device::Cuda(device) = self.pad_pages.device() {
            Cuda::synchronize(device as i64);
        }
        self.graphs.clear();
    }
}

fn copy_field(target: &Option<Tensor>, source: &Option<Tensor>, rows: i64) -> Result<()> {
    let (Some(target), Some(source)) = (target, source) else {
        return Err(ModelRunnerError::Model(
            "decode metadata is incomplete".to_owned(),
        ));
    };
    target.narrow(0, 0, rows).copy_(source);
    Ok(())
}

/// One model/device capture catalog. Rc also prevents cross-thread graph use.
pub(crate) struct NativeGraphCaptureContext {
    device: usize,
    #[cfg(has_cuda_graph)]
    ptr: *mut std::ffi::c_void,
}

impl NativeGraphCaptureContext {
    pub(crate) fn device_index(&self) -> usize {
        self.device
    }
    pub(crate) fn new(device: usize) -> Result<Rc<Self>> {
        #[cfg(has_cuda_graph)]
        {
            let ptr = unsafe { sglang_graph_context_create(device as i32) };
            if ptr.is_null() {
                return Err(native_error());
            }
            Ok(Rc::new(Self { device, ptr }))
        }
        #[cfg(not(has_cuda_graph))]
        {
            let _ = device;
            Err(ModelRunnerError::Model(
                "CUDA Graph bridge unavailable".into(),
            ))
        }
    }

    /// Check before copying inputs, planning attention or launching any graph work.
    pub(crate) fn check_execution_stream(&self) -> Result<()> {
        #[cfg(has_cuda_graph)]
        if !unsafe { sglang_graph_context_check(self.ptr) } {
            return Err(native_error());
        }
        Ok(())
    }

    pub(crate) fn memory_bytes(&self) -> Result<(i64, i64)> {
        #[cfg(has_cuda_graph)]
        {
            let (mut allocated, mut reserved) = (0, 0);
            if !unsafe { sglang_graph_memory(self.device as i32, &mut allocated, &mut reserved) } {
                return Err(native_error());
            }
            Ok((allocated, reserved))
        }
        #[cfg(not(has_cuda_graph))]
        {
            Ok((0, 0))
        }
    }
}

impl Drop for NativeGraphCaptureContext {
    fn drop(&mut self) {
        #[cfg(has_cuda_graph)]
        unsafe {
            sglang_graph_context_free(self.ptr)
        };
    }
}

pub(crate) struct NativeCudaGraph {
    #[cfg(has_cuda_graph)]
    ptr: *mut std::ffi::c_void,
    _context: Rc<NativeGraphCaptureContext>,
}

impl NativeCudaGraph {
    pub(crate) fn available() -> bool {
        cfg!(has_cuda_graph)
    }

    /// Warm and capture a GPU-only segment on a side stream. The caller owns
    /// fixed inputs and the declared storage live ranges of captured outputs.
    pub(crate) fn capture<T>(
        context: &Rc<NativeGraphCaptureContext>,
        mut forward: impl FnMut() -> Result<T>,
    ) -> Result<(Self, T)> {
        Cuda::synchronize(context.device as i64);
        let graph = Self::create(context)?;
        for _ in 0..3 {
            let _ = forward()?;
        }
        Cuda::synchronize(context.device as i64);
        graph.begin()?;
        let output = match forward() {
            Ok(output) => output,
            Err(error) => {
                let _ = graph.end();
                return Err(error);
            }
        };
        graph.end()?;
        Ok((graph, output))
    }

    fn create(context: &Rc<NativeGraphCaptureContext>) -> Result<Self> {
        #[cfg(has_cuda_graph)]
        {
            let ptr = unsafe { sglang_graph_create(context.ptr) };
            if ptr.is_null() {
                return Err(native_error());
            }
            return Ok(Self {
                ptr,
                _context: context.clone(),
            });
        }
        #[cfg(not(has_cuda_graph))]
        {
            let _ = context;
            Err(ModelRunnerError::Model(
                "CUDA Graph bridge unavailable".to_owned(),
            ))
        }
    }

    fn begin(&self) -> Result<()> {
        #[cfg(has_cuda_graph)]
        if !unsafe { sglang_graph_begin(self.ptr) } {
            return Err(native_error());
        }
        Ok(())
    }

    fn end(&self) -> Result<()> {
        #[cfg(has_cuda_graph)]
        if !unsafe { sglang_graph_end(self.ptr) } {
            return Err(native_error());
        }
        Ok(())
    }

    /// Make a metadata-only alias of a captured allocation.
    ///
    /// # Safety
    /// The output must belong to this graph's pool. The graph/context must
    /// outlive every alias use. Its owning allocation must remain live through
    /// all dependent capture operations, and replay consumers must finish before
    /// another graph can reuse the allocation. Never return this alias to users.
    pub(crate) unsafe fn non_owning_output(&self, output: &Tensor) -> Result<Tensor> {
        if output.device() != Device::Cuda(self._context.device) {
            return Err(ModelRunnerError::Model(
                "graph output device mismatch".into(),
            ));
        }
        #[cfg(has_cuda_graph)]
        {
            let ptr = unsafe { sglang_graph_non_owning_view(output.as_ptr().cast()) };
            if ptr.is_null() {
                return Err(native_error());
            }
            Ok(unsafe { Tensor::from_ptr(ptr.cast()) })
        }
        #[cfg(not(has_cuda_graph))]
        {
            Err(ModelRunnerError::Model(
                "CUDA Graph bridge unavailable".into(),
            ))
        }
    }

    pub(crate) fn replay(&self) -> Result<()> {
        #[cfg(has_cuda_graph)]
        if !unsafe { sglang_graph_replay(self.ptr) } {
            return Err(native_error());
        }
        Ok(())
    }
}

impl Drop for NativeCudaGraph {
    fn drop(&mut self) {
        #[cfg(has_cuda_graph)]
        unsafe {
            sglang_graph_free(self.ptr)
        };
    }
}

#[cfg(has_cuda_graph)]
fn native_error() -> ModelRunnerError {
    let message = unsafe { std::ffi::CStr::from_ptr(sglang_graph_error()) };
    ModelRunnerError::Model(format!("CUDA Graph: {}", message.to_string_lossy()))
}

#[cfg(has_cuda_graph)]
unsafe extern "C" {
    fn sglang_graph_non_owning_view(tensor: *const std::ffi::c_void) -> *mut std::ffi::c_void;
    fn sglang_graph_memory(device: i32, allocated: *mut i64, reserved: *mut i64) -> bool;
    fn sglang_graph_context_create(device: i32) -> *mut std::ffi::c_void;
    fn sglang_graph_context_free(context: *mut std::ffi::c_void);
    fn sglang_graph_context_check(context: *mut std::ffi::c_void) -> bool;
    fn sglang_graph_create(context: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    fn sglang_graph_begin(graph: *mut std::ffi::c_void) -> bool;
    fn sglang_graph_end(graph: *mut std::ffi::c_void) -> bool;
    fn sglang_graph_replay(graph: *mut std::ffi::c_void) -> bool;
    fn sglang_graph_free(graph: *mut std::ffi::c_void);
    fn sglang_graph_error() -> *const std::ffi::c_char;
}

#[cfg(all(test, has_flashinfer, has_cuda_graph))]
mod tests {
    use super::*;
    use crate::{
        engine::{Batch, ModelExecutor, kvcache::KVCacheLayout},
        models::attention::{Attention, AttentionSpec, BaseAttention},
    };

    struct DecodeModel {
        attention: Attention,
        cache: RefCell<BaseAttention>,
    }

    impl ModelExecutor for DecodeModel {
        fn graph_capabilities(&self) -> super::super::GraphCapabilities {
            super::super::GraphCapabilities {
                decode: GraphSupport::Supported(GraphLimits {
                    max_batch_size: 5,
                    max_tokens: 5,
                    max_context_len: 1024,
                    padding: GraphPadding::ReservedKvPageZero,
                }),
                ..Default::default()
            }
        }

        fn prepare_decode_graph(
            &self,
            metadata: &AttentionMetadata,
        ) -> Result<Option<Box<dyn DecodeGraphState>>> {
            self.attention.prepare_decode_graph(metadata)
        }

        fn forward(
            &self,
            batch: &crate::engine::ForwardBatch<'_>,
        ) -> std::result::Result<crate::engine::ForwardOutput, ModelRunnerError> {
            let ids = batch.input_ids;
            let positions = batch.positions;
            let metadata = batch.attention;
            let _ = batch.logits_indices();
            (|| -> std::result::Result<Tensor, ModelRunnerError> {
                let rows = ids.size()[0];
                let scalar = (ids + positions).to_kind(Kind::Float).view([rows, 1, 1]) * 0.001;
                let q = scalar.repeat([1, 16, 128]).to_kind(Kind::BFloat16);
                let kv = q.narrow(1, 0, 8).contiguous();
                let mut cache = self.cache.borrow_mut();
                cache.write_kv(
                    &kv,
                    &(&kv + 1.0),
                    metadata.unwrap().write_loc.as_ref(),
                    BatchPhase::Decode,
                )?;
                self.attention
                    .prepare(metadata, rows as usize)?
                    .forward(&q, &kv, &kv, &cache)
            })()
            .map(crate::engine::ForwardOutput::new)
        }
    }

    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn flashinfer_graph_replays_dynamic_pages_lengths_and_padded_batches() {
        assert!(Cuda::is_available(), "CUDA test requires an available GPU");
        let device = Device::Cuda(0);
        let pool = Rc::new(RefCell::new(
            KVCachePool::new(
                KVCacheLayout::new(1, 400, 16, 8, 128).unwrap(),
                Kind::BFloat16,
                device,
            )
            .unwrap(),
        ));
        let (mut k, mut v) = pool.borrow().get_all_kv_cache().unwrap();
        k.copy_(&Tensor::randn(k.size(), (Kind::BFloat16, device)));
        v.copy_(&Tensor::randn(v.size(), (Kind::BFloat16, device)));
        let mut cache = BaseAttention::default();
        cache.bind_kv_cache(k.get(0), v.get(0)).unwrap();
        cache.set_reserved_write_slot(0);
        let mut attention = Attention::new(
            "flashinfer",
            AttentionSpec {
                num_heads: 16,
                num_kv_heads: 8,
                head_dim: 128,
                kind: Kind::BFloat16,
                device,
            },
        )
        .unwrap();
        attention.bind_cache_layout(16);
        let runner = ModelRunner::new(
            Box::new(DecodeModel {
                attention,
                cache: RefCell::new(cache),
            }),
            device,
        );
        let mut args = ServerArgs::new("unused");
        args.max_running_req = 5;
        args.cuda_graph_bs = Some(4);
        args.max_seq_len = 1024;
        let free = pool.borrow().free_count();
        let graph = DecodeGraphRunner::capture(
            &runner,
            &args,
            pool.clone(),
            Some(&NativeGraphCaptureContext::new(0).unwrap()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(graph.graphs.keys().copied().collect::<Vec<_>>(), [1, 2, 4]);
        assert_eq!(pool.borrow().free_count(), free);
        let mut retained = None;
        for (iteration, (rows, len)) in [
            (1i64, 1i64),
            (3, 17),
            (2, 128),
            (4, 1023),
            (1, 16),
            (3, 255),
            (5, 513),
        ]
        .into_iter()
        .enumerate()
        {
            let mut table = Vec::new();
            let mut locations = Vec::new();
            let mut lengths = Vec::new();
            for row in 0..rows {
                let length = (len - row * 3).max(1);
                lengths.push(length as i32);
                let pages: Vec<i32> = (0..64)
                    .map(|page| (1 + row * 64 + (page + iteration as i64) % 64) as i32)
                    .collect();
                locations
                    .push(pages[((length - 1) / 16) as usize] * 16 + ((length - 1) % 16) as i32);
                table.extend(pages);
            }
            let mut batch = Batch::decode(
                (Tensor::arange(rows, (Kind::Int64, device)) + iteration as i64 + 1)
                    .view([rows, 1]),
                Tensor::from_slice(&lengths)
                    .to_kind(Kind::Int64)
                    .to_device(device)
                    .view([rows, 1])
                    - 1,
                Some(AttentionMetadata {
                    forward_mode: BatchPhase::Decode,
                    write_loc: Some(Tensor::from_slice(&locations).to_device(device)),
                    cache_seqlens: Some(Tensor::from_slice(&lengths).to_device(device)),
                    block_table: Some(
                        Tensor::from_slice(&table)
                            .to_device(device)
                            .view([rows, 64]),
                    ),
                    req_to_token: None,
                    cu_seqlens_q: None,
                    prefix_lens: None,
                    max_seqlen: Some(1024),
                }),
            );
            if iteration == 0 {
                let table = batch
                    .attention_metadata
                    .as_mut()
                    .unwrap()
                    .block_table
                    .take()
                    .unwrap();
                batch.attention_metadata.as_mut().unwrap().block_table =
                    Some(table.narrow(1, 0, 1));
                assert!(
                    graph
                        .replay(ForwardBatch::from_scheduler(&batch).unwrap())
                        .unwrap()
                        .is_none(),
                    "table width mismatch must fall back to eager"
                );
                batch.attention_metadata.as_mut().unwrap().block_table = Some(table);
            }
            let replay = graph
                .replay(ForwardBatch::from_scheduler(&batch).unwrap())
                .unwrap();
            assert_eq!(replay.is_some(), rows <= 4);
            let actual = replay
                .unwrap_or_else(|| runner.forward(&batch).unwrap())
                .logits;
            let expected = runner
                .run_model(&ForwardBatch::from_scheduler(&batch).unwrap())
                .unwrap()
                .logits;
            let error = (&actual.to_kind(Kind::Float) - expected.to_kind(Kind::Float))
                .abs()
                .max()
                .double_value(&[]);
            assert!(error <= 0.03125, "rows={rows}, len={len}, error={error}");
            if let Some((old, snapshot)) = retained.as_ref() {
                assert!(
                    Tensor::equal(old, snapshot),
                    "replay overwrote a returned output"
                );
            }
            retained = Some((actual.shallow_clone(), actual.copy()));
        }
        drop(graph);
        assert_eq!(pool.borrow().free_count(), free);
        let graph = DecodeGraphRunner::capture(
            &runner,
            &args,
            pool.clone(),
            Some(&NativeGraphCaptureContext::new(0).unwrap()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(graph.graphs.len(), 3);
        drop(graph);
        args.cuda_graph_bs = Some(0);
        assert!(
            DecodeGraphRunner::capture(
                &runner,
                &args,
                pool.clone(),
                Some(&NativeGraphCaptureContext::new(0).unwrap())
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(pool.borrow().free_count(), free);
    }
}

#[cfg(all(test, has_cuda_graph))]
mod shared_pool_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn non_owning_graph_output_releases_storage_preserves_views_and_replays_reused_blocks() {
        let context = NativeGraphCaptureContext::new(0).unwrap();
        let x = Tensor::ones([256, 1024], (Kind::Float, Device::Cuda(0)));
        let (a, owning) = NativeCudaGraph::capture(&context, || Ok(&x * 2.)).unwrap();
        let address = owning.data_ptr();
        let slice = owning.narrow(1, 17, 31).transpose(0, 1);
        let alias = unsafe { a.non_owning_output(&slice) }.unwrap();
        assert_eq!(alias.data_ptr(), slice.data_ptr());
        assert_eq!(alias.size(), slice.size());
        assert_eq!(alias.stride(), slice.stride());
        let before = context.memory_bytes().unwrap().0;
        drop((slice, owning));
        let after = context.memory_bytes().unwrap().0;
        assert!(
            before - after >= 256 * 1024 * 4,
            "non-owning alias retained allocator storage"
        );
        let (b, owning_b) = NativeCudaGraph::capture(&context, || Ok(&x * 5.)).unwrap();
        assert_eq!(
            owning_b.data_ptr(),
            address,
            "pool did not reuse released allocation"
        );
        let alias_b = unsafe { b.non_owning_output(&owning_b) }.unwrap();
        drop(owning_b);
        a.replay().unwrap();
        let retained = alias.copy();
        for (graph, view, expected) in [
            (&b, &alias_b, 5.),
            (&a, &alias, 2.),
            (&a, &alias, 2.),
            (&b, &alias_b, 5.),
        ] {
            graph.replay().unwrap();
            assert_eq!(view.double_value(&[0, 0]), expected);
        }
        assert_eq!(retained.double_value(&[0, 0]), 2.);
        Cuda::synchronize(0);
        drop((alias, alias_b, retained));
        drop((a, b));
        // No output owner remains. Graph-private allocations (e.g. RNG state)
        // can also disappear at graph teardown, so the total may fall further.
        assert!(context.memory_bytes().unwrap().0 <= after);
    }

    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn shared_pool_replays_retained_outputs_and_recovers_closure_failure() {
        let context = NativeGraphCaptureContext::new(0).unwrap();
        let weak = Rc::downgrade(&context);
        let x = Tensor::ones([256, 1024], (Kind::Float, Device::Cuda(0)));
        let calls = Cell::new(0);
        // Three successful warmups, then a Rust error during real capture.
        let failed = NativeCudaGraph::capture(&context, || {
            let temporary = &x * 3.;
            calls.set(calls.get() + 1);
            if calls.get() == 4 {
                return Err(ModelRunnerError::Model(
                    "injected capture closure failure".into(),
                ));
            }
            Ok(temporary)
        });
        assert!(failed.is_err());
        context.check_execution_stream().unwrap();
        let (a, output_a) =
            NativeCudaGraph::capture(&context, || Ok((&x * 2.).repeat([1, 4]))).unwrap();
        let (b, output_b) =
            NativeCudaGraph::capture(&context, || Ok((&x * 5.).repeat([1, 2]))).unwrap();
        // Check before any subsequent graph can overwrite temporary storage.
        a.replay().unwrap();
        let retained = output_a.copy();
        for graph in [&b, &a, &a, &b, &b, &a] {
            graph.replay().unwrap();
        }
        assert_eq!(output_a.double_value(&[0, 0]), 2.);
        assert_eq!(output_b.double_value(&[0, 0]), 5.);
        assert_eq!(retained.double_value(&[0, 0]), 2.);
        // A graph cannot be launched on another catalog's selected capture stream.
        let other = NativeGraphCaptureContext::new(0).unwrap();
        let unfinished = NativeCudaGraph::create(&other).unwrap();
        assert!(a.replay().is_err());
        drop(unfinished);
        a.replay().unwrap();
        Cuda::synchronize(0);
        drop(context);
        assert!(
            weak.upgrade().is_some(),
            "native graphs retain the shared context"
        );
        drop((a, b, output_a, output_b));
        assert!(weak.upgrade().is_none());
        let fresh = NativeGraphCaptureContext::new(0).unwrap();
        let (graph, output) = NativeCudaGraph::capture(&fresh, || Ok(&x + 7.)).unwrap();
        graph.replay().unwrap();
        assert_eq!(output.double_value(&[0, 0]), 8.);
        Cuda::synchronize(0);
    }

    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn shared_pool_invalid_cuda_capture_restores_stream_and_requires_new_context() {
        let context = NativeGraphCaptureContext::new(0).unwrap();
        let x = Tensor::ones([1], (Kind::Float, Device::Cuda(0)));
        let calls = Cell::new(0);
        let failed = NativeCudaGraph::capture(&context, || {
            calls.set(calls.get() + 1);
            let output = &x + 1.;
            if calls.get() == 4 {
                // A synchronous device->host read is forbidden inside capture.
                let _ = output.f_double_value(&[0])?;
            }
            Ok(output)
        });
        assert!(failed.is_err());
        assert!(context.check_execution_stream().is_err());
        drop(context);
        let fresh = NativeGraphCaptureContext::new(0).unwrap();
        fresh.check_execution_stream().unwrap();
        let (graph, output) = NativeCudaGraph::capture(&fresh, || Ok(&x + 3.)).unwrap();
        graph.replay().unwrap();
        assert_eq!(output.double_value(&[0]), 4.);
        Cuda::synchronize(0);
    }
}
