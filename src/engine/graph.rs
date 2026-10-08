//! CUDA decode graph capture and replay with fixed input buffers.

use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use tch::{Cuda, Device, Kind, Tensor};

use super::{
    AttentionMetadata, Batch, BatchPhase, DecodeGraphState, ModelRunner, ModelRunnerError,
    ServerArgs,
};
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
pub struct GraphRunner {
    graphs: BTreeMap<usize, CapturedGraph>,
    pad_pages: Tensor,
    pad_locations: Tensor,
}

impl GraphRunner {
    pub fn capture(
        runner: &ModelRunner,
        args: &ServerArgs,
        _pool: Rc<RefCell<KVCachePool>>,
    ) -> Result<Option<Self>> {
        if crate::logging::step_timing_enabled() {
            tracing::info!("CUDA Graph disabled for synchronized SGLANG_PROFILE_STEPS profiling");
            return Ok(None);
        }
        if !runner.supports_cuda_graph() {
            tracing::info!(
                "selected model does not support CUDA Graph capture; using eager decode"
            );
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
            .min(args.max_running_req);
        if limit == 0 {
            tracing::info!("CUDA Graph capture disabled; using eager decode");
            return Ok(None);
        }
        if !NativeCudaGraph::available() {
            tracing::warn!("CUDA Graph bridge unavailable; using eager decode");
            return Ok(None);
        }
        let pad_pages = Tensor::zeros([limit as i64], (Kind::Int, runner.device()));
        let pad_locations = Tensor::zeros_like(&pad_pages);
        let mut graph_runner = Self {
            graphs: BTreeMap::new(),
            pad_pages,
            pad_locations,
        };
        let mut sizes = vec![1, 2, 4, 8, 16, 32, 64, 128, 256];
        sizes.retain(|&size| size <= limit);
        if sizes.last().copied() != Some(limit) {
            sizes.push(limit);
        }
        for size in sizes {
            tracing::info!(size, "capturing CUDA decode graph");
            match graph_runner.capture_one(runner, args, device_index, size) {
                Ok(graph) => {
                    graph_runner.graphs.insert(size, graph);
                }
                Err(error) => {
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
        let native = NativeCudaGraph::create(device_index)?;
        for _ in 0..3 {
            let _ = runner.run_model(&input_ids, &positions, Some(&metadata), None)?;
        }
        Cuda::synchronize(device_index as i64);
        native.begin()?;
        let output = match runner.run_model(&input_ids, &positions, Some(&metadata), None) {
            Ok(output) => output,
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

    pub fn replay(&self, batch: &Batch) -> Result<Option<Tensor>> {
        let size = batch.input_ids.size()[0] as usize;
        let Some((_, graph)) = self.graphs.range(size..).next() else {
            return Ok(None);
        };
        let rows = size as i64;
        graph.input_ids.narrow(0, 0, rows).copy_(&batch.input_ids);
        graph.positions.narrow(0, 0, rows).copy_(&batch.positions);
        let source = batch.attention_metadata.as_ref().ok_or_else(|| {
            ModelRunnerError::Model("decode batch requires attention metadata".to_owned())
        })?;
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
            backend.update()?;
        }
        graph.native.replay()?;
        Ok(Some(graph.output.narrow(0, 0, rows).copy()))
    }
}

impl Drop for GraphRunner {
    fn drop(&mut self) {
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

pub(crate) struct NativeCudaGraph {
    #[cfg(has_cuda_graph)]
    ptr: *mut std::ffi::c_void,
}

impl NativeCudaGraph {
    pub(crate) fn available() -> bool {
        cfg!(has_cuda_graph)
    }

    /// Warm and capture a GPU-only segment on a side stream. The caller owns
    /// all fixed inputs and outputs until this graph is destroyed.
    pub(crate) fn capture<T>(
        device: usize,
        mut forward: impl FnMut() -> Result<T>,
    ) -> Result<(Self, T)> {
        Cuda::synchronize(device as i64);
        let graph = Self::create(device)?;
        for _ in 0..3 {
            let _ = forward()?;
        }
        Cuda::synchronize(device as i64);
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

    fn create(device: usize) -> Result<Self> {
        #[cfg(has_cuda_graph)]
        {
            let ptr = unsafe { sglang_graph_create(device as i32) };
            if ptr.is_null() {
                return Err(native_error());
            }
            return Ok(Self { ptr });
        }
        #[cfg(not(has_cuda_graph))]
        {
            let _ = device;
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
    fn sglang_graph_create(device: i32) -> *mut std::ffi::c_void;
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
        engine::{ModelExecutor, kvcache::KVCacheLayout},
        models::attention::{Attention, AttentionSpec, BaseAttention},
    };

    struct DecodeModel {
        attention: Attention,
        cache: RefCell<BaseAttention>,
    }

    impl ModelExecutor for DecodeModel {
        fn prepare_decode_graph(
            &self,
            metadata: &AttentionMetadata,
        ) -> Result<Option<Box<dyn DecodeGraphState>>> {
            self.attention.prepare_decode_graph(metadata)
        }

        fn forward(
            &self,
            ids: &Tensor,
            positions: &Tensor,
            metadata: Option<&AttentionMetadata>,
            _: Option<&Tensor>,
        ) -> Result<Tensor> {
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
        }
    }

    #[test]
    fn flashinfer_graph_replays_dynamic_pages_lengths_and_padded_batches() {
        if !Cuda::is_available() {
            return;
        }
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
        let graph = GraphRunner::capture(&runner, &args, pool.clone())
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
            let batch = Batch::decode(
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
            let replay = graph.replay(&batch).unwrap();
            assert_eq!(replay.is_some(), rows <= 4);
            let actual = replay.unwrap_or_else(|| runner.forward(&batch).unwrap());
            let expected = runner
                .run_model(
                    &batch.input_ids,
                    &batch.positions,
                    batch.attention_metadata.as_ref(),
                    None,
                )
                .unwrap();
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
        let graph = GraphRunner::capture(&runner, &args, pool.clone())
            .unwrap()
            .unwrap();
        assert_eq!(graph.graphs.len(), 3);
        drop(graph);
        args.cuda_graph_bs = Some(0);
        assert!(
            GraphRunner::capture(&runner, &args, pool.clone())
                .unwrap()
                .is_none()
        );
        assert_eq!(pool.borrow().free_count(), free);
    }
}
