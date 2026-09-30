//! CUDA decode graph capture and replay with fixed input buffers.

use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use tch::{Cuda, Device, Kind, Tensor};

use super::{AttentionMetadata, Batch, BatchPhase, ModelRunner, ModelRunnerError, ServerArgs};
use crate::engine::kvcache::{BaseCacheHandle, KVCachePool};

type Result<T> = std::result::Result<T, ModelRunnerError>;

struct CapturedGraph {
    native: NativeCudaGraph,
    input_ids: Tensor,
    positions: Tensor,
    metadata: AttentionMetadata,
    output: Tensor,
}

/// Holds one graph per selected batch size and one private KV page for padding.
pub struct GraphRunner {
    graphs: BTreeMap<usize, CapturedGraph>,
    pool: Rc<RefCell<KVCachePool>>,
    pad_handle: BaseCacheHandle,
    pad_page: i64,
    pad_loc: i64,
}

impl GraphRunner {
    pub fn capture(
        runner: &ModelRunner,
        args: &ServerArgs,
        pool: Rc<RefCell<KVCachePool>>,
    ) -> Result<Option<Self>> {
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
        if pool.borrow().free_count() <= 1 {
            tracing::warn!("KV cache has no spare padding page; using eager decode");
            return Ok(None);
        }
        let pad_handle = match pool.borrow_mut().alloc(1) {
            Ok(handle) => handle,
            Err(error) => {
                tracing::warn!(%error, "cannot reserve CUDA Graph padding page; using eager decode");
                return Ok(None);
            }
        };
        let pad_page = pad_handle.page_ids[0] as i64;
        let pad_loc = pad_page * args.page_size as i64;
        let mut graph_runner = Self {
            graphs: BTreeMap::new(),
            pool,
            pad_handle,
            pad_page,
            pad_loc,
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
        let metadata = AttentionMetadata {
            forward_mode: BatchPhase::Decode,
            write_loc: Some(Tensor::full([rows], self.pad_loc, (Kind::Int, device))),
            cu_seqlens_q: None,
            prefix_lens: None,
            block_table: Some(Tensor::full(
                [rows, args.max_seq_len.div_ceil(args.page_size) as i64],
                self.pad_page,
                (Kind::Int, device),
            )),
            req_to_token: Some(Tensor::full(
                [rows, args.max_seq_len as i64],
                self.pad_loc,
                (Kind::Int, device),
            )),
            cache_seqlens: Some(Tensor::ones([rows], (Kind::Int, device))),
            max_seqlen: Some(args.max_seq_len),
        };
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
        copy_field(&graph.metadata.req_to_token, &source.req_to_token, rows)?;
        if size < graph.input_ids.size()[0] as usize {
            let start = rows;
            let count = graph.input_ids.size()[0] - rows;
            let _ = graph
                .metadata
                .write_loc
                .as_ref()
                .unwrap()
                .narrow(0, start, count)
                .fill_(self.pad_loc);
            let _ = graph
                .metadata
                .cache_seqlens
                .as_ref()
                .unwrap()
                .narrow(0, start, count)
                .fill_(1);
            let _ = graph
                .metadata
                .block_table
                .as_ref()
                .unwrap()
                .narrow(0, start, count)
                .fill_(self.pad_page);
            let _ = graph
                .metadata
                .req_to_token
                .as_ref()
                .unwrap()
                .narrow(0, start, count)
                .fill_(self.pad_loc);
        }
        graph.native.replay()?;
        Ok(Some(graph.output.narrow(0, 0, rows).copy()))
    }
}

impl Drop for GraphRunner {
    fn drop(&mut self) {
        self.graphs.clear();
        self.pool.borrow_mut().free(&mut self.pad_handle);
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

struct NativeCudaGraph {
    #[cfg(has_cuda_graph)]
    ptr: *mut std::ffi::c_void,
}

impl NativeCudaGraph {
    fn available() -> bool {
        cfg!(has_cuda_graph)
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

    fn replay(&self) -> Result<()> {
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
