//! Model execution over scheduler-prepared batches.

use std::{cell::RefCell, fmt, rc::Rc};

use tch::{Device, TchError, Tensor, no_grad};

use super::{
    ModelWeights, ServerArgs,
    graph::DecodeGraphRunner,
    kvcache::{KVCachePool, ModelKvCache},
};

use super::{
    ForwardBatch, ForwardOutput, GraphCapabilities, GraphSupport, PrefillGraphProgram,
    segmented_graph::SegmentedGraphRunner,
};

/// Identifies the scheduler phase that produced a [`Batch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchPhase {
    Prefill,
    Decode,
}

/// Per-batch attention tensors passed through the model stack.
///
/// The fields mirror mini-sglang's `AttentionMetadata`. Models consume the
/// page-table fields to perform eager paged-KV reads and writes.
#[derive(Debug)]
pub struct AttentionMetadata {
    pub forward_mode: BatchPhase,
    pub write_loc: Option<Tensor>,
    pub cu_seqlens_q: Option<Tensor>,
    pub prefix_lens: Option<Tensor>,
    pub block_table: Option<Tensor>,
    pub req_to_token: Option<Tensor>,
    pub cache_seqlens: Option<Tensor>,
    pub max_seqlen: Option<usize>,
}

/// A scheduler-prepared model invocation.
///
/// Scheduler and `BatchContext` migration will construct these tensors. The
/// runner intentionally does not create request/page-table metadata itself.
#[derive(Debug)]
pub struct Batch {
    pub phase: BatchPhase,
    pub input_ids: Tensor,
    pub positions: Tensor,
    pub attention_metadata: Option<AttentionMetadata>,
    /// Last uncached-token positions for prefill LM-head gathering.
    pub logits_indices: Option<Tensor>,
}

impl Batch {
    pub fn prefill(
        input_ids: Tensor,
        positions: Tensor,
        attention_metadata: Option<AttentionMetadata>,
        logits_indices: Tensor,
    ) -> Self {
        Self {
            phase: BatchPhase::Prefill,
            input_ids,
            positions,
            attention_metadata,
            logits_indices: Some(logits_indices),
        }
    }

    pub fn decode(
        input_ids: Tensor,
        positions: Tensor,
        attention_metadata: Option<AttentionMetadata>,
    ) -> Self {
        Self {
            phase: BatchPhase::Decode,
            input_ids,
            positions,
            attention_metadata,
            logits_indices: None,
        }
    }
}

/// Owns address-stable backend planning buffers for one captured decode batch.
/// Planning updates happen before replay, outside stream capture.
pub trait DecodeGraphState {
    /// Refresh plans and metadata outside stream capture, before every replay.
    fn prepare_replay(&self) -> Result<()>;
    fn needs_req_to_token(&self) -> bool {
        true
    }
}

/// Model execution and cache-binding interface implemented by each architecture.
pub trait ModelExecutor {
    fn set_kv_reserved_slot(&mut self, _slot: i64) {}
    /// Evaluated only after dtype/backend and full-history cache binding.
    /// Each forward mode is opt-in, with explicit shape and padding limits.
    fn graph_capabilities(&self) -> GraphCapabilities {
        GraphCapabilities::default()
    }
    fn prefill_graph_program(&self) -> Option<&dyn PrefillGraphProgram> {
        None
    }

    fn prepare_decode_graph(
        &self,
        _metadata: &AttentionMetadata,
    ) -> Result<Option<Box<dyn DecodeGraphState>>> {
        Ok(None)
    }

    fn forward(&self, batch: &ForwardBatch<'_>) -> Result<ForwardOutput>;

    /// Receives native Hugging Face checkpoint tensors after model assembly.
    /// Concrete model architectures override this when their parameter naming
    /// and tensor-parallel sharding rules have been migrated.
    fn load_weights(&mut self, _weights: ModelWeights) -> Result<usize> {
        Err(ModelRunnerError::Model(
            "该 Rust 模型尚未实现 Hugging Face 权重绑定".to_owned(),
        ))
    }

    /// Bind stable, layer-local K/V views in model order, before graph capture.
    /// Cache-less executors may keep the default no-op implementation.
    fn bind_state_cache(&mut self, _cache: ModelKvCache) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
pub enum ModelRunnerError {
    InputPositionLengthMismatch { input_ids: usize, positions: usize },
    TensorOnWrongDevice { expected: Device, actual: Device },
    MissingPrefillLogitsIndices,
    Model(String),
    Torch(TchError),
}

impl fmt::Display for ModelRunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputPositionLengthMismatch {
                input_ids,
                positions,
            } => write!(
                f,
                "input_ids 数量 ({input_ids}) 必须与 positions 数量 ({positions}) 相同"
            ),
            Self::TensorOnWrongDevice { expected, actual } => {
                write!(f, "Batch tensor 位于 {actual:?}，期望位于 {expected:?}")
            }
            Self::MissingPrefillLogitsIndices => {
                write!(f, "prefill Batch 必须提供 logits_indices")
            }
            Self::Model(message) => write!(f, "模型执行失败: {message}"),
            Self::Torch(error) => write!(f, "libtorch error: {error}"),
        }
    }
}

impl std::error::Error for ModelRunnerError {}

impl From<TchError> for ModelRunnerError {
    fn from(error: TchError) -> Self {
        Self::Torch(error)
    }
}

pub type Result<T> = std::result::Result<T, ModelRunnerError>;

/// Owns model execution and both decode and segmented prefill graph lifecycles.
pub struct ModelRunner {
    // Graphs must release their backend state before model/cache ownership.
    decode_graph_runner: Option<DecodeGraphRunner>,
    pub(crate) prefill_graph_runner: Option<SegmentedGraphRunner>,
    model: Box<dyn ModelExecutor>,
    device: Device,
}

impl ModelRunner {
    pub fn new(model: Box<dyn ModelExecutor>, device: Device) -> Self {
        Self {
            model,
            device,
            decode_graph_runner: None,
            prefill_graph_runner: None,
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn graph_capabilities(&self) -> GraphCapabilities {
        self.model.graph_capabilities()
    }

    /// Hands loaded Hugging Face tensors to the concrete model architecture.
    pub fn load_weights(&mut self, weights: ModelWeights) -> Result<usize> {
        self.clear_graphs();
        self.model.load_weights(weights)
    }

    pub fn set_kv_reserved_slot(&mut self, slot: i64) {
        self.clear_graphs();
        self.model.set_kv_reserved_slot(slot);
    }

    pub fn bind_state_cache(&mut self, cache: ModelKvCache) -> Result<()> {
        self.clear_graphs();
        self.model.bind_state_cache(cache)
    }

    pub fn capture_graphs(
        &mut self,
        args: &ServerArgs,
        pool: Rc<RefCell<KVCachePool>>,
    ) -> Result<()> {
        self.clear_graphs();
        let decode = DecodeGraphRunner::capture(self, args, pool)?;
        let mut prefill = None;
        let capabilities = self.model.graph_capabilities();
        if args.prefill_cuda_graph_max_tokens > 0
            && matches!(self.device, Device::Cuda(_))
            && !crate::logging::step_timing_enabled()
            && super::NativeCudaGraph::available()
        {
            match capabilities.segmented_prefill {
                GraphSupport::Unsupported(reason) => {
                    tracing::info!(reason, "segmented prefill graphs unavailable; using eager")
                }
                GraphSupport::Supported(limits) => {
                    let program = self.model.prefill_graph_program().ok_or_else(|| {
                        ModelRunnerError::Model(
                            "model declares segmented graphs without a computation program".into(),
                        )
                    })?;
                    if args.max_seq_len > limits.max_context_len {
                        tracing::info!(
                            "segmented prefill context exceeds model capability; using eager"
                        );
                    } else {
                        prefill = Some(no_grad(|| {
                            SegmentedGraphRunner::capture(
                                program,
                                args.prefill_cuda_graph_max_tokens,
                                limits,
                            )
                        })?);
                    }
                }
            }
        }
        self.decode_graph_runner = decode;
        self.prefill_graph_runner = prefill;
        Ok(())
    }

    pub(crate) fn prepare_decode_graph(
        &self,
        metadata: &AttentionMetadata,
    ) -> Result<Option<Box<dyn DecodeGraphState>>> {
        self.model.prepare_decode_graph(metadata)
    }

    /// Eager/capture model execution uses the same typed input and output.
    pub fn run_model(&self, batch: &ForwardBatch<'_>) -> Result<ForwardOutput> {
        batch.validate(self.device)?;
        no_grad(|| self.model.forward(batch))
    }

    pub fn forward(&self, batch: &Batch) -> Result<ForwardOutput> {
        let forward = ForwardBatch::from_scheduler(batch)?;
        // Validation precedes bucket lookup, buffer copies and backend planning.
        forward.validate(self.device)?;
        if forward.phase() == BatchPhase::Decode {
            if let Some(graph) = &self.decode_graph_runner {
                if let Some(output) = graph.replay(forward)? {
                    return Ok(output);
                }
            }
        } else if let Some(graph) = &self.prefill_graph_runner {
            let program = self.model.prefill_graph_program().ok_or_else(|| {
                ModelRunnerError::Model("prefill graph program disappeared after capture".into())
            })?;
            if let Some(output) = no_grad(|| graph.replay(program, forward))? {
                return Ok(output);
            }
        }
        no_grad(|| self.model.forward(&forward))
    }

    pub fn clear_graphs(&mut self) {
        self.decode_graph_runner = None;
        self.prefill_graph_runner = None;
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use super::*;

    struct IndicesModel;

    impl ModelExecutor for IndicesModel {
        fn forward(
            &self,
            batch: &crate::engine::ForwardBatch<'_>,
        ) -> std::result::Result<crate::engine::ForwardOutput, ModelRunnerError> {
            let input_ids = batch.input_ids;
            let _positions = batch.positions;
            let _attention_metadata = batch.attention;
            let logits_indices = batch.logits_indices();
            (|| -> std::result::Result<Tensor, ModelRunnerError> {
                Ok(logits_indices
                    .map(Tensor::shallow_clone)
                    .unwrap_or_else(|| {
                        Tensor::zeros([input_ids.numel() as i64], (Kind::Int64, Device::Cpu))
                    }))
            })()
            .map(crate::engine::ForwardOutput::new)
        }
    }

    #[test]
    fn invalid_execution_inputs_never_enter_model_computation() {
        struct MustNotRun;
        impl ModelExecutor for MustNotRun {
            fn forward(&self, _: &ForwardBatch<'_>) -> Result<ForwardOutput> {
                panic!("invalid batch reached model computation")
            }
        }
        let runner = ModelRunner::new(Box::new(MustNotRun), Device::Cpu);
        let batch = Batch::decode(
            Tensor::ones([1], (Kind::Float, Device::Cpu)),
            Tensor::zeros([1], (Kind::Int64, Device::Cpu)),
            None,
        );
        assert!(runner.forward(&batch).is_err());
    }

    #[test]
    fn prefill_passes_logits_indices_to_the_model() {
        let runner = ModelRunner::new(Box::new(IndicesModel), Device::Cpu);
        let batch = Batch::prefill(
            Tensor::from_slice(&[10i64, 11, 12]),
            Tensor::from_slice(&[0i64, 1, 2]),
            None,
            Tensor::from_slice(&[2i64]),
        );

        assert_eq!(
            Vec::<i64>::try_from(&runner.forward(&batch).unwrap().logits).unwrap(),
            vec![2]
        );
    }

    #[test]
    fn decode_does_not_pass_prefill_indices() {
        let runner = ModelRunner::new(Box::new(IndicesModel), Device::Cpu);
        let batch = Batch::decode(
            Tensor::from_slice(&[10i64, 11]),
            Tensor::from_slice(&[3i64, 4]),
            None,
        );

        assert_eq!(
            Vec::<i64>::try_from(&runner.forward(&batch).unwrap().logits).unwrap(),
            vec![0, 0]
        );
    }

    #[test]
    fn rejects_mismatched_input_and_position_lengths() {
        let runner = ModelRunner::new(Box::new(IndicesModel), Device::Cpu);
        let batch = Batch::decode(
            Tensor::from_slice(&[10i64, 11]),
            Tensor::from_slice(&[3i64]),
            None,
        );

        assert!(matches!(
            runner.forward(&batch),
            Err(ModelRunnerError::InputPositionLengthMismatch { .. })
        ));
    }
}
