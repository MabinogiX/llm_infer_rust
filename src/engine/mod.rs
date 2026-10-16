mod batch_context;
mod engine;
mod forward;
mod graph;
mod graph_support;
pub mod kvcache;
pub(crate) mod metadata_transfer;
mod model_loader;
mod model_runner;
mod sampling;
pub(crate) mod segmented_graph;

pub(crate) use batch_context::BatchRequestRef;
pub use batch_context::{BatchContext, BatchContextError, BatchRequest, MixedDecodeRequest};
pub use engine::{
    Engine, EngineError, Result, RuntimeModelConfig, ServerArgs, validate_max_seq_len,
    validate_model_path,
};
pub(crate) use metadata_transfer::InputReadFence;
pub use model_loader::{ModelFactory, ModelLoadError, ModelWeights, load_hf_safetensors};
pub use model_runner::{
    AttentionMetadata, Batch, BatchPhase, DecodeGraphState, DecodeHostMetadata,
    MetadataRequirements, ModelExecutor, ModelRunner, ModelRunnerError,
};
pub(crate) use sampling::{CompletionFence, PendingSample};
pub use sampling::{Sampler, SamplingError, SamplingParams};

pub(crate) use graph::NativeCudaGraph;
#[cfg(all(test, has_cuda_graph))]
pub(crate) use graph::NativeGraphCaptureContext;

pub use forward::{ForwardBatch, ForwardMode, ForwardOutput};
pub use graph_support::{GraphCapabilities, GraphLimits, GraphPadding, GraphSupport};
pub use segmented_graph::{
    PrefillGraphInputs, PrefillGraphOutputLifetime, PrefillGraphProgram, PrefillGraphReplay,
};
