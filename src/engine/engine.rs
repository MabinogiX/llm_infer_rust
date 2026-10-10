//! Engine orchestration for the Rust migration.
//!
//! Owns model execution, sampling, configuration, and the paged KV cache.
//! The scheduler shares the pool's page allocator with the cache managers.

use std::{
    cell::{Ref, RefCell, RefMut},
    fmt,
    path::{Path, PathBuf},
    rc::Rc,
};

use tch::{Cuda, Device, Kind, Tensor};

use super::kvcache::{
    KVCacheAllocationConfig, KVCacheAllocator, KVCacheError, KVCacheModelConfig, KVCachePool,
    KVCacheServerConfig, ModelCacheSpec,
};
use super::sampling::{Sampler, SamplingError, SamplingParams};
use super::{Batch, ModelFactory, ModelRunner, ModelRunnerError, load_hf_safetensors};

/// Server-side settings consumed by [`Engine`].
///
/// This is the Rust counterpart of mini-sglang's `ServerArgs`; fields that
/// only affect HTTP serving remain outside the engine.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerArgs {
    pub model_path: PathBuf,
    pub tp_size: usize,
    pub memory_ratio: f64,
    pub max_running_req: usize,
    /// Maximum captured decode batch size. None uses max_running_req; zero disables capture.
    pub cuda_graph_bs: Option<usize>,
    /// Maximum aggregate token bucket for segmented prefill graphs; zero disables.
    pub prefill_cuda_graph_max_tokens: usize,
    pub max_seq_len: usize,
    pub page_size: usize,
    pub dtype: String,
    pub device: String,
    pub attention_backend: String,
    pub trust_remote_code: bool,
}

impl ServerArgs {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            tp_size: 1,
            memory_ratio: 0.9,
            max_running_req: 256,
            cuda_graph_bs: None,
            prefill_cuda_graph_max_tokens: 2048,
            max_seq_len: 8192,
            page_size: 16,
            dtype: "auto".to_owned(),
            device: "auto".to_owned(),
            attention_backend: "fa".to_owned(),
            trust_remote_code: false,
        }
    }
}

/// Architecture-independent information consumed by the current paged-KV runtime.
/// Model computation fields belong to the concrete model's configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeModelConfig {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub checkpoint_kind: Kind,
}

impl Default for RuntimeModelConfig {
    fn default() -> Self {
        Self {
            num_layers: 0,
            num_kv_heads: 0,
            head_dim: 0,
            vocab_size: 0,
            max_position_embeddings: 8192,
            checkpoint_kind: Kind::Float,
        }
    }
}

/// Engine construction and lifecycle failures.
#[derive(Debug)]
pub enum EngineError {
    InvalidArgument(String),
    ModelPathDoesNotExist(PathBuf),
    ModelPathIsNotDirectory(PathBuf),
    MissingModelConfig(PathBuf),
    InvalidModelConfig { path: PathBuf, message: String },
    KVCache(KVCacheError),
    ModelRunner(ModelRunnerError),
    Sampling(SamplingError),
    ModelRunnerNotAttached,
    Released,
    NotImplemented(&'static str),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(message) => write!(f, "invalid engine argument: {message}"),
            Self::ModelPathDoesNotExist(path) => write!(
                f,
                "模型路径不存在: {}。请传入包含 config.json 的目录。",
                path.display()
            ),
            Self::ModelPathIsNotDirectory(path) => {
                write!(f, "模型路径不是目录: {}", path.display())
            }
            Self::MissingModelConfig(path) => write!(
                f,
                "模型目录没有 config.json: {}。这不是 Hugging Face 模型目录。",
                path.display()
            ),
            Self::InvalidModelConfig { path, message } => {
                write!(f, "无效的模型配置 {}: {message}", path.display())
            }
            Self::KVCache(error) => write!(f, "KV cache 初始化失败: {error}"),
            Self::ModelRunner(error) => write!(f, "模型前向失败: {error}"),
            Self::Sampling(error) => write!(f, "采样失败: {error}"),
            Self::ModelRunnerNotAttached => write!(f, "Engine 尚未绑定 ModelRunner"),
            Self::Released => write!(f, "Engine 已清理，不能再使用"),
            Self::NotImplemented(feature) => write!(f, "未实现: {feature}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<KVCacheError> for EngineError {
    fn from(error: KVCacheError) -> Self {
        Self::KVCache(error)
    }
}

impl From<SamplingError> for EngineError {
    fn from(error: SamplingError) -> Self {
        Self::Sampling(error)
    }
}

impl From<ModelRunnerError> for EngineError {
    fn from(error: ModelRunnerError) -> Self {
        Self::ModelRunner(error)
    }
}

pub type Result<T> = std::result::Result<T, EngineError>;

/// Owns Rust-side engine state during the gradual migration.
pub struct Engine {
    server_args: ServerArgs,
    runtime_config: RuntimeModelConfig,
    tp_rank: usize,
    device: Device,
    kind: Kind,
    kv_cache_pool: Option<Rc<RefCell<KVCachePool>>>,
    model_runner: Option<ModelRunner>,
    sampler: Sampler,
}

impl Engine {
    /// Builds a CPU, float32 engine. Use [`Self::with_runtime`] to select a
    /// different libtorch device or element type.
    pub fn new(
        server_args: ServerArgs,
        runtime_config: RuntimeModelConfig,
        tp_rank: usize,
    ) -> Result<Self> {
        Self::with_runtime(
            server_args,
            runtime_config,
            tp_rank,
            Kind::Float,
            Device::Cpu,
        )
    }

    /// Loads model weights before sizing the KV cache against remaining GPU memory.
    pub fn load_for_serving(
        server_args: ServerArgs,
        runtime_config: RuntimeModelConfig,
        tp_rank: usize,
        factory: &dyn ModelFactory,
    ) -> Result<Self> {
        let (kind, device) =
            Self::validate_for_serving(&server_args, runtime_config, tp_rank, factory)?;
        let model = factory.create(kind, device, &server_args.attention_backend)?;
        let mut runner = ModelRunner::new(model, device);
        let weights = load_hf_safetensors(&server_args.model_path).map_err(|error| {
            EngineError::InvalidArgument(format!("Hugging Face 权重加载失败: {error}"))
        })?;
        runner.load_weights(weights)?;

        let spec = factory.cache_spec(runtime_config)?;
        let mut engine =
            Self::with_cache_spec(server_args, runtime_config, tp_rank, kind, device, spec)?;
        engine.attach_model_runner(runner)?;
        engine.capture_graphs()?;
        tracing::info!(?device, ?kind, "model loaded for inference");
        Ok(engine)
    }

    /// Resolve and validate the execution combination before startup allocates tensors.
    pub fn validate_for_serving(
        server_args: &ServerArgs,
        runtime_config: RuntimeModelConfig,
        tp_rank: usize,
        factory: &dyn ModelFactory,
    ) -> Result<(Kind, Device)> {
        validate_model_path(&server_args.model_path)?;
        validate_parallelism(server_args.tp_size, tp_rank)?;
        if server_args.tp_size > 1 {
            return Err(EngineError::NotImplemented("Rust 分布式张量并行初始化"));
        }
        validate_max_seq_len(&server_args, runtime_config)?;
        let device = resolve_device(&server_args.device, Cuda::is_available())?;
        let kind = resolve_kind(&server_args.dtype, runtime_config.checkpoint_kind, device)?;
        factory.validate_runtime(runtime_config, kind, device, &server_args.attention_backend)?;
        KVCacheAllocator::with_spec(
            allocation_config(server_args, runtime_config),
            factory.cache_spec(runtime_config)?,
        )?;
        Ok((kind, device))
    }

    /// Validates configuration and allocates the libtorch-backed KV cache.
    pub fn with_runtime(
        server_args: ServerArgs,
        runtime_config: RuntimeModelConfig,
        tp_rank: usize,
        kind: Kind,
        device: Device,
    ) -> Result<Self> {
        let spec = ModelCacheSpec::uniform(
            runtime_config.num_layers,
            runtime_config.num_kv_heads,
            runtime_config.head_dim,
        )?;
        Self::with_cache_spec(server_args, runtime_config, tp_rank, kind, device, spec)
    }

    pub fn with_cache_spec(
        server_args: ServerArgs,
        runtime_config: RuntimeModelConfig,
        tp_rank: usize,
        kind: Kind,
        device: Device,
        spec: ModelCacheSpec,
    ) -> Result<Self> {
        validate_model_path(&server_args.model_path)?;
        validate_parallelism(server_args.tp_size, tp_rank)?;
        if server_args.tp_size > 1 {
            return Err(EngineError::NotImplemented("Rust 分布式张量并行初始化"));
        }
        validate_max_seq_len(&server_args, runtime_config)?;
        let allocator =
            KVCacheAllocator::with_spec(allocation_config(&server_args, runtime_config), spec)?;
        let kv_cache_pool = allocator.allocate(kind, device, server_args.tp_size)?;

        Ok(Self {
            server_args,
            runtime_config,
            tp_rank,
            device,
            kind,
            kv_cache_pool: Some(Rc::new(RefCell::new(kv_cache_pool))),
            model_runner: None,
            sampler: Sampler,
        })
    }

    pub fn server_args(&self) -> &ServerArgs {
        &self.server_args
    }

    pub fn runtime_config(&self) -> RuntimeModelConfig {
        self.runtime_config
    }

    pub fn tp_rank(&self) -> usize {
        self.tp_rank
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn kv_cache_pool(&self) -> Result<Ref<'_, KVCachePool>> {
        Ok(self
            .kv_cache_pool
            .as_ref()
            .ok_or(EngineError::Released)?
            .borrow())
    }

    pub fn kv_cache_pool_mut(&mut self) -> Result<RefMut<'_, KVCachePool>> {
        Ok(self
            .kv_cache_pool
            .as_ref()
            .ok_or(EngineError::Released)?
            .borrow_mut())
    }

    /// Shares the pool's page allocator with the scheduler's cache manager.
    pub fn shared_kv_cache_pool(&self) -> Result<Rc<RefCell<KVCachePool>>> {
        self.kv_cache_pool.clone().ok_or(EngineError::Released)
    }

    /// Binds the Rust model and its KV cache views.
    pub fn attach_model_runner(&mut self, model_runner: ModelRunner) -> Result<()> {
        self.ensure_live()?;
        if model_runner.device() != self.device {
            return Err(EngineError::InvalidArgument(format!(
                "ModelRunner device ({:?}) must match Engine device ({:?})",
                model_runner.device(),
                self.device
            )));
        }
        let cache = self.kv_cache_pool()?.model_cache()?;
        let mut model_runner = model_runner;
        model_runner.bind_state_cache(cache)?;
        model_runner.set_kv_reserved_slot(0);
        self.model_runner = Some(model_runner);
        Ok(())
    }

    pub fn model_runner(&self) -> Result<&ModelRunner> {
        self.ensure_live()?;
        self.model_runner
            .as_ref()
            .ok_or(EngineError::ModelRunnerNotAttached)
    }

    /// Creates the selected Rust model and binds it to this Engine.
    pub fn build_model(&mut self, factory: &dyn ModelFactory) -> Result<()> {
        self.ensure_live()?;
        factory.validate_runtime(
            self.runtime_config,
            self.kind,
            self.device,
            &self.server_args.attention_backend,
        )?;
        if self.kv_cache_pool()?.spec() != &factory.cache_spec(self.runtime_config)? {
            return Err(EngineError::InvalidArgument(
                "model cache specification differs from allocated pool".into(),
            ));
        }
        let model = factory.create(self.kind, self.device, &self.server_args.attention_backend)?;
        self.attach_model_runner(ModelRunner::new(model, self.device))
    }

    /// Reads local Hugging Face safetensors and binds them to the attached model.
    pub fn load_model_weights(&mut self) -> Result<usize> {
        self.ensure_live()?;
        let weights = load_hf_safetensors(&self.server_args.model_path).map_err(|error| {
            EngineError::InvalidArgument(format!("Hugging Face 权重加载失败: {error}"))
        })?;
        let loaded = self
            .model_runner
            .as_mut()
            .ok_or(EngineError::ModelRunnerNotAttached)?
            .load_weights(weights)?;
        self.capture_graphs()?;
        Ok(loaded)
    }

    /// Capture decode only after the final weight tensors and KV views are bound.
    pub fn capture_graphs(&mut self) -> Result<()> {
        self.ensure_live()?;
        tracing::info!(
            device = ?self.device,
            max_batch_size = self.server_args.cuda_graph_bs.unwrap_or(self.server_args.max_running_req),
            "initializing CUDA Graph runners"
        );
        let pool = self.shared_kv_cache_pool()?;
        self.model_runner
            .as_mut()
            .ok_or(EngineError::ModelRunnerNotAttached)?
            .capture_graphs(&self.server_args, pool)?;
        Ok(())
    }

    /// Releases Rust-owned accelerator memory. Calling it repeatedly is safe.
    pub fn cleanup(&mut self) {
        if let Some(model_runner) = &mut self.model_runner {
            model_runner.clear_graphs();
        }
        self.model_runner.take();
        self.kv_cache_pool.take();
    }

    /// Executes a scheduler-prepared batch through the attached ModelRunner.
    pub fn forward(&self, batch: &Batch) -> Result<Tensor> {
        self.ensure_live()?;
        Ok(self
            .model_runner
            .as_ref()
            .ok_or(EngineError::ModelRunnerNotAttached)?
            .forward(batch)?)
    }

    /// Samples one token per logits row using request-aligned parameters.
    pub fn sample(&self, logits: &Tensor, params: &[SamplingParams]) -> Result<Vec<i64>> {
        self.ensure_live()?;
        Ok(self.sampler.sample_batch(logits, params)?)
    }

    fn ensure_live(&self) -> Result<()> {
        if self.kv_cache_pool.is_none() {
            return Err(EngineError::Released);
        }
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.cleanup();
    }
}

fn allocation_config(
    server_args: &ServerArgs,
    runtime_config: RuntimeModelConfig,
) -> KVCacheAllocationConfig {
    KVCacheAllocationConfig {
        server: KVCacheServerConfig {
            page_size: server_args.page_size,
            max_running_req: server_args.max_running_req,
            max_seq_len: server_args.max_seq_len,
            memory_ratio: server_args.memory_ratio,
        },
        model: KVCacheModelConfig {
            num_layers: runtime_config.num_layers,
            num_kv_heads: runtime_config.num_kv_heads,
            head_dim: runtime_config.head_dim,
        },
    }
}

/// Rejects a context window larger than the model configuration before allocating memory.
pub fn validate_max_seq_len(
    server_args: &ServerArgs,
    runtime_config: RuntimeModelConfig,
) -> Result<()> {
    if runtime_config.max_position_embeddings == 0 {
        return Err(EngineError::InvalidArgument(
            "模型配置 max_position_embeddings 必须大于 0".to_owned(),
        ));
    }
    if server_args.max_seq_len > runtime_config.max_position_embeddings {
        return Err(EngineError::InvalidArgument(format!(
            "--max-seq-len {} 超过模型配置的 max_position_embeddings={}；请降低 --max-seq-len",
            server_args.max_seq_len, runtime_config.max_position_embeddings
        )));
    }
    Ok(())
}

/// Fails early when `model_path` cannot be a local Hugging Face model directory.
pub fn validate_model_path(model_path: &Path) -> Result<()> {
    if !model_path.exists() {
        return Err(EngineError::ModelPathDoesNotExist(model_path.to_path_buf()));
    }
    if !model_path.is_dir() {
        return Err(EngineError::ModelPathIsNotDirectory(
            model_path.to_path_buf(),
        ));
    }
    if !model_path.join("config.json").is_file() {
        return Err(EngineError::MissingModelConfig(model_path.to_path_buf()));
    }
    Ok(())
}

fn resolve_kind(requested: &str, checkpoint_kind: Kind, device: Device) -> Result<Kind> {
    let kind = if requested == "auto" {
        checkpoint_kind
    } else {
        match requested {
            "bfloat16" => Kind::BFloat16,
            "float16" => Kind::Half,
            "float32" => Kind::Float,
            _ => {
                return Err(EngineError::InvalidArgument(format!(
                    "不支持的模型 dtype: {requested}"
                )));
            }
        }
    };
    if matches!(device, Device::Cpu) && kind != Kind::Float {
        tracing::warn!(?kind, "CPU 推理使用 float32，忽略模型的低精度 dtype");
        return Ok(Kind::Float);
    }
    Ok(kind)
}

fn resolve_device(requested: &str, cuda_available: bool) -> Result<Device> {
    match requested {
        "auto" if cuda_available => Ok(Device::Cuda(0)),
        "auto" | "cpu" => Ok(Device::Cpu),
        "cuda" if cuda_available => Ok(Device::Cuda(0)),
        "cuda" => Err(EngineError::InvalidArgument(
            "请求 CUDA 设备，但当前 libtorch 未检测到可用的 CUDA GPU".to_owned(),
        )),
        _ => Err(EngineError::InvalidArgument(format!(
            "无效的设备 {requested:?}；应为 auto、cpu 或 cuda"
        ))),
    }
}

fn validate_parallelism(tp_size: usize, tp_rank: usize) -> Result<()> {
    if tp_size == 0 {
        return Err(EngineError::InvalidArgument(
            "tp_size must be greater than zero".to_owned(),
        ));
    }
    if tp_rank >= tp_size {
        return Err(EngineError::InvalidArgument(format!(
            "tp_rank ({tp_rank}) must be smaller than tp_size ({tp_size})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::super::{AttentionMetadata, ModelExecutor, ModelFactory, ModelWeights};
    use super::*;

    static TEST_DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);

    fn model_dir() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after UNIX epoch")
            .as_nanos();
        let id = TEST_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("sglang-rust-engine-{nonce}-{id}"));
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("config.json"), "{}").unwrap();
        path
    }

    fn runtime_config() -> RuntimeModelConfig {
        RuntimeModelConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 1,
            max_position_embeddings: 4,
            ..Default::default()
        }
    }

    #[test]
    fn rejects_factory_runtime_mismatches_before_loading_or_binding() {
        use crate::models::{Qwen3Factory, qwen3::Qwen3Config};

        let path = model_dir();
        let factory = Qwen3Factory {
            config: Qwen3Config {
                hidden_size: 4,
                num_layers: 1,
                num_attention_heads: 2,
                num_kv_heads: 1,
                intermediate_size: 8,
                vocab_size: 8,
                head_dim: 2,
                max_position_embeddings: 4,
                ..Default::default()
            },
        };
        let expected = factory.config.runtime(Kind::Float);
        for field in [
            "num_layers",
            "num_kv_heads",
            "head_dim",
            "vocab_size",
            "context",
        ] {
            let mut runtime = expected;
            match field {
                "num_layers" => runtime.num_layers = 2,
                "num_kv_heads" => runtime.num_kv_heads = 2,
                "head_dim" => runtime.head_dim = 4,
                "vocab_size" => runtime.vocab_size = 16,
                "context" => runtime.max_position_embeddings = 8,
                _ => unreachable!(),
            }
            let mut args = ServerArgs::new(&path);
            args.device = "cpu".into();
            args.max_seq_len = runtime.max_position_embeddings;
            args.max_running_req = 1;
            args.page_size = 2;
            let error = Engine::load_for_serving(args.clone(), runtime, 0, &factory)
                .err()
                .expect("mismatch must fail before checkpoint loading");
            assert!(
                error.to_string().contains("does not match"),
                "{field}: {error}"
            );
            let mut engine = Engine::new(args, runtime, 0).unwrap();
            let error = engine
                .build_model(&factory)
                .expect_err("mismatch must fail before model binding");
            assert!(
                error.to_string().contains("does not match"),
                "{field}: {error}"
            );
            assert!(matches!(
                engine.model_runner(),
                Err(EngineError::ModelRunnerNotAttached)
            ));
        }
        let mut args = ServerArgs::new(&path);
        args.max_seq_len = 4;
        args.max_running_req = 1;
        args.page_size = 2;
        let mut engine = Engine::new(args, expected, 0).unwrap();
        engine.build_model(&factory).unwrap();
        assert!(engine.model_runner().is_ok());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn resolves_requested_device_without_silent_cuda_fallback() {
        assert_eq!(resolve_device("auto", false).unwrap(), Device::Cpu);
        assert_eq!(resolve_device("auto", true).unwrap(), Device::Cuda(0));
        assert_eq!(resolve_device("cpu", true).unwrap(), Device::Cpu);
        assert_eq!(resolve_device("cuda", true).unwrap(), Device::Cuda(0));
        assert!(resolve_device("cuda", false).is_err());
    }

    #[test]
    fn auto_dtype_uses_resolved_checkpoint_kind_and_cpu_fallback() {
        assert_eq!(
            resolve_kind("auto", Kind::BFloat16, Device::Cuda(0)).unwrap(),
            Kind::BFloat16
        );
        assert_eq!(
            resolve_kind("auto", Kind::BFloat16, Device::Cpu).unwrap(),
            Kind::Float
        );
        assert_eq!(
            resolve_kind("float32", Kind::BFloat16, Device::Cuda(0)).unwrap(),
            Kind::Float
        );
    }

    #[test]
    fn rejects_sequence_length_above_model_limit() {
        let model_dir = model_dir();
        let mut args = ServerArgs::new(&model_dir);
        args.max_running_req = 1;
        args.max_seq_len = 8;
        args.page_size = 2;

        let error = Engine::new(args, runtime_config(), 0).err().unwrap();
        assert!(error.to_string().contains("--max-seq-len 8"));
        assert!(error.to_string().contains("max_position_embeddings=4"));
        let mut at_limit = ServerArgs::new(&model_dir);
        at_limit.max_seq_len = 4;
        assert!(validate_max_seq_len(&at_limit, runtime_config()).is_ok());
        fs::remove_dir_all(model_dir).unwrap();
    }

    #[test]
    fn cleanup_releases_the_pool_idempotently() {
        let model_dir = model_dir();
        let mut args = ServerArgs::new(&model_dir);
        args.max_running_req = 1;
        args.max_seq_len = 2;
        args.page_size = 2;
        let mut engine = Engine::new(args, runtime_config(), 0).unwrap();

        engine.cleanup();
        engine.cleanup();
        assert!(matches!(engine.kv_cache_pool(), Err(EngineError::Released)));
        fs::remove_dir_all(model_dir).unwrap();
    }

    #[test]
    fn validates_model_directory_and_parallelism() {
        assert!(matches!(
            validate_model_path(Path::new("/definitely/not/a/model")),
            Err(EngineError::ModelPathDoesNotExist(_))
        ));
        assert!(matches!(
            validate_parallelism(2, 2),
            Err(EngineError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tensor_parallelism_is_explicitly_deferred() {
        let model_dir = model_dir();
        let mut args = ServerArgs::new(&model_dir);
        args.tp_size = 2;

        assert!(matches!(
            Engine::new(args, runtime_config(), 0),
            Err(EngineError::NotImplemented("Rust 分布式张量并行初始化"))
        ));
        fs::remove_dir_all(model_dir).unwrap();
    }

    #[test]
    fn delegates_sampling_to_the_rust_sampler() {
        let model_dir = model_dir();
        let mut args = ServerArgs::new(&model_dir);
        args.max_running_req = 1;
        args.max_seq_len = 2;
        args.page_size = 2;
        let engine = Engine::new(args, runtime_config(), 0).unwrap();
        let logits = Tensor::from_slice(&[0.1f32, 2.0, 0.2]).view([1, 3]);

        assert_eq!(
            engine
                .sample(&logits, &[SamplingParams::default()])
                .unwrap(),
            vec![1]
        );
        fs::remove_dir_all(model_dir).unwrap();
    }

    struct EchoModel;

    impl ModelExecutor for EchoModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _positions: &Tensor,
            _attention_metadata: Option<&AttentionMetadata>,
            _logits_indices: Option<&Tensor>,
        ) -> std::result::Result<Tensor, ModelRunnerError> {
            Ok(input_ids.shallow_clone())
        }
    }

    struct LoadingModel;

    impl ModelExecutor for LoadingModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _positions: &Tensor,
            _attention_metadata: Option<&AttentionMetadata>,
            _logits_indices: Option<&Tensor>,
        ) -> std::result::Result<Tensor, ModelRunnerError> {
            Ok(input_ids.shallow_clone())
        }

        fn load_weights(
            &mut self,
            weights: ModelWeights,
        ) -> std::result::Result<usize, ModelRunnerError> {
            Ok(weights.len())
        }
    }

    struct LoadingFactory;

    impl ModelFactory for LoadingFactory {
        fn cache_spec(
            &self,
            runtime: RuntimeModelConfig,
        ) -> std::result::Result<ModelCacheSpec, ModelRunnerError> {
            ModelCacheSpec::uniform(runtime.num_layers, runtime.num_kv_heads, runtime.head_dim)
                .map_err(|e| ModelRunnerError::Model(e.to_string()))
        }

        fn validate_runtime(
            &self,
            _runtime_config: RuntimeModelConfig,
            _kind: Kind,
            _device: Device,
            _attention_backend: &str,
        ) -> std::result::Result<(), ModelRunnerError> {
            // This test executor has no model geometry or attention backend.
            Ok(())
        }

        fn create(
            &self,
            _kind: Kind,
            _device: Device,
            _attention_backend: &str,
        ) -> std::result::Result<Box<dyn ModelExecutor>, ModelRunnerError> {
            Ok(Box::new(LoadingModel))
        }
    }

    #[test]
    fn delegates_forward_to_an_attached_model_runner() {
        let model_dir = model_dir();
        let mut args = ServerArgs::new(&model_dir);
        args.max_running_req = 1;
        args.max_seq_len = 2;
        args.page_size = 2;
        let mut engine = Engine::new(args, runtime_config(), 0).unwrap();
        engine
            .attach_model_runner(ModelRunner::new(Box::new(EchoModel), Device::Cpu))
            .unwrap();
        let batch = Batch::prefill(
            Tensor::from_slice(&[10i64, 11]),
            Tensor::from_slice(&[0i64, 1]),
            None,
            Tensor::from_slice(&[1i64]),
        );

        assert_eq!(
            Vec::<i64>::try_from(&engine.forward(&batch).unwrap()).unwrap(),
            vec![10, 11]
        );
        fs::remove_dir_all(model_dir).unwrap();
    }

    #[test]
    fn builds_a_model_and_loads_hugging_face_safetensors() {
        let model_dir = model_dir();
        let weight = Tensor::ones([2], (Kind::Float, Device::Cpu));
        Tensor::write_safetensors(
            &[("lm_head.weight", &weight)],
            model_dir.join("model.safetensors"),
        )
        .unwrap();
        let mut args = ServerArgs::new(&model_dir);
        args.max_running_req = 1;
        args.max_seq_len = 2;
        args.page_size = 2;
        let mut engine = Engine::new(args, runtime_config(), 0).unwrap();

        engine.build_model(&LoadingFactory).unwrap();
        assert_eq!(engine.load_model_weights().unwrap(), 1);
        fs::remove_dir_all(model_dir).unwrap();
    }

    #[test]
    fn serving_loader_builds_model_before_cache_on_cpu() {
        let model_dir = model_dir();
        let weight = Tensor::ones([2], (Kind::Float, Device::Cpu));
        Tensor::write_safetensors(
            &[("lm_head.weight", &weight)],
            model_dir.join("model.safetensors"),
        )
        .unwrap();
        let mut args = ServerArgs::new(&model_dir);
        args.device = "cpu".to_owned();
        args.max_running_req = 1;
        args.max_seq_len = 2;
        args.page_size = 2;

        let engine = Engine::load_for_serving(args, runtime_config(), 0, &LoadingFactory).unwrap();
        assert_eq!(engine.device(), Device::Cpu);
        assert!(engine.model_runner().is_ok());
        let (k_cache, _) = engine.kv_cache_pool().unwrap().get_all_kv_cache().unwrap();
        assert_eq!(k_cache.device(), Device::Cpu);
        fs::remove_dir_all(model_dir).unwrap();
    }
}
