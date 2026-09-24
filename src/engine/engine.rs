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

use tch::{Device, Kind, Tensor};

use super::kvcache::{
    KVCacheAllocationConfig, KVCacheAllocator, KVCacheError, KVCacheModelConfig, KVCachePool,
    KVCacheServerConfig,
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
    pub max_seq_len: usize,
    pub page_size: usize,
    pub dtype: String,
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
            max_seq_len: 8192,
            page_size: 16,
            dtype: "auto".to_owned(),
            attention_backend: "pt".to_owned(),
            trust_remote_code: false,
        }
    }
}

/// Model architecture values required to size the KV cache.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelArgs {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub qk_norm: bool,
}

impl Default for ModelArgs {
    fn default() -> Self {
        Self {
            hidden_size: 0,
            num_layers: 0,
            num_attention_heads: 0,
            num_kv_heads: 0,
            intermediate_size: 0,
            vocab_size: 0,
            head_dim: 0,
            max_position_embeddings: 8192,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            qk_norm: false,
        }
    }
}

impl ModelArgs {
    /// Reads Qwen-compatible architecture fields from Hugging Face `config.json`.
    pub fn from_pretrained(model_path: impl AsRef<Path>) -> Result<Self> {
        let config_path = model_path.as_ref().join("config.json");
        let contents = std::fs::read_to_string(&config_path).map_err(|error| {
            EngineError::InvalidModelConfig {
                path: config_path.clone(),
                message: error.to_string(),
            }
        })?;
        let config: serde_json::Value =
            serde_json::from_str(&contents).map_err(|error| EngineError::InvalidModelConfig {
                path: config_path.clone(),
                message: error.to_string(),
            })?;
        let required_usize = |name: &str| -> Result<usize> {
            config
                .get(name)
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| EngineError::InvalidModelConfig {
                    path: config_path.clone(),
                    message: format!("missing or invalid integer field {name}"),
                })
        };
        let optional_usize = |name: &str, default: usize| -> Result<usize> {
            match config.get(name) {
                None | Some(serde_json::Value::Null) => Ok(default),
                Some(value) => value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| EngineError::InvalidModelConfig {
                        path: config_path.clone(),
                        message: format!("invalid integer field {name}"),
                    }),
            }
        };
        let hidden_size = required_usize("hidden_size")?;
        let num_attention_heads = required_usize("num_attention_heads")?;
        let head_dim = optional_usize(
            "head_dim",
            hidden_size
                .checked_div(num_attention_heads)
                .ok_or_else(|| EngineError::InvalidModelConfig {
                    path: config_path.clone(),
                    message: "num_attention_heads must be greater than zero".to_owned(),
                })?,
        )?;

        Ok(Self {
            hidden_size,
            num_layers: optional_usize("num_hidden_layers", 0)?,
            num_attention_heads,
            num_kv_heads: optional_usize("num_key_value_heads", num_attention_heads)?,
            intermediate_size: optional_usize("intermediate_size", 0)?,
            vocab_size: optional_usize("vocab_size", 0)?,
            head_dim,
            max_position_embeddings: optional_usize("max_position_embeddings", 8192)?,
            rope_theta: config
                .get("rope_theta")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(10_000.0),
            rms_norm_eps: config
                .get("rms_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-6),
            tie_word_embeddings: config
                .get("tie_word_embeddings")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            qk_norm: config
                .get("qk_norm")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        })
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
    model_args: ModelArgs,
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
    pub fn new(server_args: ServerArgs, model_args: ModelArgs, tp_rank: usize) -> Result<Self> {
        Self::with_runtime(server_args, model_args, tp_rank, Kind::Float, Device::Cpu)
    }

    /// Validates configuration and allocates the libtorch-backed KV cache.
    pub fn with_runtime(
        mut server_args: ServerArgs,
        model_args: ModelArgs,
        tp_rank: usize,
        kind: Kind,
        device: Device,
    ) -> Result<Self> {
        validate_model_path(&server_args.model_path)?;
        validate_parallelism(server_args.tp_size, tp_rank)?;
        if server_args.tp_size > 1 {
            return Err(EngineError::NotImplemented("Rust 分布式张量并行初始化"));
        }
        clamp_max_seq_len(&mut server_args, model_args);

        let allocator = KVCacheAllocator::new(KVCacheAllocationConfig {
            server: KVCacheServerConfig {
                page_size: server_args.page_size,
                max_running_req: server_args.max_running_req,
                max_seq_len: server_args.max_seq_len,
                memory_ratio: server_args.memory_ratio,
            },
            model: KVCacheModelConfig {
                num_layers: model_args.num_layers,
                num_kv_heads: model_args.num_kv_heads,
                head_dim: model_args.head_dim,
            },
        })?;
        let kv_cache_pool = allocator.allocate(kind, device, server_args.tp_size)?;

        Ok(Self {
            server_args,
            model_args,
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

    pub fn model_args(&self) -> ModelArgs {
        self.model_args
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

    /// Binds the migrated eager execution path after the Rust model is built.
    pub fn attach_model_runner(&mut self, model_runner: ModelRunner) -> Result<()> {
        self.ensure_live()?;
        if model_runner.device() != self.device {
            return Err(EngineError::InvalidArgument(format!(
                "ModelRunner device ({:?}) must match Engine device ({:?})",
                model_runner.device(),
                self.device
            )));
        }
        let (k_cache, v_cache) = self.kv_cache_pool()?.get_all_kv_cache()?;
        let mut model_runner = model_runner;
        model_runner.bind_kv_cache(k_cache, v_cache)?;
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
        let model = factory.create_with_attention_backend(
            self.model_args,
            self.kind,
            self.device,
            &self.server_args.attention_backend,
        )?;
        self.attach_model_runner(ModelRunner::new(model, self.device))
    }

    /// Reads local Hugging Face safetensors and binds them to the attached model.
    pub fn load_model_weights(&mut self) -> Result<usize> {
        self.ensure_live()?;
        let weights = load_hf_safetensors(&self.server_args.model_path).map_err(|error| {
            EngineError::InvalidArgument(format!("Hugging Face 权重加载失败: {error}"))
        })?;
        Ok(self
            .model_runner
            .as_mut()
            .ok_or(EngineError::ModelRunnerNotAttached)?
            .load_weights(weights)?)
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

/// Clamps scheduler and cache sizing to the trained context window.
pub fn clamp_max_seq_len(server_args: &mut ServerArgs, model_args: ModelArgs) {
    if model_args.max_position_embeddings > 0
        && server_args.max_seq_len > model_args.max_position_embeddings
    {
        server_args.max_seq_len = model_args.max_position_embeddings;
    }
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

    fn model_args() -> ModelArgs {
        ModelArgs {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 1,
            max_position_embeddings: 4,
            ..Default::default()
        }
    }

    #[test]
    fn initializes_kv_cache_and_clamps_sequence_length() {
        let model_dir = model_dir();
        let mut args = ServerArgs::new(&model_dir);
        args.max_running_req = 1;
        args.max_seq_len = 8;
        args.page_size = 2;

        let engine = Engine::new(args, model_args(), 0).unwrap();
        assert_eq!(engine.server_args().max_seq_len, 4);
        assert_eq!(engine.kv_cache_pool().unwrap().layout.num_pages, 3);
        fs::remove_dir_all(model_dir).unwrap();
    }

    #[test]
    fn cleanup_releases_the_pool_idempotently() {
        let model_dir = model_dir();
        let mut args = ServerArgs::new(&model_dir);
        args.max_running_req = 1;
        args.max_seq_len = 2;
        args.page_size = 2;
        let mut engine = Engine::new(args, model_args(), 0).unwrap();

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
    fn parses_qwen_architecture_from_hugging_face_config() {
        let path = model_dir();
        fs::write(
            path.join("config.json"),
            r#"{
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 32,
                "vocab_size": 64,
                "max_position_embeddings": 128,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 0.00001,
                "tie_word_embeddings": true,
                "qk_norm": true
            }"#,
        )
        .unwrap();

        let args = ModelArgs::from_pretrained(&path).unwrap();
        assert_eq!(args.head_dim, 4);
        assert_eq!(args.num_kv_heads, 2);
        assert!(args.tie_word_embeddings);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn tensor_parallelism_is_explicitly_deferred() {
        let model_dir = model_dir();
        let mut args = ServerArgs::new(&model_dir);
        args.tp_size = 2;

        assert!(matches!(
            Engine::new(args, model_args(), 0),
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
        let engine = Engine::new(args, model_args(), 0).unwrap();
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
        fn create(
            &self,
            _model_args: ModelArgs,
            _kind: Kind,
            _device: Device,
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
        let mut engine = Engine::new(args, model_args(), 0).unwrap();
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
        let mut engine = Engine::new(args, model_args(), 0).unwrap();

        engine.build_model(&LoadingFactory).unwrap();
        assert_eq!(engine.load_model_weights().unwrap(), 1);
        fs::remove_dir_all(model_dir).unwrap();
    }
}
