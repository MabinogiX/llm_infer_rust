//! Hugging Face safetensors discovery and loading.
//!
//! This module loads actual checkpoint tensors. Binding their names to Rust
//! model parameters is left to the registered model implementation.

use std::{
    collections::BTreeSet,
    fmt, fs,
    path::{Component, Path, PathBuf},
};

use tch::{Device, Kind, TchError, Tensor};

use super::{ModelExecutor, ModelRunnerError, RuntimeModelConfig};

#[derive(Debug)]
pub enum ModelLoadError {
    NoSafeTensors(PathBuf),
    ReadFile { path: PathBuf, message: String },
    InvalidIndex { path: PathBuf, message: String },
    UnsafeShardPath(String),
    DuplicateTensor(String),
    Torch(TchError),
}

impl fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSafeTensors(path) => write!(
                f,
                "模型目录中没有 model.safetensors 或 model.safetensors.index.json: {}",
                path.display()
            ),
            Self::ReadFile { path, message } => write!(f, "无法读取 {}: {message}", path.display()),
            Self::InvalidIndex { path, message } => {
                write!(f, "无效的 safetensors index {}: {message}", path.display())
            }
            Self::UnsafeShardPath(path) => write!(f, "不安全的 safetensors shard 路径: {path}"),
            Self::DuplicateTensor(name) => write!(f, "多个 shard 包含同名权重: {name}"),
            Self::Torch(error) => write!(f, "读取 safetensors 失败: {error}"),
        }
    }
}

impl std::error::Error for ModelLoadError {}

impl From<TchError> for ModelLoadError {
    fn from(error: TchError) -> Self {
        Self::Torch(error)
    }
}

pub type Result<T> = std::result::Result<T, ModelLoadError>;

/// Creates a concrete Rust model from the architecture selected by the caller.
///
/// The registry itself belongs with the model implementations; keeping this
/// trait here lets Engine own the same construction lifecycle as mini-sglang
/// without hard-coding a model architecture.
pub trait ModelFactory {
    /// Declare full-attention state in model layer order. The runtime never
    /// infers heterogeneous geometry from a representative global head count.
    fn cache_spec(
        &self,
        runtime: RuntimeModelConfig,
    ) -> std::result::Result<super::kvcache::ModelCacheSpec, ModelRunnerError>;

    /// Check that runtime geometry matches the model configuration and that the
    /// execution combination is supported, before model or weight allocation.
    fn validate_runtime(
        &self,
        runtime_config: RuntimeModelConfig,
        kind: Kind,
        device: Device,
        attention_backend: &str,
    ) -> std::result::Result<(), ModelRunnerError>;

    /// Construct the model using its already parsed, model-specific configuration.
    fn create(
        &self,
        kind: Kind,
        device: Device,
        attention_backend: &str,
    ) -> std::result::Result<Box<dyn ModelExecutor>, ModelRunnerError>;
}

/// Named tensors read from a Hugging Face safetensors checkpoint.
pub struct ModelWeights {
    tensors: Vec<(String, Tensor)>,
}

impl ModelWeights {
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn into_tensors(self) -> Vec<(String, Tensor)> {
        self.tensors
    }
}

/// Loads `model.safetensors` or all shards listed in
/// `model.safetensors.index.json` into libtorch tensors.
pub fn load_hf_safetensors(model_path: impl AsRef<Path>) -> Result<ModelWeights> {
    let model_path = model_path.as_ref();
    let files = checkpoint_files(model_path)?;
    let mut tensors = Vec::new();
    let mut names = BTreeSet::new();
    for file in files {
        for (name, tensor) in Tensor::read_safetensors(&file)? {
            if !names.insert(name.clone()) {
                return Err(ModelLoadError::DuplicateTensor(name));
            }
            tensors.push((name, tensor));
        }
    }
    Ok(ModelWeights { tensors })
}

fn checkpoint_files(model_path: &Path) -> Result<Vec<PathBuf>> {
    let single_file = model_path.join("model.safetensors");
    if single_file.is_file() {
        return Ok(vec![single_file]);
    }

    let index_path = model_path.join("model.safetensors.index.json");
    if !index_path.is_file() {
        return Err(ModelLoadError::NoSafeTensors(model_path.to_path_buf()));
    }
    let contents = fs::read_to_string(&index_path).map_err(|error| ModelLoadError::ReadFile {
        path: index_path.clone(),
        message: error.to_string(),
    })?;
    let index: serde_json::Value =
        serde_json::from_str(&contents).map_err(|error| ModelLoadError::InvalidIndex {
            path: index_path.clone(),
            message: error.to_string(),
        })?;
    let weight_map = index
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| ModelLoadError::InvalidIndex {
            path: index_path,
            message: "missing object field weight_map".to_owned(),
        })?;

    let mut shard_names = BTreeSet::new();
    for value in weight_map.values() {
        let shard = value.as_str().ok_or_else(|| ModelLoadError::InvalidIndex {
            path: model_path.join("model.safetensors.index.json"),
            message: "weight_map values must be strings".to_owned(),
        })?;
        let shard_path = Path::new(shard);
        if shard_path.is_absolute()
            || shard_path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(ModelLoadError::UnsafeShardPath(shard.to_owned()));
        }
        shard_names.insert(shard.to_owned());
    }

    Ok(shard_names
        .into_iter()
        .map(|shard| model_path.join(shard))
        .collect())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use tch::{Device, Kind};

    use super::*;

    static TEST_DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);

    fn model_dir() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let id = TEST_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("sglang-rust-weights-{nonce}-{id}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn loads_a_single_safetensors_checkpoint() {
        let path = model_dir();
        let tensor = Tensor::ones([2, 3], (Kind::Float, Device::Cpu));
        Tensor::write_safetensors(
            &[("lm_head.weight", &tensor)],
            path.join("model.safetensors"),
        )
        .unwrap();

        let weights = load_hf_safetensors(&path).unwrap();
        assert_eq!(weights.len(), 1);
        let tensors = weights.into_tensors();
        assert_eq!(tensors[0].0, "lm_head.weight");
        assert_eq!(tensors[0].1.size(), vec![2, 3]);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn rejects_index_path_traversal() {
        let path = model_dir();
        fs::write(
            path.join("model.safetensors.index.json"),
            r#"{"weight_map":{"weight":"../outside.safetensors"}}"#,
        )
        .unwrap();

        assert!(matches!(
            load_hf_safetensors(&path),
            Err(ModelLoadError::UnsafeShardPath(_))
        ));
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn loads_indexed_shards_once_and_preserves_tensor_values() {
        let path = model_dir();
        let first = Tensor::from_slice(&[1f32, 2.]);
        let second = Tensor::from_slice(&[3f32, 4.]);
        Tensor::write_safetensors(
            &[("a", &first), ("b", &second)],
            path.join("part-1.safetensors"),
        )
        .unwrap();
        Tensor::write_safetensors(&[("c", &second)], path.join("part-2.safetensors")).unwrap();
        fs::write(path.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"part-1.safetensors","b":"part-1.safetensors","c":"part-2.safetensors"}}"#,
        ).unwrap();
        let weights = load_hf_safetensors(&path).unwrap().into_tensors();
        assert_eq!(weights.len(), 3);
        for (name, actual) in weights {
            let expected = if name == "a" { &first } else { &second };
            assert!(actual.equal(expected), "{name}");
        }
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn rejects_duplicate_tensor_names_across_shards() {
        let path = model_dir();
        let tensor = Tensor::ones([2], (Kind::Float, Device::Cpu));
        for shard in ["part-1.safetensors", "part-2.safetensors"] {
            Tensor::write_safetensors(&[("duplicate", &tensor)], path.join(shard)).unwrap();
        }
        fs::write(
            path.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"part-1.safetensors","b":"part-2.safetensors"}}"#,
        )
        .unwrap();
        assert!(matches!(load_hf_safetensors(&path),
            Err(ModelLoadError::DuplicateTensor(name)) if name == "duplicate"));
        fs::remove_dir_all(path).unwrap();
    }
}
