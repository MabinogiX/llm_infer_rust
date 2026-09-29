use std::fmt;

use tch::TchError;

pub type Result<T> = std::result::Result<T, KVCacheError>;

#[derive(Debug)]
pub enum KVCacheError {
    InvalidArgument(String),
    OutOfMemory { requested: usize, available: usize },
    MemoryQuery(String),
    NotImplemented(&'static str),
    Torch(TchError),
}

impl fmt::Display for KVCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(message) => write!(f, "invalid KV-cache argument: {message}"),
            Self::OutOfMemory {
                requested,
                available,
            } => write!(
                f,
                "KV cache out of memory: requested {requested} pages, only {available} free"
            ),
            Self::MemoryQuery(message) => write!(f, "CUDA 显存查询失败: {message}"),
            Self::NotImplemented(feature) => write!(f, "未实现: {feature}"),
            Self::Torch(error) => write!(f, "libtorch error: {error}"),
        }
    }
}

impl std::error::Error for KVCacheError {}

impl From<TchError> for KVCacheError {
    fn from(error: TchError) -> Self {
        Self::Torch(error)
    }
}
