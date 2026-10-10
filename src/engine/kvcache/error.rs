use std::fmt;

use tch::TchError;

pub type Result<T> = std::result::Result<T, KVCacheError>;

#[derive(Debug)]
pub enum KVCacheError {
    InvalidArgument(String),
    OutOfMemory {
        requested: usize,
        available: usize,
    },
    InsufficientBudget {
        required_bytes: usize,
        available_bytes: usize,
    },
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
            Self::InsufficientBudget {
                required_bytes,
                available_bytes,
            } => write!(
                f,
                "KV cache needs at least {required_bytes} bytes (sink plus one page), budget is {available_bytes} bytes"
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
