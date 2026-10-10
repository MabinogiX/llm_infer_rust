//! Scheduler construction and execution errors.

use std::fmt;

use crate::engine::kvcache::KVCacheError;
use crate::engine::{BatchContextError, EngineError};

#[derive(Debug)]
pub enum SchedulerError {
    RequestIdExhausted,
    Cache(KVCacheError),
    Batch(BatchContextError),
    InvalidDecode(String),
    Engine(EngineError),
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestIdExhausted => write!(f, "scheduler request ID space exhausted"),
            Self::Cache(error) => write!(f, "KV cache scheduling failed: {error}"),
            Self::Batch(error) => write!(f, "prefill batch preparation failed: {error}"),
            Self::InvalidDecode(message) => write!(f, "decode batch preparation failed: {message}"),
            Self::Engine(error) => write!(f, "engine scheduling failed: {error}"),
        }
    }
}

impl std::error::Error for SchedulerError {}

impl From<KVCacheError> for SchedulerError {
    fn from(error: KVCacheError) -> Self {
        Self::Cache(error)
    }
}

impl From<BatchContextError> for SchedulerError {
    fn from(error: BatchContextError) -> Self {
        Self::Batch(error)
    }
}

impl From<EngineError> for SchedulerError {
    fn from(error: EngineError) -> Self {
        Self::Engine(error)
    }
}

pub type Result<T> = std::result::Result<T, SchedulerError>;
