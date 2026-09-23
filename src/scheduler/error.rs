//! Scheduler construction and execution errors.

use std::{fmt, path::PathBuf};

#[derive(Debug)]
pub enum SchedulerError {
    InvalidModelConfig { path: PathBuf, message: String },
    RequestIdExhausted,
    ExecutionNotAvailable,
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModelConfig { path, message } => {
                write!(f, "invalid model config {}: {message}", path.display())
            }
            Self::RequestIdExhausted => write!(f, "scheduler request ID space exhausted"),
            Self::ExecutionNotAvailable => {
                write!(f, "prefill/decode scheduling is not connected yet")
            }
        }
    }
}

impl std::error::Error for SchedulerError {}

pub type Result<T> = std::result::Result<T, SchedulerError>;
