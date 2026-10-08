//! Shared paged-KV attention state.

mod backend;
mod base;
#[cfg(has_cuda_kv_store)]
mod kv_store;

pub use backend::{Attention, AttentionBatch, AttentionSpec};
pub use base::BaseAttention;
