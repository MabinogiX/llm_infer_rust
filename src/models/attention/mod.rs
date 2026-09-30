//! Shared paged-KV attention state.

mod backend;
mod base;

pub use backend::{Attention, AttentionBatch, AttentionSpec};
pub use base::BaseAttention;
