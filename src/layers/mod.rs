//! Shared computation for model implementations.
//!
//! These layers deliberately implement specific semantics: bias-free linear,
//! packed [Q, K, V], direct-scale RMSNorm, full-head half-split RoPE and SwiGLU.
//! Models choose and compose them, own checkpoint names, and bind weights before
//! graph capture. No model selection or runtime assembly belongs here.

mod linear;
mod mlp;
mod ops;

pub(crate) use linear::{PackedQkv, embedding, linear, logits};
pub(crate) use mlp::DenseSwiGlu;
pub(crate) use ops::{HalfSplitRope, add_rms_norm, qk_norm, rms_norm};
#[cfg(test)]
pub(crate) use ops::{rotate_half, silu_and_mul};
