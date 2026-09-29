//! Paged KV-cache ownership and allocation.
//!
//! Page bookkeeping and backing K/V tensors both live in Rust through
//! `tch-rs`, the Rust bindings for libtorch.

mod allocator;
mod error;
mod naive;
mod pool;
mod radix;

pub use allocator::{
    KVCacheAllocationConfig, KVCacheAllocator, KVCacheModelConfig, KVCacheServerConfig,
};
pub use error::{KVCacheError, Result};
pub use naive::NaiveCacheManager;
pub use pool::{AcquireOutcome, BaseCacheHandle, CacheManager, KVCacheLayout, KVCachePool};
pub use radix::{RadixCacheManager, RadixNode};
pub use tch::{Device, Kind, Tensor};
