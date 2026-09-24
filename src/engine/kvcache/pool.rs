use tch::{Device, Kind, Tensor};

use super::{KVCacheError, Result};

/// A request's complete KV page table.
///
/// When prefix caching is introduced, the leading `num_shared` page IDs are
/// borrowed from that cache. The remaining IDs belong to this request.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BaseCacheHandle {
    pub page_ids: Vec<usize>,
    pub cached_len: usize,
    pub num_shared: usize,
}

impl BaseCacheHandle {
    pub fn num_pages(&self) -> usize {
        self.page_ids.len()
    }
}

/// Interface shared by naive and radix cache managers.
pub trait CacheManager {
    fn match_prefix(&self, input_ids: &[i64]) -> Result<(usize, Vec<usize>)>;
    fn insert(&mut self, input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()>;
    fn evict(&mut self, num_pages: usize) -> Result<Vec<usize>>;
    fn remove(&mut self, input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()>;
    fn rollback_insert(&mut self, input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()>;
}

/// Shape metadata for the backing `(K, V)` tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KVCacheLayout {
    pub num_layers: usize,
    pub num_pages: usize,
    pub page_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl KVCacheLayout {
    pub fn new(
        num_layers: usize,
        num_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let layout = Self {
            num_layers,
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
        };
        if [num_layers, num_pages, page_size, num_kv_heads, head_dim].contains(&0) {
            return Err(KVCacheError::InvalidArgument(
                "all KV-cache dimensions must be greater than zero".to_owned(),
            ));
        }
        Ok(layout)
    }

    fn tensor_shape(self) -> Result<[i64; 6]> {
        [
            2,
            self.num_layers,
            self.num_pages,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
        ]
        .map(|dimension| {
            i64::try_from(dimension).map_err(|_| {
                KVCacheError::InvalidArgument("KV-cache dimension exceeds i64".to_owned())
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .and_then(|dimensions| {
            dimensions.try_into().map_err(|_| {
                KVCacheError::InvalidArgument("invalid KV-cache tensor rank".to_owned())
            })
        })
    }
}

/// Owns page allocation state and a libtorch-backed K/V buffer.
pub struct KVCachePool {
    pub layout: KVCacheLayout,
    buffer: Option<Tensor>,
    free_pages: Vec<usize>,
}

impl KVCachePool {
    /// Allocate the backing tensor with libtorch's `at::empty`.
    pub fn new(layout: KVCacheLayout, kind: Kind, device: Device) -> Result<Self> {
        let shape = layout.tensor_shape()?;
        let buffer = Tensor::f_empty(shape, (kind, device))?;

        Ok(Self {
            layout,
            buffer: Some(buffer),
            free_pages: (0..layout.num_pages).collect(),
        })
    }

    /// Create only the page allocator. It is useful for deterministic unit
    /// tests; requesting tensors from it returns an explicit 未实现 error.
    pub fn without_tensor(layout: KVCacheLayout) -> Self {
        Self {
            layout,
            buffer: None,
            free_pages: (0..layout.num_pages).collect(),
        }
    }

    /// Allocate pages in the same LIFO order as mini-sglang's Python pool.
    pub fn alloc(&mut self, num_pages: usize) -> Result<BaseCacheHandle> {
        let available = self.free_pages.len();
        if available < num_pages {
            return Err(KVCacheError::OutOfMemory {
                requested: num_pages,
                available,
            });
        }

        let page_ids = (0..num_pages)
            .map(|_| self.free_pages.pop().expect("checked free page count"))
            .collect();
        Ok(BaseCacheHandle {
            page_ids,
            ..Default::default()
        })
    }

    /// Return a handle's pages. Clearing the handle makes repeated frees safe.
    pub fn free(&mut self, handle: &mut BaseCacheHandle) {
        self.free_pages.append(&mut handle.page_ids);
    }

    /// Return pages owned by a future cache manager, such as a radix tree.
    pub fn free_pages_by_id(&mut self, page_ids: impl IntoIterator<Item = usize>) {
        self.free_pages.extend(page_ids);
    }

    pub fn free_count(&self) -> usize {
        self.free_pages.len()
    }

    /// Return `(k_cache, v_cache)`, each shaped
    /// `(num_layers, num_pages, page_size, num_kv_heads, head_dim)`.
    pub fn get_all_kv_cache(&self) -> Result<(Tensor, Tensor)> {
        let buffer = self.buffer.as_ref().ok_or(KVCacheError::NotImplemented(
            "无 libtorch buffer 的 KVCachePool 不能提供 K/V tensor",
        ))?;
        Ok((buffer.f_get(0)?, buffer.f_get(1)?))
    }

    /// Binding the pool to Rust attention modules awaits migration of the model
    /// layer and remains deliberately unsupported for this step.
    pub fn bind_layers(&self) -> Result<()> {
        Err(KVCacheError::NotImplemented(
            "Rust attention layer 的 KV-cache 绑定",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(num_pages: usize) -> KVCachePool {
        KVCachePool::without_tensor(KVCacheLayout::new(2, num_pages, 4, 4, 32).unwrap())
    }

    #[test]
    fn alloc_and_free_match_the_python_pool_contract() {
        let mut pool = pool(10);
        let mut first = pool.alloc(3).unwrap();
        let mut second = pool.alloc(2).unwrap();

        assert_eq!(first.page_ids, vec![9, 8, 7]);
        assert_eq!(pool.free_count(), 5);
        pool.free(&mut first);
        assert_eq!(pool.free_count(), 8);
        pool.free(&mut second);
        assert_eq!(pool.free_count(), 10);
        pool.free(&mut second);
        assert_eq!(pool.free_count(), 10);
    }

    #[test]
    fn allocation_fails_without_mutating_the_pool() {
        let mut pool = pool(2);
        let error = pool.alloc(3).unwrap_err();

        assert!(matches!(
            error,
            KVCacheError::OutOfMemory {
                requested: 3,
                available: 2
            }
        ));
        assert_eq!(pool.free_count(), 2);
    }

    #[test]
    fn layout_rejects_zero_dimensions() {
        assert!(KVCacheLayout::new(0, 1, 1, 1, 1).is_err());
    }

    #[test]
    fn libtorch_backed_pool_exposes_k_and_v_slices() {
        let layout = KVCacheLayout::new(2, 3, 4, 5, 6).unwrap();
        let pool = KVCachePool::new(layout, Kind::Float, Device::Cpu).unwrap();
        let (k_cache, v_cache) = pool.get_all_kv_cache().unwrap();

        assert_eq!(k_cache.size(), vec![2, 3, 4, 5, 6]);
        assert_eq!(v_cache.size(), vec![2, 3, 4, 5, 6]);
    }
}
