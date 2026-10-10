use tch::{Device, Kind, Tensor};

use super::{KVCacheError, ModelCacheSpec, PagedKvSpec, Result};

/// One page is either private to a request or pinned in the radix tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PageOwner {
    Private,
    Tree(usize),
}

/// A request's unique KV page table and its exact page ownership.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct BaseCacheHandle {
    pub page_ids: Vec<usize>,
    pub cached_len: usize,
    pub(crate) owners: Vec<PageOwner>,
    pub(crate) published_pages: usize,
    pub(crate) written_len: usize,
}

impl BaseCacheHandle {
    pub fn num_pages(&self) -> usize {
        self.page_ids.len()
    }
}

/// Result of an atomic cache lookup and allocation.
pub enum AcquireOutcome {
    Ready(BaseCacheHandle),
    DeferredBudget,
    DeferredMemory,
    Impossible,
}

/// Interface shared by naive and radix cache managers.
pub trait CacheManager {
    fn acquire(
        &mut self,
        input_ids: &[i64],
        capacity_tokens: usize,
        budget: Option<usize>,
    ) -> Result<AcquireOutcome>;
    fn publish(&mut self, handle: &mut BaseCacheHandle, written_ids: &[i64]) -> Result<()>;
    fn release(&mut self, handle: &mut BaseCacheHandle);
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
            self.num_pages.checked_add(1).ok_or_else(|| {
                KVCacheError::InvalidArgument("KV page count overflow".to_owned())
            })?,
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

/// Geometry common to every cache group. Physical page zero is reserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KVCachePageLayout {
    pub num_layers: usize,
    pub num_pages: usize,
    pub page_size: usize,
}

/// Stable layer-local tensor views, without allocator or ownership operations.
#[derive(Debug)]
pub struct LayerKvCache {
    pub k: Tensor,
    pub v: Tensor,
}

#[derive(Debug)]
pub struct ModelKvCache {
    pub layers: Vec<LayerKvCache>,
    pub page_size: usize,
}

struct CacheGroup {
    geometry: PagedKvSpec,
    buffer: Option<Tensor>,
}

/// One logical page allocator owns all full-attention groups atomically.
/// A radix-tree page ID identifies the same prefix in every group's tensors.
pub struct KVCachePool {
    pub layout: KVCachePageLayout,
    spec: ModelCacheSpec,
    groups: Vec<CacheGroup>,
    layer_mapping: Vec<(usize, usize)>,
    free_pages: Vec<usize>,
}

impl KVCachePool {
    pub fn new(layout: KVCacheLayout, kind: Kind, device: Device) -> Result<Self> {
        let spec =
            ModelCacheSpec::uniform(layout.num_layers, layout.num_kv_heads, layout.head_dim)?;
        Self::with_spec(spec, layout.num_pages, layout.page_size, kind, device)
    }

    pub fn with_spec(
        spec: ModelCacheSpec,
        num_pages: usize,
        page_size: usize,
        kind: Kind,
        device: Device,
    ) -> Result<Self> {
        let mut pool = Self::metadata(spec, num_pages, page_size)?;
        // Build locally; partial allocation failures drop all prior groups.
        for group_id in 0..pool.groups.len() {
            let group = &pool.groups[group_id];
            let num_layers = pool
                .layer_mapping
                .iter()
                .filter(|(id, _)| *id == group_id)
                .count();
            let shape = KVCacheLayout::new(
                num_layers,
                num_pages,
                page_size,
                group.geometry.num_kv_heads,
                group.geometry.head_dim,
            )?
            .tensor_shape()?;
            let buffer = Tensor::f_empty(shape, (kind, device))?;
            let _ = buffer.f_narrow(2, 0, 1)?.f_zero_()?;
            pool.groups[group_id].buffer = Some(buffer);
        }
        Ok(pool)
    }

    fn metadata(spec: ModelCacheSpec, num_pages: usize, page_size: usize) -> Result<Self> {
        if num_pages == 0 || page_size == 0 {
            return Err(KVCacheError::InvalidArgument(
                "page count and size must be positive".into(),
            ));
        }
        spec.bytes_per_page(page_size, 1)?;
        let mut groups: Vec<CacheGroup> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();
        let mut layer_mapping = Vec::new();
        for &geometry in spec.layers() {
            let group_id = groups
                .iter()
                .position(|g| g.geometry == geometry)
                .unwrap_or_else(|| {
                    groups.push(CacheGroup {
                        geometry,
                        buffer: None,
                    });
                    counts.push(0);
                    groups.len() - 1
                });
            layer_mapping.push((group_id, counts[group_id]));
            counts[group_id] += 1;
        }
        // Validate shape conversions before creating the free-page vector.
        for (group, &count) in groups.iter().zip(&counts) {
            KVCacheLayout::new(
                count,
                num_pages,
                page_size,
                group.geometry.num_kv_heads,
                group.geometry.head_dim,
            )?
            .tensor_shape()?;
        }
        Ok(Self {
            layout: KVCachePageLayout {
                num_layers: spec.layers().len(),
                num_pages,
                page_size,
            },
            spec,
            groups,
            layer_mapping,
            free_pages: (1..=num_pages).collect(),
        })
    }

    /// Allocator-only test pool.
    pub fn without_tensor(layout: KVCacheLayout) -> Self {
        Self::metadata(
            ModelCacheSpec::uniform(layout.num_layers, layout.num_kv_heads, layout.head_dim)
                .expect("valid cache geometry"),
            layout.num_pages,
            layout.page_size,
        )
        .expect("valid page geometry")
    }

    pub fn spec(&self) -> &ModelCacheSpec {
        &self.spec
    }

    pub fn get_layer_kv_cache(&self, layer_id: usize) -> Result<LayerKvCache> {
        let &(group_id, local_id) = self.layer_mapping.get(layer_id).ok_or_else(|| {
            KVCacheError::InvalidArgument(format!("cache layer {layer_id} out of range"))
        })?;
        let buffer = self.groups[group_id]
            .buffer
            .as_ref()
            .ok_or(KVCacheError::NotImplemented(
                "allocator-only cache has no tensor buffer",
            ))?;
        Ok(LayerKvCache {
            k: buffer.f_get(0)?.f_get(local_id as i64)?,
            v: buffer.f_get(1)?.f_get(local_id as i64)?,
        })
    }

    pub fn model_cache(&self) -> Result<ModelKvCache> {
        Ok(ModelKvCache {
            layers: (0..self.layout.num_layers)
                .map(|i| self.get_layer_kv_cache(i))
                .collect::<Result<Vec<_>>>()?,
            page_size: self.layout.page_size,
        })
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
            owners: vec![PageOwner::Private; num_pages],
            ..Default::default()
        })
    }

    /// Return a handle's pages. Clearing the handle makes repeated frees safe.
    pub fn free(&mut self, handle: &mut BaseCacheHandle) {
        debug_assert!(
            handle
                .owners
                .iter()
                .all(|owner| *owner == PageOwner::Private)
        );
        assert!(
            handle
                .page_ids
                .iter()
                .all(|&id| id > 0 && id <= self.layout.num_pages),
            "invalid or reserved KV page returned to pool"
        );
        self.free_pages.append(&mut handle.page_ids);
        handle.owners.clear();
        handle.cached_len = 0;
        handle.published_pages = 0;
        handle.written_len = 0;
    }

    /// Return pages owned by a future cache manager, such as a radix tree.
    pub fn free_pages_by_id(&mut self, page_ids: impl IntoIterator<Item = usize>) {
        for id in page_ids {
            assert!(
                id > 0 && id <= self.layout.num_pages,
                "invalid or reserved KV page returned to pool"
            );
            self.free_pages.push(id);
        }
    }

    pub fn free_count(&self) -> usize {
        self.free_pages.len()
    }

    #[cfg(test)]
    pub(crate) fn free_page_ids(&self) -> &[usize] {
        &self.free_pages
    }

    /// Compatibility inspection helper for uniform pools only. Model execution
    /// uses layer views and never depends on a global homogeneous tensor.
    pub fn get_all_kv_cache(&self) -> Result<(Tensor, Tensor)> {
        if self.groups.len() != 1 {
            return Err(KVCacheError::InvalidArgument(
                "heterogeneous cache has no global K/V tensor; use layer views".into(),
            ));
        }
        let buffer = self.groups[0]
            .buffer
            .as_ref()
            .ok_or(KVCacheError::NotImplemented(
                "allocator-only cache has no tensor buffer",
            ))?;
        Ok((buffer.f_get(0)?, buffer.f_get(1)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heterogeneous_lifecycle(device: Device) {
        use super::super::RadixCacheManager;
        use crate::engine::BatchPhase;
        use crate::models::attention::BaseAttention;
        use std::{cell::RefCell, rc::Rc};

        // Nonadjacent layers with equal geometry must retain model layer order.
        let a = PagedKvSpec {
            num_kv_heads: 2,
            head_dim: 4,
        };
        let b = PagedKvSpec {
            num_kv_heads: 1,
            head_dim: 8,
        };
        let spec = ModelCacheSpec::new(vec![a, b, a]).unwrap();
        let pool = Rc::new(RefCell::new(
            KVCachePool::with_spec(spec, 4, 2, Kind::Float, device).unwrap(),
        ));
        assert_eq!(pool.borrow().groups.len(), 2);
        assert!(pool.borrow().get_all_kv_cache().is_err());
        assert!(pool.borrow().get_layer_kv_cache(3).is_err());
        let mut caches = Vec::new();
        let mut addresses = Vec::new();
        for view in pool.borrow().model_cache().unwrap().layers {
            addresses.push(view.k.data_ptr());
            assert_eq!(view.k.get(0).abs().sum(Kind::Float).double_value(&[]), 0.0);
            assert_eq!(view.v.get(0).abs().sum(Kind::Float).double_value(&[]), 0.0);
            let mut cache = BaseAttention::default();
            cache.set_reserved_write_slot(0);
            cache.bind_kv_cache(view.k, view.v).unwrap();
            caches.push(cache);
        }
        let mut manager = RadixCacheManager::new(pool.clone(), 2).unwrap();
        let AcquireOutcome::Ready(mut first) = manager.acquire(&[10, 11], 4, None).unwrap() else {
            panic!("allocation failed")
        };
        let prefix_page = first.page_ids[0];
        let prefix_locs =
            Tensor::from_slice(&[(prefix_page * 2) as i32, (prefix_page * 2 + 1) as i32])
                .to_device(device);
        for (i, cache) in caches.iter_mut().enumerate() {
            let geometry = pool.borrow().spec.layers()[i];
            let shape = [2, geometry.num_kv_heads as i64, geometry.head_dim as i64];
            let kv = Tensor::full(shape, (i + 1) as f64, (Kind::Float, device));
            cache
                .write_kv(&kv, &(&kv + 10.0), Some(&prefix_locs), BatchPhase::Prefill)
                .unwrap();
        }
        manager.publish(&mut first, &[10, 11]).unwrap();
        // A branch shares only the completed prefix and owns its unwritten tail.
        let AcquireOutcome::Ready(mut branch) = manager.acquire(&[10, 11, 12], 4, None).unwrap()
        else {
            panic!("prefix lookup failed")
        };
        assert_eq!(branch.cached_len, 2);
        assert_eq!(branch.page_ids[0], prefix_page);
        assert_ne!(branch.page_ids[1], first.page_ids[1]);
        for (i, cache) in caches.iter_mut().enumerate() {
            let geometry = pool.borrow().spec.layers()[i];
            let kv = Tensor::full(
                [2, geometry.num_kv_heads as i64, geometry.head_dim as i64],
                99.,
                (Kind::Float, device),
            );
            // Include graph padding alongside a real branch write.
            let locs =
                Tensor::from_slice(&[0i32, (branch.page_ids[1] * 2) as i32]).to_device(device);
            cache
                .write_kv(&kv, &kv, Some(&locs), BatchPhase::Decode)
                .unwrap();
            let table =
                Tensor::from_slice(&[(prefix_page * 2) as i32, (prefix_page * 2 + 1) as i32])
                    .view([1, 2])
                    .to_device(device);
            let (k, v) = cache.read_kv(&table, 0, 2).unwrap();
            assert_eq!(k.mean(Kind::Float).double_value(&[]), (i + 1) as f64);
            assert_eq!(v.mean(Kind::Float).double_value(&[]), (i + 11) as f64);
            assert_eq!(
                cache
                    .cache_tensors()
                    .unwrap()
                    .0
                    .get(0)
                    .abs()
                    .sum(Kind::Float)
                    .double_value(&[]),
                0.0
            );
            assert_eq!(cache.cache_tensors().unwrap().0.data_ptr(), addresses[i]);
        }
        // Cancellation frees branch-private pages but cannot evict the pinned prefix.
        manager.release(&mut branch);
        manager.release(&mut branch);
        assert_eq!(manager.evict(4).len(), 0);
        manager.release(&mut first);
        assert_eq!(manager.evict(4).len(), 1);
        assert_eq!(pool.borrow().free_count(), 4);
        assert!(!pool.borrow().free_page_ids().contains(&0));
        if matches!(device, Device::Cuda(_)) {
            tch::Cuda::synchronize(0);
        }
    }

    #[test]
    fn grouped_pool_rejects_overflow_before_allocating_pages() {
        let spec = ModelCacheSpec::uniform(1, 1, 4).unwrap();
        assert!(
            KVCachePool::with_spec(spec.clone(), usize::MAX, 1, Kind::Float, Device::Cpu).is_err()
        );
        assert!(KVCachePool::with_spec(spec, 1, usize::MAX, Kind::Float, Device::Cpu).is_err());
    }

    #[test]
    fn heterogeneous_cache_preserves_prefix_branch_cleanup_and_padding() {
        heterogeneous_lifecycle(Device::Cpu);
    }

    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn cuda_heterogeneous_cache_preserves_prefix_branch_cleanup_and_padding() {
        assert!(
            tch::Cuda::is_available(),
            "CUDA test requires an available GPU"
        );
        heterogeneous_lifecycle(Device::Cuda(0));
    }

    fn pool(num_pages: usize) -> KVCachePool {
        KVCachePool::without_tensor(KVCacheLayout::new(2, num_pages, 4, 4, 32).unwrap())
    }

    #[test]
    fn alloc_and_free_match_the_python_pool_contract() {
        let mut pool = pool(10);
        let mut first = pool.alloc(3).unwrap();
        let mut second = pool.alloc(2).unwrap();

        assert_eq!(first.page_ids, vec![10, 9, 8]);
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

        assert_eq!(k_cache.size(), vec![2, 4, 4, 5, 6]);
        assert_eq!(v_cache.size(), vec![2, 4, 4, 5, 6]);
    }
    #[test]
    fn reserved_page_zero_is_initialized_and_never_allocated() {
        let layout = KVCacheLayout::new(1, 3, 4, 1, 2).unwrap();
        let mut pool = KVCachePool::new(layout, Kind::Float, Device::Cpu).unwrap();
        let (k, v) = pool.get_all_kv_cache().unwrap();
        assert_eq!(
            k.get(0).get(0).abs().sum(Kind::Float).double_value(&[]),
            0.0
        );
        assert_eq!(
            v.get(0).get(0).abs().sum(Kind::Float).double_value(&[]),
            0.0
        );
        let mut handle = pool.alloc(3).unwrap();
        assert_eq!(handle.page_ids, vec![3, 2, 1]);
        pool.free(&mut handle);
        assert!(!pool.free_page_ids().contains(&0));
    }
}
