//! Model-declared full-attention cache geometry. All groups share logical pages.

use super::{KVCacheError, Result};

/// One full-attention layer's K/V geometry, before tensor-parallel sharding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvSpec {
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

/// Layer order is model order. Equal geometries are packed into one tensor group.
/// Only full-history paged K/V is supported: recurrent state and independently
/// evicted windows must not be represented by this descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCacheSpec {
    layers: Vec<PagedKvSpec>,
}

impl ModelCacheSpec {
    pub fn new(layers: Vec<PagedKvSpec>) -> Result<Self> {
        if layers.is_empty()
            || layers
                .iter()
                .any(|s| s.num_kv_heads == 0 || s.head_dim == 0)
        {
            return Err(KVCacheError::InvalidArgument(
                "cache requires nonempty layers with positive KV heads and head dimension".into(),
            ));
        }
        let spec = Self { layers };
        spec.bytes_per_page(1, 1)?;
        Ok(spec)
    }

    pub fn uniform(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Result<Self> {
        Self::new(vec![
            PagedKvSpec {
                num_kv_heads,
                head_dim
            };
            num_layers
        ])
    }

    pub fn layers(&self) -> &[PagedKvSpec] {
        &self.layers
    }

    pub fn bytes_per_page(&self, page_size: usize, itemsize: usize) -> Result<usize> {
        if page_size == 0 || itemsize == 0 {
            return Err(KVCacheError::InvalidArgument(
                "page size and itemsize must be positive".into(),
            ));
        }
        self.layers.iter().try_fold(0usize, |total, layer| {
            let bytes = layer
                .num_kv_heads
                .checked_mul(layer.head_dim)
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| n.checked_mul(page_size))
                .and_then(|n| n.checked_mul(itemsize))
                .and_then(|n| n.checked_add(total))
                .ok_or_else(|| {
                    KVCacheError::InvalidArgument("cache bytes per page overflow".into())
                })?;
            Ok(bytes)
        })
    }

    pub(crate) fn per_rank(&self, tp_size: usize) -> Result<Self> {
        if tp_size == 0 {
            return Err(KVCacheError::InvalidArgument(
                "tp_size must be positive".into(),
            ));
        }
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let heads = layer.num_kv_heads;
                if heads % tp_size != 0 && tp_size % heads != 0 {
                    return Err(KVCacheError::InvalidArgument(
                        "KV heads must divide TP size or be divisible by it".into(),
                    ));
                }
                Ok(PagedKvSpec {
                    num_kv_heads: (heads / tp_size).max(1),
                    head_dim: layer.head_dim,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new(layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_geometry_and_counts_all_layers() {
        let spec = ModelCacheSpec::new(vec![
            PagedKvSpec {
                num_kv_heads: 2,
                head_dim: 8,
            },
            PagedKvSpec {
                num_kv_heads: 1,
                head_dim: 16,
            },
        ])
        .unwrap();
        assert_eq!(spec.bytes_per_page(4, 2).unwrap(), 512);
        assert!(ModelCacheSpec::new(vec![]).is_err());
        assert!(ModelCacheSpec::uniform(1, 0, 4).is_err());
        assert!(ModelCacheSpec::uniform(1, usize::MAX, 2).is_err());
        assert!(spec.bytes_per_page(usize::MAX, 4).is_err());
        assert!(
            ModelCacheSpec::uniform(1, 3, 8)
                .unwrap()
                .per_rank(2)
                .is_err()
        );
        assert_eq!(spec.per_rank(2).unwrap().layers()[0].num_kv_heads, 1);
    }
}
