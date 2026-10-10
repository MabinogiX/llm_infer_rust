//! Bias-free projections, packed QKV geometry and generation row selection.

use tch::{Device, Kind, Tensor};

pub(crate) fn linear(x: &Tensor, weight: &Tensor) -> Tensor {
    x.matmul(&weight.transpose(0, 1))
}

pub(crate) fn embedding(ids: &Tensor, weight: &Tensor) -> Tensor {
    weight.index_select(0, ids)
}

/// Select live prefill rows before projection; decode projects every row.
/// Tied embeddings and any model-specific logits transforms belong to the model.
pub(crate) fn logits(hidden: &Tensor, weight: &Tensor, indices: Option<&Tensor>) -> Tensor {
    match indices {
        Some(indices) => linear(&hidden.index_select(0, indices), weight),
        None => linear(hidden, weight),
    }
}

/// [Q, K, V] projection with independently sized query and KV heads.
/// Returned tensors are views of one packed allocation; Q/K norm and RoPE may
/// mutate those views in place. The model binds checkpoint tensors to `weight`.
pub(crate) struct PackedQkv {
    pub(crate) weight: Tensor,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
}

impl PackedQkv {
    pub(crate) fn new(
        hidden: i64,
        num_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
        kind: Kind,
        device: Device,
    ) -> Self {
        // Model configuration validation must precede parameter allocation.
        Self {
            weight: Tensor::zeros(
                [(num_heads + 2 * num_kv_heads) * head_dim, hidden],
                (kind, device),
            ),
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    pub(crate) fn widths(&self) -> [i64; 3] {
        [
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        ]
    }

    pub(crate) fn forward(&self, hidden: &Tensor) -> (Tensor, Tensor, Tensor) {
        let rows = hidden.size()[0];
        let packed = linear(hidden, &self.weight);
        let [q_width, kv_width, _] = self.widths();
        let q = packed
            .narrow(-1, 0, q_width)
            .view([rows, self.num_heads, self.head_dim]);
        let k = packed
            .narrow(-1, q_width, kv_width)
            .view([rows, self.num_kv_heads, self.head_dim]);
        let v = packed.narrow(-1, q_width + kv_width, kv_width).view([
            rows,
            self.num_kv_heads,
            self.head_dim,
        ]);
        (q, k, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_qkv_matches_separate_projections_with_independent_attention_width() {
        let mut layer = PackedQkv::new(4, 2, 1, 4, Kind::Float, Device::Cpu);
        let x = Tensor::arange(12, (Kind::Float, Device::Cpu)).view([3, 4]) / 10.0;
        let q = Tensor::arange(32, (Kind::Float, Device::Cpu)).view([8, 4]);
        let k = Tensor::arange(16, (Kind::Float, Device::Cpu)).view([4, 4]) + 100.0;
        let v = &k + 100.0;
        layer.weight = Tensor::cat(&[&q, &k, &v], 0);
        let (query, key, value) = layer.forward(&x);
        assert_eq!(query.size(), [3, 2, 4]);
        assert_eq!(key.size(), [3, 1, 4]);
        assert_eq!(layer.widths(), [8, 4, 4]);
        for (actual, weight) in [(query, &q), (key, &k), (value, &v)] {
            let error = (actual.flatten(1, -1) - linear(&x, weight))
                .abs()
                .max()
                .double_value(&[]);
            assert!(error < 1e-4);
        }
    }

    #[test]
    fn logits_gather_live_rows_and_accept_tied_embedding_weights() {
        let weights = Tensor::arange(32, (Kind::Float, Device::Cpu)).view([8, 4]);
        let hidden = embedding(&Tensor::from_slice(&[1i64, 5, 2]), &weights);
        let indices = Tensor::from_slice(&[2i64, 0]);
        let all = logits(&hidden, &weights, None);
        let selected = logits(&hidden, &weights.shallow_clone(), Some(&indices));
        assert_eq!(selected.size(), [2, 8]);
        assert!(selected.equal(&all.index_select(0, &indices)));
    }
}
