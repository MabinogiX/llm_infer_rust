//! Dense, bias-free SwiGLU FFN. Models select this concrete computation;
//! expert routing and other activations are separate implementations.

use super::{linear, ops::silu_and_mul};
use tch::{Device, Kind, Tensor};

pub(crate) struct DenseSwiGlu {
    pub(crate) gate_up_proj: Tensor,
    pub(crate) down_proj: Tensor,
}

impl DenseSwiGlu {
    /// Called after the model validates dimensions. Weights are bound by the model.
    pub(crate) fn new(hidden: i64, intermediate: i64, kind: Kind, device: Device) -> Self {
        Self {
            gate_up_proj: Tensor::zeros([2 * intermediate, hidden], (kind, device)),
            down_proj: Tensor::zeros([hidden, intermediate], (kind, device)),
        }
    }

    pub(crate) fn forward(&self, hidden: &Tensor) -> Tensor {
        let gate_up = linear(hidden, &self.gate_up_proj);
        linear(&silu_and_mul(&gate_up), &self.down_proj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_swiglu_matches_separate_gate_up_and_down() {
        let mut layer = DenseSwiGlu::new(4, 7, Kind::Float, Device::Cpu);
        let x = Tensor::arange(12, (Kind::Float, Device::Cpu)).view([3, 4]) / 10.0;
        let gate = Tensor::arange(28, (Kind::Float, Device::Cpu)).view([7, 4]) / 100.0;
        let up = &gate + 0.1;
        layer.gate_up_proj = Tensor::cat(&[&gate, &up], 0);
        layer.down_proj = Tensor::arange(28, (Kind::Float, Device::Cpu)).view([4, 7]) / 100.0;
        let expected = linear(
            &(linear(&x, &gate).silu() * linear(&x, &up)),
            &layer.down_proj,
        );
        let error = (layer.forward(&x) - expected).abs().max().double_value(&[]);
        assert!(error < 1e-5);
    }
}
