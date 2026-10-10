//! Shared operators with fixed numerical semantics and native CUDA dispatch.
//! RMSNorm uses direct scales and FP32 accumulation; RoPE is unscaled, full-head
//! half-split rotation. Other semantics require a separate implementation.

use tch::{Device, Kind, Tensor};

/// Direct scale RMSNorm: FP32 variance and multiplication, cast to activation dtype.
pub(crate) fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    #[cfg(has_layer_cuda)]
    if supports_cuda(x) {
        return cuda::rms_norm(x, weight, eps);
    }
    let x_float = x.to_kind(Kind::Float);
    let variance = x_float
        .pow_tensor_scalar(2)
        .mean_dim(&[-1i64][..], true, Kind::Float);
    (x_float * (variance + eps).rsqrt() * weight.to_kind(Kind::Float)).to_kind(x.kind())
}

/// Consumes layer-owned activations; CUDA updates both buffers in place.
/// Normalize the full FP32 sum; store the residual rounded to activation dtype.
pub(crate) fn add_rms_norm(
    x: Tensor,
    residual: Tensor,
    weight: &Tensor,
    eps: f64,
) -> (Tensor, Tensor) {
    #[cfg(has_layer_cuda)]
    if supports_cuda(&x) {
        return cuda::add_rms_norm(x, residual, weight, eps);
    }
    let sum = x.to_kind(Kind::Float) + residual.to_kind(Kind::Float);
    let normalized = rms_norm(&sum, weight, eps).to_kind(x.kind());
    (normalized, sum.to_kind(x.kind()))
}

/// Normalize owned Q/K projection views; CUDA mutates those views in place.
/// Callers must not reuse their pre-normalization contents. V is left untouched.
pub(crate) fn qk_norm(
    q: &Tensor,
    k: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    eps: f64,
) -> (Tensor, Tensor) {
    #[cfg(has_layer_cuda)]
    if supports_cuda(q) {
        return cuda::qk_norm(q, k, q_weight, k_weight, eps);
    }
    (rms_norm(q, q_weight, eps), rms_norm(k, k_weight, eps))
}

/// Packed [gate, up] SwiGLU activation; input is a 2-D tensor of even width.
pub(crate) fn silu_and_mul(gate_up: &Tensor) -> Tensor {
    #[cfg(has_layer_cuda)]
    if supports_cuda(gate_up) {
        return cuda::silu_and_mul(gate_up);
    }
    let width = gate_up.size()[1] / 2;
    gate_up.narrow(-1, 0, width).silu() * gate_up.narrow(-1, width, width)
}

#[cfg(has_layer_cuda)]
fn supports_cuda(tensor: &Tensor) -> bool {
    matches!(tensor.device(), Device::Cuda(_))
        && matches!(tensor.kind(), Kind::BFloat16 | Kind::Half)
}

/// Shared by all decoder layers. Frequencies are materialized once at model creation.
pub(crate) struct HalfSplitRope {
    cos: Tensor,
    sin: Tensor,
    half_dim: i64,
}

impl HalfSplitRope {
    pub(crate) fn new(
        max_positions: i64,
        head_dim: i64,
        rope_theta: f64,
        kind: Kind,
        device: Device,
    ) -> Self {
        let inv_freq = (Tensor::arange_start_step(0, head_dim, 2, (Kind::Float, device))
            * (-(rope_theta.ln() / head_dim as f64)))
            .exp();
        let positions = Tensor::arange(max_positions, (Kind::Float, device));
        let frequencies = positions.unsqueeze(-1) * inv_freq.unsqueeze(0);
        // CUDA RoPE reads the FP32 cache directly, matching SGLang's CUDA path.
        // Keep the CPU cache in the activation dtype for the eager fallback.
        let cache_kind = if matches!(device, Device::Cuda(_)) {
            Kind::Float
        } else {
            kind
        };
        let cos = frequencies.cos().to_kind(cache_kind);
        let sin = frequencies.sin().to_kind(cache_kind);
        Self {
            cos,
            sin,
            half_dim: head_dim / 2,
        }
    }

    /// CUDA mutates owned Q/K views in place; positions must index the cache.
    pub(crate) fn apply(&self, q: &Tensor, k: &Tensor, positions: &Tensor) -> (Tensor, Tensor) {
        #[cfg(has_layer_cuda)]
        if supports_cuda(q) {
            return cuda::rope(q, k, positions, &self.cos, &self.sin);
        }
        let cos = self.cos.index_select(0, positions).unsqueeze(1);
        let sin = self.sin.index_select(0, positions).unsqueeze(1);
        (
            rotate_half(q, &cos, &sin, self.half_dim),
            rotate_half(k, &cos, &sin, self.half_dim),
        )
    }
}

pub(crate) fn rotate_half(x: &Tensor, cos: &Tensor, sin: &Tensor, half_dim: i64) -> Tensor {
    let first = x.narrow(-1, 0, half_dim);
    let second = x.narrow(-1, half_dim, half_dim);
    Tensor::cat(
        &[&first * cos - &second * sin, &second * cos + &first * sin],
        -1,
    )
}

#[cfg(has_layer_cuda)]
mod cuda {
    use std::ffi::{CStr, c_char, c_void};
    use tch::Tensor;

    unsafe extern "C" {
        fn sglang_layers_rms_norm(x: *const c_void, weight: *const c_void, eps: f64)
        -> *mut c_void;
        fn sglang_layers_add_rms_norm_inplace(
            x: *const c_void,
            residual: *const c_void,
            weight: *const c_void,
            eps: f64,
        ) -> bool;
        fn sglang_layers_qk_norm_inplace(
            q: *const c_void,
            k: *const c_void,
            qw: *const c_void,
            kw: *const c_void,
            eps: f64,
        ) -> bool;
        fn sglang_layers_rope_inplace(
            q: *const c_void,
            k: *const c_void,
            positions: *const c_void,
            cos: *const c_void,
            sin: *const c_void,
        ) -> bool;
        fn sglang_layers_silu_and_mul(gate_up: *const c_void) -> *mut c_void;
        fn sglang_layers_error() -> *const c_char;
    }

    fn check(output: *mut c_void) -> Tensor {
        if output.is_null() {
            let error = unsafe { CStr::from_ptr(sglang_layers_error()) };
            panic!(
                "shared layer CUDA kernel failed: {}",
                error.to_string_lossy()
            );
        }
        unsafe { Tensor::from_ptr(output.cast()) }
    }

    fn check_inplace(ok: bool) {
        if !ok {
            let error = unsafe { CStr::from_ptr(sglang_layers_error()) };
            panic!(
                "shared layer CUDA kernel failed: {}",
                error.to_string_lossy()
            );
        }
    }

    pub(super) fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
        check(unsafe { sglang_layers_rms_norm(x.as_ptr().cast(), weight.as_ptr().cast(), eps) })
    }

    pub(super) fn add_rms_norm(
        x: Tensor,
        residual: Tensor,
        weight: &Tensor,
        eps: f64,
    ) -> (Tensor, Tensor) {
        check_inplace(unsafe {
            sglang_layers_add_rms_norm_inplace(
                x.as_ptr().cast(),
                residual.as_ptr().cast(),
                weight.as_ptr().cast(),
                eps,
            )
        });
        (x, residual)
    }

    pub(super) fn qk_norm(
        q: &Tensor,
        k: &Tensor,
        qw: &Tensor,
        kw: &Tensor,
        eps: f64,
    ) -> (Tensor, Tensor) {
        let ok = unsafe {
            sglang_layers_qk_norm_inplace(
                q.as_ptr().cast(),
                k.as_ptr().cast(),
                qw.as_ptr().cast(),
                kw.as_ptr().cast(),
                eps,
            )
        };
        check_inplace(ok);
        (q.shallow_clone(), k.shallow_clone())
    }

    pub(super) fn rope(
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> (Tensor, Tensor) {
        let ok = unsafe {
            sglang_layers_rope_inplace(
                q.as_ptr().cast(),
                k.as_ptr().cast(),
                positions.as_ptr().cast(),
                cos.as_ptr().cast(),
                sin.as_ptr().cast(),
            )
        };
        check_inplace(ok);
        (q.shallow_clone(), k.shallow_clone())
    }

    pub(super) fn silu_and_mul(gate_up: &Tensor) -> Tensor {
        check(unsafe { sglang_layers_silu_and_mul(gate_up.as_ptr().cast()) })
    }
}

#[cfg(all(test, has_layer_cuda))]
mod tests {
    use super::*;
    use tch::Cuda;

    fn max_error(actual: &Tensor, expected: &Tensor) -> f64 {
        (actual.to_kind(Kind::Float) - expected.to_kind(Kind::Float))
            .abs()
            .max()
            .double_value(&[])
    }

    fn eager_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
        let x_float = x.to_kind(Kind::Float);
        let variance = x_float
            .pow_tensor_scalar(2)
            .mean_dim(&[-1i64][..], true, Kind::Float);
        (x_float * (variance + eps).rsqrt() * weight.to_kind(Kind::Float)).to_kind(x.kind())
    }

    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn fused_cuda_ops_match_eager_bfloat16() {
        assert!(Cuda::is_available(), "CUDA test requires an available GPU");
        let device = Device::Cuda(0);
        tch::manual_seed(7);
        let hidden = Tensor::randn([3, 1024], (Kind::BFloat16, device));
        let weight = Tensor::randn([1024], (Kind::BFloat16, device));
        assert!(
            max_error(
                &rms_norm(&hidden, &weight, 1e-6),
                &eager_norm(&hidden, &weight, 1e-6)
            ) <= 0.03125
        );

        let packed = Tensor::randn([3, 4096], (Kind::BFloat16, device));
        let q = packed.narrow(1, 0, 2048).view([3, 16, 128]);
        let k = packed.narrow(1, 2048, 1024).view([3, 8, 128]);
        let qw = Tensor::randn([128], (Kind::BFloat16, device));
        let kw = Tensor::randn([128], (Kind::BFloat16, device));
        let expected_q = eager_norm(&q, &qw, 1e-6);
        let expected_k = eager_norm(&k, &kw, 1e-6);
        let v_before = packed.narrow(1, 3072, 1024).to_kind(Kind::Float);
        let q_storage = q.data_ptr();
        let k_storage = k.data_ptr();
        let (qn, kn) = qk_norm(&q, &k, &qw, &kw, 1e-6);
        assert_eq!(qn.data_ptr(), q_storage);
        assert_eq!(kn.data_ptr(), k_storage);
        assert!(max_error(&qn, &expected_q) <= 0.03125);
        assert!(max_error(&kn, &expected_k) <= 0.03125);

        let rope = HalfSplitRope::new(32, 128, 1_000_000.0, Kind::BFloat16, device);
        assert_eq!(rope.cos.kind(), Kind::Float);
        assert_eq!(rope.sin.kind(), Kind::Float);
        let positions = Tensor::from_slice(&[0i64, 31, 7]).to_device(device);
        let cos = rope.cos.index_select(0, &positions).unsqueeze(1);
        let sin = rope.sin.index_select(0, &positions).unsqueeze(1);
        let expected_qr = rotate_half(&qn, &cos, &sin, 64).to_kind(Kind::BFloat16);
        let expected_kr = rotate_half(&kn, &cos, &sin, 64).to_kind(Kind::BFloat16);
        let (qr, kr) = rope.apply(&qn, &kn, &positions);
        assert_eq!(qr.data_ptr(), q_storage);
        assert_eq!(kr.data_ptr(), k_storage);
        assert!(max_error(&qr, &expected_qr) <= 0.0625);
        assert!(max_error(&kr, &expected_kr) <= 0.0625);
        assert_eq!(max_error(&packed.narrow(1, 3072, 1024), &v_before), 0.0);

        let gate_up = Tensor::randn([3, 6144], (Kind::BFloat16, device));
        let expected = gate_up.narrow(1, 0, 3072).silu() * gate_up.narrow(1, 3072, 3072);
        assert!(max_error(&silu_and_mul(&gate_up), &expected) <= 0.03125);
        let unaligned = Tensor::randn([3, 26], (Kind::BFloat16, device));
        let expected_unaligned = unaligned.narrow(1, 0, 13).silu() * unaligned.narrow(1, 13, 13);
        assert!(max_error(&silu_and_mul(&unaligned), &expected_unaligned) <= 0.03125);
    }

    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn fused_add_norm_matches_fp32_sum_for_both_cuda_dtypes() {
        assert!(Cuda::is_available(), "CUDA test requires an available GPU");
        tch::manual_seed(19);
        for kind in [Kind::BFloat16, Kind::Half] {
            for tokens in [1, 3, 65] {
                for width in [13, 1024] {
                    let x = Tensor::randn([tokens, width], (kind, Device::Cuda(0)));
                    let residual = Tensor::randn_like(&x);
                    let weight = Tensor::randn([width], (kind, Device::Cuda(0)));
                    let expected_residual = &x + &residual;
                    let expected = eager_norm(
                        &(x.to_kind(Kind::Float) + residual.to_kind(Kind::Float)),
                        &weight,
                        1e-6,
                    )
                    .to_kind(kind);
                    let x_ptr = x.data_ptr();
                    let residual_ptr = residual.data_ptr();
                    let (actual, summed) = add_rms_norm(x, residual, &weight, 1e-6);
                    assert_eq!(actual.data_ptr(), x_ptr);
                    assert_eq!(summed.data_ptr(), residual_ptr);
                    assert_eq!(max_error(&summed, &expected_residual), 0.0);
                    assert!(
                        max_error(&actual, &expected)
                            <= if kind == Kind::Half {
                                0.0078125
                            } else {
                                0.0625
                            }
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn cuda_norm_handles_padded_rows_and_partial_head_warps() {
        assert!(Cuda::is_available(), "CUDA test requires an available GPU");
        tch::manual_seed(41);
        for kind in [Kind::BFloat16, Kind::Half] {
            let tolerance = if kind == Kind::Half {
                0.0078125
            } else {
                0.0625
            };
            for padding in [1, 16] {
                let storage = Tensor::randn([5, 1024 + padding], (kind, Device::Cuda(0)));
                let x = storage.narrow(1, 0, 1024);
                let residual_storage = Tensor::randn_like(&storage);
                let residual = residual_storage.narrow(1, 0, 1024);
                let weight = Tensor::randn([1024], (kind, Device::Cuda(0)));
                let expected_residual = &x + &residual;
                let expected = eager_norm(
                    &(x.to_kind(Kind::Float) + residual.to_kind(Kind::Float)),
                    &weight,
                    1e-6,
                )
                .to_kind(kind);
                let padding_before = storage.narrow(1, 1024, padding).copy();
                let residual_padding = residual_storage.narrow(1, 1024, padding).copy();
                let (actual, summed) = add_rms_norm(x, residual, &weight, 1e-6);
                assert!(summed.equal(&expected_residual));
                assert!(max_error(&actual, &expected) <= tolerance);
                assert!(storage.narrow(1, 1024, padding).equal(&padding_before));
                assert!(
                    residual_storage
                        .narrow(1, 1024, padding)
                        .equal(&residual_padding)
                );
            }
            let packed = Tensor::randn([3, 7, 128], (kind, Device::Cuda(0)));
            let q = packed.narrow(1, 0, 3);
            let k = packed.narrow(1, 3, 2);
            let v = packed.narrow(1, 5, 2).copy();
            let qw = Tensor::randn([128], (kind, Device::Cuda(0)));
            let kw = Tensor::randn([128], (kind, Device::Cuda(0)));
            let expected_q = eager_norm(&q, &qw, 1e-6);
            let expected_k = eager_norm(&k, &kw, 1e-6);
            let (q, k) = qk_norm(&q, &k, &qw, &kw, 1e-6);
            assert!(max_error(&q, &expected_q) <= tolerance);
            assert!(max_error(&k, &expected_k) <= tolerance);
            assert!(packed.narrow(1, 5, 2).equal(&v));
        }
    }
}

#[cfg(test)]
mod residual_semantics_tests {
    use super::*;

    #[test]
    fn normalization_uses_unrounded_fp32_residual_sum() {
        check_normalization_uses_unrounded_fp32_residual_sum(Device::Cpu);
    }

    #[cfg(has_layer_cuda)]
    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn cuda_normalization_uses_unrounded_fp32_residual_sum() {
        assert!(
            tch::Cuda::is_available(),
            "CUDA test requires an available GPU"
        );
        check_normalization_uses_unrounded_fp32_residual_sum(Device::Cuda(0));
    }

    fn check_normalization_uses_unrounded_fp32_residual_sum(device: Device) {
        let x = Tensor::ones([1, 1024], (Kind::BFloat16, device));
        let residual = Tensor::from_slice(&[0.00390625f32, 0.0078125])
            .to_kind(Kind::BFloat16)
            .to_device(device)
            .view([1, 2])
            .repeat([1, 512]);
        let weight = Tensor::ones([1024], (Kind::BFloat16, device));
        let rounded = rms_norm(&(&x + &residual), &weight, 1e-6);
        let expected = Tensor::ones_like(&x);
        assert!(
            !rounded.equal(&expected),
            "fixture must distinguish the old semantics"
        );
        let (actual, sum) = add_rms_norm(x, residual, &weight, 1e-6);
        assert!(actual.equal(&expected));
        assert!(
            sum.narrow(1, 0, 2).equal(
                &Tensor::from_slice(&[1f32, 1.0078125])
                    .to_kind(Kind::BFloat16)
                    .to_device(device)
                    .view([1, 2])
            )
        );
    }
}

#[cfg(test)]
mod rope_tests {
    use super::*;

    #[test]
    fn rope_keeps_bfloat16_query_and_key() {
        let q = Tensor::ones([2, 2, 4], (Kind::BFloat16, Device::Cpu));
        let k = Tensor::ones([2, 1, 4], (Kind::BFloat16, Device::Cpu));
        let positions = Tensor::from_slice(&[0i64, 1]);
        let rope = HalfSplitRope::new(16, 4, 10_000.0, Kind::BFloat16, Device::Cpu);
        let (q, k) = rope.apply(&q, &k, &positions);
        assert_eq!(q.kind(), Kind::BFloat16);
        assert_eq!(k.kind(), Kind::BFloat16);
    }

    #[test]
    fn cached_rope_matches_eager_formula_for_reordered_and_last_positions() {
        for kind in [Kind::Float, Kind::BFloat16] {
            let q = Tensor::arange(32, (Kind::Float, Device::Cpu))
                .view([4, 2, 4])
                .to_kind(kind);
            let k = Tensor::arange(16, (Kind::Float, Device::Cpu))
                .view([4, 1, 4])
                .to_kind(kind);
            let positions = Tensor::from_slice(&[0i64, 15, 3, 15]);
            let rope = HalfSplitRope::new(16, 4, 10_000.0, kind, Device::Cpu);
            let (cached_q, cached_k) = rope.apply(&q, &k, &positions);

            let inv_freq = (Tensor::arange_start_step(0, 4, 2, (Kind::Float, Device::Cpu))
                * (-(10_000.0_f64.ln() / 4.0)))
                .exp();
            let frequencies = positions.to_kind(Kind::Float).unsqueeze(-1) * inv_freq.unsqueeze(0);
            let cos = frequencies.cos().to_kind(kind).unsqueeze(1);
            let sin = frequencies.sin().to_kind(kind).unsqueeze(1);
            let eager_q = rotate_half(&q, &cos, &sin, 2);
            let eager_k = rotate_half(&k, &cos, &sin, 2);
            let q_error = (cached_q - eager_q).abs().max().double_value(&[]);
            let k_error = (cached_k - eager_k).abs().max().double_value(&[]);
            assert!(q_error <= 1e-5, "kind={kind:?}, q_error={q_error}");
            assert!(k_error <= 1e-5, "kind={kind:?}, k_error={k_error}");
        }
    }
}
