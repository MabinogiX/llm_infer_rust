//! Qwen3 elementwise operators. CUDA dispatch stays behind this module so the
//! decoder only expresses model operations, irrespective of the kernel backend.

use tch::{Device, Kind, Tensor};

pub(super) fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    #[cfg(has_qwen3_cuda)]
    if supports_cuda(x) {
        return cuda::rms_norm(x, weight, eps);
    }
    let x_float = x.to_kind(Kind::Float);
    let variance = x_float
        .pow_tensor_scalar(2)
        .mean_dim(&[-1i64][..], true, Kind::Float);
    (x_float * (variance + eps).rsqrt() * weight.to_kind(Kind::Float)).to_kind(x.kind())
}

pub(super) fn qk_norm(
    q: &Tensor,
    k: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    eps: f64,
) -> (Tensor, Tensor) {
    #[cfg(has_qwen3_cuda)]
    if supports_cuda(q) {
        return cuda::qk_norm(q, k, q_weight, k_weight, eps);
    }
    (rms_norm(q, q_weight, eps), rms_norm(k, k_weight, eps))
}

pub(super) fn silu_and_mul(gate_up: &Tensor) -> Tensor {
    #[cfg(has_qwen3_cuda)]
    if supports_cuda(gate_up) {
        return cuda::silu_and_mul(gate_up);
    }
    let width = gate_up.size()[1] / 2;
    gate_up.narrow(-1, 0, width).silu() * gate_up.narrow(-1, width, width)
}

#[cfg(has_qwen3_cuda)]
fn supports_cuda(tensor: &Tensor) -> bool {
    matches!(tensor.device(), Device::Cuda(_))
        && matches!(tensor.kind(), Kind::BFloat16 | Kind::Half)
}

/// Shared by all decoder layers. Frequencies are materialized once at model creation.
pub(super) struct RopeCache {
    cos: Tensor,
    sin: Tensor,
    half_dim: i64,
}

impl RopeCache {
    pub(super) fn new(
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

    pub(super) fn apply(&self, q: &Tensor, k: &Tensor, positions: &Tensor) -> (Tensor, Tensor) {
        #[cfg(has_qwen3_cuda)]
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

pub(super) fn rotate_half(x: &Tensor, cos: &Tensor, sin: &Tensor, half_dim: i64) -> Tensor {
    let first = x.narrow(-1, 0, half_dim);
    let second = x.narrow(-1, half_dim, half_dim);
    Tensor::cat(
        &[&first * cos - &second * sin, &second * cos + &first * sin],
        -1,
    )
}

#[cfg(has_qwen3_cuda)]
mod cuda {
    use std::ffi::{CStr, c_char, c_void};
    use tch::Tensor;

    unsafe extern "C" {
        fn sglang_qwen3_rms_norm(x: *const c_void, weight: *const c_void, eps: f64) -> *mut c_void;
        fn sglang_qwen3_qk_norm_inplace(
            q: *const c_void,
            k: *const c_void,
            qw: *const c_void,
            kw: *const c_void,
            eps: f64,
        ) -> bool;
        fn sglang_qwen3_rope_inplace(
            q: *const c_void,
            k: *const c_void,
            positions: *const c_void,
            cos: *const c_void,
            sin: *const c_void,
        ) -> bool;
        fn sglang_qwen3_silu_and_mul(gate_up: *const c_void) -> *mut c_void;
        fn sglang_qwen3_error() -> *const c_char;
    }

    fn check(output: *mut c_void) -> Tensor {
        if output.is_null() {
            let error = unsafe { CStr::from_ptr(sglang_qwen3_error()) };
            panic!("Qwen3 CUDA kernel failed: {}", error.to_string_lossy());
        }
        unsafe { Tensor::from_ptr(output.cast()) }
    }

    fn check_inplace(ok: bool) {
        if !ok {
            let error = unsafe { CStr::from_ptr(sglang_qwen3_error()) };
            panic!("Qwen3 CUDA kernel failed: {}", error.to_string_lossy());
        }
    }

    pub(super) fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
        check(unsafe { sglang_qwen3_rms_norm(x.as_ptr().cast(), weight.as_ptr().cast(), eps) })
    }

    pub(super) fn qk_norm(
        q: &Tensor,
        k: &Tensor,
        qw: &Tensor,
        kw: &Tensor,
        eps: f64,
    ) -> (Tensor, Tensor) {
        let ok = unsafe {
            sglang_qwen3_qk_norm_inplace(
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
            sglang_qwen3_rope_inplace(
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
        check(unsafe { sglang_qwen3_silu_and_mul(gate_up.as_ptr().cast()) })
    }
}

#[cfg(all(test, has_qwen3_cuda))]
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
    fn fused_cuda_ops_match_eager_bfloat16() {
        if !Cuda::is_available() {
            return;
        }
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

        let rope = RopeCache::new(32, 128, 1_000_000.0, Kind::BFloat16, device);
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
}
