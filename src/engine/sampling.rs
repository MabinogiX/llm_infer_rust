//! Token sampling strategies backed by libtorch.

use std::fmt;

use tch::{Kind, TchError, Tensor};

/// Per-request generation settings, mirroring mini-sglang's `SamplingParams`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_k: i64,
    pub top_p: f64,
    pub ignore_eos: bool,
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            ignore_eos: false,
            max_tokens: 1024,
        }
    }
}

impl SamplingParams {
    /// Applies the same safe defaults as mini-sglang's HTTP-facing config.
    pub fn normalized(self) -> Self {
        Self {
            max_tokens: self.max_tokens.max(1),
            top_p: if self.top_p.is_finite() && (0.0..=1.0).contains(&self.top_p) {
                self.top_p
            } else {
                1.0
            },
            ..self
        }
    }

    fn has_same_distribution(self, other: Self) -> bool {
        self.temperature.to_bits() == other.temperature.to_bits()
            && self.top_k == other.top_k
            && self.top_p.to_bits() == other.top_p.to_bits()
    }
}

#[derive(Debug)]
pub enum SamplingError {
    InvalidLogitsShape(Vec<i64>),
    ParamsLengthMismatch { logits_rows: usize, params: usize },
    Torch(TchError),
    Native(String),
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLogitsShape(shape) => {
                write!(
                    f,
                    "logits 必须是二维 (num_reqs, vocab_size)，实际为 {shape:?}"
                )
            }
            Self::ParamsLengthMismatch {
                logits_rows,
                params,
            } => write!(
                f,
                "SamplingParams 数量 ({params}) 必须与 logits 行数 ({logits_rows}) 相同"
            ),
            Self::Native(error) => write!(f, "FlashInfer sampling error: {error}"),
            Self::Torch(error) => write!(f, "libtorch sampling error: {error}"),
        }
    }
}

impl std::error::Error for SamplingError {}

impl From<TchError> for SamplingError {
    fn from(error: TchError) -> Self {
        Self::Torch(error)
    }
}

pub type Result<T> = std::result::Result<T, SamplingError>;

/// Samples one next-token ID per logits row.
#[derive(Debug, Default, Clone, Copy)]
pub struct Sampler;

impl Sampler {
    /// Samples rows that all share one set of sampling parameters.
    pub fn sample(&self, logits: &Tensor, params: SamplingParams) -> Result<Tensor> {
        validate_logits(logits)?;
        let params = params.normalized();
        if params.temperature <= 0.0 {
            return Ok(logits.argmax(-1, false));
        }

        if params.top_p == 0.0 {
            return Ok(logits.argmax(-1, false));
        }
        let scaled = logits / params.temperature;
        let vocab_size = logits.size()[1];
        let top_k = if params.top_k > 0 {
            params.top_k.min(vocab_size)
        } else {
            vocab_size
        };
        #[cfg(has_flashinfer)]
        if matches!(logits.device(), tch::Device::Cuda(_)) {
            // Joint filtering evaluates both cutoffs on the original softmax.
            return sample_flashinfer(&scaled.softmax(-1, Kind::Float), top_k, params.top_p);
        }
        let filtered = apply_joint_filters(&scaled, top_k, params.top_p);

        Ok(filtered
            .softmax(-1, Kind::Float)
            .multinomial(1, false)
            .squeeze_dim(-1))
    }

    /// Samples a heterogeneous batch while preserving its original row order.
    /// Requests with equal temperature/top-k/top-p settings share one
    /// sampling call. CUDA uses FlashInfer joint top-k/top-p; CPU uses a matching
    /// probability-threshold fallback.
    pub fn sample_batch(
        &self,
        logits: &Tensor,
        params_list: &[SamplingParams],
    ) -> Result<Vec<i64>> {
        validate_logits(logits)?;
        let num_rows = usize::try_from(logits.size()[0]).expect("validated non-negative shape");
        if num_rows != params_list.len() {
            return Err(SamplingError::ParamsLengthMismatch {
                logits_rows: num_rows,
                params: params_list.len(),
            });
        }
        if num_rows == 0 {
            return Ok(Vec::new());
        }

        if params_list.iter().all(|params| params.temperature <= 0.0) {
            return Vec::<i64>::try_from(&logits.argmax(-1, false)).map_err(Into::into);
        }

        let mut groups: Vec<(SamplingParams, Vec<i64>)> = Vec::new();
        for (row, params) in params_list.iter().copied().enumerate() {
            let params = params.normalized();
            if let Some((_, rows)) = groups
                .iter_mut()
                .find(|(group_params, _)| group_params.has_same_distribution(params))
            {
                rows.push(row as i64);
            } else {
                groups.push((params, vec![row as i64]));
            }
        }

        if groups.len() == 1 {
            let output = self.sample(logits, groups[0].0)?;
            return sampled_ids(&output);
        }
        let mut output = Tensor::zeros([num_rows as i64], (Kind::Int64, logits.device()));
        for (params, rows) in groups {
            let row_indices = Tensor::from_slice(&rows).to_device(logits.device());
            let sampled = self.sample(&logits.index_select(0, &row_indices), params)?;
            output = output.index_copy(0, &row_indices, &sampled);
        }
        sampled_ids(&output)
    }
}

fn sampled_ids(output: &Tensor) -> Result<Vec<i64>> {
    let ids = Vec::<i64>::try_from(output)?;
    if ids.iter().any(|&id| id < 0) {
        return Err(SamplingError::Native(
            "invalid probability distribution".to_owned(),
        ));
    }
    Ok(ids)
}

#[cfg(has_flashinfer)]
fn sample_flashinfer(probs: &Tensor, top_k: i64, top_p: f64) -> Result<Tensor> {
    use std::ffi::{CStr, c_void};
    unsafe extern "C" {
        fn sglang_sampling_flashinfer(probs: *const c_void, top_k: i64, top_p: f64) -> *mut c_void;
        fn sglang_sampling_error() -> *const std::ffi::c_char;
    }
    let output = unsafe { sglang_sampling_flashinfer(probs.as_ptr().cast(), top_k, top_p) };
    if output.is_null() {
        return Err(SamplingError::Native(
            unsafe { CStr::from_ptr(sglang_sampling_error()) }
                .to_string_lossy()
                .into_owned(),
        ));
    }
    Ok(unsafe { Tensor::from_ptr(output.cast()) })
}

fn validate_logits(logits: &Tensor) -> Result<()> {
    let shape = logits.size();
    if shape.len() != 2 || shape[1] <= 0 {
        return Err(SamplingError::InvalidLogitsShape(shape));
    }
    Ok(())
}

// Both cutoffs are computed before either filter is applied. Preserve ties at
// the boundary, matching FlashInfer's probability threshold semantics.
fn apply_joint_filters(logits: &Tensor, top_k: i64, top_p: f64) -> Tensor {
    let mut remove = Tensor::zeros_like(logits).to_kind(Kind::Bool);
    let vocab = logits.size()[1];
    if top_k > 0 && top_k < vocab {
        let (values, _) = logits.topk(top_k, -1, true, true);
        remove = logits.lt_tensor(&values.select(-1, top_k - 1).unsqueeze(-1));
    }
    if top_p < 1.0 {
        let probs = logits.softmax(-1, Kind::Float);
        let (sorted, _) = probs.sort(-1, true);
        let crossing = sorted
            .cumsum(-1, Kind::Float)
            .lt(top_p)
            .sum_dim_intlist([-1].as_slice(), true, Kind::Int64)
            .clamp_max(vocab - 1);
        let threshold = sorted.gather(-1, &crossing, false);
        remove = remove.logical_or(&probs.lt_tensor(&threshold));
    }
    logits.masked_fill(&remove, f64::NEG_INFINITY)
}

#[cfg(all(test, has_flashinfer))]
fn apply_top_p(logits: &Tensor, top_p: f64) -> Tensor {
    let (sorted_logits, sorted_indices) = logits.sort(-1, true);
    let cumulative_probs = sorted_logits
        .softmax(-1, Kind::Float)
        .cumsum(-1, Kind::Float);
    let remove = cumulative_probs.gt(top_p);

    // Keep the first token whose cumulative probability crosses the threshold.
    let mut first_column_shape = remove.size();
    let last_dim = first_column_shape.len() - 1;
    first_column_shape[last_dim] = 1;
    let first_column = Tensor::zeros(first_column_shape, (Kind::Bool, logits.device()));
    let shifted = Tensor::cat(
        &[first_column, remove.slice(-1, 0, logits.size()[1] - 1, 1)],
        -1,
    );
    let removal_mask = Tensor::zeros_like(&shifted).scatter(-1, &sorted_indices, &shifted);
    logits.masked_fill(&removal_mask, f64::NEG_INFINITY)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits() -> Tensor {
        Tensor::from_slice(&[0.1f32, 2.0, 0.2, 3.0, 1.0, 0.5]).view([2, 3])
    }

    #[test]
    fn samples_greedily_for_non_positive_temperature() {
        let sampled = Sampler
            .sample_batch(&logits(), &[SamplingParams::default(); 2])
            .unwrap();
        assert_eq!(sampled, vec![1, 0]);
    }

    #[test]
    fn top_k_one_makes_sampling_deterministic() {
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            ..Default::default()
        };
        let sampled = Sampler.sample_batch(&logits(), &[params; 2]).unwrap();
        assert_eq!(sampled, vec![1, 0]);
    }

    #[test]
    fn top_p_zero_keeps_the_highest_logit() {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.0,
            ..Default::default()
        };
        let sampled = Sampler.sample_batch(&logits(), &[params; 2]).unwrap();
        assert_eq!(sampled, vec![1, 0]);
    }

    #[cfg(has_flashinfer)]
    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn flashinfer_sampling_preserves_support_distribution_and_rng() {
        assert!(
            tch::Cuda::is_available(),
            "CUDA test requires an available GPU"
        );
        let device = tch::Device::Cuda(0);
        let logits = Tensor::from_slice(&[4f32, 3., 2., 1., 0.])
            .view([1, 5])
            .to_device(device);
        for (top_k, top_p) in [(-1, 1.0), (-1, 0.8), (3, 0.9), (1, 1.0), (-1, 0.0)] {
            let params = SamplingParams {
                temperature: 1.,
                top_k,
                top_p,
                ..Default::default()
            };
            let filtered = apply_joint_filters(&logits, top_k, top_p);
            let expected = Vec::<f32>::try_from(
                &filtered
                    .softmax(-1, Kind::Float)
                    .to_device(tch::Device::Cpu)
                    .view([-1]),
            )
            .unwrap();
            let repeated = logits.repeat([20000, 1]);
            tch::Cuda::manual_seed(12345);
            let first = Sampler
                .sample_batch(&repeated, &vec![params; 20000])
                .unwrap();
            tch::Cuda::manual_seed(12345);
            assert_eq!(
                first,
                Sampler
                    .sample_batch(&repeated, &vec![params; 20000])
                    .unwrap()
            );
            let mut counts = [0usize; 5];
            for token in first {
                counts[token as usize] += 1;
            }
            for i in 0..5 {
                if expected[i] == 0. {
                    assert_eq!(counts[i], 0);
                }
                assert!(
                    (counts[i] as f32 / 20000. - expected[i]).abs() < 0.02,
                    "k={top_k},p={top_p},counts={counts:?},expected={expected:?}"
                );
            }
        }
        let params = [
            SamplingParams::default(),
            SamplingParams {
                temperature: 1.,
                top_k: 1,
                ..Default::default()
            },
            SamplingParams {
                temperature: 1.,
                top_p: 0.,
                ..Default::default()
            },
        ];
        assert_eq!(
            Sampler
                .sample_batch(&logits.repeat([3, 1]), &params)
                .unwrap(),
            vec![0, 0, 0]
        );
        // Exercise the vectorized kernels at the actual Qwen3 vocabulary size.
        // Small odd vocabularies instantiate a different CUDA kernel variant.
        let wide = Tensor::randn([2, 151936], (Kind::BFloat16, device));
        for (top_k, top_p) in [(-1, 1.0), (-1, 0.9), (50, 0.9)] {
            let ids = Sampler
                .sample_batch(
                    &wide,
                    &[SamplingParams {
                        temperature: 0.8,
                        top_k,
                        top_p,
                        ..Default::default()
                    }; 2],
                )
                .unwrap();
            assert!(ids.iter().all(|&id| (0..151936).contains(&id)));
        }
        let tied = Tensor::from_slice(&[1f32, 1., 0.])
            .view([1, 3])
            .to_device(device)
            .repeat([1024, 1]);
        let tied_samples = Sampler
            .sample_batch(
                &tied,
                &vec![
                    SamplingParams {
                        temperature: 1.,
                        top_k: 1,
                        ..Default::default()
                    };
                    1024
                ],
            )
            .unwrap();
        assert!(tied_samples.contains(&0) && tied_samples.contains(&1));
        assert!(!tied_samples.contains(&2));
    }

    #[cfg(has_flashinfer)]
    #[test]
    #[ignore = "GPU microbenchmark; run alone with --nocapture"]
    fn benchmark_sampling_cuda() {
        assert!(
            tch::Cuda::is_available(),
            "CUDA test requires an available GPU"
        );
        let device = tch::Device::Cuda(0);
        for rows in [1i64, 8] {
            let logits = Tensor::randn([rows, 151936], (Kind::Float, device));
            for (top_k, top_p) in [(-1, 1.0), (-1, 0.9), (50, 0.9)] {
                let params = SamplingParams {
                    temperature: 0.8,
                    top_k,
                    top_p,
                    ..Default::default()
                };
                let old = || {
                    let mut filtered = &logits / params.temperature;
                    if top_k > 0 {
                        let (values, _) = filtered.topk(top_k, -1, true, true);
                        filtered = filtered.masked_fill(
                            &filtered.lt_tensor(&values.select(-1, top_k - 1).unsqueeze(-1)),
                            f64::NEG_INFINITY,
                        );
                    }
                    if top_p < 1. {
                        filtered = apply_top_p(&filtered, top_p);
                    }
                    filtered.softmax(-1, Kind::Float).multinomial(1, false)
                };
                for _ in 0..10 {
                    let _ = old();
                    let _ = Sampler.sample(&logits, params).unwrap();
                }
                let mut times = Vec::new();
                for optimized in [false, true] {
                    tch::Cuda::synchronize(0);
                    let start = std::time::Instant::now();
                    for _ in 0..100 {
                        let output = if optimized {
                            Sampler.sample(&logits, params).unwrap()
                        } else {
                            old()
                        };
                        let _ = Vec::<i64>::try_from(&output.view([-1])).unwrap();
                    }
                    tch::Cuda::synchronize(0);
                    times.push(start.elapsed().as_secs_f64() * 1000. / 100.);
                }
                println!(
                    "sampling rows={rows} k={top_k} p={top_p} old_ms={:.4} new_ms={:.4}",
                    times[0], times[1]
                );
            }
        }
    }

    #[test]
    fn joint_top_k_top_p_uses_original_probability_mass() {
        check_joint_top_k_top_p_uses_original_probability_mass(tch::Device::Cpu);
    }

    #[cfg(has_flashinfer)]
    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn cuda_joint_top_k_top_p_uses_original_probability_mass() {
        assert!(
            tch::Cuda::is_available(),
            "CUDA test requires an available GPU"
        );
        check_joint_top_k_top_p_uses_original_probability_mass(tch::Device::Cuda(0));
    }

    fn check_joint_top_k_top_p_uses_original_probability_mass(device: tch::Device) {
        // Original nucleus keeps 0.4 and 0.3 for p=0.5; top-k=2 keeps
        // both. Sequential renormalization would incorrectly keep only 0.4.
        let logits = Tensor::from_slice(&[0.4f32, 0.3, 0.2, 0.1])
            .log()
            .view([1, 4])
            .to_device(device)
            .repeat([20000, 1]);
        let params = SamplingParams {
            temperature: 1.,
            top_k: 2,
            top_p: 0.5,
            ..Default::default()
        };
        let ids = Sampler.sample_batch(&logits, &vec![params; 20000]).unwrap();
        assert!(ids.iter().all(|&id| id == 0 || id == 1));
        let frequency = ids.iter().filter(|&&id| id == 1).count() as f64 / 20000.;
        assert!(
            (frequency - 3. / 7.).abs() < 0.02,
            "device={device:?},frequency={frequency}"
        );
    }

    #[test]
    fn rejects_mismatched_batch_metadata() {
        assert!(matches!(
            Sampler.sample_batch(&logits(), &[SamplingParams::default()]),
            Err(SamplingError::ParamsLengthMismatch { .. })
        ));
    }
}
