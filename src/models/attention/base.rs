//! Per-layer paged KV-cache binding, writes, and reads.

use tch::{Device, Kind, Tensor};

use crate::engine::{BatchPhase, ModelRunnerError};

type Result<T> = std::result::Result<T, ModelRunnerError>;

/// Owns one attention layer's K/V views into the global [`KVCachePool`].
#[derive(Debug)]
pub struct BaseAttention {
    k_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
    reserved_write_slot: i64,
}

impl Default for BaseAttention {
    fn default() -> Self {
        Self {
            k_cache: None,
            v_cache: None,
            reserved_write_slot: -1,
        }
    }
}

impl BaseAttention {
    /// Pool-backed production models skip slot 0; raw test caches can disable it.
    pub fn set_reserved_write_slot(&mut self, slot: i64) {
        self.reserved_write_slot = slot;
    }

    /// Binds layer-local tensors shaped `(pages, page_size, kv_heads, head_dim)`.
    pub fn bind_kv_cache(&mut self, k_cache: Tensor, v_cache: Tensor) -> Result<()> {
        if k_cache.size() != v_cache.size() || k_cache.dim() != 4 {
            return Err(model_error(
                "K/V cache must have matching (pages, page_size, kv_heads, head_dim) shapes",
            ));
        }
        self.k_cache = Some(k_cache);
        self.v_cache = Some(v_cache);
        Ok(())
    }

    pub fn is_bound(&self) -> bool {
        self.k_cache.is_some()
    }

    pub fn cache_tensors(&self) -> Result<(&Tensor, &Tensor)> {
        match (&self.k_cache, &self.v_cache) {
            (Some(k), Some(v)) => Ok((k, v)),
            _ => Err(model_error("paged-KV attention requires a bound KV cache")),
        }
    }

    /// Writes `(tokens, kv_heads, head_dim)` K/V values to flattened page slots.
    /// A configured legal reserved slot is skipped; invalid indices are errors.
    pub fn write_kv(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        write_loc: Option<&Tensor>,
        _phase: BatchPhase,
    ) -> Result<()> {
        let (Some(k_cache), Some(v_cache), Some(write_loc)) =
            (&mut self.k_cache, &mut self.v_cache, write_loc)
        else {
            return Ok(());
        };
        if k.size() != v.size() || k.dim() != 3 {
            return Err(model_error(
                "K/V input must have matching (tokens, kv_heads, head_dim) shapes",
            ));
        }
        if write_loc.numel() != k.size()[0] as usize {
            return Err(model_error(
                "write_loc length must equal the number of K/V tokens",
            ));
        }
        let flat_size = k_cache.size();
        if flat_size[2] != k.size()[1] || flat_size[3] != k.size()[2] {
            return Err(model_error(
                "K/V input head shape does not match the bound cache",
            ));
        }

        // Prefill and decode locations are checked by the scheduler before
        // upload. Keep both CUDA paths free of device-to-host synchronization.
        if matches!(k_cache.device(), Device::Cuda(_)) {
            #[cfg(has_cuda_kv_store)]
            if matches!(k.kind(), Kind::BFloat16 | Kind::Half) {
                return super::kv_store::store(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    write_loc,
                    self.reserved_write_slot,
                );
            }
            let indices = write_loc.to_kind(Kind::Int64);
            // Keep the reserved slot unchanged without a variable-shape mask.
            let reserved = indices.eq(self.reserved_write_slot).view([-1, 1, 1]);
            let old_k = k_cache
                .view([-1, flat_size[2], flat_size[3]])
                .index_select(0, &indices);
            let old_v = v_cache
                .view([-1, flat_size[2], flat_size[3]])
                .index_select(0, &indices);
            let k = old_k.where_self(&reserved, k);
            let v = old_v.where_self(&reserved, v);
            let mut flat_k = k_cache.view([-1, flat_size[2], flat_size[3]]);
            let mut flat_v = v_cache.view([-1, flat_size[2], flat_size[3]]);
            let _ = flat_k.index_copy_(0, &indices, &k);
            let _ = flat_v.index_copy_(0, &indices, &v);
            return Ok(());
        }

        let locations = Vec::<i32>::try_from(&write_loc.to_device(Device::Cpu))
            .map_err(ModelRunnerError::Torch)?;

        let capacity = flat_size[0] * flat_size[1];
        let mut cache_indices = Vec::new();
        let mut input_indices = Vec::new();
        for (input_index, location) in locations.into_iter().enumerate() {
            if location < 0 || i64::from(location) >= capacity {
                return Err(model_error("write_loc points outside the bound KV cache"));
            }
            if i64::from(location) == self.reserved_write_slot {
                continue;
            }
            cache_indices.push(i64::from(location));
            input_indices.push(input_index as i64);
        }
        if cache_indices.is_empty() {
            return Ok(());
        }

        let cache_indices = Tensor::from_slice(&cache_indices).to_device(k_cache.device());
        let input_indices = Tensor::from_slice(&input_indices).to_device(k.device());
        let selected_k = k.index_select(0, &input_indices);
        let selected_v = v.index_select(0, &input_indices);
        let mut flat_k = k_cache.view([-1, flat_size[2], flat_size[3]]);
        let mut flat_v = v_cache.view([-1, flat_size[2], flat_size[3]]);
        let _ = flat_k.index_copy_(0, &cache_indices, &selected_k);
        let _ = flat_v.index_copy_(0, &cache_indices, &selected_v);
        Ok(())
    }

    /// Reads one request's valid cache entries from its `req_to_token` row.
    pub fn read_kv(
        &self,
        req_to_token: &Tensor,
        request_index: i64,
        length: i64,
    ) -> Result<(Tensor, Tensor)> {
        let (Some(k_cache), Some(v_cache)) = (&self.k_cache, &self.v_cache) else {
            return Err(model_error("paged-KV attention requires a bound KV cache"));
        };
        if req_to_token.dim() != 2 || request_index < 0 || request_index >= req_to_token.size()[0] {
            return Err(model_error("invalid req_to_token request index"));
        }
        if length <= 0 || length > req_to_token.size()[1] {
            return Err(model_error("invalid cache sequence length"));
        }
        let locations = Vec::<i32>::try_from(
            &req_to_token
                .get(request_index)
                .narrow(0, 0, length)
                .to_device(Device::Cpu),
        )
        .map_err(ModelRunnerError::Torch)?;
        if locations.iter().any(|&location| location < 0) {
            return Err(model_error(
                "req_to_token contains an invalid cache location",
            ));
        }
        let capacity = k_cache.size()[0] * k_cache.size()[1];
        if locations
            .iter()
            .any(|&location| i64::from(location) >= capacity)
        {
            return Err(model_error(
                "req_to_token points outside the bound KV cache",
            ));
        }
        let indices = Tensor::from_slice(&locations.into_iter().map(i64::from).collect::<Vec<_>>())
            .to_device(k_cache.device());
        let shape = k_cache.size();
        let flat_k = k_cache.view([-1, shape[2], shape[3]]);
        let flat_v = v_cache.view([-1, shape[2], shape[3]]);
        Ok((
            flat_k.index_select(0, &indices),
            flat_v.index_select(0, &indices),
        ))
    }

    /// Gather a fixed number of cache slots per request without CPU reads.
    pub fn read_kv_padded(&self, indices: &Tensor) -> Result<(Tensor, Tensor)> {
        let (Some(k_cache), Some(v_cache)) = (&self.k_cache, &self.v_cache) else {
            return Err(model_error("paged-KV attention requires a bound KV cache"));
        };
        let shape = k_cache.size();
        let output_shape = [indices.size()[0], indices.size()[1], shape[2], shape[3]];
        let flat_k = k_cache.view([-1, shape[2], shape[3]]);
        let flat_v = v_cache.view([-1, shape[2], shape[3]]);
        let flat_indices = indices.view([-1]);
        Ok((
            flat_k.index_select(0, &flat_indices).view(output_shape),
            flat_v.index_select(0, &flat_indices).view(output_shape),
        ))
    }
}

fn model_error(message: &str) -> ModelRunnerError {
    ModelRunnerError::Model(message.to_owned())
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use super::*;

    #[cfg(has_cuda_kv_store)]
    #[test]
    fn cuda_kv_store_preserves_packed_strides_padding_and_unwritten_slots() {
        if !tch::Cuda::is_available() {
            return;
        }
        for kind in [Kind::BFloat16, Kind::Half] {
            for width in [7, 128] {
                for index_kind in [Kind::Int, Kind::Int64] {
                    let device = Device::Cuda(0);
                    let packed = Tensor::randn([5, 6, width], (kind, device));
                    let k = packed.narrow(1, 2, 2);
                    let v = packed.narrow(1, 4, 2);
                    let kc = Tensor::full([3, 4, 2, width], -3., (kind, device));
                    let vc = Tensor::full_like(&kc, 5.);
                    let expected_k = kc.copy();
                    let expected_v = vc.copy();
                    // Cross page boundaries, skip padding, and guard the upper bound.
                    let locations = Tensor::from_slice(&[3i64, 4, 0, 11, 0])
                        .to_kind(index_kind)
                        .to_device(device);
                    for (row, slot) in [(0, 3), (1, 4), (3, 11)] {
                        expected_k.view([12, 2, width]).get(slot).copy_(&k.get(row));
                        expected_v.view([12, 2, width]).get(slot).copy_(&v.get(row));
                    }
                    let packed_before = packed.copy();
                    let mut attention = BaseAttention::default();
                    attention.set_reserved_write_slot(0);
                    attention.bind_kv_cache(kc, vc).unwrap();
                    attention
                        .write_kv(&k, &v, Some(&locations), BatchPhase::Prefill)
                        .unwrap();
                    let (actual_k, actual_v) = attention.cache_tensors().unwrap();
                    assert!(actual_k.equal(&expected_k));
                    assert!(actual_v.equal(&expected_v));
                    assert!(packed.equal(&packed_before));
                    // Decode overwrites only the selected slot with a one-token view.
                    let next = Tensor::from_slice(&[4i32]).to_device(device);
                    attention
                        .write_kv(
                            &k.narrow(0, 0, 1),
                            &v.narrow(0, 0, 1),
                            Some(&next),
                            BatchPhase::Decode,
                        )
                        .unwrap();
                    expected_k.view([12, 2, width]).get(4).copy_(&k.get(0));
                    expected_v.view([12, 2, width]).get(4).copy_(&v.get(0));
                    let (actual_k, actual_v) = attention.cache_tensors().unwrap();
                    assert!(actual_k.equal(&expected_k));
                    assert!(actual_v.equal(&expected_v));
                }
            }
        }
    }

    #[cfg(has_cuda_kv_store)]
    #[test]
    fn cuda_invalid_kv_indices_fail_in_isolated_process() {
        if !tch::Cuda::is_available() {
            return;
        }
        if let Ok(index) = std::env::var("SGLANG_TEST_INVALID_KV_INDEX") {
            let device = Device::Cuda(0);
            let mut attention = BaseAttention::default();
            attention.set_reserved_write_slot(0);
            attention
                .bind_kv_cache(
                    Tensor::zeros([3, 4, 2, 128], (Kind::BFloat16, device)),
                    Tensor::zeros([3, 4, 2, 128], (Kind::BFloat16, device)),
                )
                .unwrap();
            let kv = Tensor::ones([1, 2, 128], (Kind::BFloat16, device));
            attention
                .write_kv(
                    &kv,
                    &kv,
                    Some(&Tensor::from_slice(&[index.parse::<i32>().unwrap()]).to_device(device)),
                    BatchPhase::Decode,
                )
                .unwrap();
            tch::Cuda::synchronize(0);
            return;
        }
        for index in ["-1", "12"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "models::attention::base::tests::cuda_invalid_kv_indices_fail_in_isolated_process", "--nocapture"])
                .env("SGLANG_TEST_INVALID_KV_INDEX", index).output().unwrap();
            let message = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                !output.status.success(),
                "invalid slot {index} unexpectedly succeeded"
            );
            assert!(
                message.contains("device-side assert") || message.contains("Assertion"),
                "{message}"
            );
        }
    }

    #[test]
    fn writes_and_reads_paged_cache_slots() {
        let mut attention = BaseAttention::default();
        attention
            .bind_kv_cache(
                Tensor::zeros([2, 2, 1, 2], (Kind::Float, Device::Cpu)),
                Tensor::zeros([2, 2, 1, 2], (Kind::Float, Device::Cpu)),
            )
            .unwrap();
        let k = Tensor::from_slice(&[1f32, 2., 3., 4.]).view([2, 1, 2]);
        let v = Tensor::from_slice(&[5f32, 6., 7., 8.]).view([2, 1, 2]);
        attention
            .write_kv(
                &k,
                &v,
                Some(&Tensor::from_slice(&[1i32, 3])),
                BatchPhase::Prefill,
            )
            .unwrap();
        let table = Tensor::from_slice(&[1i32, 3]).view([1, 2]);
        let (cached_k, cached_v) = attention.read_kv(&table, 0, 2).unwrap();

        assert_eq!(
            Vec::<f32>::try_from(&cached_k.view([-1])).unwrap(),
            vec![1., 2., 3., 4.]
        );
        assert_eq!(
            Vec::<f32>::try_from(&cached_v.view([-1])).unwrap(),
            vec![5., 6., 7., 8.]
        );
    }
    #[test]
    fn cpu_store_skips_reserved_slot_and_rejects_invalid_locations() {
        let mut attention = BaseAttention::default();
        attention.set_reserved_write_slot(0);
        attention
            .bind_kv_cache(
                Tensor::zeros([1, 4, 1, 2], (Kind::Float, Device::Cpu)),
                Tensor::zeros([1, 4, 1, 2], (Kind::Float, Device::Cpu)),
            )
            .unwrap();
        let kv = Tensor::ones([2, 1, 2], (Kind::Float, Device::Cpu));
        attention
            .write_kv(
                &kv,
                &kv,
                Some(&Tensor::from_slice(&[0i32, 1])),
                BatchPhase::Decode,
            )
            .unwrap();
        let (k, _) = attention.cache_tensors().unwrap();
        assert_eq!(
            k.view([4, 2]).get(0).sum(Kind::Float).double_value(&[]),
            0.0
        );
        assert_eq!(
            k.view([4, 2]).get(1).sum(Kind::Float).double_value(&[]),
            2.0
        );
        for index in [-1i32, 4] {
            assert!(
                attention
                    .write_kv(
                        &kv.narrow(0, 0, 1),
                        &kv.narrow(0, 0, 1),
                        Some(&Tensor::from_slice(&[index])),
                        BatchPhase::Decode
                    )
                    .is_err()
            );
        }
    }
}
