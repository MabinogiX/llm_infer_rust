//! Per-layer paged KV-cache binding, writes, and reads.

use tch::{Device, Kind, Tensor};

use crate::engine::{BatchPhase, ModelRunnerError};

type Result<T> = std::result::Result<T, ModelRunnerError>;

/// Owns one attention layer's K/V views into the global [`KVCachePool`].
#[derive(Debug, Default)]
pub struct BaseAttention {
    k_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
}

impl BaseAttention {
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

    /// Writes `(tokens, kv_heads, head_dim)` K/V values to flattened page slots.
    /// `-1` write locations are deliberately skipped.
    pub fn write_kv(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        write_loc: Option<&Tensor>,
        phase: BatchPhase,
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

        // Decode locations are checked by the scheduler while still on the
        // host. Keep the CUDA path free of device-to-host synchronization.
        if phase == BatchPhase::Decode && matches!(k_cache.device(), Device::Cuda(_)) {
            let indices = write_loc.to_kind(Kind::Int64);
            let mut flat_k = k_cache.view([-1, flat_size[2], flat_size[3]]);
            let mut flat_v = v_cache.view([-1, flat_size[2], flat_size[3]]);
            let _ = flat_k.index_copy_(0, &indices, k);
            let _ = flat_v.index_copy_(0, &indices, v);
            return Ok(());
        }

        let locations = Vec::<i32>::try_from(&write_loc.to_device(Device::Cpu))
            .map_err(ModelRunnerError::Torch)?;

        let capacity = flat_size[0] * flat_size[1];
        let mut cache_indices = Vec::new();
        let mut input_indices = Vec::new();
        for (input_index, location) in locations.into_iter().enumerate() {
            if location < 0 {
                continue;
            }
            if i64::from(location) >= capacity {
                return Err(model_error("write_loc points outside the bound KV cache"));
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
}
