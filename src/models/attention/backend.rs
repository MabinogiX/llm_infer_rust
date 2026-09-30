//! Attention backend dispatch matching mini-sglang's `pt` / `fa` seam.

use tch::{Device, Kind, Tensor};

use crate::engine::{AttentionMetadata, BatchPhase, ModelRunnerError};

use super::BaseAttention;

type Result<T> = std::result::Result<T, ModelRunnerError>;

/// Backend identifiers accepted by the engine configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionBackendKind {
    /// Eager libtorch operations, corresponding to Python's PyTorch/SDPA path.
    Pt,
    /// LibTorch SDPA, which selects the fused FlashAttention kernel on supported CUDA inputs.
    FlashAttention,
}

impl AttentionBackendKind {
    pub fn parse(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "pt" | "pytorch" => Ok(Self::Pt),
            "fa" | "flashattention" | "flash-attention" => Ok(Self::FlashAttention),
            _ => Err(model_error(&format!(
                "unknown attention backend {name:?}; expected \"pt\" or \"fa\""
            ))),
        }
    }
}

/// Architecture-independent attention dispatch boundary.
pub trait AttentionBackend {
    fn kind(&self) -> AttentionBackendKind;

    /// Computes attention after model-specific QKV projection, RoPE, and KV write.
    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cache: &BaseAttention,
        sequence_boundaries: &[i64],
        metadata: Option<&AttentionMetadata>,
        num_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
    ) -> Result<Tensor>;
}

pub fn create_attention_backend(kind: AttentionBackendKind) -> Box<dyn AttentionBackend> {
    match kind {
        AttentionBackendKind::Pt => Box::new(PyTorchAttentionBackend),
        AttentionBackendKind::FlashAttention => Box::new(FlashAttentionBackend),
    }
}

/// Eager libtorch implementation of mini-sglang's Python `PyTorchBackend`.
struct PyTorchAttentionBackend;

impl AttentionBackend for PyTorchAttentionBackend {
    fn kind(&self) -> AttentionBackendKind {
        AttentionBackendKind::Pt
    }

    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cache: &BaseAttention,
        sequence_boundaries: &[i64],
        metadata: Option<&AttentionMetadata>,
        num_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
    ) -> Result<Tensor> {
        if let Some(metadata) = metadata {
            if metadata.forward_mode == BatchPhase::Decode {
                return decode_with_cache(q, cache, metadata, num_heads, num_kv_heads, head_dim);
            }
            if has_cached_prefix(metadata)? {
                return prefill_with_cache(q, cache, metadata, num_heads, num_kv_heads, head_dim);
            }
        }

        let mut outputs = Vec::with_capacity(sequence_boundaries.len().saturating_sub(1));
        for boundaries in sequence_boundaries.windows(2) {
            let start = boundaries[0];
            let length = boundaries[1] - start;
            outputs.push(causal_attention(
                &q.narrow(0, start, length),
                &k.narrow(0, start, length),
                &v.narrow(0, start, length),
                num_heads,
                num_kv_heads,
                head_dim,
            ));
        }
        Ok(Tensor::cat(&outputs, 0))
    }
}

/// Uses LibTorch SDPA so supported CUDA BF16/FP16 inputs take its FlashAttention path.
struct FlashAttentionBackend;

impl AttentionBackend for FlashAttentionBackend {
    fn kind(&self) -> AttentionBackendKind {
        AttentionBackendKind::FlashAttention
    }

    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cache: &BaseAttention,
        sequence_boundaries: &[i64],
        metadata: Option<&AttentionMetadata>,
        num_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
    ) -> Result<Tensor> {
        if let Some(metadata) = metadata {
            if metadata.forward_mode == BatchPhase::Decode {
                return sdpa_decode_with_cache(
                    q,
                    cache,
                    metadata,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );
            }
        }
        let prefixes = if let Some(metadata) = metadata {
            tensor_i32(metadata.prefix_lens.as_ref(), "prefix_lens")?
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>()
        } else {
            vec![0; sequence_boundaries.len().saturating_sub(1)]
        };
        batched_prefill_sdpa(
            q,
            k,
            v,
            cache,
            sequence_boundaries,
            &prefixes,
            metadata,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    }
}

/// Packs variable-length requests into one padded SDPA invocation per model layer.
fn batched_prefill_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cache: &BaseAttention,
    boundaries: &[i64],
    prefixes: &[i64],
    metadata: Option<&AttentionMetadata>,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Result<Tensor> {
    let batch_size = prefixes.len();
    if batch_size == 0
        || boundaries.len() != batch_size + 1
        || boundaries[0] != 0
        || boundaries[batch_size] != q.size()[0]
        || prefixes.iter().any(|&prefix| prefix < 0)
    {
        return Err(model_error(
            "invalid prefill batch boundaries or prefix lengths",
        ));
    }
    let lengths = boundaries
        .windows(2)
        .map(|bounds| bounds[1] - bounds[0])
        .collect::<Vec<_>>();
    if lengths.iter().any(|&length| length <= 0) {
        return Err(model_error("prefill requests must contain uncached tokens"));
    }
    if batch_size == 1 && prefixes[0] == 0 {
        return sdpa_attention(q, k, v, 0, num_heads, num_kv_heads, head_dim);
    }
    let max_query = *lengths.iter().max().unwrap();
    let max_key = lengths
        .iter()
        .zip(prefixes)
        .map(|(&length, &prefix)| length + prefix)
        .max()
        .unwrap();
    let mut padded_indices = Vec::with_capacity(batch_size * max_query as usize);
    let mut output_indices = Vec::with_capacity(q.size()[0] as usize);
    for (row, bounds) in boundaries.windows(2).enumerate() {
        for position in 0..max_query {
            padded_indices.push(bounds[0] + position.min(lengths[row] - 1));
        }
        for position in 0..lengths[row] {
            output_indices.push(row as i64 * max_query + position);
        }
    }
    let indices = Tensor::from_slice(&padded_indices).to_device(q.device());
    let query = q
        .index_select(0, &indices)
        .view([batch_size as i64, max_query, num_heads, head_dim])
        .permute([0, 2, 1, 3]);
    let has_prefix = prefixes.iter().any(|&prefix| prefix > 0);
    let (keys, values) = if has_prefix {
        let table = metadata
            .and_then(|meta| meta.req_to_token.as_ref())
            .ok_or_else(|| model_error("cached prefill requires req_to_token"))?;
        if table.size()[0] != batch_size as i64 || table.size()[1] < max_key {
            return Err(model_error("cached prefill page table is too small"));
        }
        let locations = table.narrow(1, 0, max_key);
        cache.read_kv_padded(&locations.clamp_min(0).to_kind(Kind::Int64))?
    } else {
        let keys = k.index_select(0, &indices).view([
            batch_size as i64,
            max_query,
            num_kv_heads,
            head_dim,
        ]);
        let values = v.index_select(0, &indices).view([
            batch_size as i64,
            max_query,
            num_kv_heads,
            head_dim,
        ]);
        (keys, values)
    };
    let repeats = num_heads / num_kv_heads;
    let keys = keys
        .repeat_interleave_self_int(repeats, 2, Some(num_heads))
        .permute([0, 2, 1, 3]);
    let values = values
        .repeat_interleave_self_int(repeats, 2, Some(num_heads))
        .permute([0, 2, 1, 3]);
    let output = if has_prefix {
        let query_positions = Tensor::arange(max_query, (Kind::Int64, q.device())).unsqueeze(0)
            + Tensor::from_slice(prefixes)
                .to_device(q.device())
                .unsqueeze(1);
        let key_positions = Tensor::arange(max_key, (Kind::Int64, q.device()));
        let key_lengths = lengths
            .iter()
            .zip(prefixes)
            .map(|(&length, &prefix)| length + prefix)
            .collect::<Vec<_>>();
        let mask = key_positions
            .unsqueeze(0)
            .unsqueeze(0)
            .le_tensor(&query_positions.unsqueeze(2))
            .logical_and(
                &key_positions
                    .unsqueeze(0)
                    .lt_tensor(
                        &Tensor::from_slice(&key_lengths)
                            .to_device(q.device())
                            .unsqueeze(1),
                    )
                    .unsqueeze(1),
            )
            .unsqueeze(1);
        Tensor::f_scaled_dot_product_attention(
            &query,
            &keys,
            &values,
            Some(&mask),
            0.0,
            false,
            None,
            false,
        )?
    } else {
        // Right padding is already hidden from every real query by causal masking.
        // Keeping attn_mask=None lets LibTorch select its fused Flash kernel.
        Tensor::f_scaled_dot_product_attention(
            &query,
            &keys,
            &values,
            None::<&Tensor>,
            0.0,
            true,
            None,
            false,
        )?
    };
    let output = output
        .permute([0, 2, 1, 3])
        .reshape([batch_size as i64 * max_query, num_heads * head_dim]);
    Ok(output.index_select(
        0,
        &Tensor::from_slice(&output_indices).to_device(q.device()),
    ))
}

fn sdpa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prefix: i64,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Result<Tensor> {
    let query_len = q.size()[0];
    let key_len = k.size()[0];
    let repeats = num_heads / num_kv_heads;
    let q = q.transpose(0, 1).unsqueeze(0);
    let k = k
        .transpose(0, 1)
        .repeat_interleave_self_int(repeats, 0, Some(num_heads))
        .unsqueeze(0);
    let v = v
        .transpose(0, 1)
        .repeat_interleave_self_int(repeats, 0, Some(num_heads))
        .unsqueeze(0);
    let mask = if prefix > 0 {
        let queries = Tensor::arange(query_len, (Kind::Int64, q.device())) + prefix;
        let keys = Tensor::arange(key_len, (Kind::Int64, q.device()));
        Some(
            keys.unsqueeze(0)
                .le_tensor(&queries.unsqueeze(1))
                .unsqueeze(0)
                .unsqueeze(0),
        )
    } else {
        None
    };
    let output = Tensor::f_scaled_dot_product_attention(
        &q,
        &k,
        &v,
        mask.as_ref(),
        0.0,
        prefix == 0,
        None,
        false,
    )?;
    Ok(output
        .squeeze_dim(0)
        .transpose(0, 1)
        .reshape([query_len, num_heads * head_dim]))
}

fn sdpa_decode_with_cache(
    q: &Tensor,
    cache: &BaseAttention,
    metadata: &AttentionMetadata,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Result<Tensor> {
    let table = metadata
        .req_to_token
        .as_ref()
        .ok_or_else(|| model_error("paged-KV decode requires req_to_token"))?;
    let lengths = metadata
        .cache_seqlens
        .as_ref()
        .ok_or_else(|| model_error("paged-KV decode requires cache_seqlens"))?;
    let max_len = metadata.max_seqlen.unwrap_or(table.size()[1] as usize) as i64;
    let batch_size = q.size()[0];
    if table.size()[0] != batch_size || lengths.size()[0] != batch_size {
        return Err(model_error("decode metadata row count must match queries"));
    }
    let indices = table.narrow(1, 0, max_len);
    let valid = indices.ge(0).logical_and(
        &Tensor::arange(max_len, (Kind::Int64, q.device()))
            .unsqueeze(0)
            .lt_tensor(&lengths.to_kind(Kind::Int64).unsqueeze(1)),
    );
    let (k, v) = cache.read_kv_padded(&indices.clamp_min(0).to_kind(Kind::Int64))?;
    let repeats = num_heads / num_kv_heads;
    let k = k
        .repeat_interleave_self_int(repeats, 2, Some(num_heads))
        .permute([0, 2, 1, 3]);
    let v = v
        .repeat_interleave_self_int(repeats, 2, Some(num_heads))
        .permute([0, 2, 1, 3]);
    let query = q.unsqueeze(2);
    let mask = valid.unsqueeze(1).unsqueeze(2);
    let output = Tensor::f_scaled_dot_product_attention(
        &query,
        &k,
        &v,
        Some(&mask),
        0.0,
        false,
        None,
        false,
    )?;
    Ok(output
        .squeeze_dim(2)
        .reshape([batch_size, num_heads * head_dim]))
}

fn prefill_with_cache(
    q: &Tensor,
    cache: &BaseAttention,
    metadata: &AttentionMetadata,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Result<Tensor> {
    let boundaries = prefill_boundaries(metadata, q.size()[0] as usize)?;
    let prefix_lens = tensor_i32(metadata.prefix_lens.as_ref(), "prefix_lens")?;
    if prefix_lens.len() + 1 != boundaries.len() {
        return Err(model_error(
            "prefix_lens must contain one value per prefill request",
        ));
    }
    let table = metadata
        .req_to_token
        .as_ref()
        .ok_or_else(|| model_error("cached prefill requires req_to_token"))?;
    let mut outputs = Vec::with_capacity(prefix_lens.len());
    for (request_index, (&prefix_len, boundaries)) in
        prefix_lens.iter().zip(boundaries.windows(2)).enumerate()
    {
        if prefix_len < 0 {
            return Err(model_error("prefix_lens cannot be negative"));
        }
        let start = boundaries[0];
        let query_len = boundaries[1] - start;
        let (cached_k, cached_v) = cache.read_kv(
            table,
            request_index as i64,
            i64::from(prefix_len) + query_len,
        )?;
        outputs.push(attention_against_cache(
            &q.narrow(0, start, query_len),
            &cached_k,
            &cached_v,
            i64::from(prefix_len),
            num_heads,
            num_kv_heads,
            head_dim,
        ));
    }
    Ok(Tensor::cat(&outputs, 0))
}

fn decode_with_cache(
    q: &Tensor,
    cache: &BaseAttention,
    metadata: &AttentionMetadata,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Result<Tensor> {
    let table = metadata
        .req_to_token
        .as_ref()
        .ok_or_else(|| model_error("paged-KV decode requires req_to_token"))?;
    if matches!(q.device(), Device::Cuda(_)) {
        return decode_with_cache_padded(q, cache, metadata, num_heads, num_kv_heads, head_dim);
    }
    let cache_seqlens = tensor_i32(metadata.cache_seqlens.as_ref(), "cache_seqlens")?;
    if q.size()[0] != cache_seqlens.len() as i64 || table.size()[0] != q.size()[0] {
        return Err(model_error(
            "decode K/V metadata must contain one row per query",
        ));
    }
    let mut outputs = Vec::with_capacity(cache_seqlens.len());
    for (request_index, &cache_len) in cache_seqlens.iter().enumerate() {
        if cache_len <= 0 {
            return Err(model_error("cache_seqlens must be positive"));
        }
        let (cached_k, cached_v) =
            cache.read_kv(table, request_index as i64, i64::from(cache_len))?;
        outputs.push(attention_against_cache(
            &q.narrow(0, request_index as i64, 1),
            &cached_k,
            &cached_v,
            i64::from(cache_len - 1),
            num_heads,
            num_kv_heads,
            head_dim,
        ));
    }
    Ok(Tensor::cat(&outputs, 0))
}

/// Fixed-shape decode used by CUDA Graph capture. Sequence lengths stay on
/// device; invalid columns are masked before softmax.
fn decode_with_cache_padded(
    q: &Tensor,
    cache: &BaseAttention,
    metadata: &AttentionMetadata,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Result<Tensor> {
    let table = metadata
        .req_to_token
        .as_ref()
        .ok_or_else(|| model_error("paged-KV decode requires req_to_token"))?;
    let lengths = metadata
        .cache_seqlens
        .as_ref()
        .ok_or_else(|| model_error("paged-KV decode requires cache_seqlens"))?;
    let max_len = metadata.max_seqlen.unwrap_or(table.size()[1] as usize) as i64;
    let batch_size = q.size()[0];
    if table.size()[0] != batch_size || lengths.size()[0] != batch_size {
        return Err(model_error("decode metadata row count must match queries"));
    }
    let indices = table.narrow(1, 0, max_len);
    let valid = indices.ge(0).logical_and(
        &Tensor::arange(max_len, (Kind::Int64, q.device()))
            .unsqueeze(0)
            .lt_tensor(&lengths.to_kind(Kind::Int64).unsqueeze(1)),
    );
    let (k, v) = cache.read_kv_padded(&indices.clamp_min(0).to_kind(Kind::Int64))?;
    let k = k
        .repeat_interleave_self_int(num_heads / num_kv_heads, 2, Some(num_heads))
        .permute([0, 2, 1, 3]);
    let v = v
        .repeat_interleave_self_int(num_heads / num_kv_heads, 2, Some(num_heads))
        .permute([0, 2, 1, 3]);
    let query = q.unsqueeze(2);
    let scores = (query.matmul(&k.transpose(2, 3)) * (head_dim as f64).sqrt().recip()).masked_fill(
        &valid.logical_not().unsqueeze(1).unsqueeze(2),
        f64::NEG_INFINITY,
    );
    Ok(scores
        .softmax(-1, Kind::Float)
        .to_kind(v.kind())
        .matmul(&v)
        .squeeze_dim(2)
        .reshape([batch_size, num_heads * head_dim]))
}

fn causal_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Tensor {
    let q = q.transpose(0, 1);
    let repeats = num_heads / num_kv_heads;
    let k = k
        .transpose(0, 1)
        .repeat_interleave_self_int(repeats, 0, Some(num_heads));
    let v = v
        .transpose(0, 1)
        .repeat_interleave_self_int(repeats, 0, Some(num_heads));
    let sequence_length = q.size()[1];
    let scale = (head_dim as f64).sqrt().recip();
    let scores = (q.bmm(&k.transpose(1, 2)) * scale).masked_fill(
        &Tensor::ones([sequence_length, sequence_length], (Kind::Bool, q.device()))
            .triu(1)
            .unsqueeze(0),
        f64::NEG_INFINITY,
    );
    scores
        .softmax(-1, Kind::Float)
        .to_kind(v.kind())
        .bmm(&v)
        .transpose(0, 1)
        .reshape([sequence_length, num_heads * head_dim])
}

fn attention_against_cache(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prefix_len: i64,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> Tensor {
    let q = q.transpose(0, 1);
    let repeats = num_heads / num_kv_heads;
    let k = k
        .transpose(0, 1)
        .repeat_interleave_self_int(repeats, 0, Some(num_heads));
    let v = v
        .transpose(0, 1)
        .repeat_interleave_self_int(repeats, 0, Some(num_heads));
    let query_len = q.size()[1];
    let key_len = k.size()[1];
    let query_positions = Tensor::arange(query_len, (Kind::Int64, q.device())) + prefix_len;
    let key_positions = Tensor::arange(key_len, (Kind::Int64, q.device()));
    let mask = key_positions
        .unsqueeze(0)
        .le_tensor(&query_positions.unsqueeze(1));
    let scale = (head_dim as f64).sqrt().recip();
    (q.bmm(&k.transpose(1, 2)) * scale)
        .masked_fill(&mask.logical_not().unsqueeze(0), f64::NEG_INFINITY)
        .softmax(-1, Kind::Float)
        .to_kind(v.kind())
        .bmm(&v)
        .transpose(0, 1)
        .reshape([query_len, num_heads * head_dim])
}

fn prefill_boundaries(metadata: &AttentionMetadata, total_tokens: usize) -> Result<Vec<i64>> {
    let total_tokens =
        i64::try_from(total_tokens).map_err(|_| model_error("token count exceeds i64"))?;
    let cumulative = metadata
        .cu_seqlens_q
        .as_ref()
        .ok_or_else(|| model_error("cached prefill requires cu_seqlens_q"))?;
    let boundaries = tensor_i32(Some(cumulative), "cu_seqlens_q")?;
    if boundaries.len() < 2
        || boundaries.first().copied() != Some(0)
        || boundaries.last().copied().map(i64::from) != Some(total_tokens)
        || boundaries.windows(2).any(|window| window[0] >= window[1])
    {
        return Err(model_error("invalid prefill cu_seqlens_q"));
    }
    Ok(boundaries.into_iter().map(i64::from).collect())
}

fn has_cached_prefix(metadata: &AttentionMetadata) -> Result<bool> {
    Ok(tensor_i32(metadata.prefix_lens.as_ref(), "prefix_lens")?
        .into_iter()
        .any(|prefix| prefix > 0))
}

fn tensor_i32(tensor: Option<&Tensor>, field: &str) -> Result<Vec<i32>> {
    let tensor = tensor.ok_or_else(|| model_error(&format!("missing {field}")))?;
    Vec::<i32>::try_from(&tensor.to_device(Device::Cpu)).map_err(ModelRunnerError::Torch)
}

fn model_error(message: &str) -> ModelRunnerError {
    ModelRunnerError::Model(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_decode_matches_eager_for_different_sequence_lengths() {
        let mut cache = BaseAttention::default();
        cache
            .bind_kv_cache(
                Tensor::zeros([1, 4, 1, 2], (Kind::Float, Device::Cpu)),
                Tensor::zeros([1, 4, 1, 2], (Kind::Float, Device::Cpu)),
            )
            .unwrap();
        let k = Tensor::from_slice(&[1f32, 0., 0., 1., 1., 1.]).view([3, 1, 2]);
        let v = Tensor::from_slice(&[2f32, 0., 0., 4., 6., 6.]).view([3, 1, 2]);
        cache
            .write_kv(
                &k,
                &v,
                Some(&Tensor::from_slice(&[0i32, 1, 2])),
                BatchPhase::Prefill,
            )
            .unwrap();
        let metadata = AttentionMetadata {
            forward_mode: BatchPhase::Decode,
            write_loc: None,
            cu_seqlens_q: None,
            prefix_lens: None,
            block_table: None,
            req_to_token: Some(Tensor::from_slice(&[0i32, 1, -1, -1, 2, -1, -1, -1]).view([2, 4])),
            cache_seqlens: Some(Tensor::from_slice(&[2i32, 1])),
            max_seqlen: Some(4),
        };
        let q = Tensor::from_slice(&[1f32, 0., 0., 1.]).view([2, 1, 2]);
        let eager = decode_with_cache(&q, &cache, &metadata, 1, 1, 2).unwrap();
        let padded = decode_with_cache_padded(&q, &cache, &metadata, 1, 1, 2).unwrap();
        let difference = (eager - padded).abs().max().double_value(&[]);
        assert!(difference < 1e-5, "decode difference: {difference}");
        let sdpa = sdpa_decode_with_cache(&q, &cache, &metadata, 1, 1, 2).unwrap();
        let reference = decode_with_cache_padded(&q, &cache, &metadata, 1, 1, 2).unwrap();
        let difference = (reference - sdpa).abs().max().double_value(&[]);
        assert!(difference < 1e-5, "SDPA decode difference: {difference}");
    }

    #[test]
    fn sdpa_causal_prefill_matches_reference() {
        let q = Tensor::from_slice(&[1f32, 0., 0., 1., 1., 1.]).view([3, 1, 2]);
        let k = q.shallow_clone();
        let v = Tensor::from_slice(&[2f32, 1., 3., 4., 5., 6.]).view([3, 1, 2]);
        let reference = causal_attention(&q, &k, &v, 1, 1, 2);
        let sdpa = sdpa_attention(&q, &k, &v, 0, 1, 1, 2).unwrap();
        let difference = (reference - sdpa).abs().max().double_value(&[]);
        assert!(difference < 1e-5, "SDPA prefill difference: {difference}");
    }

    #[test]
    fn batched_prefill_matches_per_request_attention_for_unequal_lengths() {
        let q = Tensor::from_slice(&[1f32, 0., 0., 1., 1., 1., 2., 1., 1., 2.]).view([5, 1, 2]);
        let k = q.shallow_clone();
        let v = Tensor::from_slice(&[2f32, 1., 3., 4., 5., 6., 7., 8., 9., 10.]).view([5, 1, 2]);
        let boundaries = [0, 2, 3, 5];
        let actual = batched_prefill_sdpa(
            &q,
            &k,
            &v,
            &BaseAttention::default(),
            &boundaries,
            &[0, 0, 0],
            None,
            1,
            1,
            2,
        )
        .unwrap();
        let expected = Tensor::cat(
            &boundaries
                .windows(2)
                .map(|bounds| {
                    let start = bounds[0];
                    let len = bounds[1] - start;
                    sdpa_attention(
                        &q.narrow(0, start, len),
                        &k.narrow(0, start, len),
                        &v.narrow(0, start, len),
                        0,
                        1,
                        1,
                        2,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>(),
            0,
        );
        let difference = (actual - expected).abs().max().double_value(&[]);
        assert!(
            difference < 1e-5,
            "batched prefill difference: {difference}"
        );
    }

    #[test]
    fn batched_cached_prefill_matches_per_request_attention() {
        let mut cache = BaseAttention::default();
        cache
            .bind_kv_cache(
                Tensor::zeros([1, 8, 1, 2], (Kind::Float, Device::Cpu)),
                Tensor::zeros([1, 8, 1, 2], (Kind::Float, Device::Cpu)),
            )
            .unwrap();
        let all_k = Tensor::from_slice(&[
            1f32, 0., 0., 1., 1., 1., 2., 0., 0., 2., 1., 2., 2., 1., 1., 3.,
        ])
        .view([8, 1, 2]);
        let all_v = Tensor::from_slice(&[
            2f32, 1., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ])
        .view([8, 1, 2]);
        cache
            .write_kv(
                &all_k,
                &all_v,
                Some(&Tensor::from_slice(&[0i32, 1, 2, 3, 4, 5, 6, 7])),
                BatchPhase::Prefill,
            )
            .unwrap();
        let q = Tensor::from_slice(&[1f32, 0., 0., 1., 1., 1., 2., 1., 1., 2.]).view([5, 1, 2]);
        let table = Tensor::from_slice(&[0i32, 1, 2, -1, 3, -1, -1, -1, 4, 5, 6, 7]).view([3, 4]);
        let metadata = AttentionMetadata {
            forward_mode: BatchPhase::Prefill,
            write_loc: None,
            cu_seqlens_q: None,
            prefix_lens: None,
            block_table: None,
            req_to_token: Some(table.shallow_clone()),
            cache_seqlens: None,
            max_seqlen: Some(2),
        };
        let boundaries = [0, 2, 3, 5];
        let prefixes = [1, 0, 2];
        let actual = batched_prefill_sdpa(
            &q,
            &q,
            &q,
            &cache,
            &boundaries,
            &prefixes,
            Some(&metadata),
            1,
            1,
            2,
        )
        .unwrap();
        let expected = Tensor::cat(
            &boundaries
                .windows(2)
                .enumerate()
                .map(|(row, bounds)| {
                    let len = bounds[1] - bounds[0];
                    let (k, v) = cache
                        .read_kv(&table, row as i64, prefixes[row] + len)
                        .unwrap();
                    sdpa_attention(&q.narrow(0, bounds[0], len), &k, &v, prefixes[row], 1, 1, 2)
                        .unwrap()
                })
                .collect::<Vec<_>>(),
            0,
        );
        let difference = (actual - expected).abs().max().double_value(&[]);
        assert!(
            difference < 1e-5,
            "batched cached prefill difference: {difference}"
        );
    }
}
