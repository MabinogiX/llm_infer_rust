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
    /// Reserved for a future FlashAttention binding.
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

/// Placeholder preserving the Python dispatch surface until a Rust binding is added.
struct FlashAttentionBackend;

impl AttentionBackend for FlashAttentionBackend {
    fn kind(&self) -> AttentionBackendKind {
        AttentionBackendKind::FlashAttention
    }

    fn forward(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _cache: &BaseAttention,
        _sequence_boundaries: &[i64],
        _metadata: Option<&AttentionMetadata>,
        _num_heads: i64,
        _num_kv_heads: i64,
        _head_dim: i64,
    ) -> Result<Tensor> {
        Err(model_error(
            "FlashAttention Rust binding has not been implemented; use attention_backend=\"pt\"",
        ))
    }
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
    }
}
