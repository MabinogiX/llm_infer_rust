//! Dense Qwen3 causal language model implemented with libtorch tensors.
//!
//! The dense path supports eager prefill plus paged-KV decode and cached-prefix
//! prefill. Qwen3-MoE remains outside this implementation.

use std::{cell::RefCell, collections::HashMap};

use tch::{Device, Kind, Tensor};

use crate::engine::{
    AttentionMetadata, BatchPhase, ModelArgs, ModelExecutor, ModelFactory, ModelRunnerError,
    ModelWeights,
};
use crate::models::attention::{Attention, AttentionBatch, AttentionSpec, BaseAttention};
use crate::profiling::{ModelProfiler, ModelStage};

type Result<T> = std::result::Result<T, ModelRunnerError>;

/// Factory for the dense `Qwen3ForCausalLM` architecture.
#[derive(Debug, Default, Clone, Copy)]
pub struct Qwen3Factory;

impl ModelFactory for Qwen3Factory {
    fn create(
        &self,
        model_args: ModelArgs,
        kind: Kind,
        device: Device,
    ) -> Result<Box<dyn ModelExecutor>> {
        Ok(Box::new(Qwen3ForCausalLM::new(model_args, kind, device)?))
    }

    fn create_with_attention_backend(
        &self,
        model_args: ModelArgs,
        kind: Kind,
        device: Device,
        attention_backend: &str,
    ) -> Result<Box<dyn ModelExecutor>> {
        Ok(Box::new(Qwen3ForCausalLM::new_with_attention_backend(
            model_args,
            kind,
            device,
            attention_backend,
        )?))
    }
}

/// Dense Qwen3 decoder-only model with QK-RMSNorm, RoPE, GQA, and SwiGLU.
pub struct Qwen3ForCausalLM {
    config: ModelArgs,
    device: Device,
    kind: Kind,
    attention: Attention,
    embed_tokens: Tensor,
    layers: Vec<DecoderLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

impl Qwen3ForCausalLM {
    pub fn new(config: ModelArgs, kind: Kind, device: Device) -> Result<Self> {
        Self::new_with_attention_backend(config, kind, device, "pt")
    }

    pub fn new_with_attention_backend(
        config: ModelArgs,
        kind: Kind,
        device: Device,
        attention_backend: &str,
    ) -> Result<Self> {
        validate_config(config)?;
        let hidden = as_i64(config.hidden_size, "hidden_size")?;
        let vocab = as_i64(config.vocab_size, "vocab_size")?;
        let intermediate = as_i64(config.intermediate_size, "intermediate_size")?;
        let heads = as_i64(config.num_attention_heads, "num_attention_heads")?;
        let kv_heads = as_i64(config.num_kv_heads, "num_kv_heads")?;
        let head_dim = as_i64(config.head_dim, "head_dim")?;
        let attention = Attention::new(
            attention_backend,
            AttentionSpec {
                num_heads: heads,
                num_kv_heads: kv_heads,
                head_dim,
                kind,
                device,
            },
        )?;

        Ok(Self {
            config,
            device,
            kind,
            attention,
            embed_tokens: parameter([vocab, hidden], kind, device),
            layers: (0..config.num_layers)
                .map(|_| {
                    DecoderLayer::new(
                        hidden,
                        intermediate,
                        heads,
                        kv_heads,
                        head_dim,
                        kind,
                        device,
                    )
                })
                .collect(),
            norm: parameter([hidden], kind, device),
            lm_head: parameter([vocab, hidden], kind, device),
        })
    }

    fn forward_impl(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        attention_metadata: Option<&AttentionMetadata>,
        logits_indices: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut profiler = ModelProfiler::new(self.device);
        let ids = input_ids.view([-1]);
        let positions = positions.view([-1]);
        if ids.numel() != positions.numel() {
            return Err(model_error(
                "input_ids and positions must have identical lengths",
            ));
        }
        let timer = profiler.start(ModelStage::Plan);
        let attention = self.attention.prepare(attention_metadata, ids.numel());
        profiler.finish(timer);
        let attention = attention?;

        let timer = profiler.start(ModelStage::Embed);
        let mut hidden_states = self.embed_tokens.index_select(0, &ids);
        profiler.finish(timer);

        let timer = profiler.start(ModelStage::Layers);
        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                &positions,
                &attention,
                attention_metadata,
                self.config.rms_norm_eps,
                self.config.rope_theta,
                &mut profiler,
            )?;
        }
        profiler.finish(timer);

        let timer = profiler.start(ModelStage::Head);
        hidden_states = rms_norm(&hidden_states, &self.norm, self.config.rms_norm_eps);
        if let Some(indices) = logits_indices {
            hidden_states = hidden_states.index_select(0, indices);
        }
        let logits = linear(&hidden_states, &self.lm_head);
        profiler.finish(timer);

        let timer = profiler.start(ModelStage::PlanDrop);
        drop(attention);
        profiler.finish(timer);
        profiler.log(
            attention_metadata.map_or(BatchPhase::Prefill, |metadata| metadata.forward_mode),
            ids.numel(),
            self.layers.len(),
        );
        Ok(logits)
    }

    fn load_weights_impl(&mut self, weights: ModelWeights) -> Result<usize> {
        let mut weights = weights
            .into_tensors()
            .into_iter()
            .collect::<HashMap<_, _>>();
        let mut loaded = 0;
        load_parameter(
            &mut self.embed_tokens,
            "model.embed_tokens.weight",
            &mut weights,
            self.kind,
            self.device,
        )?;
        loaded += 1;

        for (index, layer) in self.layers.iter_mut().enumerate() {
            loaded += layer.load_weights(index, &mut weights, self.kind, self.device)?;
        }
        load_parameter(
            &mut self.norm,
            "model.norm.weight",
            &mut weights,
            self.kind,
            self.device,
        )?;
        loaded += 1;

        if weights.contains_key("lm_head.weight") {
            load_parameter(
                &mut self.lm_head,
                "lm_head.weight",
                &mut weights,
                self.kind,
                self.device,
            )?;
            loaded += 1;
        } else if self.config.tie_word_embeddings {
            self.lm_head = self.embed_tokens.shallow_clone();
        } else {
            return Err(model_error("checkpoint is missing lm_head.weight"));
        }
        Ok(loaded)
    }
}

impl ModelExecutor for Qwen3ForCausalLM {
    fn supports_cuda_graph(&self) -> bool {
        self.attention.supports_cuda_graph()
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        attention_metadata: Option<&AttentionMetadata>,
        logits_indices: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_impl(input_ids, positions, attention_metadata, logits_indices)
    }

    fn load_weights(&mut self, weights: ModelWeights) -> Result<usize> {
        self.load_weights_impl(weights)
    }

    fn bind_kv_cache(&mut self, k_cache: Tensor, v_cache: Tensor) -> Result<()> {
        if k_cache.dim() != 5
            || v_cache.size() != k_cache.size()
            || k_cache.size()[0] != self.layers.len() as i64
        {
            return Err(model_error(
                "Qwen3 KV cache must be (layers, pages, page_size, kv_heads, head_dim)",
            ));
        }
        self.attention.bind_cache_layout(k_cache.size()[2]);
        for (index, layer) in self.layers.iter().enumerate() {
            layer
                .base_attention
                .borrow_mut()
                .bind_kv_cache(k_cache.get(index as i64), v_cache.get(index as i64))?;
        }
        Ok(())
    }
}

struct DecoderLayer {
    base_attention: RefCell<BaseAttention>,
    input_layernorm: Tensor,
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    post_attention_layernorm: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
}

impl DecoderLayer {
    fn new(
        hidden: i64,
        intermediate: i64,
        num_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
        kind: Kind,
        device: Device,
    ) -> Self {
        Self {
            base_attention: RefCell::new(BaseAttention::default()),
            input_layernorm: parameter([hidden], kind, device),
            q_proj: parameter([num_heads * head_dim, hidden], kind, device),
            k_proj: parameter([num_kv_heads * head_dim, hidden], kind, device),
            v_proj: parameter([num_kv_heads * head_dim, hidden], kind, device),
            o_proj: parameter([hidden, num_heads * head_dim], kind, device),
            q_norm: parameter([head_dim], kind, device),
            k_norm: parameter([head_dim], kind, device),
            post_attention_layernorm: parameter([hidden], kind, device),
            gate_proj: parameter([intermediate, hidden], kind, device),
            up_proj: parameter([intermediate, hidden], kind, device),
            down_proj: parameter([hidden, intermediate], kind, device),
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        attention_batch: &AttentionBatch<'_>,
        attention_metadata: Option<&AttentionMetadata>,
        eps: f64,
        rope_theta: f64,
        profiler: &mut ModelProfiler,
    ) -> Result<Tensor> {
        let residual = hidden_states.shallow_clone();
        let timer = profiler.start(ModelStage::Norm);
        let normalized = rms_norm(hidden_states, &self.input_layernorm, eps);
        profiler.finish(timer);
        let attention = self.attention(
            &normalized,
            positions,
            attention_batch,
            attention_metadata,
            eps,
            rope_theta,
            profiler,
        )?;
        let timer = profiler.start(ModelStage::Norm);
        let residual = attention + residual;
        let normalized = rms_norm(&residual, &self.post_attention_layernorm, eps);
        profiler.finish(timer);

        let timer = profiler.start(ModelStage::Mlp);
        let mlp = linear(&normalized, &self.gate_proj).silu() * linear(&normalized, &self.up_proj);
        let output = linear(&mlp, &self.down_proj) + residual;
        profiler.finish(timer);
        Ok(output)
    }

    fn attention(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        attention_batch: &AttentionBatch<'_>,
        attention_metadata: Option<&AttentionMetadata>,
        eps: f64,
        rope_theta: f64,
        profiler: &mut ModelProfiler,
    ) -> Result<Tensor> {
        let timer = profiler.start(ModelStage::QkvLinear);
        let total_tokens = hidden_states.size()[0];
        let q =
            linear(hidden_states, &self.q_proj).view([total_tokens, self.num_heads, self.head_dim]);
        let k = linear(hidden_states, &self.k_proj).view([
            total_tokens,
            self.num_kv_heads,
            self.head_dim,
        ]);
        let v = linear(hidden_states, &self.v_proj).view([
            total_tokens,
            self.num_kv_heads,
            self.head_dim,
        ]);
        profiler.finish(timer);

        let timer = profiler.start(ModelStage::QkNorm);
        let q = rms_norm(&q, &self.q_norm, eps);
        let k = rms_norm(&k, &self.k_norm, eps);
        profiler.finish(timer);

        let timer = profiler.start(ModelStage::Rope);
        let (q, k) = apply_rope(&q, &k, positions, rope_theta);
        profiler.finish(timer);

        let timer = profiler.start(ModelStage::KvWrite);
        let write_result = self.base_attention.borrow_mut().write_kv(
            &k,
            &v,
            attention_metadata.and_then(|metadata| metadata.write_loc.as_ref()),
            attention_metadata.map_or(BatchPhase::Prefill, |metadata| metadata.forward_mode),
        );
        profiler.finish(timer);
        write_result?;

        let timer = profiler.start(ModelStage::AttentionBackend);
        let output = attention_batch.forward(&q, &k, &v, &self.base_attention.borrow());
        profiler.finish(timer);
        let output = output?;

        let timer = profiler.start(ModelStage::AttentionOutput);
        let output = linear(&output, &self.o_proj);
        profiler.finish(timer);
        Ok(output)
    }

    fn load_weights(
        &mut self,
        index: usize,
        weights: &mut HashMap<String, Tensor>,
        kind: Kind,
        device: Device,
    ) -> Result<usize> {
        let prefix = format!("model.layers.{index}");
        for (parameter, suffix) in [
            (&mut self.input_layernorm, "input_layernorm.weight"),
            (&mut self.q_proj, "self_attn.q_proj.weight"),
            (&mut self.k_proj, "self_attn.k_proj.weight"),
            (&mut self.v_proj, "self_attn.v_proj.weight"),
            (&mut self.o_proj, "self_attn.o_proj.weight"),
            (&mut self.q_norm, "self_attn.q_norm.weight"),
            (&mut self.k_norm, "self_attn.k_norm.weight"),
            (
                &mut self.post_attention_layernorm,
                "post_attention_layernorm.weight",
            ),
            (&mut self.gate_proj, "mlp.gate_proj.weight"),
            (&mut self.up_proj, "mlp.up_proj.weight"),
            (&mut self.down_proj, "mlp.down_proj.weight"),
        ] {
            load_parameter(
                parameter,
                &format!("{prefix}.{suffix}"),
                weights,
                kind,
                device,
            )?;
        }
        Ok(11)
    }
}

fn validate_config(config: ModelArgs) -> Result<()> {
    if config.hidden_size == 0
        || config.num_layers == 0
        || config.num_attention_heads == 0
        || config.num_kv_heads == 0
        || config.intermediate_size == 0
        || config.vocab_size == 0
        || config.head_dim == 0
    {
        return Err(model_error(
            "Qwen3 configuration dimensions must be greater than zero",
        ));
    }
    if config.num_attention_heads % config.num_kv_heads != 0 {
        return Err(model_error(
            "num_attention_heads must be divisible by num_kv_heads",
        ));
    }
    if config.head_dim % 2 != 0 {
        return Err(model_error("Qwen3 RoPE requires an even head_dim"));
    }
    Ok(())
}

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let x_float = x.to_kind(Kind::Float);
    let variance = x_float
        .pow_tensor_scalar(2)
        .mean_dim(&[-1i64][..], true, Kind::Float);
    (x_float * (variance + eps).rsqrt() * weight.to_kind(Kind::Float)).to_kind(x.kind())
}

fn apply_rope(q: &Tensor, k: &Tensor, positions: &Tensor, rope_theta: f64) -> (Tensor, Tensor) {
    let head_dim = q.size()[2];
    let half_dim = head_dim / 2;
    let inv_freq = (Tensor::arange_start_step(0, head_dim, 2, (Kind::Float, q.device()))
        * (-(rope_theta.ln() / head_dim as f64)))
        .exp();
    let frequencies = positions.to_kind(Kind::Float).unsqueeze(-1) * inv_freq.unsqueeze(0);
    let cos = frequencies.cos().to_kind(q.kind()).unsqueeze(1);
    let sin = frequencies.sin().to_kind(q.kind()).unsqueeze(1);
    (
        rotate_half(q, &cos, &sin, half_dim),
        rotate_half(k, &cos, &sin, half_dim),
    )
}

fn rotate_half(x: &Tensor, cos: &Tensor, sin: &Tensor, half_dim: i64) -> Tensor {
    let first = x.narrow(-1, 0, half_dim);
    let second = x.narrow(-1, half_dim, half_dim);
    Tensor::cat(
        &[&first * cos - &second * sin, &second * cos + &first * sin],
        -1,
    )
}

fn linear(x: &Tensor, weight: &Tensor) -> Tensor {
    x.matmul(&weight.transpose(0, 1))
}

fn parameter(shape: impl AsRef<[i64]>, kind: Kind, device: Device) -> Tensor {
    Tensor::zeros(shape.as_ref(), (kind, device))
}

fn load_parameter(
    target: &mut Tensor,
    name: &str,
    weights: &mut HashMap<String, Tensor>,
    kind: Kind,
    device: Device,
) -> Result<()> {
    let weight = weights
        .remove(name)
        .ok_or_else(|| model_error(&format!("checkpoint is missing {name}")))?;
    if weight.size() != target.size() {
        return Err(model_error(&format!(
            "checkpoint tensor {name} has shape {:?}, expected {:?}",
            weight.size(),
            target.size()
        )));
    }
    *target = weight.to_device(device).to_kind(kind);
    Ok(())
}

fn as_i64(value: usize, field: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| model_error(&format!("{field} exceeds i64")))
}

fn model_error(message: &str) -> ModelRunnerError {
    ModelRunnerError::Model(message.to_owned())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;
    use crate::engine::load_hf_safetensors;

    static TEST_DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);

    fn config() -> ModelArgs {
        ModelArgs {
            hidden_size: 4,
            num_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 8,
            vocab_size: 8,
            head_dim: 2,
            max_position_embeddings: 16,
            ..Default::default()
        }
    }

    #[test]
    fn produces_prefill_logits_with_the_expected_shape() {
        let model = Qwen3ForCausalLM::new(config(), Kind::Float, Device::Cpu).unwrap();
        let logits = model
            .forward(
                &Tensor::from_slice(&[1i64, 2, 3]),
                &Tensor::from_slice(&[0i64, 1, 2]),
                None,
                Some(&Tensor::from_slice(&[2i64])),
            )
            .unwrap();
        assert_eq!(logits.size(), vec![1, 8]);
    }

    #[test]
    fn allows_qwen3_attention_width_to_exceed_hidden_size() {
        let mut config = config();
        config.head_dim = 4;
        let model = Qwen3ForCausalLM::new(config, Kind::Float, Device::Cpu).unwrap();
        let logits = model
            .forward(
                &Tensor::from_slice(&[1i64]),
                &Tensor::from_slice(&[0i64]),
                None,
                None,
            )
            .unwrap();
        assert_eq!(logits.size(), vec![1, 8]);
    }

    #[test]
    fn flash_attention_backend_runs_on_cpu_via_sdpa() {
        let model =
            Qwen3ForCausalLM::new_with_attention_backend(config(), Kind::Float, Device::Cpu, "fa")
                .unwrap();
        let logits = model
            .forward(
                &Tensor::from_slice(&[1i64]),
                &Tensor::from_slice(&[0i64]),
                None,
                None,
            )
            .unwrap();
        assert_eq!(logits.size(), vec![1, 8]);
    }

    #[test]
    fn rope_keeps_bfloat16_query_and_key() {
        let q = Tensor::ones([2, 2, 4], (Kind::BFloat16, Device::Cpu));
        let k = Tensor::ones([2, 1, 4], (Kind::BFloat16, Device::Cpu));
        let positions = Tensor::from_slice(&[0i64, 1]);
        let (q, k) = apply_rope(&q, &k, &positions, 10_000.0);
        assert_eq!(q.kind(), Kind::BFloat16);
        assert_eq!(k.kind(), Kind::BFloat16);
    }

    #[test]
    fn runs_paged_decode_and_cached_prefix_prefill() {
        let mut model = Qwen3ForCausalLM::new(config(), Kind::Float, Device::Cpu).unwrap();
        let k_cache = Tensor::zeros([1, 2, 2, 1, 2], (Kind::Float, Device::Cpu));
        let v_cache = Tensor::zeros([1, 2, 2, 1, 2], (Kind::Float, Device::Cpu));
        model
            .bind_kv_cache(k_cache.shallow_clone(), v_cache.shallow_clone())
            .unwrap();

        let first_prefill = AttentionMetadata {
            forward_mode: BatchPhase::Prefill,
            write_loc: Some(Tensor::from_slice(&[0i32, 1])),
            cu_seqlens_q: Some(Tensor::from_slice(&[0i32, 2])),
            prefix_lens: Some(Tensor::from_slice(&[0i32])),
            block_table: None,
            req_to_token: Some(Tensor::from_slice(&[0i32, 1, 2, 3]).view([1, 4])),
            cache_seqlens: None,
            max_seqlen: Some(2),
        };
        assert_eq!(
            model
                .forward(
                    &Tensor::from_slice(&[1i64, 2]),
                    &Tensor::from_slice(&[0i64, 1]),
                    Some(&first_prefill),
                    None,
                )
                .unwrap()
                .size(),
            vec![2, 8]
        );

        let decode = AttentionMetadata {
            forward_mode: BatchPhase::Decode,
            write_loc: Some(Tensor::from_slice(&[2i32])),
            cu_seqlens_q: None,
            prefix_lens: None,
            block_table: None,
            req_to_token: Some(Tensor::from_slice(&[0i32, 1, 2, 3]).view([1, 4])),
            cache_seqlens: Some(Tensor::from_slice(&[3i32])),
            max_seqlen: Some(3),
        };
        assert_eq!(
            model
                .forward(
                    &Tensor::from_slice(&[3i64]),
                    &Tensor::from_slice(&[2i64]),
                    Some(&decode),
                    None,
                )
                .unwrap()
                .size(),
            vec![1, 8]
        );

        let cached_prefill = AttentionMetadata {
            forward_mode: BatchPhase::Prefill,
            write_loc: Some(Tensor::from_slice(&[3i32])),
            cu_seqlens_q: Some(Tensor::from_slice(&[0i32, 1])),
            prefix_lens: Some(Tensor::from_slice(&[3i32])),
            block_table: None,
            req_to_token: Some(Tensor::from_slice(&[0i32, 1, 2, 3]).view([1, 4])),
            cache_seqlens: None,
            max_seqlen: Some(4),
        };
        assert_eq!(
            model
                .forward(
                    &Tensor::from_slice(&[4i64]),
                    &Tensor::from_slice(&[3i64]),
                    Some(&cached_prefill),
                    None,
                )
                .unwrap()
                .size(),
            vec![1, 8]
        );
    }

    #[test]
    fn loads_all_dense_qwen3_hugging_face_weight_names() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let id = TEST_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("sglang-rust-qwen3-{nonce}-{id}"));
        fs::create_dir_all(&path).unwrap();

        let weights = vec![
            (
                "model.embed_tokens.weight".to_owned(),
                Tensor::ones([8, 4], (Kind::Float, Device::Cpu)),
            ),
            (
                "model.layers.0.input_layernorm.weight".to_owned(),
                Tensor::ones([4], (Kind::Float, Device::Cpu)),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight".to_owned(),
                Tensor::ones([4, 4], (Kind::Float, Device::Cpu)),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight".to_owned(),
                Tensor::ones([2, 4], (Kind::Float, Device::Cpu)),
            ),
            (
                "model.layers.0.self_attn.v_proj.weight".to_owned(),
                Tensor::ones([2, 4], (Kind::Float, Device::Cpu)),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight".to_owned(),
                Tensor::ones([4, 4], (Kind::Float, Device::Cpu)),
            ),
            (
                "model.layers.0.self_attn.q_norm.weight".to_owned(),
                Tensor::ones([2], (Kind::Float, Device::Cpu)),
            ),
            (
                "model.layers.0.self_attn.k_norm.weight".to_owned(),
                Tensor::ones([2], (Kind::Float, Device::Cpu)),
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".to_owned(),
                Tensor::ones([4], (Kind::Float, Device::Cpu)),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight".to_owned(),
                Tensor::ones([8, 4], (Kind::Float, Device::Cpu)),
            ),
            (
                "model.layers.0.mlp.up_proj.weight".to_owned(),
                Tensor::ones([8, 4], (Kind::Float, Device::Cpu)),
            ),
            (
                "model.layers.0.mlp.down_proj.weight".to_owned(),
                Tensor::ones([4, 8], (Kind::Float, Device::Cpu)),
            ),
            (
                "model.norm.weight".to_owned(),
                Tensor::ones([4], (Kind::Float, Device::Cpu)),
            ),
            (
                "lm_head.weight".to_owned(),
                Tensor::ones([8, 4], (Kind::Float, Device::Cpu)),
            ),
        ];
        let named = weights
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        Tensor::write_safetensors(&named, path.join("model.safetensors")).unwrap();

        let mut model = Qwen3ForCausalLM::new(config(), Kind::Float, Device::Cpu).unwrap();
        assert_eq!(
            model
                .load_weights(load_hf_safetensors(&path).unwrap())
                .unwrap(),
            14
        );
        fs::remove_dir_all(path).unwrap();
    }
}
