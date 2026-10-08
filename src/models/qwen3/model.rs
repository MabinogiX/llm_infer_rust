//! Dense Qwen3 causal language model implemented with libtorch tensors.
//!
//! The dense path supports segmented prefill graphs, eager cached-prefix
//! prefill, and paged-KV decode graphs. Qwen3-MoE remains outside this implementation.

use std::{cell::RefCell, collections::HashMap};

use tch::{Device, Kind, Tensor};

use super::ops::{RopeCache, add_rms_norm, qk_norm, rms_norm, silu_and_mul};
use crate::engine::{
    AttentionMetadata, BatchPhase, DecodeGraphState, ModelArgs, ModelExecutor, ModelFactory,
    ModelRunnerError, ModelWeights,
};
use crate::models::attention::{Attention, AttentionBatch, AttentionSpec, BaseAttention};
use crate::profiling::{ModelProfiler, ModelStage};

#[path = "prefill_graph.rs"]
mod prefill_graph;
use prefill_graph::PrefillGraphCache;

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
    // Drop captured segments before any parameters that their kernels reference.
    prefill_graphs: RefCell<PrefillGraphCache>,
    config: ModelArgs,
    device: Device,
    kind: Kind,
    attention: Attention,
    rope: RopeCache,
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
        let max_positions = as_i64(config.max_position_embeddings, "max_position_embeddings")?;
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
            prefill_graphs: RefCell::new(PrefillGraphCache::default()),
            config,
            device,
            kind,
            attention,
            rope: RopeCache::new(max_positions, head_dim, config.rope_theta, kind, device),
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
        if attention_metadata.is_some_and(|meta| meta.forward_mode == BatchPhase::Prefill) {
            if let Some(logits) = self.forward_prefill_graph(
                input_ids,
                positions,
                attention_metadata,
                logits_indices,
            )? {
                return Ok(logits);
            }
        }
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
        let mut residual = None;
        for layer in &self.layers {
            let (output, next_residual) = layer.forward(
                &hidden_states,
                residual,
                &positions,
                &attention,
                attention_metadata,
                self.config.rms_norm_eps,
                &self.rope,
                &mut profiler,
            )?;
            hidden_states = output;
            residual = Some(next_residual);
        }
        profiler.finish(timer);

        let timer = profiler.start(ModelStage::Head);
        hidden_states = if let Some(residual) = residual {
            add_rms_norm(
                hidden_states,
                residual,
                &self.norm,
                self.config.rms_norm_eps,
            )
            .0
        } else {
            rms_norm(&hidden_states, &self.norm, self.config.rms_norm_eps)
        };
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
        self.prefill_graphs.get_mut().clear();
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
    fn configure_prefill_graph(&mut self, max_tokens: usize) {
        let limit = if matches!(self.device, Device::Cuda(_))
            && matches!(self.kind, Kind::BFloat16 | Kind::Half)
            && crate::engine::NativeCudaGraph::available()
            && !crate::logging::step_timing_enabled()
        {
            max_tokens
        } else {
            0
        };
        *self.prefill_graphs.get_mut() = PrefillGraphCache::new(limit);
        if limit > 0 {
            tracing::info!(
                max_tokens = limit,
                "prefill segmented CUDA Graph configured for startup capture"
            );
        }
    }

    fn capture_prefill_graphs(&self) -> Result<()> {
        self.capture_prefill_buckets()
    }

    fn supports_cuda_graph(&self) -> bool {
        self.attention.supports_cuda_graph()
    }

    fn prepare_decode_graph(
        &self,
        metadata: &AttentionMetadata,
    ) -> Result<Option<Box<dyn DecodeGraphState>>> {
        self.attention.prepare_decode_graph(metadata)
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

    fn set_kv_reserved_slot(&mut self, slot: i64) {
        for layer in &self.layers {
            layer
                .base_attention
                .borrow_mut()
                .set_reserved_write_slot(slot);
        }
    }

    fn bind_kv_cache(&mut self, k_cache: Tensor, v_cache: Tensor) -> Result<()> {
        self.prefill_graphs.get_mut().clear();
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
    qkv_proj: Tensor,
    o_proj: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    post_attention_layernorm: Tensor,
    gate_up_proj: Tensor,
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
            qkv_proj: parameter(
                [(num_heads + 2 * num_kv_heads) * head_dim, hidden],
                kind,
                device,
            ),
            o_proj: parameter([hidden, num_heads * head_dim], kind, device),
            q_norm: parameter([head_dim], kind, device),
            k_norm: parameter([head_dim], kind, device),
            post_attention_layernorm: parameter([hidden], kind, device),
            gate_up_proj: parameter([2 * intermediate, hidden], kind, device),
            down_proj: parameter([hidden, intermediate], kind, device),
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        residual: Option<Tensor>,
        positions: &Tensor,
        attention_batch: &AttentionBatch<'_>,
        attention_metadata: Option<&AttentionMetadata>,
        eps: f64,
        rope: &RopeCache,
        profiler: &mut ModelProfiler,
    ) -> Result<(Tensor, Tensor)> {
        let timer = profiler.start(ModelStage::Norm);
        let (normalized, residual) = self.input_norm(hidden_states, residual, eps);
        profiler.finish(timer);
        let (q, k, v) = self.project_qkv(&normalized, positions, eps, rope, profiler)?;
        let attention = self.attend(&q, &k, &v, attention_batch, attention_metadata, profiler)?;
        Ok(self.output_mlp(&attention, residual, eps, profiler))
    }

    fn input_norm(&self, hidden: &Tensor, residual: Option<Tensor>, eps: f64) -> (Tensor, Tensor) {
        if let Some(residual) = residual {
            add_rms_norm(hidden.shallow_clone(), residual, &self.input_layernorm, eps)
        } else {
            (
                rms_norm(hidden, &self.input_layernorm, eps),
                hidden.shallow_clone(),
            )
        }
    }

    fn project_qkv(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        eps: f64,
        rope: &RopeCache,
        profiler: &mut ModelProfiler,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let timer = profiler.start(ModelStage::QkvLinear);
        let total_tokens = hidden_states.size()[0];
        let qkv = linear(hidden_states, &self.qkv_proj);
        let q_width = self.num_heads * self.head_dim;
        let kv_width = self.num_kv_heads * self.head_dim;
        let q = qkv
            .narrow(-1, 0, q_width)
            .view([total_tokens, self.num_heads, self.head_dim]);
        let k = qkv.narrow(-1, q_width, kv_width).view([
            total_tokens,
            self.num_kv_heads,
            self.head_dim,
        ]);
        let v = qkv.narrow(-1, q_width + kv_width, kv_width).view([
            total_tokens,
            self.num_kv_heads,
            self.head_dim,
        ]);
        profiler.finish(timer);

        let timer = profiler.start(ModelStage::QkNorm);
        let (q, k) = qk_norm(&q, &k, &self.q_norm, &self.k_norm, eps);
        profiler.finish(timer);

        let timer = profiler.start(ModelStage::Rope);
        let (q, k) = rope.apply(&q, &k, positions);
        profiler.finish(timer);

        Ok((q, k, v))
    }

    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_batch: &AttentionBatch<'_>,
        attention_metadata: Option<&AttentionMetadata>,
        profiler: &mut ModelProfiler,
    ) -> Result<Tensor> {
        let timer = profiler.start(ModelStage::KvWrite);
        let write_result = self.base_attention.borrow_mut().write_kv(
            k,
            v,
            attention_metadata.and_then(|metadata| metadata.write_loc.as_ref()),
            attention_metadata.map_or(BatchPhase::Prefill, |metadata| metadata.forward_mode),
        );
        profiler.finish(timer);
        write_result?;

        let timer = profiler.start(ModelStage::AttentionBackend);
        let output = attention_batch.forward(q, k, v, &self.base_attention.borrow());
        profiler.finish(timer);
        output
    }

    fn output_mlp(
        &self,
        attention: &Tensor,
        residual: Tensor,
        eps: f64,
        profiler: &mut ModelProfiler,
    ) -> (Tensor, Tensor) {
        let timer = profiler.start(ModelStage::AttentionOutput);
        let output = linear(attention, &self.o_proj);
        profiler.finish(timer);
        let timer = profiler.start(ModelStage::Norm);
        let (normalized, residual) =
            add_rms_norm(output, residual, &self.post_attention_layernorm, eps);
        profiler.finish(timer);
        let timer = profiler.start(ModelStage::Mlp);
        let gate_up = linear(&normalized, &self.gate_up_proj);
        let mlp = silu_and_mul(&gate_up);
        let output = linear(&mlp, &self.down_proj);
        profiler.finish(timer);
        (output, residual)
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
            (&mut self.o_proj, "self_attn.o_proj.weight"),
            (&mut self.q_norm, "self_attn.q_norm.weight"),
            (&mut self.k_norm, "self_attn.k_norm.weight"),
            (
                &mut self.post_attention_layernorm,
                "post_attention_layernorm.weight",
            ),
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
        load_packed_parameter(
            &mut self.qkv_proj,
            &prefix,
            &[
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ],
            &[
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            weights,
            kind,
            device,
        )?;
        let intermediate = self.gate_up_proj.size()[0] / 2;
        load_packed_parameter(
            &mut self.gate_up_proj,
            &prefix,
            &["mlp.gate_proj.weight", "mlp.up_proj.weight"],
            &[intermediate, intermediate],
            weights,
            kind,
            device,
        )?;
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
        || config.max_position_embeddings == 0
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
    if !config.rope_theta.is_finite() || config.rope_theta <= 0.0 {
        return Err(model_error(
            "Qwen3 RoPE requires a positive finite rope_theta",
        ));
    }
    Ok(())
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

fn load_packed_parameter(
    target: &mut Tensor,
    prefix: &str,
    suffixes: &[&str],
    rows: &[i64],
    weights: &mut HashMap<String, Tensor>,
    kind: Kind,
    device: Device,
) -> Result<()> {
    let mut parts = Vec::with_capacity(suffixes.len());
    for (&suffix, &row_count) in suffixes.iter().zip(rows) {
        let name = format!("{prefix}.{suffix}");
        let weight = weights
            .remove(&name)
            .ok_or_else(|| model_error(&format!("checkpoint is missing {name}")))?;
        let expected = vec![row_count, target.size()[1]];
        if weight.size() != expected {
            return Err(model_error(&format!(
                "checkpoint tensor {name} has shape {:?}, expected {expected:?}",
                weight.size()
            )));
        }
        parts.push(weight.to_device(device).to_kind(kind));
    }
    *target = Tensor::cat(&parts, 0);
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

    use super::super::ops::rotate_half;
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
        let rope = RopeCache::new(16, 4, 10_000.0, Kind::BFloat16, Device::Cpu);
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
            let rope = RopeCache::new(16, 4, 10_000.0, kind, Device::Cpu);
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

    #[test]
    fn packed_projections_match_separate_projections() {
        let x = Tensor::arange(12, (Kind::Float, Device::Cpu)).view([3, 4]) / 10.0;
        let q = Tensor::arange(16, (Kind::Float, Device::Cpu)).view([4, 4]);
        let k = &q + 100.0;
        let v = &q + 200.0;
        let packed = Tensor::cat(&[&q, &k, &v], 0);
        let output = linear(&x, &packed);
        for (offset, weight) in [(0, &q), (4, &k), (8, &v)] {
            let difference = (output.narrow(1, offset, 4) - linear(&x, weight))
                .abs()
                .max()
                .double_value(&[]);
            assert!(difference < 1e-4);
        }

        let gate = &q / 10.0;
        let up = &k / 10.0;
        let gate_up = linear(&x, &Tensor::cat(&[&gate, &up], 0));
        let expected = linear(&x, &gate).silu() * linear(&x, &up);
        let difference = (silu_and_mul(&gate_up) - expected)
            .abs()
            .max()
            .double_value(&[]);
        assert!(difference < 1e-4);
    }
}
