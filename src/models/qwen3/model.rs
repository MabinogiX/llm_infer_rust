//! Dense Qwen3 causal language model implemented with libtorch tensors.
//!
//! The dense path supports segmented prefill graphs, eager cached-prefix
//! prefill, and paged-KV decode graphs. Qwen3-MoE remains outside this implementation.

use std::{cell::RefCell, collections::HashMap};

use tch::{Device, Kind, Tensor};

use super::config::Qwen3Config;
use crate::engine::{
    AttentionMetadata, BatchPhase, DecodeGraphState, ModelExecutor, ModelFactory, ModelRunnerError,
    ModelWeights, RuntimeModelConfig,
};
use crate::layers::{
    DenseSwiGlu, HalfSplitRope, PackedQkv, add_rms_norm, embedding, linear, logits, qk_norm,
    rms_norm,
};
use crate::models::attention::{Attention, AttentionBatch, AttentionSpec, BaseAttention};
use crate::profiling::{ModelProfiler, ModelStage};

#[path = "prefill_graph.rs"]
mod prefill_graph;

type Result<T> = std::result::Result<T, ModelRunnerError>;

/// Factory for the dense `Qwen3ForCausalLM` architecture.
#[derive(Debug, Clone, Copy)]
pub struct Qwen3Factory {
    pub config: Qwen3Config,
}

impl ModelFactory for Qwen3Factory {
    fn cache_spec(
        &self,
        _runtime: RuntimeModelConfig,
    ) -> Result<crate::engine::kvcache::ModelCacheSpec> {
        crate::engine::kvcache::ModelCacheSpec::uniform(
            self.config.num_layers,
            self.config.num_kv_heads,
            self.config.head_dim,
        )
        .map_err(|e| model_error(&e.to_string()))
    }

    fn validate_runtime(
        &self,
        runtime_config: RuntimeModelConfig,
        kind: Kind,
        device: Device,
        attention_backend: &str,
    ) -> Result<()> {
        validate_config(self.config)?;
        let expected = self.config.runtime(runtime_config.checkpoint_kind);
        if runtime_config != expected {
            return Err(model_error(&format!(
                "runtime model configuration does not match Qwen3 config: expected {expected:?}, received {runtime_config:?}"
            )));
        }
        Attention::validate(
            attention_backend,
            AttentionSpec {
                num_heads: self.config.num_attention_heads as i64,
                num_kv_heads: self.config.num_kv_heads as i64,
                head_dim: self.config.head_dim as i64,
                kind,
                device,
            },
        )
    }

    fn create(
        &self,
        kind: Kind,
        device: Device,
        attention_backend: &str,
    ) -> Result<Box<dyn ModelExecutor>> {
        Ok(Box::new(Qwen3ForCausalLM::new_with_attention_backend(
            self.config,
            kind,
            device,
            attention_backend,
        )?))
    }
}

/// Dense Qwen3 decoder-only model with QK-RMSNorm, RoPE, GQA, and SwiGLU.
pub struct Qwen3ForCausalLM {
    config: Qwen3Config,
    device: Device,
    kind: Kind,
    attention: Attention,
    rope: HalfSplitRope,
    embed_tokens: Tensor,
    layers: Vec<DecoderLayer>,
    norm: Tensor,
    lm_head: Tensor,
}

impl Qwen3ForCausalLM {
    pub fn new(config: Qwen3Config, kind: Kind, device: Device) -> Result<Self> {
        Self::new_with_attention_backend(config, kind, device, "pt")
    }

    pub fn new_with_attention_backend(
        config: Qwen3Config,
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
            config,
            device,
            kind,
            attention,
            rope: HalfSplitRope::new(max_positions, head_dim, config.rope_theta, kind, device),
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
        let mut hidden_states = embedding(&ids, &self.embed_tokens);
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
        let logits = logits(&hidden_states, &self.lm_head, logits_indices);
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
    fn graph_capabilities(&self) -> crate::engine::GraphCapabilities {
        use crate::engine::{GraphCapabilities, GraphLimits, GraphPadding, GraphSupport};
        let reason = if !matches!(self.device, Device::Cuda(_)) {
            Some("Qwen3 graphs require CUDA")
        } else if !matches!(self.kind, Kind::BFloat16 | Kind::Half) {
            Some("Qwen3 graphs require BF16 or FP16")
        } else if !self
            .layers
            .iter()
            .all(|layer| layer.base_attention.borrow().is_bound())
        {
            Some("Qwen3 graphs require bound full-history KV")
        } else {
            None
        };
        if let Some(reason) = reason {
            return GraphCapabilities {
                decode: GraphSupport::Unsupported(reason),
                segmented_prefill: GraphSupport::Unsupported(reason),
            };
        }
        let limit = |padding| GraphLimits {
            max_batch_size: usize::MAX,
            max_tokens: usize::MAX,
            max_context_len: self.config.max_position_embeddings,
            padding,
        };
        GraphCapabilities {
            decode: if !self
                .layers
                .iter()
                .all(|layer| layer.base_attention.borrow().reserved_write_slot() == 0)
            {
                GraphSupport::Unsupported("decode graph requires reserved KV slot zero")
            } else if self.attention.supports_cuda_graph() {
                GraphSupport::Supported(limit(GraphPadding::ReservedKvPageZero))
            } else {
                GraphSupport::Unsupported(
                    "attention backend has no verified full decode graph path",
                )
            },
            segmented_prefill: GraphSupport::Supported(limit(GraphPadding::InertTokenRows)),
        }
    }
    fn prefill_graph_program(&self) -> Option<&dyn crate::engine::PrefillGraphProgram> {
        Some(self)
    }

    fn prepare_decode_graph(
        &self,
        metadata: &AttentionMetadata,
    ) -> Result<Option<Box<dyn DecodeGraphState>>> {
        self.attention.prepare_decode_graph(metadata)
    }

    fn forward(
        &self,
        batch: &crate::engine::ForwardBatch<'_>,
    ) -> Result<crate::engine::ForwardOutput> {
        self.forward_impl(
            batch.input_ids,
            batch.positions,
            batch.attention,
            batch.logits_indices(),
        )
        .map(crate::engine::ForwardOutput::new)
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

    fn bind_state_cache(&mut self, cache: crate::engine::kvcache::ModelKvCache) -> Result<()> {
        // Validate every view before mutating bindings, so a bad later layer
        // cannot leave the model half-bound to a new cache.
        if cache.layers.len() != self.layers.len() || cache.page_size == 0 {
            return Err(model_error("Qwen3 cache layer count or page size mismatch"));
        }
        let pages = cache.layers[0].k.size().first().copied().unwrap_or(0);
        for layer in &cache.layers {
            let expected = vec![
                pages,
                cache.page_size as i64,
                self.config.num_kv_heads as i64,
                self.config.head_dim as i64,
            ];
            if pages < 2
                || layer.k.size() != expected
                || layer.v.size() != expected
                || layer.k.device() != self.device
                || layer.v.device() != self.device
                || layer.k.kind() != self.kind
                || layer.v.kind() != self.kind
            {
                return Err(model_error(
                    "Qwen3 cache view geometry, dtype or device mismatch",
                ));
            }
        }
        self.attention.bind_cache_layout(cache.page_size as i64);
        for (layer, view) in self.layers.iter().zip(cache.layers) {
            layer
                .base_attention
                .borrow_mut()
                .bind_kv_cache(view.k, view.v)?;
        }
        Ok(())
    }
}

struct DecoderLayer {
    base_attention: RefCell<BaseAttention>,
    input_layernorm: Tensor,
    qkv_proj: PackedQkv,
    o_proj: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    post_attention_layernorm: Tensor,
    mlp: DenseSwiGlu,
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
            qkv_proj: PackedQkv::new(hidden, num_heads, num_kv_heads, head_dim, kind, device),
            o_proj: parameter([hidden, num_heads * head_dim], kind, device),
            q_norm: parameter([head_dim], kind, device),
            k_norm: parameter([head_dim], kind, device),
            post_attention_layernorm: parameter([hidden], kind, device),
            mlp: DenseSwiGlu::new(hidden, intermediate, kind, device),
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
        rope: &HalfSplitRope,
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
        rope: &HalfSplitRope,
        profiler: &mut ModelProfiler,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let timer = profiler.start(ModelStage::QkvLinear);
        let (q, k, v) = self.qkv_proj.forward(hidden_states);
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
        let output = self.mlp.forward(&normalized);
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
            (&mut self.mlp.down_proj, "mlp.down_proj.weight"),
        ] {
            load_parameter(
                parameter,
                &format!("{prefix}.{suffix}"),
                weights,
                kind,
                device,
            )?;
        }
        let qkv_rows = self.qkv_proj.widths();
        load_packed_parameter(
            &mut self.qkv_proj.weight,
            &prefix,
            &[
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ],
            &qkv_rows,
            weights,
            kind,
            device,
        )?;
        let intermediate = self.mlp.gate_up_proj.size()[0] / 2;
        load_packed_parameter(
            &mut self.mlp.gate_up_proj,
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

fn validate_config(config: Qwen3Config) -> Result<()> {
    config.validate().map_err(|message| model_error(&message))
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

    use super::*;
    use crate::engine::load_hf_safetensors;

    static TEST_DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);

    fn config() -> Qwen3Config {
        Qwen3Config {
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
            .forward_impl(
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
            .forward_impl(
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
            .forward_impl(
                &Tensor::from_slice(&[1i64]),
                &Tensor::from_slice(&[0i64]),
                None,
                None,
            )
            .unwrap();
        assert_eq!(logits.size(), vec![1, 8]);
    }

    #[test]
    fn paged_decode_and_cached_prefix_prefill_match_full_forward() {
        let mut model = Qwen3ForCausalLM::new(config(), Kind::Float, Device::Cpu).unwrap();
        // Zero-initialized parameters only checked shapes and could not detect
        // missing KV writes. Use nonuniform deterministic weights and compare
        // each cached step to the full causal forward.
        for weight in [&mut model.embed_tokens, &mut model.lm_head] {
            weight.copy_(
                &(Tensor::arange(weight.numel() as i64, (Kind::Float, Device::Cpu))
                    .sin()
                    .view(weight.size().as_slice())
                    * 0.1),
            );
        }
        let _ = model.norm.fill_(1.0);
        for layer in &mut model.layers {
            for weight in [
                &mut layer.input_layernorm,
                &mut layer.q_norm,
                &mut layer.k_norm,
                &mut layer.post_attention_layernorm,
            ] {
                let _ = weight.fill_(1.0);
            }
            for weight in [
                &mut layer.qkv_proj.weight,
                &mut layer.o_proj,
                &mut layer.mlp.gate_up_proj,
                &mut layer.mlp.down_proj,
            ] {
                weight.copy_(
                    &(Tensor::arange(weight.numel() as i64, (Kind::Float, Device::Cpu))
                        .cos()
                        .view(weight.size().as_slice())
                        * 0.2),
                );
            }
        }
        let full = model
            .forward_impl(
                &Tensor::from_slice(&[1i64, 2, 3, 4]),
                &Tensor::from_slice(&[0i64, 1, 2, 3]),
                None,
                None,
            )
            .unwrap();
        assert!(full.abs().max().double_value(&[]) > 0.01);
        let check = |actual: Tensor, start, rows| {
            assert_eq!(actual.size(), vec![rows, 8]);
            let error = (actual - full.narrow(0, start, rows))
                .abs()
                .max()
                .double_value(&[]);
            assert!(error < 1e-5, "start={start}, error={error}");
        };
        let k_cache = Tensor::zeros([1, 2, 2, 1, 2], (Kind::Float, Device::Cpu));
        let v_cache = Tensor::zeros([1, 2, 2, 1, 2], (Kind::Float, Device::Cpu));
        model
            .bind_state_cache(crate::engine::kvcache::ModelKvCache {
                page_size: k_cache.size()[2] as usize,
                layers: (0..k_cache.size()[0])
                    .map(|i| crate::engine::kvcache::LayerKvCache {
                        k: k_cache.get(i),
                        v: v_cache.get(i),
                    })
                    .collect(),
            })
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
        check(
            model
                .forward_impl(
                    &Tensor::from_slice(&[1i64, 2]),
                    &Tensor::from_slice(&[0i64, 1]),
                    Some(&first_prefill),
                    None,
                )
                .unwrap(),
            0,
            2,
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
        check(
            model
                .forward_impl(
                    &Tensor::from_slice(&[3i64]),
                    &Tensor::from_slice(&[2i64]),
                    Some(&decode),
                    None,
                )
                .unwrap(),
            2,
            1,
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
        check(
            model
                .forward_impl(
                    &Tensor::from_slice(&[4i64]),
                    &Tensor::from_slice(&[3i64]),
                    Some(&cached_prefill),
                    None,
                )
                .unwrap(),
            3,
            1,
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
