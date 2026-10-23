//! 直接使用 libtorch 的混合注意力实现；循环状态独立于分页 KV。
use super::Config;
use crate::{
    engine::{
        BatchPhase, ForwardBatch, ForwardOutput, MetadataRequirements, ModelExecutor, ModelFactory,
        ModelRunnerError, ModelWeights, RuntimeModelConfig,
        kvcache::{
            LinearState, ModelCacheSpec, ModelKvCache, RecurrentStatePool, RequestState,
            SharedRecurrentPool, StateSlotId,
        },
    },
    layers::{embedding, linear, logits},
    models::attention::{Attention, AttentionSpec, BaseAttention},
};
use std::{cell::RefCell, collections::HashMap, rc::Rc};
use tch::{Device, Kind, Tensor};
type Result<T> = std::result::Result<T, ModelRunnerError>;
fn error(s: impl Into<String>) -> ModelRunnerError {
    ModelRunnerError::Model(s.into())
}

pub struct Factory {
    pub config: Config,
}
impl Factory {
    fn spec(&self, kind: Kind, device: Device) -> AttentionSpec {
        AttentionSpec {
            num_heads: self.config.num_attention_heads,
            num_kv_heads: self.config.num_key_value_heads,
            head_dim: self.config.head_dim,
            kind,
            device,
        }
    }
}
impl ModelFactory for Factory {
    fn cache_spec(&self, _: RuntimeModelConfig) -> Result<ModelCacheSpec> {
        ModelCacheSpec::uniform(
            self.config.full_layers(),
            self.config.num_key_value_heads as usize,
            self.config.head_dim as usize,
        )
        .map_err(|e| error(e.to_string()))
    }
    fn validate_runtime(
        &self,
        runtime: RuntimeModelConfig,
        kind: Kind,
        device: Device,
        backend: &str,
    ) -> Result<()> {
        if runtime != self.config.runtime(runtime.checkpoint_kind) {
            return Err(error("Qwen3.5 runtime 配置不匹配"));
        }
        Attention::validate(backend, self.spec(kind, device))
    }
    fn create(&self, kind: Kind, device: Device, backend: &str) -> Result<Box<dyn ModelExecutor>> {
        Ok(Box::new(Model {
            config: self.config.clone(),
            kind,
            device,
            weights: HashMap::new(),
            attention: Attention::new(backend, self.spec(kind, device))?,
            caches: (0..self.config.full_layers())
                .map(|_| RefCell::new(BaseAttention::default()))
                .collect(),
            state_pool: Rc::new(RefCell::new(RecurrentStatePool::empty())),
            page_size: 1,
        }))
    }
}
struct Row {
    key: StateSlotId,
    start: i64,
    len: i64,
    position: i64,
    checkpoint: Option<usize>,
}
struct Model {
    config: Config,
    kind: Kind,
    device: Device,
    weights: HashMap<String, Tensor>,
    attention: Attention,
    caches: Vec<RefCell<BaseAttention>>,
    state_pool: SharedRecurrentPool,
    page_size: usize,
}
impl Model {
    fn w(&self, name: &str) -> &Tensor {
        &self.weights[name]
    }
    fn norm(&self, x: &Tensor, name: &str) -> Tensor {
        // Qwen3.5 的 RMSNorm 权重表示相对 1 的偏移，乘法在 FP32 中完成。
        let xf = x.to_kind(Kind::Float);
        let variance = xf.square().mean_dim(&[-1i64][..], true, Kind::Float);
        (xf * (variance + self.config.rms_norm_eps).rsqrt()
            * (self.w(name).to_kind(Kind::Float) + 1.0))
            .to_kind(x.kind())
    }
    fn rotate(&self, x: &Tensor, positions: &Tensor) -> Tensor {
        let dim = self.config.rotary_dim();
        let inv = (Tensor::arange_start_step(0, dim, 2, (Kind::Float, self.device))
            * (-self.config.rope_parameters.rope_theta.ln() / dim as f64))
            .exp();
        let angle = positions.to_kind(Kind::Float).view([-1, 1]) * inv;
        let cos = angle.cos().to_kind(self.kind).unsqueeze(1);
        let sin = angle.sin().to_kind(self.kind).unsqueeze(1);
        let a = x.narrow(-1, 0, dim / 2);
        let b = x.narrow(-1, dim / 2, dim / 2);
        Tensor::cat(
            &[
                &a * &cos - &b * &sin,
                &b * cos + &a * sin,
                x.narrow(-1, dim, self.config.head_dim - dim),
            ],
            -1,
        )
    }
    fn rows(&self, batch: &ForwardBatch<'_>) -> Result<Vec<Row>> {
        let meta = batch
            .attention
            .ok_or_else(|| error("混合模型需要分页缓存元数据"))?;
        let keys = meta
            .state_slots
            .as_ref()
            .ok_or_else(|| error("缺少独立循环状态槽 ID"))?;
        let bounds: Vec<i32> = if batch.phase() == BatchPhase::Prefill {
            Vec::try_from(&meta.cu_seqlens_q.as_ref().unwrap().to_device(Device::Cpu))?
        } else {
            (0..=batch.tokens() as i32).collect()
        };
        // Decode lengths are immutable CPU snapshots from the scheduler. A
        // device-to-host position read here would drain the preceding forward
        // and defeat the one-batch pipeline even without CUDA Graphs.
        let positions: Vec<i64> = if batch.phase() == BatchPhase::Decode {
            if let Some(host) = &meta.decode_host {
                host.cache_seqlens
                    .iter()
                    .map(|&len| i64::from(len) - 1)
                    .collect()
            } else {
                Vec::try_from(&batch.positions.view([-1]).to_device(Device::Cpu))?
            }
        } else {
            Vec::try_from(&batch.positions.view([-1]).to_device(Device::Cpu))?
        };
        if keys.len() + 1 != bounds.len() {
            return Err(error("循环状态槽数量与请求数不匹配"));
        }
        let branching: Vec<_> = keys
            .iter()
            .map(|&key| {
                self.state_pool
                    .borrow()
                    .branching_seqlen(key)
                    .map_err(|e| error(e.to_string()))
            })
            .collect::<Result<_>>()?;
        Ok(keys
            .iter()
            .copied()
            .enumerate()
            .map(|(i, key)| Row {
                key,
                start: bounds[i] as i64,
                len: (bounds[i + 1] - bounds[i]) as i64,
                position: positions[bounds[i] as usize],
                checkpoint: {
                    let start = positions[bounds[i] as usize] as usize;
                    let end = start + (bounds[i + 1] - bounds[i]) as usize;
                    let target = branching[i]
                        .filter(|&target| target > start && target <= end)
                        .unwrap_or(end / self.page_size * self.page_size);
                    (target > start).then_some(target)
                },
            })
            .collect())
    }
    fn new_state(&self) -> RequestState {
        let c = &self.config;
        RequestState {
            position: 0,
            layers: c
                .layer_types
                .iter()
                .filter(|s| *s == "linear_attention")
                .map(|_| LinearState {
                    conv: Tensor::zeros(
                        [
                            1,
                            2 * c.linear_num_key_heads * c.linear_key_head_dim
                                + c.linear_num_value_heads * c.linear_value_head_dim,
                            c.linear_conv_kernel_dim - 1,
                        ],
                        (self.kind, self.device),
                    ),
                    recurrent: Tensor::zeros(
                        [
                            c.linear_num_value_heads,
                            c.linear_key_head_dim,
                            c.linear_value_head_dim,
                        ],
                        (Kind::Float, self.device),
                    ),
                })
                .collect(),
        }
    }
    fn linear_attention(
        &self,
        x: &Tensor,
        prefix: &str,
        state: &mut LinearState,
    ) -> Result<Tensor> {
        let c = &self.config;
        let n = x.size()[0];
        let kd = c.linear_num_key_heads * c.linear_key_head_dim;
        let vd = c.linear_num_value_heads * c.linear_value_head_dim;
        let raw = linear(x, self.w(&format!("{prefix}.in_proj_qkv.weight")))
            .transpose(0, 1)
            .unsqueeze(0);
        let joined = Tensor::cat(&[&state.conv, &raw], -1);
        state.conv = joined
            .narrow(-1, n, c.linear_conv_kernel_dim - 1)
            .contiguous();
        let convolved = joined
            .f_conv1d(
                self.w(&format!("{prefix}.conv1d.weight")),
                None::<&Tensor>,
                [1],
                [0],
                [1],
                2 * kd + vd,
            )?
            .silu()
            .squeeze_dim(0)
            .transpose(0, 1);
        let repeat = c.linear_num_value_heads / c.linear_num_key_heads;
        let q = convolved
            .narrow(1, 0, kd)
            .view([n, c.linear_num_key_heads, c.linear_key_head_dim])
            .repeat_interleave_self_int(repeat, 1, None)
            .to_kind(Kind::Float);
        let k = convolved
            .narrow(1, kd, kd)
            .view([n, c.linear_num_key_heads, c.linear_key_head_dim])
            .repeat_interleave_self_int(repeat, 1, None)
            .to_kind(Kind::Float);
        let q = l2norm(&q) / (c.linear_key_head_dim as f64).sqrt();
        let k = l2norm(&k);
        let v = convolved
            .narrow(1, 2 * kd, vd)
            .view([n, c.linear_num_value_heads, c.linear_value_head_dim])
            .to_kind(Kind::Float);
        let beta = linear(x, self.w(&format!("{prefix}.in_proj_b.weight")))
            .sigmoid()
            .to_kind(Kind::Float);
        let a = linear(x, self.w(&format!("{prefix}.in_proj_a.weight"))).to_kind(Kind::Float);
        let g = -self
            .w(&format!("{prefix}.A_log"))
            .to_kind(Kind::Float)
            .exp()
            * (a + self.w(&format!("{prefix}.dt_bias")).to_kind(Kind::Float)).softplus();
        let (out, recurrent) = delta_scan(&q, &k, &v, &g, &beta, &state.recurrent);
        state.recurrent = recurrent;
        let out = out.to_kind(self.kind);
        let variance = out
            .to_kind(Kind::Float)
            .square()
            .mean_dim(&[-1i64][..], true, Kind::Float);
        // 门控 norm 的 scale 是直接权重，与解码器 norm 不同。
        let normed = (out.to_kind(Kind::Float) * (variance + c.rms_norm_eps).rsqrt())
            .to_kind(self.kind)
            * self.w(&format!("{prefix}.norm.weight"));
        let z = linear(x, self.w(&format!("{prefix}.in_proj_z.weight"))).view([
            n,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
        ]);
        let gated = (normed.to_kind(Kind::Float) * z.to_kind(Kind::Float).silu())
            .to_kind(self.kind)
            .view([n, vd]);
        Ok(linear(&gated, self.w(&format!("{prefix}.out_proj.weight"))))
    }
    fn weight_shapes(&self) -> Vec<(String, Vec<i64>)> {
        let c = &self.config;
        let h = c.hidden_size;
        let mut shapes = vec![
            (
                "model.language_model.embed_tokens.weight".into(),
                vec![c.vocab_size as i64, h],
            ),
            ("model.language_model.norm.weight".into(), vec![h]),
            ("lm_head.weight".into(), vec![c.vocab_size as i64, h]),
        ];
        for (i, ty) in c.layer_types.iter().enumerate() {
            let p = format!("model.language_model.layers.{i}");
            for (name, shape) in [
                ("input_layernorm.weight", vec![h]),
                ("post_attention_layernorm.weight", vec![h]),
                ("mlp.gate_proj.weight", vec![c.intermediate_size, h]),
                ("mlp.up_proj.weight", vec![c.intermediate_size, h]),
                ("mlp.down_proj.weight", vec![h, c.intermediate_size]),
            ] {
                shapes.push((format!("{p}.{name}"), shape));
            }
            let mut add = |name: &str, shape| shapes.push((format!("{p}.{name}"), shape));
            if ty == "full_attention" {
                for (name, width) in [
                    ("q_proj", 2 * c.num_attention_heads * c.head_dim),
                    ("k_proj", c.num_key_value_heads * c.head_dim),
                    ("v_proj", c.num_key_value_heads * c.head_dim),
                ] {
                    add(&format!("self_attn.{name}.weight"), vec![width, h]);
                }
                add(
                    "self_attn.o_proj.weight",
                    vec![h, c.num_attention_heads * c.head_dim],
                );
                add("self_attn.q_norm.weight", vec![c.head_dim]);
                add("self_attn.k_norm.weight", vec![c.head_dim]);
            } else {
                let kd = c.linear_num_key_heads * c.linear_key_head_dim;
                let vd = c.linear_num_value_heads * c.linear_value_head_dim;
                for (name, width) in [
                    ("in_proj_qkv", 2 * kd + vd),
                    ("in_proj_z", vd),
                    ("in_proj_a", c.linear_num_value_heads),
                    ("in_proj_b", c.linear_num_value_heads),
                ] {
                    add(&format!("linear_attn.{name}.weight"), vec![width, h]);
                }
                add("linear_attn.out_proj.weight", vec![h, vd]);
                add(
                    "linear_attn.conv1d.weight",
                    vec![2 * kd + vd, 1, c.linear_conv_kernel_dim],
                );
                add("linear_attn.A_log", vec![c.linear_num_value_heads]);
                add("linear_attn.dt_bias", vec![c.linear_num_value_heads]);
                add("linear_attn.norm.weight", vec![c.linear_value_head_dim]);
            }
        }
        shapes
    }
}
impl ModelExecutor for Model {
    fn recurrent_state_pool(&self) -> Option<SharedRecurrentPool> {
        Some(Rc::clone(&self.state_pool))
    }
    fn recurrent_state_bytes(&self) -> Result<Option<usize>> {
        let c = &self.config;
        let layers = c
            .layer_types
            .iter()
            .filter(|ty| *ty == "linear_attention")
            .count() as u128;
        let hk = c.linear_num_key_heads as u128;
        let hv = c.linear_num_value_heads as u128;
        let k = c.linear_key_head_dim as u128;
        let v = c.linear_value_head_dim as u128;
        let conv = (2 * hk * k + hv * v)
            * (c.linear_conv_kernel_dim - 1) as u128
            * self.kind.elt_size_in_bytes() as u128;
        let bytes = layers * (conv + hv * k * v * 4); // recurrent 固定 FP32。
        usize::try_from(bytes)
            .map(Some)
            .map_err(|_| error("循环状态字节数溢出"))
    }
    fn reserve_request_states(
        &mut self,
        capacity: usize,
        checkpoint_capacity: usize,
    ) -> Result<()> {
        *self.state_pool.borrow_mut() =
            RecurrentStatePool::new(capacity, checkpoint_capacity, || self.new_state());
        tracing::info!(
            request_slots = capacity,
            checkpoint_slots = checkpoint_capacity,
            storage_slots = self.state_pool.borrow().storage_capacity(),
            "统一循环状态池已预分配，历史状态按需申请并移交"
        );
        Ok(())
    }
    fn metadata_requirements(&self, _: BatchPhase) -> MetadataRequirements {
        MetadataRequirements {
            block_table: true,
            req_to_token: true,
        }
    }
    fn set_kv_reserved_slot(&mut self, slot: i64) {
        for cache in &mut self.caches {
            cache.get_mut().set_reserved_write_slot(slot);
        }
    }
    fn bind_state_cache(&mut self, cache: ModelKvCache) -> Result<()> {
        if cache.layers.len() != self.caches.len() {
            return Err(error("Qwen3.5 KV 层数不匹配"));
        }
        self.attention.bind_cache_layout(cache.page_size as i64);
        for (target, source) in self.caches.iter_mut().zip(cache.layers) {
            target.get_mut().bind_kv_cache(source.k, source.v)?;
        }
        self.page_size = cache.page_size;
        Ok(())
    }
    fn load_weights(&mut self, weights: ModelWeights) -> Result<usize> {
        let mut source: HashMap<_, _> = weights.into_tensors().into_iter().collect();
        let shapes = self.weight_shapes();
        for (name, shape) in &shapes {
            let tensor = source
                .remove(name)
                .ok_or_else(|| error(format!("缺少权重 {name}")))?;
            if tensor.size() != *shape {
                return Err(error(format!(
                    "权重 {name} 形状 {:?}，期望 {shape:?}",
                    tensor.size()
                )));
            }
            self.weights.insert(
                name.clone(),
                tensor.to_device(self.device).to_kind(self.kind),
            );
        }
        for key in source.keys() {
            if !key.starts_with("model.visual.")
                && !key.starts_with("mtp.")
                && !key.starts_with("model.mtp.")
            {
                return Err(error(format!("未识别的文本权重 {key}")));
            }
        }
        Ok(shapes.len())
    }
    fn forward(&self, batch: &ForwardBatch<'_>) -> Result<ForwardOutput> {
        let mut rows = self.rows(batch)?;
        let mut states = self.state_pool.borrow_mut();
        for row in &rows {
            let state = states
                .current_mut(row.key)
                .map_err(|e| error(e.to_string()))?;
            if state.position != row.position {
                return Err(error("循环状态位置与 KV 前缀不匹配"));
            }
        }
        for row in &mut rows {
            if row.checkpoint.is_some()
                && !states
                    .begin_snapshot(row.key)
                    .map_err(|e| error(e.to_string()))?
            {
                row.checkpoint = None;
            }
        }
        let prepared = self.attention.prepare(batch.attention, batch.tokens())?;
        let mut hidden = embedding(
            &batch.input_ids.view([-1]),
            self.w("model.language_model.embed_tokens.weight"),
        );
        let c = &self.config;
        let mut full = 0;
        let mut recurrent = 0;
        for (i, ty) in c.layer_types.iter().enumerate() {
            let p = format!("model.language_model.layers.{i}");
            let x = self.norm(&hidden, &format!("{p}.input_layernorm.weight"));
            let mixed =
                if ty == "linear_attention" {
                    let outputs: Result<Vec<_>> = rows
                        .iter()
                        .map(|row| {
                            let x = x.narrow(0, row.start, row.len);
                            if let Some(target) = row.checkpoint {
                                let (state, staged) = states
                                    .snapshot_states_mut(row.key)
                                    .map_err(|e| error(e.to_string()))?;
                                let split = target as i64 - row.position;
                                let first = self.linear_attention(
                                    &x.narrow(0, 0, split),
                                    &format!("{p}.linear_attn"),
                                    &mut state.layers[recurrent],
                                )?;
                                staged.layers[recurrent]
                                    .conv
                                    .copy_(&state.layers[recurrent].conv);
                                staged.layers[recurrent]
                                    .recurrent
                                    .copy_(&state.layers[recurrent].recurrent);
                                if split == row.len {
                                    Ok(first)
                                } else {
                                    let tail = self.linear_attention(
                                        &x.narrow(0, split, row.len - split),
                                        &format!("{p}.linear_attn"),
                                        &mut state.layers[recurrent],
                                    )?;
                                    Ok(Tensor::cat(&[first, tail], 0))
                                }
                            } else {
                                self.linear_attention(
                                    &x,
                                    &format!("{p}.linear_attn"),
                                    &mut states
                                        .current_mut(row.key)
                                        .map_err(|e| error(e.to_string()))?
                                        .layers[recurrent],
                                )
                            }
                        })
                        .collect();
                    recurrent += 1;
                    Tensor::cat(&outputs?, 0)
                } else {
                    let qg = linear(&x, self.w(&format!("{p}.self_attn.q_proj.weight"))).view([
                        -1,
                        c.num_attention_heads,
                        2 * c.head_dim,
                    ]);
                    let q = self.rotate(
                        &self.norm(
                            &qg.narrow(-1, 0, c.head_dim),
                            &format!("{p}.self_attn.q_norm.weight"),
                        ),
                        batch.positions,
                    );
                    let k =
                        self.rotate(
                            &self.norm(
                                &linear(&x, self.w(&format!("{p}.self_attn.k_proj.weight")))
                                    .view([-1, c.num_key_value_heads, c.head_dim]),
                                &format!("{p}.self_attn.k_norm.weight"),
                            ),
                            batch.positions,
                        );
                    let v = linear(&x, self.w(&format!("{p}.self_attn.v_proj.weight"))).view([
                        -1,
                        c.num_key_value_heads,
                        c.head_dim,
                    ]);
                    self.caches[full].borrow_mut().write_kv(
                        &k,
                        &v,
                        batch.attention.and_then(|meta| meta.write_loc.as_ref()),
                        batch.phase(),
                    )?;
                    let out = prepared.forward(
                        &q.contiguous(),
                        &k.contiguous(),
                        &v,
                        &self.caches[full].borrow(),
                    )?;
                    full += 1;
                    linear(
                        &(out.view([-1, c.num_attention_heads, c.head_dim])
                            * qg.narrow(-1, c.head_dim, c.head_dim).sigmoid())
                        .view([-1, c.num_attention_heads * c.head_dim]),
                        self.w(&format!("{p}.self_attn.o_proj.weight")),
                    )
                };
            hidden += mixed;
            let x = self.norm(&hidden, &format!("{p}.post_attention_layernorm.weight"));
            let gate = linear(&x, self.w(&format!("{p}.mlp.gate_proj.weight"))).silu();
            let up = linear(&x, self.w(&format!("{p}.mlp.up_proj.weight")));
            hidden += linear(&(gate * up), self.w(&format!("{p}.mlp.down_proj.weight")));
        }
        let output = ForwardOutput::new(logits(
            &self.norm(&hidden, "model.language_model.norm.weight"),
            self.w("lm_head.weight"),
            batch.logits_indices(),
        ));
        for row in &rows {
            let end = (row.position + row.len) as usize;
            states
                .commit_forward(row.key, end, row.checkpoint)
                .map_err(|e| error(e.to_string()))?;
        }
        Ok(output)
    }
}
fn l2norm(x: &Tensor) -> Tensor {
    x * (x.square().sum_dim_intlist(&[-1i64][..], true, Kind::Float) + 1e-6).rsqrt()
}

// 以小块三角求解实现 delta 更新，避免 prefill 为每个 token 启动整套算子。
fn delta_scan(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial: &Tensor,
) -> (Tensor, Tensor) {
    let mut state = initial.shallow_clone();
    let mut outputs = Vec::new();
    let n = q.size()[0];
    for start in (0..n).step_by(64) {
        let len = (n - start).min(64);
        let q = q.narrow(0, start, len).transpose(0, 1);
        let k = k.narrow(0, start, len).transpose(0, 1);
        let v = v.narrow(0, start, len).transpose(0, 1);
        let beta = beta.narrow(0, start, len).transpose(0, 1).unsqueeze(-1);
        let decay = g
            .narrow(0, start, len)
            .transpose(0, 1)
            .cumsum(-1, Kind::Float);
        let future = Tensor::ones([len, len], (Kind::Bool, q.device())).triu(1);
        let pair = (decay.unsqueeze(-1) - decay.unsqueeze(-2))
            .masked_fill(&future, f64::NEG_INFINITY)
            .exp();
        let kb = &k * &beta;
        let system = (kb.matmul(&k.transpose(-1, -2)) * &pair).tril(-1)
            + Tensor::eye(len, (Kind::Float, q.device()));
        let values = system.linalg_solve_triangular(&(&v * &beta), false, true, true);
        let keys =
            system.linalg_solve_triangular(&(&kb * decay.exp().unsqueeze(-1)), false, true, true);
        let delta = values - keys.matmul(&state);
        let out = (&q * decay.exp().unsqueeze(-1)).matmul(&state)
            + (q.matmul(&k.transpose(-1, -2)) * pair).matmul(&delta);
        let last = decay.narrow(-1, len - 1, 1);
        state = &state * last.exp().unsqueeze(-1)
            + (k * (last - decay).exp().unsqueeze(-1))
                .transpose(-1, -2)
                .matmul(&delta);
        outputs.push(out.transpose(0, 1));
    }
    (Tensor::cat(&outputs, 0), state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::kvcache::LayerKvCache;
    use crate::engine::{AttentionMetadata, ForwardMode};
    pub(super) fn tiny() -> Model {
        let config = Config::parse(&serde_json::json!({"hidden_size":8,"num_hidden_layers":2,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"intermediate_size":12,
            "vocab_size":16,"max_position_embeddings":32,"rms_norm_eps":1e-6,
            "layer_types":["linear_attention","full_attention"],"linear_conv_kernel_dim":4,
            "linear_key_head_dim":3,"linear_value_head_dim":4,"linear_num_key_heads":1,"linear_num_value_heads":2,
            "hidden_act":"silu","rope_parameters":{"rope_type":"default","rope_theta":10000.0,"partial_rotary_factor":0.5}})).unwrap();
        let mut model = Model {
            attention: Attention::new(
                "pt",
                AttentionSpec {
                    num_heads: 2,
                    num_kv_heads: 1,
                    head_dim: 4,
                    kind: Kind::Float,
                    device: Device::Cpu,
                },
            )
            .unwrap(),
            caches: vec![RefCell::new(BaseAttention::default())],
            config,
            weights: HashMap::new(),
            kind: Kind::Float,
            device: Device::Cpu,
            state_pool: Rc::new(RefCell::new(RecurrentStatePool::empty())),
            page_size: 1,
        };
        tch::manual_seed(17);
        for (name, shape) in model.weight_shapes() {
            model.weights.insert(
                name,
                Tensor::randn(&shape, (Kind::Float, Device::Cpu)) * 0.15,
            );
        }
        model
            .bind_state_cache(ModelKvCache {
                page_size: 2,
                layers: vec![LayerKvCache {
                    k: Tensor::zeros([33, 2, 1, 4], (Kind::Float, Device::Cpu)),
                    v: Tensor::zeros([33, 2, 1, 4], (Kind::Float, Device::Cpu)),
                }],
            })
            .unwrap();
        model
    }
    pub(super) fn run(
        model: &Model,
        ids: &[i64],
        position: i64,
        page: i32,
        decode: bool,
        slot: StateSlotId,
    ) -> Tensor {
        let pages: Vec<i32> = (page..page + 16).collect();
        run_pages(model, ids, position, &pages, decode, slot)
    }
    pub(super) fn run_pages(
        model: &Model,
        ids: &[i64],
        position: i64,
        pages: &[i32],
        decode: bool,
        slot: StateSlotId,
    ) -> Tensor {
        let n = ids.len() as i64;
        let tokens: Vec<i32> = pages
            .iter()
            .flat_map(|page| [page * 2, page * 2 + 1])
            .collect();
        let loc: Vec<i32> = tokens[position as usize..(position + n) as usize].to_vec();
        let phase = if decode {
            BatchPhase::Decode
        } else {
            BatchPhase::Prefill
        };
        let metadata = AttentionMetadata {
            state_slots: Some(vec![slot]),
            forward_mode: phase,
            decode_host: None,
            write_loc: Some(Tensor::from_slice(&loc)),
            block_table: Some(Tensor::from_slice(pages).view([1, pages.len() as i64])),
            req_to_token: Some(Tensor::from_slice(&tokens).view([1, tokens.len() as i64])),
            cu_seqlens_q: (!decode).then(|| Tensor::from_slice(&[0i32, n as i32])),
            prefix_lens: (!decode).then(|| Tensor::from_slice(&[position as i32])),
            cache_seqlens: decode.then(|| Tensor::from_slice(&[(position + n) as i32])),
            max_seqlen: Some((position + n) as usize),
        };
        let input = Tensor::from_slice(ids);
        let positions = Tensor::arange_start(position, position + n, (Kind::Int64, Device::Cpu));
        let indices = Tensor::from_slice(&[n - 1]);
        model
            .forward(&ForwardBatch {
                input_ids: &input,
                positions: &positions,
                attention: Some(&metadata),
                mode: if decode {
                    ForwardMode::Decode
                } else {
                    ForwardMode::Prefill {
                        logits_indices: &indices,
                    }
                },
            })
            .unwrap()
            .logits
    }
    #[test]
    fn decode_rows_use_host_lengths_without_reading_device_positions() {
        let mut model = tiny();
        model.reserve_request_states(1, 4).unwrap();
        let slot = model.state_pool.borrow_mut().alloc_request().unwrap();
        let metadata = AttentionMetadata {
            state_slots: Some(vec![slot]),
            forward_mode: BatchPhase::Decode,
            decode_host: Some(crate::engine::DecodeHostMetadata {
                cache_seqlens: vec![8],
                block_table_versions: None,
            }),
            write_loc: None,
            cu_seqlens_q: None,
            prefix_lens: None,
            block_table: None,
            req_to_token: None,
            cache_seqlens: None,
            max_seqlen: None,
        };
        let input = Tensor::from_slice(&[1i64]);
        // A deliberately inaccessible position element: reading it would
        // fail. Scheduler-owned lengths carry all decode row information.
        let positions = Tensor::zeros([0], (Kind::Int64, Device::Cpu));
        let rows = model
            .rows(&ForwardBatch {
                input_ids: &input,
                positions: &positions,
                attention: Some(&metadata),
                mode: ForwardMode::Decode,
            })
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].position, 7);
        assert_eq!(rows[0].checkpoint, Some(8));
    }

    #[test]
    fn hybrid_chunk_decode_interleaving_reset_and_state_retirement() {
        let mut model = tiny();
        model.reserve_request_states(2, 4).unwrap();
        let a = model.state_pool.borrow_mut().alloc_request().unwrap();
        let expected = run(&model, &[1, 2, 3, 4, 5, 6], 0, 1, false, a);
        model.state_pool.borrow_mut().free_request(a).unwrap();
        let a = model.state_pool.borrow_mut().alloc_request().unwrap();
        let b = model.state_pool.borrow_mut().alloc_request().unwrap();
        let _ = run(&model, &[1, 2], 0, 1, false, a);
        let other = run(&model, &[7, 8, 9], 0, 17, false, b);
        let _ = run(&model, &[3, 4, 5], 2, 1, false, a);
        let actual = run(&model, &[6], 5, 1, true, a);
        assert!((expected - actual).abs().max().double_value(&[]) < 1e-5);
        model.state_pool.borrow_mut().free_request(b).unwrap();
        let b = model.state_pool.borrow_mut().alloc_request().unwrap();
        assert!(
            (other - run(&model, &[7, 8, 9], 0, 17, false, b))
                .abs()
                .max()
                .double_value(&[])
                < 1e-5
        );
        model.state_pool.borrow_mut().free_request(a).unwrap();
        model.state_pool.borrow_mut().free_request(b).unwrap();
        assert_eq!(model.state_pool.borrow().usage(), (0, 0));
    }
    #[test]
    fn block_delta_matches_independent_token_recurrence_with_nonzero_state() {
        tch::manual_seed(29);
        let q = l2norm(&Tensor::randn([135, 2, 3], (Kind::Float, Device::Cpu))) / 3f64.sqrt();
        let k = l2norm(&Tensor::randn([135, 2, 3], (Kind::Float, Device::Cpu)));
        let v = Tensor::randn([135, 2, 4], (Kind::Float, Device::Cpu));
        let g = -Tensor::rand([135, 2], (Kind::Float, Device::Cpu)) * 0.5;
        let beta = Tensor::rand([135, 2], (Kind::Float, Device::Cpu));
        let initial = Tensor::randn([2, 3, 4], (Kind::Float, Device::Cpu));
        let (actual, last) = delta_scan(&q, &k, &v, &g, &beta, &initial);
        let mut state = initial;
        let mut expected = Vec::new();
        for t in 0..135 {
            state *= g.get(t).exp().view([2, 1, 1]);
            let key = k.get(t);
            let predicted =
                (&state * key.unsqueeze(-1)).sum_dim_intlist(&[1i64][..], false, Kind::Float);
            let delta = (v.get(t) - predicted) * beta.get(t).unsqueeze(-1);
            state += key.unsqueeze(-1) * delta.unsqueeze(-2);
            expected.push((&state * q.get(t).unsqueeze(-1)).sum_dim_intlist(
                &[1i64][..],
                false,
                Kind::Float,
            ));
        }
        assert!(
            (actual - Tensor::stack(&expected, 0))
                .abs()
                .max()
                .double_value(&[])
                < 2e-5
        );
        assert!((last - state).abs().max().double_value(&[]) < 2e-5);
    }
}

#[cfg(test)]
mod cached_model_tests {
    use super::tests::{run, run_pages, tiny};
    use super::*;
    use crate::engine::kvcache::{
        AcquireOutcome, CacheManager, KVCacheLayout, KVCachePool, RadixCacheManager,
    };
    #[test]
    fn declared_recurrent_bytes_match_tensors_and_full_checkpoint_config() {
        for kind in [Kind::Float, Kind::BFloat16] {
            let mut model = tiny();
            model.kind = kind;
            let state = model.new_state();
            let actual: usize = state
                .layers
                .iter()
                .map(|s| {
                    s.conv.numel() * s.conv.kind().elt_size_in_bytes()
                        + s.recurrent.numel() * s.recurrent.kind().elt_size_in_bytes()
                })
                .sum();
            assert_eq!(model.recurrent_state_bytes().unwrap(), Some(actual));
        }
        let raw = serde_json::from_str(include_str!("../../../tests/fixtures/qwen3_8_config.json"))
            .unwrap();
        let config = Config::parse(&raw).unwrap();
        let model = Factory { config }
            .create(Kind::BFloat16, Device::Cpu, "pt")
            .unwrap();
        assert_eq!(model.recurrent_state_bytes().unwrap(), Some(153_944_064));
    }

    #[test]
    fn disabled_checkpoint_pool_runs_prefill_decode_without_history_storage() {
        let mut model = tiny();
        model.state_pool = Rc::new(RefCell::new(RecurrentStatePool::new(1, 0, || {
            model.new_state()
        })));
        let slot = model.state_pool.borrow_mut().alloc_request().unwrap();
        let _ = run(&model, &[1, 2, 3, 4, 5], 0, 1, false, slot);
        let actual = run(&model, &[6], 5, 1, true, slot);
        assert_eq!(model.state_pool.borrow().storage_capacity(), 1);
        assert_eq!(
            model.state_pool.borrow().staged_position(slot).unwrap(),
            None
        );
        assert!(
            model
                .state_pool
                .borrow_mut()
                .snapshot_states_mut(slot)
                .is_err()
        );
        let mut reference = tiny();
        reference.reserve_request_states(1, 4).unwrap();
        let other = reference.state_pool.borrow_mut().alloc_request().unwrap();
        let expected = run(&reference, &[1, 2, 3, 4, 5, 6], 0, 1, false, other);
        assert!((actual - expected).abs().max().double_value(&[]) < 2e-5);
        model.state_pool.borrow_mut().free_request(slot).unwrap();
    }

    #[test]
    fn aligned_chunk_ends_are_published_without_replaying_live_tokens() {
        let mut model = tiny();
        model.reserve_request_states(2, 4).unwrap();
        let kv = Rc::new(RefCell::new(
            KVCachePool::new(
                KVCacheLayout::new(1, 32, 2, 1, 4).unwrap(),
                Kind::Float,
                Device::Cpu,
            )
            .unwrap(),
        ));
        model
            .bind_state_cache(kv.borrow().model_cache().unwrap())
            .unwrap();
        let mut cache = RadixCacheManager::new(kv, 2)
            .unwrap()
            .with_recurrent_state(Some(model.state_pool.clone()));
        let ids = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let AcquireOutcome::Ready(mut active) = cache.acquire(&ids, 12, None).unwrap() else {
            panic!()
        };
        let slot = active.state_slot.unwrap();
        for start in [0, 4] {
            let end = start + 4;
            let pages: Vec<_> = active.page_ids.iter().map(|&p| p as i32).collect();
            let _ = run_pages(&model, &ids[start..end], start as i64, &pages, false, slot);
            assert_eq!(
                model.state_pool.borrow().staged_position(slot).unwrap(),
                Some(end)
            );
            assert_eq!(
                model
                    .state_pool
                    .borrow_mut()
                    .current_mut(slot)
                    .unwrap()
                    .position,
                end as i64
            );
            cache.publish(&mut active, &ids[..end]).unwrap();
            assert_eq!(active.published_pages, end / 2);
        }
        let pages: Vec<_> = active.page_ids.iter().map(|&p| p as i32).collect();
        let actual = run_pages(&model, &ids[8..], 8, &pages, false, slot);
        assert_eq!(
            model.state_pool.borrow().staged_position(slot).unwrap(),
            None
        );
        cache.publish(&mut active, &ids).unwrap();
        cache.release(&mut active);
        let AcquireOutcome::Ready(mut exact) = cache.acquire(&ids[..8], 12, None).unwrap() else {
            panic!()
        };
        assert_eq!(exact.cached_len, 4); // N−1 排除末尾 8，只能回退到较早的检查点 4。
        cache.release(&mut exact);
        let AcquireOutcome::Ready(mut first_chunk) = cache.acquire(&ids[..4], 12, None).unwrap()
        else {
            panic!()
        };
        assert_eq!(first_chunk.cached_len, 0); // 末尾有状态，也不能绕过 N−1 匹配规则。
        cache.release(&mut first_chunk);
        let AcquireOutcome::Ready(mut extended) = cache.acquire(&ids, 12, None).unwrap() else {
            panic!()
        };
        assert_eq!(extended.cached_len, 8); // 更长 prompt 可使用之前 chunk 末尾的状态。
        let pages: Vec<_> = extended.page_ids.iter().map(|&p| p as i32).collect();
        let restored = run_pages(
            &model,
            &ids[8..],
            8,
            &pages,
            false,
            extended.state_slot.unwrap(),
        );
        let mut reference = tiny();
        reference.reserve_request_states(1, 4).unwrap();
        let other = reference.state_pool.borrow_mut().alloc_request().unwrap();
        let expected = run(&reference, &ids, 0, 1, false, other);
        assert!((actual - &expected).abs().max().double_value(&[]) < 2e-5);
        assert!((restored - expected).abs().max().double_value(&[]) < 2e-5);
        cache.release(&mut extended);
        cache.evict(32);
        assert_eq!(model.state_pool.borrow().usage(), (0, 0));
    }

    #[test]
    fn longer_kv_branch_is_captured_across_chunks_and_reused_with_correct_logits() {
        let mut model = tiny();
        model.reserve_request_states(2, 4).unwrap();
        let kv = Rc::new(RefCell::new(
            KVCachePool::new(
                KVCacheLayout::new(1, 32, 2, 1, 4).unwrap(),
                Kind::Float,
                Device::Cpu,
            )
            .unwrap(),
        ));
        model
            .bind_state_cache(kv.borrow().model_cache().unwrap())
            .unwrap();
        let mut cache = RadixCacheManager::new(kv, 2)
            .unwrap()
            .with_recurrent_state(Some(model.state_pool.clone()));
        let ids = [1, 2, 3, 4, 5, 6, 7, 8];
        let AcquireOutcome::Ready(mut original) = cache.acquire(&ids, 10, None).unwrap() else {
            panic!()
        };
        let pages: Vec<_> = original.page_ids.iter().map(|&p| p as i32).collect();
        let _ = run_pages(&model, &ids, 0, &pages, false, original.state_slot.unwrap());
        cache.publish(&mut original, &ids).unwrap();
        cache.release(&mut original);
        let branch = [1, 2, 3, 4, 9, 10, 11, 12];
        let AcquireOutcome::Ready(mut h) = cache.acquire(&branch, 10, None).unwrap() else {
            panic!()
        };
        let slot = h.state_slot.unwrap();
        assert_eq!(h.cached_len, 0);
        assert_eq!(
            model.state_pool.borrow().branching_seqlen(slot).unwrap(),
            Some(4)
        );
        let pages: Vec<_> = h.page_ids.iter().map(|&p| p as i32).collect();
        let _ = run_pages(&model, &branch[..3], 0, &pages, false, slot);
        cache.publish(&mut h, &branch[..3]).unwrap();
        assert_eq!(
            model.state_pool.borrow().branching_seqlen(slot).unwrap(),
            Some(4)
        );
        let pages: Vec<_> = h.page_ids.iter().map(|&p| p as i32).collect();
        let actual = run_pages(&model, &branch[3..], 3, &pages, false, slot);
        assert_eq!(
            model.state_pool.borrow().staged_position(slot).unwrap(),
            Some(4)
        );
        assert_eq!(
            model.state_pool.borrow().branching_seqlen(slot).unwrap(),
            None
        );
        cache.publish(&mut h, &branch).unwrap();
        assert_eq!(h.published_pages, 2); // 优先补分支 4，不能误发布到默认位置 8。
        cache.release(&mut h);
        let AcquireOutcome::Ready(mut warm) = cache.acquire(&branch, 10, None).unwrap() else {
            panic!()
        };
        assert_eq!(warm.cached_len, 4);
        let pages: Vec<_> = warm.page_ids.iter().map(|&p| p as i32).collect();
        let restored = run_pages(
            &model,
            &branch[4..],
            4,
            &pages,
            false,
            warm.state_slot.unwrap(),
        );
        let mut reference = tiny();
        reference.reserve_request_states(1, 4).unwrap();
        let slot = reference.state_pool.borrow_mut().alloc_request().unwrap();
        let expected = run(&reference, &branch, 0, 1, false, slot);
        assert!((actual - &expected).abs().max().double_value(&[]) < 2e-5);
        assert!((restored - expected).abs().max().double_value(&[]) < 2e-5);
        cache.release(&mut warm);
        cache.evict(32);
        assert_eq!(model.state_pool.borrow().usage(), (0, 0));
    }

    #[test]
    fn branching_at_chunk_end_is_captured_and_hint_is_cleared() {
        let mut model = tiny();
        model.reserve_request_states(1, 4).unwrap();
        let slot = model.state_pool.borrow_mut().alloc_request().unwrap();
        model
            .state_pool
            .borrow_mut()
            .set_branching_seqlen(slot, Some(4))
            .unwrap();
        let _ = run(&model, &[1, 2, 3, 4], 0, 1, false, slot);
        assert_eq!(
            model.state_pool.borrow().staged_position(slot).unwrap(),
            Some(4)
        );
        assert_eq!(
            model.state_pool.borrow().branching_seqlen(slot).unwrap(),
            None
        );
        model.state_pool.borrow_mut().free_request(slot).unwrap();
        let slot = model.state_pool.borrow_mut().alloc_request().unwrap();
        assert_eq!(
            model.state_pool.borrow().branching_seqlen(slot).unwrap(),
            None
        );
    }

    #[test]
    fn restored_hybrid_prefix_and_two_branches_match_cold_full_forward() {
        let mut model = tiny();
        model.reserve_request_states(3, 4).unwrap();
        let kv = Rc::new(RefCell::new(
            KVCachePool::new(
                KVCacheLayout::new(1, 32, 2, 1, 4).unwrap(),
                Kind::Float,
                Device::Cpu,
            )
            .unwrap(),
        ));
        model
            .bind_state_cache(kv.borrow().model_cache().unwrap())
            .unwrap();
        let mut cache = RadixCacheManager::new(kv, 2)
            .unwrap()
            .with_recurrent_state(Some(model.state_pool.clone()));
        let ids = [1, 2, 3, 4, 5];
        let AcquireOutcome::Ready(mut cold) = cache.acquire(&ids, 8, None).unwrap() else {
            panic!()
        };
        let pages: Vec<_> = cold.page_ids.iter().map(|&p| p as i32).collect();
        let expected = run_pages(&model, &ids, 0, &pages, false, cold.state_slot.unwrap());
        cache.publish(&mut cold, &ids).unwrap();
        cache.release(&mut cold);
        let AcquireOutcome::Ready(mut a) = cache.acquire(&ids, 8, None).unwrap() else {
            panic!()
        };
        let branch = [1, 2, 3, 4, 7, 8];
        let AcquireOutcome::Ready(mut b) = cache.acquire(&branch, 8, None).unwrap() else {
            panic!()
        };
        assert_eq!(a.cached_len, 4);
        assert_eq!(b.cached_len, 4);
        assert_ne!(a.state_slot, b.state_slot);
        for (handle, tail) in [(&mut a, &ids[4..]), (&mut b, &branch[4..])] {
            let pages: Vec<_> = handle.page_ids.iter().map(|&p| p as i32).collect();
            let actual = run_pages(&model, tail, 4, &pages, false, handle.state_slot.unwrap());
            let expected = if tail == &ids[4..] {
                expected.shallow_clone()
            } else {
                let mut reference = tiny();
                reference.reserve_request_states(1, 4).unwrap();
                let slot = reference.state_pool.borrow_mut().alloc_request().unwrap();
                run(&reference, &branch, 0, 1, false, slot)
            };
            assert!((expected - actual).abs().max().double_value(&[]) < 2e-5);
        }
        cache.publish(&mut a, &ids).unwrap();
        cache.publish(&mut b, &branch).unwrap();
        cache.release(&mut a);
        cache.release(&mut b);
        cache.evict(32);
        assert_eq!(model.state_pool.borrow().usage(), (0, 0));
    }
}

#[cfg(test)]
mod scheduler_state_tests {
    use super::tests::tiny;
    use super::*;
    use crate::{
        engine::{Engine, SamplingParams, ServerArgs},
        scheduler::Scheduler,
    };
    #[test]
    fn scheduler_chunks_warm_hits_cancel_and_return_request_slots() {
        let model = tiny();
        let path = std::env::temp_dir().join(format!(
            "qwen-state-scheduler-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("config.json"), "{}").unwrap();
        let weights: Vec<_> = model
            .weights
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&weights, path.join("model.safetensors")).unwrap();
        let mut args = ServerArgs::new(&path);
        args.device = "cpu".into();
        args.attention_backend = "pt".into();
        args.max_running_req = 2;
        args.max_seq_len = 32;
        args.max_total_tokens = Some(64);
        args.page_size = 2;
        args.chunked_prefill_size = 4;
        args.chunked_prefill_size_explicit = true;
        args.disable_cpu_gpu_overlap = true;
        args.cuda_graph_bs = Some(0);
        args.prefill_cuda_graph_max_tokens = 0;
        let factory = Factory {
            config: model.config.clone(),
        };
        let automatic =
            Engine::load_for_serving(args.clone(), model.config.runtime(Kind::Float), 0, &factory)
                .unwrap();
        assert_eq!(
            automatic
                .recurrent_state_pool()
                .unwrap()
                .borrow()
                .storage_capacity(),
            64
        );
        drop(automatic);
        args.recurrent_checkpoint_slots = Some(20);
        let engine =
            Engine::load_for_serving(args, model.config.runtime(Kind::Float), 0, &factory).unwrap();
        assert_eq!(
            engine
                .recurrent_state_pool()
                .unwrap()
                .borrow()
                .checkpoint_capacity(),
            20
        );
        assert_eq!(
            engine
                .recurrent_state_pool()
                .unwrap()
                .borrow()
                .storage_capacity(),
            24
        );
        let pool = engine.recurrent_state_pool().unwrap();
        let mut scheduler = Scheduler::new(engine, Default::default()).unwrap();
        let ids = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let params = SamplingParams {
            max_tokens: 3,
            ..Default::default()
        };
        let mut outputs = Vec::new();
        for _ in 0..2 {
            let uid = scheduler.add_request(ids.clone(), params).unwrap();
            let mut tokens = Vec::new();
            for _ in 0..20 {
                for token in scheduler.step().unwrap() {
                    assert_eq!(token.uid, uid);
                    assert!(!matches!(
                        token.finish_reason,
                        Some(crate::scheduler::FinishReason::Error)
                    ));
                    tokens.push(token.token_id);
                }
                if scheduler.is_idle() {
                    break;
                }
            }
            assert!(scheduler.is_idle());
            assert_eq!(pool.borrow().usage().0, 0);
            outputs.push(tokens);
        }
        assert_eq!(outputs[0], outputs[1]);
        assert!(pool.borrow().usage().1 > 0);
        let uid = scheduler
            .add_request(vec![10, 11, 12, 13, 14, 15, 1, 2, 3], params)
            .unwrap();
        scheduler.step().unwrap();
        assert_eq!(pool.borrow().usage().0, 1);
        assert!(scheduler.abort_request(uid));
        for _ in 0..5 {
            let _ = scheduler.step().unwrap();
            if scheduler.is_idle() {
                break;
            }
        }
        assert_eq!(pool.borrow().usage().0, 0);
        std::fs::remove_dir_all(path).unwrap();
    }
    fn exercise_hybrid_overlap(device: Device) {
        let model = tiny();
        let path = std::env::temp_dir().join(format!(
            "qwen-overlap-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("config.json"), "{}").unwrap();
        let weights: Vec<_> = model
            .weights
            .iter()
            .map(|(name, t)| (name.as_str(), t))
            .collect();
        Tensor::write_safetensors(&weights, path.join("model.safetensors")).unwrap();
        let mut histories = Vec::new();
        for overlap in [false, true] {
            let mut args = ServerArgs::new(&path);
            args.device = if device.is_cuda() { "cuda" } else { "cpu" }.into();
            args.attention_backend = "pt".into();
            args.max_running_req = 3;
            args.max_seq_len = 32;
            args.max_total_tokens = Some(96);
            args.page_size = 2;
            args.chunked_prefill_size = 4;
            args.chunked_prefill_size_explicit = true;
            args.recurrent_checkpoint_slots = Some(2);
            args.disable_cpu_gpu_overlap = !overlap;
            args.cuda_graph_bs = Some(0);
            args.prefill_cuda_graph_max_tokens = 0;
            let factory = Factory {
                config: model.config.clone(),
            };
            let engine =
                Engine::load_for_serving(args, model.config.runtime(Kind::Float), 0, &factory)
                    .unwrap();
            let pool = engine.recurrent_state_pool().unwrap();
            let mut scheduler = Scheduler::new(engine, Default::default()).unwrap();
            let step = |scheduler: &mut Scheduler| {
                if overlap {
                    scheduler.step_overlap_for_test().unwrap()
                } else {
                    scheduler.step().unwrap()
                }
            };
            let params = |max_tokens| SamplingParams {
                max_tokens,
                ignore_eos: true,
                ..Default::default()
            };
            let a = scheduler
                .add_request(vec![1, 2, 3, 4, 5], params(8))
                .unwrap();
            let mut seen = std::collections::BTreeMap::<_, Vec<_>>::new();
            // Admit a second prompt only after decode starts. In the pipeline
            // this creates a real mixed prefill + relayed decode invocation.
            let mut added = false;
            for _ in 0..80 {
                for token in step(&mut scheduler) {
                    assert_ne!(
                        token.finish_reason,
                        Some(crate::scheduler::FinishReason::Error)
                    );
                    seen.entry(token.uid).or_default().push(token.token_id);
                }
                if !added && seen.contains_key(&a) {
                    scheduler
                        .add_request(vec![6, 7, 8, 9, 10, 11, 12, 13, 14], params(4))
                        .unwrap();
                    added = true;
                }
                if scheduler.is_idle() {
                    break;
                }
            }
            assert!(scheduler.is_idle());
            assert_eq!(seen.get(&a).unwrap().len(), 8);
            assert_eq!(seen.values().map(Vec::len).sum::<usize>(), 12);
            assert_eq!(pool.borrow().usage().0, 0);
            // Warm prefix restore after a small history pool has seen pressure.
            let warm = scheduler
                .add_request(vec![1, 2, 3, 4, 5], params(8))
                .unwrap();
            let mut warm_tokens = Vec::new();
            for _ in 0..80 {
                for token in step(&mut scheduler) {
                    assert_eq!(token.uid, warm);
                    assert_ne!(
                        token.finish_reason,
                        Some(crate::scheduler::FinishReason::Error)
                    );
                    warm_tokens.push(token.token_id);
                }
                if scheduler.is_idle() {
                    break;
                }
            }
            assert!(scheduler.is_idle());
            assert_eq!(warm_tokens, seen[&a]);
            // Cancel while the next decode is queued, then immediately reuse
            // the request capacity. Old state stays protected by its fence.
            let cancelled = scheduler
                .add_request(vec![9, 8, 7, 6, 5], params(10))
                .unwrap();
            let mut started = false;
            for _ in 0..80 {
                if step(&mut scheduler).iter().any(|t| t.uid == cancelled) {
                    started = true;
                    break;
                }
            }
            assert!(started);
            assert!(scheduler.abort_request(cancelled));
            let replacement = scheduler
                .add_request(vec![15, 14, 13, 12, 11], params(3))
                .unwrap();
            let mut replacement_tokens = Vec::new();
            let mut aborts = 0;
            for _ in 0..80 {
                for token in step(&mut scheduler) {
                    assert_ne!(
                        token.finish_reason,
                        Some(crate::scheduler::FinishReason::Error)
                    );
                    if token.uid == cancelled {
                        assert_eq!(
                            token.finish_reason,
                            Some(crate::scheduler::FinishReason::Abort)
                        );
                        aborts += 1;
                    } else {
                        assert_eq!(token.uid, replacement);
                        replacement_tokens.push(token.token_id);
                    }
                }
                if scheduler.is_idle() {
                    break;
                }
            }
            assert!(scheduler.is_idle());
            assert_eq!(aborts, 0); // Explicit cancellation is acknowledged by abort_request.
            assert_eq!(replacement_tokens.len(), 3);
            assert_eq!(pool.borrow().usage().0, 0);
            histories.push((seen, replacement_tokens));
        }
        assert_eq!(histories[0], histories[1]);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn hybrid_overlap_matches_serial_mixed_chunks_restore_and_cancel() {
        exercise_hybrid_overlap(Device::Cpu);
    }

    #[cfg(has_cuda_graph)]
    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn cuda_hybrid_overlap_matches_serial_mixed_chunks_restore_and_cancel() {
        exercise_hybrid_overlap(Device::Cuda(0));
    }
}
