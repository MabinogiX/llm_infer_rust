//! Qwen3 computation segments. Buckets, buffers and graph lifetime belong to Runner.
use super::*;
use crate::engine::{
    ForwardBatch, ForwardOutput, PrefillGraphInputs, PrefillGraphProgram, PrefillGraphReplay,
};

impl PrefillGraphProgram for Qwen3ForCausalLM {
    fn create_inputs(&self, tokens: usize) -> Result<PrefillGraphInputs> {
        let rows = tokens as i64;
        let mut segments = vec![vec![]];
        for layer in &self.layers {
            segments.push(vec![
                Tensor::zeros([rows, layer.qkv_proj.widths()[0]], (self.kind, self.device)),
                Tensor::zeros(
                    [rows, self.config.hidden_size as i64],
                    (self.kind, self.device),
                ),
            ]);
        }
        Ok(PrefillGraphInputs {
            ids: Tensor::ones([rows], (Kind::Int64, self.device)),
            positions: Tensor::zeros([rows], (Kind::Int64, self.device)),
            segments,
        })
    }
    fn run_segment(&self, index: usize, inputs: &PrefillGraphInputs) -> Result<Vec<Tensor>> {
        let eps = self.config.rms_norm_eps;
        let mut profiler = ModelProfiler::new(self.device);
        if index == 0 {
            let hidden = embedding(&inputs.ids, &self.embed_tokens);
            let (normalized, residual) = self.layers[0].input_norm(&hidden, None, eps);
            let (q, k, v) = self.layers[0].project_qkv(
                &normalized,
                &inputs.positions,
                eps,
                &self.rope,
                &mut profiler,
            )?;
            return Ok(vec![q, k, v, residual]);
        }
        let layer = &self.layers[index - 1];
        let segment = &inputs.segments[index];
        let (hidden, residual) =
            layer.output_mlp(&segment[0], segment[1].shallow_clone(), eps, &mut profiler);
        if let Some(next) = self.layers.get(index) {
            let (normalized, residual) = next.input_norm(&hidden, Some(residual), eps);
            let (q, k, v) = next.project_qkv(
                &normalized,
                &inputs.positions,
                eps,
                &self.rope,
                &mut profiler,
            )?;
            Ok(vec![q, k, v, residual])
        } else {
            Ok(vec![add_rms_norm(hidden, residual, &self.norm, eps).0])
        }
    }
    fn prepare_replay<'a>(
        &'a self,
        batch: ForwardBatch<'a>,
    ) -> Result<Box<dyn PrefillGraphReplay + 'a>> {
        // Real request boundaries and all planning stay outside GPU graph segments.
        let attention = self.attention.prepare(batch.attention, batch.tokens())?;
        Ok(Box::new(QwenPrefillReplay {
            model: self,
            attention,
            batch,
        }))
    }
}
struct QwenPrefillReplay<'a> {
    model: &'a Qwen3ForCausalLM,
    attention: AttentionBatch<'a>,
    batch: ForwardBatch<'a>,
}
impl PrefillGraphReplay for QwenPrefillReplay<'_> {
    fn run_eager(&self, index: usize, outputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let count = self.batch.tokens() as i64;
        let raw = self.model.layers[index].attend(
            &outputs[0].narrow(0, 0, count),
            &outputs[1].narrow(0, 0, count),
            &outputs[2].narrow(0, 0, count),
            &self.attention,
            self.batch.attention,
            &mut ModelProfiler::new(self.model.device),
        )?;
        Ok(vec![raw, outputs[3].shallow_clone()])
    }
    fn finish(&self, outputs: &[Tensor]) -> Result<ForwardOutput> {
        let hidden = outputs[0].narrow(0, 0, self.batch.tokens() as i64);
        Ok(ForwardOutput::new(logits(
            &hidden,
            &self.model.lm_head,
            self.batch.logits_indices(),
        )))
    }
}

#[cfg(all(test, has_flashinfer, has_cuda_graph))]
mod cuda_tests {
    use super::*;
    use crate::engine::{Batch, ModelRunner, ServerArgs, segmented_graph::capture_sizes};
    use tch::no_grad;
    fn capture(runner: &mut ModelRunner, limit: usize) {
        use crate::engine::kvcache::{KVCacheLayout, KVCachePool};
        use std::rc::Rc;
        let mut args = ServerArgs::new("unused");
        args.max_seq_len = 256;
        args.max_running_req = 2;
        args.cuda_graph_bs = Some(0);
        args.prefill_cuda_graph_max_tokens = limit;
        let pool = Rc::new(RefCell::new(KVCachePool::without_tensor(
            KVCacheLayout::new(3, 32, 16, 1, 128).unwrap(),
        )));
        runner.capture_graphs(&args, pool).unwrap();
    }

    fn pair(
        kind: Kind,
        backend: &str,
    ) -> (
        Qwen3ForCausalLM,
        ModelRunner,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ) {
        let config = Qwen3Config {
            hidden_size: 64,
            num_layers: 3,
            num_attention_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 96,
            vocab_size: 32,
            head_dim: 128,
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
            ..Default::default()
        };
        let device = Device::Cuda(0);
        let mut eager =
            Qwen3ForCausalLM::new_with_attention_backend(config, kind, device, backend).unwrap();
        let mut graph =
            Qwen3ForCausalLM::new_with_attention_backend(config, kind, device, backend).unwrap();
        for (a, b) in [
            (&mut eager.embed_tokens, &mut graph.embed_tokens),
            (&mut eager.norm, &mut graph.norm),
            (&mut eager.lm_head, &mut graph.lm_head),
        ] {
            a.copy_(&(Tensor::randn(a.size(), (kind, device)) * 0.02));
            b.copy_(a);
        }
        for (a, b) in eager.layers.iter_mut().zip(graph.layers.iter_mut()) {
            for (x, y, norm) in [
                (&mut a.input_layernorm, &mut b.input_layernorm, true),
                (&mut a.qkv_proj.weight, &mut b.qkv_proj.weight, false),
                (&mut a.o_proj, &mut b.o_proj, false),
                (&mut a.q_norm, &mut b.q_norm, true),
                (&mut a.k_norm, &mut b.k_norm, true),
                (
                    &mut a.post_attention_layernorm,
                    &mut b.post_attention_layernorm,
                    true,
                ),
                (&mut a.mlp.gate_up_proj, &mut b.mlp.gate_up_proj, false),
                (&mut a.mlp.down_proj, &mut b.mlp.down_proj, false),
            ] {
                if norm {
                    let _ = x.fill_(1.0);
                } else {
                    x.copy_(&(Tensor::randn(x.size(), (kind, device)) * 0.02));
                }
                y.copy_(x);
            }
        }
        let ek = Tensor::zeros([3, 32, 16, 1, 128], (kind, device));
        let ev = Tensor::zeros_like(&ek);
        let gk = ek.copy();
        let gv = ev.copy();
        for (model, k, v) in [(&mut eager, &ek, &ev), (&mut graph, &gk, &gv)] {
            model
                .bind_state_cache(crate::engine::kvcache::ModelKvCache {
                    page_size: k.size()[2] as usize,
                    layers: (0..k.size()[0])
                        .map(|i| crate::engine::kvcache::LayerKvCache {
                            k: k.get(i),
                            v: v.get(i),
                        })
                        .collect(),
                })
                .unwrap();
        }
        let mut graph = ModelRunner::new(Box::new(graph), device);
        capture(&mut graph, 64);
        assert_eq!(
            graph
                .prefill_graph_runner
                .as_ref()
                .unwrap()
                .captured_sizes(),
            capture_sizes(64)
        );
        (eager, graph, ek, ev, gk, gv)
    }

    #[test]
    fn segmented_prefill_matches_eager_logits_and_kv_for_live_request_boundaries() {
        if !tch::Cuda::is_available() {
            return;
        }
        no_grad(|| {
            for (kind, backend) in [
                (Kind::BFloat16, "flashinfer"),
                (Kind::Half, "flashinfer"),
                (Kind::BFloat16, "fa"),
                (Kind::Half, "fa"),
                (Kind::BFloat16, "pt"),
                (Kind::Half, "pt"),
            ] {
                let (eager, mut graph, ek, ev, gk, gv) = pair(kind, backend);
                for (iteration, (lengths, prefixes)) in [
                    (vec![17, 6], vec![0, 0]),
                    (vec![3, 5], vec![17, 6]),
                    (vec![8, 7], vec![0, 0]),
                    (vec![1, 2], vec![8, 0]),
                    (vec![40, 25], vec![0, 0]),
                    (vec![17, 6], vec![0, 0]),
                    (vec![64, 65], vec![0, 0]),
                    (vec![96, 97], vec![0, 0]),
                    (vec![128, 129], vec![0, 0]),
                    (vec![160, 161], vec![0, 0]),
                    (vec![192, 193], vec![0, 0]),
                    (vec![224, 225], vec![0, 0]),
                    (vec![64, 65], vec![0, 0]),
                ]
                .into_iter()
                .enumerate()
                {
                    if iteration == 4 {
                        graph.clear_graphs();
                        assert!(graph.prefill_graph_runner.is_none());
                        capture(&mut graph, 64);
                    }
                    if iteration == 6 {
                        let cache = graph.prefill_graph_runner.as_ref().unwrap();
                        assert!(cache.captured_sizes().contains(&32));
                        assert!(
                            !cache.captured_sizes().contains(&128),
                            "over-limit prefill was captured"
                        );
                        capture(&mut graph, 512);
                    }
                    if iteration == 2 {
                        graph
                            .prefill_graph_runner
                            .as_mut()
                            .unwrap()
                            .remove_bucket(16);
                    }
                    let device = Device::Cuda(0);
                    let tokens: i64 = lengths.iter().sum();
                    let mut cumulative = vec![0i32];
                    let mut positions = vec![];
                    let mut writes = vec![];
                    for (row, (&len, &prefix)) in lengths.iter().zip(&prefixes).enumerate() {
                        cumulative.push(cumulative.last().unwrap() + len as i32);
                        for pos in prefix..prefix + len {
                            positions.push(pos);
                            writes.push((row as i64 * 256 + pos) as i32);
                        }
                    }
                    let metadata = AttentionMetadata {
                        forward_mode: BatchPhase::Prefill,
                        write_loc: Some(Tensor::from_slice(&writes).to_device(device)),
                        cu_seqlens_q: Some(Tensor::from_slice(&cumulative).to_device(device)),
                        prefix_lens: Some(
                            Tensor::from_slice(&prefixes)
                                .to_kind(Kind::Int)
                                .to_device(device),
                        ),
                        block_table: Some(Tensor::arange(32, (Kind::Int, device)).view([2, 16])),
                        req_to_token: Some(Tensor::arange(512, (Kind::Int, device)).view([2, 256])),
                        cache_seqlens: None,
                        max_seqlen: Some(
                            lengths
                                .iter()
                                .zip(&prefixes)
                                .map(|(len, prefix)| len + prefix)
                                .max()
                                .unwrap() as usize,
                        ),
                    };
                    let ids = (Tensor::arange(tokens, (Kind::Int64, device)) + 3).remainder(32);
                    let pos = Tensor::from_slice(&positions).to_device(device);
                    let indices =
                        Tensor::from_slice(&[lengths[0] - 1, tokens - 1]).to_device(device);
                    let expected = eager
                        .forward_impl(&ids, &pos, Some(&metadata), Some(&indices))
                        .unwrap();
                    let actual = graph
                        .forward(&Batch::prefill(
                            ids.shallow_clone(),
                            pos.shallow_clone(),
                            Some(metadata),
                            indices,
                        ))
                        .unwrap()
                        .logits;
                    assert_eq!(actual.size(), [2, 32]);
                    if iteration == 2 {
                        assert!(
                            !graph
                                .prefill_graph_runner
                                .as_ref()
                                .unwrap()
                                .captured_sizes()
                                .contains(&16),
                            "request lazily captured a missing bucket"
                        );
                    }
                    let error = (actual.to_kind(Kind::Float) - expected.to_kind(Kind::Float))
                        .abs()
                        .max()
                        .double_value(&[]);
                    assert!(error < 0.005, "{kind:?}/{backend}: logits error={error}");
                    for (a, b) in [(&ek, &gk), (&ev, &gv)] {
                        let error = (a.to_kind(Kind::Float) - b.to_kind(Kind::Float))
                            .abs()
                            .max()
                            .double_value(&[]);
                        assert!(error < 0.03125, "{kind:?}/{backend}: KV error={error}");
                    }
                }
                let cache = graph.prefill_graph_runner.as_ref().unwrap();
                assert_eq!(cache.captured_sizes(), capture_sizes(512));
                assert!(cache.failed_sizes().is_empty());
                graph.clear_graphs();
                assert!(graph.prefill_graph_runner.is_none());
                capture(&mut graph, 64);
                assert_eq!(
                    graph
                        .prefill_graph_runner
                        .as_ref()
                        .unwrap()
                        .captured_sizes(),
                    capture_sizes(64)
                );
            }
        });
    }
}
