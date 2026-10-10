//! Qwen3 computation segments. Buckets, buffers and graph lifetime belong to Runner.
use super::*;
use crate::engine::{
    ForwardBatch, ForwardOutput, PrefillGraphInputs, PrefillGraphOutputLifetime,
    PrefillGraphProgram, PrefillGraphReplay,
};

impl PrefillGraphProgram for Qwen3ForCausalLM {
    fn create_shared_buffers(&self, max_tokens: usize) -> Result<Vec<Tensor>> {
        // Attention consumes QKV first, then writes this slot. The next graph
        // reads it before another layer or bucket can overwrite it. QKV and
        // residual use separately captured storage.
        Ok(vec![Tensor::zeros(
            [max_tokens as i64, self.layers[0].qkv_proj.widths()[0]],
            (self.kind, self.device),
        )])
    }
    fn create_inputs(&self, tokens: usize, shared: &[Tensor]) -> Result<PrefillGraphInputs> {
        let rows = tokens as i64;
        let mut segments = vec![vec![]];
        for layer in &self.layers {
            if shared.len() != 1
                || shared[0].size().len() != 2
                || shared[0].size()[0] < rows
                || shared[0].size()[1] != layer.qkv_proj.widths()[0]
                || shared[0].kind() != self.kind
                || shared[0].device() != self.device
                || !shared[0].is_contiguous()
            {
                return Err(ModelRunnerError::Model(
                    "invalid shared attention output buffer".into(),
                ));
            }
            segments.push(vec![shared[0].narrow(0, 0, rows)]);
        }
        Ok(PrefillGraphInputs {
            ids: Tensor::ones([rows], (Kind::Int64, self.device)),
            positions: Tensor::zeros([rows], (Kind::Int64, self.device)),
            segments,
        })
    }
    fn output_lifetime(&self, segment: usize, output: usize) -> PrefillGraphOutputLifetime {
        if segment < self.layers.len() && output == 3 {
            // The same residual allocation is read and updated by every later
            // graph. Keep its owning alias until all bucket segments are captured.
            PrefillGraphOutputLifetime::BucketCapture
        } else {
            // QKV is consumed by attention before launching the next segment;
            // final hidden is consumed by finish before another forward starts.
            PrefillGraphOutputLifetime::EagerSeam
        }
    }
    fn run_segment(
        &self,
        index: usize,
        inputs: &PrefillGraphInputs,
        previous: &[Tensor],
    ) -> Result<Vec<Tensor>> {
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
            layer.output_mlp(&segment[0], previous[3].shallow_clone(), eps, &mut profiler);
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
    fn run_eager(&self, index: usize, outputs: &[Tensor], inputs: &[Tensor]) -> Result<()> {
        let count = self.batch.tokens() as i64;
        let output = &inputs[0];
        let _ = self.model.layers[index].attend(
            &outputs[0].narrow(0, 0, count),
            &outputs[1].narrow(0, 0, count),
            &outputs[2].narrow(0, 0, count),
            &self.attention,
            self.batch.attention,
            &mut ModelProfiler::new(self.model.device),
            Some(&output.narrow(0, 0, count)),
        )?;
        // Only the attention output is padded; residual remains at its captured address.
        if count < output.size()[0] {
            let _ = output.narrow(0, count, output.size()[0] - count).zero_();
        }
        Ok(())
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
    struct FailLaterSegment<'a> {
        model: &'a Qwen3ForCausalLM,
        calls: std::cell::Cell<usize>,
    }
    impl PrefillGraphProgram for FailLaterSegment<'_> {
        fn create_shared_buffers(&self, max: usize) -> Result<Vec<Tensor>> {
            self.model.create_shared_buffers(max)
        }
        fn create_inputs(&self, tokens: usize, shared: &[Tensor]) -> Result<PrefillGraphInputs> {
            self.model.create_inputs(tokens, shared)
        }
        fn output_lifetime(&self, segment: usize, role: usize) -> PrefillGraphOutputLifetime {
            self.model.output_lifetime(segment, role)
        }
        fn run_segment(
            &self,
            index: usize,
            inputs: &PrefillGraphInputs,
            previous: &[Tensor],
        ) -> Result<Vec<Tensor>> {
            let result = self.model.run_segment(index, inputs, previous)?;
            if index == 1 {
                let calls = self.calls.get() + 1;
                self.calls.set(calls);
                if calls == 4 {
                    return Err(ModelRunnerError::Model(
                        "injected later-segment capture failure".into(),
                    ));
                }
            }
            Ok(result)
        }
        fn prepare_replay<'a>(
            &'a self,
            batch: ForwardBatch<'a>,
        ) -> Result<Box<dyn PrefillGraphReplay + 'a>> {
            self.model.prepare_replay(batch)
        }
    }
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
        let cache = runner.prefill_graph_runner.as_ref().unwrap();
        let buffers = cache.bridge_inputs();
        assert!(!buffers.is_empty());
        let address = buffers[0].data_ptr();
        for buffer in buffers {
            assert_eq!(
                buffer.data_ptr(),
                address,
                "layers and buckets must share one attention slot"
            );
            assert!(buffer.is_contiguous());
            assert_eq!(buffer.size()[1], 256);
        }
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
        assert_eq!(
            graph
                .prefill_graph_runner
                .as_ref()
                .unwrap()
                .bridge_input_counts(64),
            vec![0, 1, 1, 1],
            "only attention outputs need bridge buffers; residuals are graph outputs",
        );
        (eager, graph, ek, ev, gk, gv)
    }

    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn segmented_prefill_matches_eager_logits_and_kv_for_live_request_boundaries() {
        assert!(
            tch::Cuda::is_available(),
            "CUDA test requires an available GPU"
        );
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
                let mut retained: Option<(Tensor, Tensor)> = None;
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
                    (vec![3, 5], vec![0, 0]),
                    (vec![224, 225], vec![0, 0]),
                    // Deterministic irregular order after every bucket is live.
                    (vec![31, 2], vec![0, 0]),
                    (vec![192, 191], vec![0, 0]),
                    (vec![7, 2], vec![0, 0]),
                    (vec![128, 127], vec![0, 0]),
                    (vec![63, 64], vec![0, 0]),
                    (vec![224, 223], vec![0, 0]),
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
                    // Poison the entire shared slot, including rows beyond this
                    // bucket, to expose missing live writes and padding clears.
                    for buffer in graph.prefill_graph_runner.as_ref().unwrap().bridge_inputs() {
                        let _ = buffer.shallow_clone().fill_(f64::NAN);
                    }
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
                    if let Some((previous, snapshot)) = &retained {
                        assert!(
                            previous.equal(snapshot),
                            "next forward overwrote retained logits"
                        );
                    }
                    retained = Some((actual.shallow_clone(), actual.copy()));
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
                // Fail inside capture after the preceding segment's QKV owners
                // have been released. Partial bucket cleanup must synchronize
                // before dropping the retained residual, then allow other sizes.
                let context = crate::engine::NativeGraphCaptureContext::new(0).unwrap();
                let failing = FailLaterSegment {
                    model: &eager,
                    calls: std::cell::Cell::new(0),
                };
                let partial = crate::engine::segmented_graph::SegmentedGraphRunner::capture(
                    &failing,
                    64,
                    crate::engine::GraphLimits {
                        max_tokens: 64,
                        max_batch_size: 2,
                        max_context_len: 256,
                        padding: crate::engine::GraphPadding::InertTokenRows,
                    },
                    &context,
                )
                .unwrap();
                assert_eq!(partial.failed_sizes(), vec![64]);
                assert_eq!(
                    partial.captured_sizes(),
                    capture_sizes(64)
                        .into_iter()
                        .filter(|&n| n != 64)
                        .collect::<Vec<_>>()
                );
                context.check_execution_stream().unwrap();
                for count in [9, 17, 3, 9] {
                    let ids = Tensor::arange(count, (Kind::Int64, Device::Cuda(0))).remainder(32);
                    let positions = Tensor::arange(count, (Kind::Int64, Device::Cuda(0)));
                    let indices = Tensor::from_slice(&[count - 1]).to_device(Device::Cuda(0));
                    let expected = eager
                        .forward_impl(&ids, &positions, None, Some(&indices))
                        .unwrap();
                    let batch = ForwardBatch {
                        input_ids: &ids,
                        positions: &positions,
                        attention: None,
                        mode: crate::engine::ForwardMode::Prefill {
                            logits_indices: &indices,
                        },
                    };
                    batch.validate(Device::Cuda(0)).unwrap();
                    let actual = partial.replay(&failing, batch).unwrap().unwrap().logits;
                    let error = (actual.to_kind(Kind::Float) - expected.to_kind(Kind::Float))
                        .abs()
                        .max()
                        .double_value(&[]);
                    assert!(
                        error < 0.005,
                        "{kind:?}/{backend}: replay after partial failure error={error}"
                    );
                }
                drop((partial, context));
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
