//! Token-bucketed transformer segments separated by eager attention calls.
use std::collections::{BTreeMap, HashSet};

use super::*;
use crate::engine::NativeCudaGraph;

#[derive(Default)]
pub(super) struct PrefillGraphCache {
    limit: usize,
    graphs: BTreeMap<usize, PrefillGraph>,
    failed: HashSet<usize>,
}

impl PrefillGraphCache {
    pub(super) fn clear(&mut self) {
        self.graphs.clear();
        self.failed.clear();
    }

    pub(super) fn new(limit: usize) -> Self {
        Self {
            limit,
            ..Self::default()
        }
    }
}

struct Segment {
    native: NativeCudaGraph,
    // Inputs and outputs must outlive the native graph.
    attention_input: Option<Tensor>,
    residual_input: Option<Tensor>,
    outputs: Vec<Tensor>,
}

struct PrefillGraph {
    segments: Vec<Segment>,
    ids: Tensor,
    positions: Tensor,
}

impl Drop for PrefillGraph {
    fn drop(&mut self) {
        // A returned LM-head tensor may still be consuming segment outputs.
        // Finish queued work before releasing the graph's private allocation pools.
        if let Device::Cuda(device) = self.ids.device() {
            tch::Cuda::synchronize(device as i64);
        }
    }
}

fn bucket(tokens: usize, limit: usize) -> Option<usize> {
    if tokens == 0 || tokens > limit {
        return None;
    }
    // Small prompts are dominated by launch overhead. Larger buckets use
    // finer steps to avoid doubling GEMM work immediately past a power of two.
    let size = if tokens <= 128 {
        tokens.checked_next_power_of_two()?
    } else {
        let step = if tokens <= 512 {
            64
        } else if tokens <= 1024 {
            128
        } else {
            256
        };
        tokens.checked_add(step - 1)? / step * step
    };
    Some(size.min(limit))
}

/// Enumerate every shape reachable by bucket(), including a non-aligned limit.
fn capture_sizes(limit: usize) -> Vec<usize> {
    if limit == 0 {
        return vec![];
    }
    let mut sizes: Vec<usize> = [1, 2, 4, 8, 16, 32, 64, 128]
        .into_iter()
        .filter(|&size| size <= limit)
        .collect();
    sizes.extend((192..=limit.min(512)).step_by(64));
    sizes.extend((640..=limit.min(1024)).step_by(128));
    sizes.extend((1280..=limit).step_by(256));
    if sizes.last().copied() != Some(limit) {
        sizes.push(limit);
    }
    sizes
}

impl Qwen3ForCausalLM {
    pub(super) fn capture_prefill_buckets(&self) -> Result<()> {
        let mut cache = self.prefill_graphs.borrow_mut();
        let sizes = capture_sizes(cache.limit);
        if sizes.is_empty() {
            return Ok(());
        }
        tracing::info!(sizes = ?sizes, "capturing prefill segmented CUDA Graphs at startup");
        // Match SGLang's largest-to-smallest capture order. All successful
        // buckets remain resident; request processing never captures a graph.
        for &size in sizes.iter().rev() {
            if cache.graphs.contains_key(&size) || cache.failed.contains(&size) {
                continue;
            }
            tracing::info!(tokens = size, "capturing prefill segmented CUDA Graph");
            match self.capture_prefill_segments(size) {
                Ok(graph) => {
                    tracing::info!(
                        tokens = size,
                        segments = graph.segments.len(),
                        "prefill segmented CUDA Graph capture complete"
                    );
                    cache.graphs.insert(size, graph);
                }
                Err(error) => {
                    cache.failed.insert(size);
                    tracing::warn!(tokens = size, %error, "prefill CUDA Graph capture failed; bucket uses eager");
                }
            }
        }
        tracing::info!(sizes = ?cache.graphs.keys().collect::<Vec<_>>(), failed = ?cache.failed,
            "prefill CUDA Graph startup capture complete");
        Ok(())
    }

    pub(super) fn forward_prefill_graph(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        metadata: Option<&AttentionMetadata>,
        logits_indices: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        if !NativeCudaGraph::available()
            || !matches!(self.device, Device::Cuda(_))
            || !matches!(self.kind, Kind::BFloat16 | Kind::Half)
            || crate::logging::step_timing_enabled()
        {
            return Ok(None);
        }
        let tokens = input_ids.numel();
        let cache = self.prefill_graphs.borrow();
        let Some(size) = bucket(tokens, cache.limit) else {
            return Ok(None);
        };
        let Some(graph) = cache.graphs.get(&size) else {
            return Ok(None);
        };
        if positions.numel() != tokens {
            return Err(model_error(
                "input_ids and positions must have identical lengths",
            ));
        }
        // Plan only on real request boundaries, outside captured segments.
        let attention = self.attention.prepare(metadata, tokens)?;
        let count = tokens as i64;
        let _ = graph.ids.shallow_clone().fill_(1);
        let _ = graph.positions.shallow_clone().zero_();
        graph.ids.narrow(0, 0, count).copy_(&input_ids.view([-1]));
        graph
            .positions
            .narrow(0, 0, count)
            .copy_(&positions.view([-1]));
        graph.segments[0].native.replay()?;
        let mut profiler = ModelProfiler::new(self.device);
        for (index, layer) in self.layers.iter().enumerate() {
            let outputs = &graph.segments[index].outputs;
            let q = outputs[0].narrow(0, 0, count);
            let k = outputs[1].narrow(0, 0, count);
            let v = outputs[2].narrow(0, 0, count);
            let raw = layer.attend(&q, &k, &v, &attention, metadata, &mut profiler)?;
            let next = &graph.segments[index + 1];
            let mut slot = next.attention_input.as_ref().unwrap().shallow_clone();
            let _ = slot.zero_();
            slot.narrow(0, 0, count).copy_(&raw);
            next.residual_input
                .as_ref()
                .unwrap()
                .shallow_clone()
                .copy_(&outputs[3]);
            next.native.replay()?;
        }
        let hidden = graph.segments.last().unwrap().outputs[0].narrow(0, 0, count);
        // Output row count depends on the live requests, not the token bucket.
        Ok(Some(logits(&hidden, &self.lm_head, logits_indices)))
    }

    fn capture_prefill_segments(&self, tokens: usize) -> Result<PrefillGraph> {
        let Device::Cuda(device) = self.device else {
            unreachable!()
        };
        let rows = tokens as i64;
        let ids = Tensor::ones([rows], (Kind::Int64, self.device));
        let positions = Tensor::zeros([rows], (Kind::Int64, self.device));
        let eps = self.config.rms_norm_eps;
        let layer = &self.layers[0];
        let (native, outputs) = NativeCudaGraph::capture(device, || {
            let hidden = embedding(&ids, &self.embed_tokens);
            let (normalized, residual) = layer.input_norm(&hidden, None, eps);
            let (q, k, v) = layer.project_qkv(
                &normalized,
                &positions,
                eps,
                &self.rope,
                &mut ModelProfiler::new(self.device),
            )?;
            Ok(vec![q, k, v, residual])
        })?;
        let mut segments = vec![Segment {
            native,
            outputs,
            attention_input: None,
            residual_input: None,
        }];
        for (index, layer) in self.layers.iter().enumerate() {
            let attention_input =
                Tensor::zeros([rows, layer.qkv_proj.widths()[0]], (self.kind, self.device));
            let residual_input = Tensor::zeros(
                [rows, self.config.hidden_size as i64],
                (self.kind, self.device),
            );
            let (native, outputs) = NativeCudaGraph::capture(device, || {
                let mut profiler = ModelProfiler::new(self.device);
                let (hidden, residual) = layer.output_mlp(
                    &attention_input,
                    residual_input.shallow_clone(),
                    eps,
                    &mut profiler,
                );
                if let Some(next) = self.layers.get(index + 1) {
                    let (normalized, residual) = next.input_norm(&hidden, Some(residual), eps);
                    let (q, k, v) =
                        next.project_qkv(&normalized, &positions, eps, &self.rope, &mut profiler)?;
                    Ok(vec![q, k, v, residual])
                } else {
                    Ok(vec![add_rms_norm(hidden, residual, &self.norm, eps).0])
                }
            })?;
            segments.push(Segment {
                native,
                outputs,
                attention_input: Some(attention_input),
                residual_input: Some(residual_input),
            });
        }
        Ok(PrefillGraph {
            segments,
            ids,
            positions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn startup_catalog_covers_all_admitted_request_shapes() {
        for limit in [0, 1, 3, 64, 150, 512, 777, 1000, 2048] {
            let expected: std::collections::BTreeSet<_> = (1..=limit)
                .filter_map(|tokens| bucket(tokens, limit))
                .collect();
            assert_eq!(
                capture_sizes(limit),
                expected.into_iter().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn token_bucket_limits_and_padding() {
        assert_eq!(bucket(0, 2048), None);
        assert_eq!(bucket(1, 2048), Some(1));
        assert_eq!(bucket(17, 2048), Some(32));
        assert_eq!(bucket(265, 2048), Some(320));
        assert_eq!(bucket(516, 2048), Some(640));
        assert_eq!(bucket(513, 1000), Some(640));
        assert_eq!(bucket(950, 1000), Some(1000));
        assert_eq!(bucket(1001, 1000), None);
        assert_eq!(bucket(1, 0), None);
    }
}

#[cfg(all(test, has_flashinfer, has_cuda_graph))]
mod cuda_tests {
    use super::*;
    use tch::no_grad;

    fn pair(
        kind: Kind,
    ) -> (
        Qwen3ForCausalLM,
        Qwen3ForCausalLM,
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
            Qwen3ForCausalLM::new_with_attention_backend(config, kind, device, "flashinfer")
                .unwrap();
        let mut graph =
            Qwen3ForCausalLM::new_with_attention_backend(config, kind, device, "flashinfer")
                .unwrap();
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
        eager
            .bind_kv_cache(ek.shallow_clone(), ev.shallow_clone())
            .unwrap();
        graph
            .bind_kv_cache(gk.shallow_clone(), gv.shallow_clone())
            .unwrap();
        graph.configure_prefill_graph(64);
        graph.capture_prefill_graphs().unwrap();
        assert_eq!(
            graph
                .prefill_graphs
                .borrow()
                .graphs
                .keys()
                .copied()
                .collect::<Vec<_>>(),
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
            for kind in [Kind::BFloat16, Kind::Half] {
                let (eager, mut graph, ek, ev, gk, gv) = pair(kind);
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
                        graph.configure_prefill_graph(0);
                        assert!(graph.prefill_graphs.borrow().graphs.is_empty());
                        graph.configure_prefill_graph(64);
                        graph.capture_prefill_graphs().unwrap();
                    }
                    if iteration == 6 {
                        let cache = graph.prefill_graphs.borrow();
                        assert!(cache.graphs.contains_key(&32));
                        assert!(
                            !cache.graphs.contains_key(&128),
                            "over-limit prefill was captured"
                        );
                        drop(cache);
                        graph.configure_prefill_graph(512);
                        graph.capture_prefill_graphs().unwrap();
                    }
                    if iteration == 2 {
                        graph.prefill_graphs.borrow_mut().graphs.remove(&16);
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
                        req_to_token: None,
                        cache_seqlens: None,
                        max_seqlen: Some(*lengths.iter().max().unwrap() as usize),
                    };
                    let ids = (Tensor::arange(tokens, (Kind::Int64, device)) + 3).remainder(32);
                    let pos = Tensor::from_slice(&positions).to_device(device);
                    let indices =
                        Tensor::from_slice(&[lengths[0] - 1, tokens - 1]).to_device(device);
                    let expected = eager
                        .forward(&ids, &pos, Some(&metadata), Some(&indices))
                        .unwrap();
                    let actual = graph
                        .forward(&ids, &pos, Some(&metadata), Some(&indices))
                        .unwrap();
                    assert_eq!(actual.size(), [2, 32]);
                    if iteration == 2 {
                        assert!(
                            !graph.prefill_graphs.borrow().graphs.contains_key(&16),
                            "request lazily captured a missing bucket"
                        );
                    }
                    let error = (actual.to_kind(Kind::Float) - expected.to_kind(Kind::Float))
                        .abs()
                        .max()
                        .double_value(&[]);
                    assert!(error < 0.005, "{kind:?}: logits error={error}");
                    for (a, b) in [(&ek, &gk), (&ev, &gv)] {
                        let error = (a.to_kind(Kind::Float) - b.to_kind(Kind::Float))
                            .abs()
                            .max()
                            .double_value(&[]);
                        assert!(error < 0.03125, "{kind:?}: KV error={error}");
                    }
                }
                let cache = graph.prefill_graphs.borrow();
                assert_eq!(
                    cache.graphs.keys().copied().collect::<Vec<_>>(),
                    capture_sizes(512)
                );
                assert!(cache.failed.is_empty());
                drop(cache);
                graph.configure_prefill_graph(0);
                assert!(graph.prefill_graphs.borrow().graphs.is_empty());
                graph.configure_prefill_graph(64);
                assert_eq!(graph.prefill_graphs.borrow().limit, 64);
            }
        });
    }
}
