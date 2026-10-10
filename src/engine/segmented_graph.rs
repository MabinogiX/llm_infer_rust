//! Model-independent startup capture and replay of GPU segments with eager seams.
use super::{
    ForwardBatch, ForwardOutput, GraphLimits, GraphPadding, ModelRunnerError, NativeCudaGraph,
};
use std::collections::{BTreeMap, HashSet};
use tch::{Device, Kind, Tensor};
type Result<T> = std::result::Result<T, ModelRunnerError>;

/// Model-specific static buffers. Segment inputs and outputs use model-defined
/// tensor roles; all inputs have bucket-sized leading token axes.
pub struct PrefillGraphInputs {
    pub ids: Tensor,
    pub positions: Tensor,
    pub segments: Vec<Vec<Tensor>>,
}

/// A model describes computation only; the runner owns native graphs and buckets.
pub trait PrefillGraphProgram {
    fn create_inputs(&self, tokens: usize) -> Result<PrefillGraphInputs>;
    fn run_segment(&self, index: usize, inputs: &PrefillGraphInputs) -> Result<Vec<Tensor>>;
    /// Prepare backend plans outside graphs, once per live invocation.
    fn prepare_replay<'a>(
        &'a self,
        batch: ForwardBatch<'a>,
    ) -> Result<Box<dyn PrefillGraphReplay + 'a>>;
}

pub trait PrefillGraphReplay {
    /// Run the eager seam after segment `index`; return the next segment's inputs.
    fn run_eager(&self, index: usize, outputs: &[Tensor]) -> Result<Vec<Tensor>>;
    /// Produce independently owned live-request logits, outside captured segments.
    fn finish(&self, outputs: &[Tensor]) -> Result<ForwardOutput>;
}

struct Segment {
    native: NativeCudaGraph,
    outputs: Vec<Tensor>,
}
struct PrefillGraph {
    segments: Vec<Segment>,
    inputs: PrefillGraphInputs,
}
impl Drop for PrefillGraph {
    fn drop(&mut self) {
        if let Device::Cuda(device) = self.inputs.ids.device() {
            tch::Cuda::synchronize(device as i64);
        }
    }
}

pub(crate) struct SegmentedGraphRunner {
    graphs: BTreeMap<usize, PrefillGraph>,
    failed: HashSet<usize>,
    limit: usize,
    admission: GraphLimits,
}
impl SegmentedGraphRunner {
    pub(crate) fn capture(
        program: &dyn PrefillGraphProgram,
        limit: usize,
        admission: GraphLimits,
    ) -> Result<Self> {
        if admission.padding != GraphPadding::InertTokenRows {
            return Err(error("segmented prefill requires inert token padding"));
        }
        let limit = limit.min(admission.max_tokens);
        let mut runner = Self {
            graphs: BTreeMap::new(),
            failed: HashSet::new(),
            limit,
            admission,
        };
        let sizes = capture_sizes(limit);
        tracing::info!(
            max_tokens = limit,
            "prefill segmented CUDA Graph configured for startup capture"
        );
        for &size in sizes.iter().rev() {
            match Self::capture_one(program, size) {
                Ok(graph) => {
                    tracing::info!(
                        tokens = size,
                        segments = graph.segments.len(),
                        "prefill segmented CUDA Graph capture complete"
                    );
                    runner.graphs.insert(size, graph);
                }
                Err(err) => {
                    runner.failed.insert(size);
                    tracing::warn!(tokens = size, %err, "prefill CUDA Graph capture failed; bucket uses eager");
                }
            }
        }
        Ok(runner)
    }
    #[cfg(all(test, has_cuda_graph))]
    pub(crate) fn captured_sizes(&self) -> Vec<usize> {
        self.graphs.keys().copied().collect()
    }
    #[cfg(all(test, has_cuda_graph))]
    pub(crate) fn failed_sizes(&self) -> Vec<usize> {
        self.failed.iter().copied().collect()
    }
    #[cfg(all(test, has_cuda_graph))]
    pub(crate) fn remove_bucket(&mut self, size: usize) {
        self.graphs.remove(&size);
    }

    fn capture_one(program: &dyn PrefillGraphProgram, tokens: usize) -> Result<PrefillGraph> {
        let inputs = program.create_inputs(tokens)?;
        let Device::Cuda(device) = inputs.ids.device() else {
            return Err(error("segment inputs require CUDA"));
        };
        if inputs.segments.is_empty()
            || inputs.ids.size() != [tokens as i64]
            || inputs.positions.size() != [tokens as i64]
            || inputs.ids.kind() != Kind::Int64
            || inputs.positions.kind() != Kind::Int64
            || inputs.positions.device() != inputs.ids.device()
        {
            return Err(error("invalid static segment input layout"));
        }
        for input in inputs.segments.iter().flatten() {
            if input.size().first() != Some(&(tokens as i64))
                || input.device() != inputs.ids.device()
            {
                return Err(error(
                    "segment inputs must have bucket-sized token axes on the same device",
                ));
            }
        }
        let mut segments = Vec::new();
        for index in 0..inputs.segments.len() {
            let (native, outputs) =
                NativeCudaGraph::capture(device, || program.run_segment(index, &inputs))?;
            if outputs.is_empty() {
                return Err(error("segment must return outputs"));
            }
            segments.push(Segment { native, outputs });
        }
        Ok(PrefillGraph { segments, inputs })
    }
    /// Replay a batch already validated by ModelRunner, or request eager fallback.
    pub(crate) fn replay(
        &self,
        program: &dyn PrefillGraphProgram,
        batch: ForwardBatch<'_>,
    ) -> Result<Option<ForwardOutput>> {
        let tokens = batch.tokens();
        let rows = batch.logits_indices().map_or(0, Tensor::numel);
        if rows == 0
            || rows > self.admission.max_batch_size
            || batch
                .attention
                .and_then(|m| m.max_seqlen)
                .is_some_and(|n| n > self.admission.max_context_len)
        {
            return Ok(None);
        }
        let Some(size) = bucket(tokens, self.limit) else {
            tracing::debug!(tokens, "prefill exceeds captured token limit; using eager");
            return Ok(None);
        };
        let Some(graph) = self.graphs.get(&size) else {
            tracing::debug!(
                tokens,
                bucket = size,
                "prefill bucket was not captured; using eager"
            );
            return Ok(None);
        };
        let replay = program.prepare_replay(batch)?;
        let _ = graph.inputs.ids.shallow_clone().fill_(1);
        let _ = graph.inputs.positions.shallow_clone().zero_();
        graph
            .inputs
            .ids
            .narrow(0, 0, tokens as i64)
            .copy_(&batch.input_ids.view([-1]));
        graph
            .inputs
            .positions
            .narrow(0, 0, tokens as i64)
            .copy_(&batch.positions.view([-1]));
        graph.segments[0].native.replay()?;
        for index in 0..graph.segments.len() - 1 {
            let values = replay.run_eager(index, &graph.segments[index].outputs)?;
            copy_segment_inputs(&graph.inputs.segments[index + 1], &values)?;
            graph.segments[index + 1].native.replay()?;
        }
        replay
            .finish(&graph.segments.last().unwrap().outputs)
            .map(Some)
    }
}

fn copy_segment_inputs(targets: &[Tensor], sources: &[Tensor]) -> Result<()> {
    if targets.len() != sources.len() {
        return Err(error("eager seam returned the wrong input count"));
    }
    for (target, source) in targets.iter().zip(sources) {
        let a = target.size();
        let b = source.size();
        if a.is_empty()
            || b.is_empty()
            || a.len() != b.len()
            || a[1..] != b[1..]
            || b[0] > a[0]
            || source.kind() != target.kind()
            || source.device() != target.device()
        {
            return Err(error("eager seam returned incompatible segment inputs"));
        }
    }
    for (target, source) in targets.iter().zip(sources) {
        if source.size()[0] < target.size()[0] {
            let _ = target.shallow_clone().zero_();
        }
        target.narrow(0, 0, source.size()[0]).copy_(source);
    }
    Ok(())
}
fn error(message: &str) -> ModelRunnerError {
    ModelRunnerError::Model(message.into())
}

pub(crate) fn bucket(tokens: usize, limit: usize) -> Option<usize> {
    if tokens == 0 || tokens > limit {
        return None;
    }
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
pub(crate) fn capture_sizes(limit: usize) -> Vec<usize> {
    if limit == 0 {
        return vec![];
    }
    let mut sizes: Vec<usize> = [1, 2, 4, 8, 16, 32, 64, 128]
        .into_iter()
        .filter(|&n| n <= limit)
        .collect();
    sizes.extend((192..=limit.min(512)).step_by(64));
    sizes.extend((640..=limit.min(1024)).step_by(128));
    sizes.extend((1280..=limit).step_by(256));
    if sizes.last().copied() != Some(limit) {
        sizes.push(limit);
    }
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn startup_catalog_covers_all_admitted_request_shapes() {
        for limit in [0, 1, 3, 64, 150, 512, 777, 1000, 2048] {
            let expected: std::collections::BTreeSet<_> =
                (1..=limit).filter_map(|n| bucket(n, limit)).collect();
            assert_eq!(
                capture_sizes(limit),
                expected.into_iter().collect::<Vec<_>>()
            );
        }
        assert_eq!(bucket(17, 2048), Some(32));
        assert_eq!(bucket(265, 2048), Some(320));
        assert_eq!(bucket(950, 1000), Some(1000));
        assert_eq!(bucket(1001, 1000), None);
        assert_eq!(bucket(1, 0), None);
    }
    #[test]
    fn validates_all_segment_inputs_before_copying_and_zeroes_padding() {
        let target = Tensor::ones([4, 2], (Kind::Float, Device::Cpu));
        let other = Tensor::ones([4, 3], (Kind::Float, Device::Cpu));
        let source = Tensor::full([2, 2], 7.0, (Kind::Float, Device::Cpu));
        assert!(
            copy_segment_inputs(
                &[target.shallow_clone(), other],
                &[source.shallow_clone(), source.shallow_clone()]
            )
            .is_err()
        );
        assert_eq!(target.sum(Kind::Float).double_value(&[]), 8.0);
        copy_segment_inputs(&[target.shallow_clone()], &[source]).unwrap();
        assert_eq!(
            target.narrow(0, 0, 2).sum(Kind::Float).double_value(&[]),
            28.0
        );
        assert_eq!(
            target.narrow(0, 2, 2).sum(Kind::Float).double_value(&[]),
            0.0
        );
    }
}

#[cfg(all(test, has_cuda_graph))]
mod cuda_tests {
    use super::*;
    use crate::engine::{
        Batch, GraphCapabilities, GraphSupport, ModelExecutor, ModelRunner, ModelWeights,
        ServerArgs,
        kvcache::{KVCacheLayout, KVCachePool},
        load_hf_safetensors,
    };
    use std::{cell::RefCell, rc::Rc};

    struct ArithmeticProgram {
        weight: Tensor,
    }
    impl ModelExecutor for ArithmeticProgram {
        fn graph_capabilities(&self) -> GraphCapabilities {
            GraphCapabilities {
                segmented_prefill: GraphSupport::Supported(GraphLimits {
                    max_batch_size: 4,
                    max_tokens: 17,
                    max_context_len: 64,
                    padding: GraphPadding::InertTokenRows,
                }),
                ..Default::default()
            }
        }
        fn prefill_graph_program(&self) -> Option<&dyn PrefillGraphProgram> {
            Some(self)
        }
        fn forward(&self, batch: &ForwardBatch<'_>) -> Result<ForwardOutput> {
            let x = (batch.input_ids.view([-1]).to_kind(Kind::Float) * &self.weight
                + batch.positions.view([-1]).to_kind(Kind::Float)
                + 1.)
                * 6.;
            Ok(ForwardOutput::new(
                x.view([-1, 1])
                    .index_select(0, batch.logits_indices().unwrap()),
            ))
        }
        fn load_weights(&mut self, weights: ModelWeights) -> Result<usize> {
            self.weight = weights
                .into_tensors()
                .remove(0)
                .1
                .to_device(Device::Cuda(0));
            Ok(1)
        }
    }
    impl PrefillGraphProgram for ArithmeticProgram {
        fn create_inputs(&self, tokens: usize) -> Result<PrefillGraphInputs> {
            // Simulate a rejected bucket: other shapes must stay usable.
            if tokens == 8 {
                return Err(error("test bucket unavailable"));
            }
            let options = (Kind::Float, Device::Cuda(0));
            Ok(PrefillGraphInputs {
                ids: Tensor::ones([tokens as i64], (Kind::Int64, options.1)),
                positions: Tensor::zeros([tokens as i64], (Kind::Int64, options.1)),
                segments: vec![
                    vec![],
                    vec![Tensor::zeros([tokens as i64, 1], options)],
                    vec![
                        Tensor::zeros([tokens as i64, 2], options),
                        Tensor::zeros([tokens as i64, 1], options),
                    ],
                ],
            })
        }
        fn run_segment(&self, index: usize, inputs: &PrefillGraphInputs) -> Result<Vec<Tensor>> {
            Ok(vec![match index {
                0 => (inputs.ids.to_kind(Kind::Float) * &self.weight
                    + inputs.positions.to_kind(Kind::Float))
                .view([-1, 1]),
                1 => &inputs.segments[1][0] * 2.,
                _ => {
                    inputs.segments[2][0].sum_dim_intlist([1i64].as_slice(), true, Kind::Float)
                        + &inputs.segments[2][1]
                }
            }])
        }
        fn prepare_replay<'a>(
            &'a self,
            batch: ForwardBatch<'a>,
        ) -> Result<Box<dyn PrefillGraphReplay + 'a>> {
            Ok(Box::new(ArithmeticReplay { batch }))
        }
    }
    struct ArithmeticReplay<'a> {
        batch: ForwardBatch<'a>,
    }
    impl PrefillGraphReplay for ArithmeticReplay<'_> {
        fn run_eager(&self, index: usize, outputs: &[Tensor]) -> Result<Vec<Tensor>> {
            let live = outputs[0].narrow(0, 0, self.batch.tokens() as i64);
            Ok(if index == 0 {
                vec![live + 1.]
            } else {
                vec![live.repeat([1, 2]), outputs[0].shallow_clone()]
            })
        }
        fn finish(&self, outputs: &[Tensor]) -> Result<ForwardOutput> {
            Ok(ForwardOutput::new(
                outputs[0].index_select(0, self.batch.logits_indices().unwrap()),
            ))
        }
    }
    fn capture(runner: &mut ModelRunner) {
        let mut args = ServerArgs::new("unused");
        args.max_seq_len = 64;
        args.max_running_req = 4;
        args.cuda_graph_bs = Some(0);
        args.prefill_cuda_graph_max_tokens = 17;
        runner
            .capture_graphs(
                &args,
                Rc::new(RefCell::new(KVCachePool::without_tensor(
                    KVCacheLayout::new(1, 4, 2, 1, 1).unwrap(),
                ))),
            )
            .unwrap();
    }
    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn non_transformer_segments_handle_padding_failure_fallback_and_rebinding() {
        assert!(
            tch::Cuda::is_available(),
            "CUDA test requires an available GPU"
        );
        tch::no_grad(|| {
            let device = Device::Cuda(0);
            let mut runner = ModelRunner::new(
                Box::new(ArithmeticProgram {
                    weight: Tensor::ones([1], (Kind::Float, device)),
                }),
                device,
            );
            capture(&mut runner);
            assert_eq!(
                runner.prefill_graph_runner.as_ref().unwrap().failed_sizes(),
                vec![8]
            );
            runner
                .prefill_graph_runner
                .as_mut()
                .unwrap()
                .remove_bucket(4);
            let original_sizes = runner
                .prefill_graph_runner
                .as_ref()
                .unwrap()
                .captured_sizes();
            let mut retained: Option<(Tensor, Tensor)> = None;
            for tokens in [1, 3, 8, 17, 18, 3] {
                let batch = Batch::prefill(
                    Tensor::arange(tokens, (Kind::Int64, device)) + 2,
                    Tensor::arange(tokens, (Kind::Int64, device)),
                    None,
                    Tensor::from_slice(&[0i64, tokens - 1]).to_device(device),
                );
                let actual = runner.forward(&batch).unwrap().logits;
                let expected = runner
                    .run_model(&ForwardBatch::from_scheduler(&batch).unwrap())
                    .unwrap()
                    .logits;
                assert!((&actual - &expected).abs().max().double_value(&[]) < 1e-5);
                if let Some((old, snapshot)) = &retained {
                    assert!(Tensor::equal(old, snapshot));
                }
                retained = Some((actual.shallow_clone(), actual.copy()));
                assert_eq!(
                    runner
                        .prefill_graph_runner
                        .as_ref()
                        .unwrap()
                        .captured_sizes(),
                    original_sizes
                );
            }
            let path =
                std::env::temp_dir().join(format!("sglang-graph-weight-{}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Tensor::write_safetensors(
                &[("weight", Tensor::full([1], 2., (Kind::Float, Device::Cpu)))],
                path.join("model.safetensors"),
            )
            .unwrap();
            runner
                .load_weights(load_hf_safetensors(&path).unwrap())
                .unwrap();
            std::fs::remove_dir_all(path).unwrap();
            assert!(runner.prefill_graph_runner.is_none());
            capture(&mut runner);
            let batch = Batch::prefill(
                Tensor::from_slice(&[2i64]).to_device(device),
                Tensor::zeros([1], (Kind::Int64, device)),
                None,
                Tensor::zeros([1], (Kind::Int64, device)),
            );
            assert_eq!(
                runner.forward(&batch).unwrap().logits.double_value(&[0, 0]),
                30.
            );
            let pool = KVCachePool::new(
                KVCacheLayout::new(1, 4, 2, 1, 1).unwrap(),
                Kind::Float,
                device,
            )
            .unwrap();
            runner
                .bind_state_cache(pool.model_cache().unwrap())
                .unwrap();
            assert!(runner.prefill_graph_runner.is_none());
            capture(&mut runner);
            runner.set_kv_reserved_slot(0);
            assert!(runner.prefill_graph_runner.is_none());
            runner.clear_graphs();
            runner.clear_graphs();
        });
    }
}
