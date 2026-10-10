//! Exercise the public model/runtime seam with heterogeneous full-attention layers.
use sglang_rust::{
    engine::{
        AttentionMetadata, Batch, BatchPhase, Engine, ModelExecutor, ModelFactory,
        ModelRunnerError, ModelWeights, RuntimeModelConfig, ServerArgs,
        kvcache::{ModelCacheSpec, ModelKvCache, PagedKvSpec},
    },
    models::attention::{Attention, AttentionSpec, BaseAttention},
};
use std::{
    cell::RefCell,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
use tch::{Device, Kind, Tensor};

type Result<T> = std::result::Result<T, ModelRunnerError>;

fn specification() -> ModelCacheSpec {
    ModelCacheSpec::new(vec![
        PagedKvSpec {
            num_kv_heads: 2,
            head_dim: 4,
        },
        PagedKvSpec {
            num_kv_heads: 1,
            head_dim: 8,
        },
        PagedKvSpec {
            num_kv_heads: 2,
            head_dim: 4,
        },
    ])
    .unwrap()
}

struct HeterogeneousFactory;
impl ModelFactory for HeterogeneousFactory {
    fn cache_spec(&self, _: RuntimeModelConfig) -> Result<ModelCacheSpec> {
        Ok(specification())
    }
    fn validate_runtime(&self, _: RuntimeModelConfig, _: Kind, _: Device, _: &str) -> Result<()> {
        Ok(())
    }
    fn create(&self, kind: Kind, device: Device, backend: &str) -> Result<Box<dyn ModelExecutor>> {
        let attention = specification()
            .layers()
            .iter()
            .map(|s| {
                Attention::new(
                    backend,
                    AttentionSpec {
                        num_heads: 2,
                        num_kv_heads: s.num_kv_heads as i64,
                        head_dim: s.head_dim as i64,
                        kind,
                        device,
                    },
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Box::new(HeterogeneousModel {
            attention,
            caches: (0..3)
                .map(|_| RefCell::new(BaseAttention::default()))
                .collect(),
            kind,
        }))
    }
}

struct HeterogeneousModel {
    attention: Vec<Attention>,
    caches: Vec<RefCell<BaseAttention>>,
    kind: Kind,
}
impl ModelExecutor for HeterogeneousModel {
    fn load_weights(&mut self, weights: ModelWeights) -> Result<usize> {
        // This seam test uses a marker checkpoint, rather than a second production model.
        assert_eq!(weights.len(), 1);
        Ok(1)
    }
    fn bind_state_cache(&mut self, cache: ModelKvCache) -> Result<()> {
        assert_eq!(cache.layers.len(), 3);
        for ((attention, base), view) in self
            .attention
            .iter_mut()
            .zip(&self.caches)
            .zip(cache.layers)
        {
            attention.bind_cache_layout(cache.page_size as i64);
            base.borrow_mut().bind_kv_cache(view.k, view.v)?;
        }
        Ok(())
    }
    fn set_kv_reserved_slot(&mut self, slot: i64) {
        for cache in &self.caches {
            cache.borrow_mut().set_reserved_write_slot(slot);
        }
    }
    fn forward(
        &self,
        ids: &Tensor,
        positions: &Tensor,
        metadata: Option<&AttentionMetadata>,
        indices: Option<&Tensor>,
    ) -> Result<Tensor> {
        let tokens = ids.numel() as i64;
        let scalar = (ids + positions).to_kind(self.kind).view([tokens, 1, 1]) * 0.1;
        let mut outputs = Vec::new();
        for (i, geometry) in specification().layers().iter().enumerate() {
            let q = scalar.repeat([1, 2, geometry.head_dim as i64]);
            let kv = scalar.repeat([1, geometry.num_kv_heads as i64, geometry.head_dim as i64]);
            let v = &kv + i as f64;
            let mut cache = self.caches[i].borrow_mut();
            cache.write_kv(
                &kv,
                &v,
                metadata.and_then(|m| m.write_loc.as_ref()),
                metadata.map_or(BatchPhase::Prefill, |m| m.forward_mode),
            )?;
            outputs.push(
                self.attention[i]
                    .prepare(metadata, tokens as usize)?
                    .forward(&q, &kv, &v, &cache)?
                    .view([tokens, -1]),
            );
        }
        let result = Tensor::cat(&outputs, 1);
        Ok(match indices {
            Some(indices) => result.index_select(0, indices),
            None => result,
        })
    }
}

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn compare_cached_and_full_forward(device: Device) {
    let path = std::env::temp_dir().join(format!(
        "sglang-layer-state-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    let fixture = Fixture(path);
    fs::write(fixture.0.join("config.json"), "{}").unwrap();
    Tensor::write_safetensors(
        &[("marker", Tensor::zeros([1], (Kind::Float, Device::Cpu)))],
        fixture.0.join("model.safetensors"),
    )
    .unwrap();
    let mut args = ServerArgs::new(&fixture.0);
    args.device = if matches!(device, Device::Cuda(_)) {
        "cuda"
    } else {
        "cpu"
    }
    .into();
    args.dtype = "float32".into();
    args.attention_backend = "pt".into();
    args.max_seq_len = 8;
    args.max_running_req = 2;
    args.page_size = 2;
    args.cuda_graph_bs = Some(0);
    args.prefill_cuda_graph_max_tokens = 0;
    let runtime = RuntimeModelConfig {
        num_layers: 3,
        num_kv_heads: 2,
        head_dim: 4,
        max_position_embeddings: 8,
        ..Default::default()
    };
    let mut engine = Engine::load_for_serving(args, runtime, 0, &HeterogeneousFactory).unwrap();
    assert_eq!(engine.kv_cache_pool().unwrap().spec(), &specification());
    let mut handle = engine.kv_cache_pool_mut().unwrap().alloc(2).unwrap();
    let base = handle.page_ids[0] * 2;
    let tail = handle.page_ids[1] * 2;
    let table = Tensor::from_slice(&[
        base as i32,
        (base + 1) as i32,
        tail as i32,
        (tail + 1) as i32,
    ])
    .view([1, 4])
    .to_device(device);
    let ids = Tensor::from_slice(&[1i64, 2, 3, 4]).to_device(device);
    let positions = Tensor::arange(4, (Kind::Int64, device));
    let reference = HeterogeneousFactory
        .create(Kind::Float, device, "pt")
        .unwrap();
    let full = reference.forward(&ids, &positions, None, None).unwrap();
    let prefill = Batch::prefill(
        ids.narrow(0, 0, 2),
        positions.narrow(0, 0, 2),
        Some(AttentionMetadata {
            forward_mode: BatchPhase::Prefill,
            write_loc: Some(table.get(0).narrow(0, 0, 2)),
            cu_seqlens_q: Some(Tensor::from_slice(&[0i32, 2]).to_device(device)),
            prefix_lens: Some(Tensor::zeros([1], (Kind::Int, device))),
            block_table: None,
            req_to_token: Some(table.shallow_clone()),
            cache_seqlens: None,
            max_seqlen: Some(2),
        }),
        Tensor::from_slice(&[1i64]).to_device(device),
    );
    let first = engine.forward(&prefill).unwrap();
    assert!((first - full.narrow(0, 1, 1)).abs().max().double_value(&[]) < 1e-5);
    for (phase, index) in [(BatchPhase::Decode, 2i64), (BatchPhase::Prefill, 3i64)] {
        let metadata = AttentionMetadata {
            forward_mode: phase,
            write_loc: Some(table.get(0).narrow(0, index, 1)),
            cu_seqlens_q: (phase == BatchPhase::Prefill)
                .then(|| Tensor::from_slice(&[0i32, 1]).to_device(device)),
            prefix_lens: (phase == BatchPhase::Prefill)
                .then(|| Tensor::from_slice(&[index as i32]).to_device(device)),
            block_table: None,
            req_to_token: Some(table.shallow_clone()),
            cache_seqlens: (phase == BatchPhase::Decode)
                .then(|| Tensor::from_slice(&[(index + 1) as i32]).to_device(device)),
            max_seqlen: Some(index as usize + 1),
        };
        let batch = if phase == BatchPhase::Decode {
            Batch::decode(
                ids.narrow(0, index, 1),
                positions.narrow(0, index, 1),
                Some(metadata),
            )
        } else {
            Batch::prefill(
                ids.narrow(0, index, 1),
                positions.narrow(0, index, 1),
                Some(metadata),
                Tensor::zeros([1], (Kind::Int64, device)),
            )
        };
        let actual = engine.forward(&batch).unwrap();
        assert!(
            (actual - full.narrow(0, index, 1))
                .abs()
                .max()
                .double_value(&[])
                < 1e-5
        );
    }
    engine.kv_cache_pool_mut().unwrap().free(&mut handle);
    engine.cleanup();
    engine.cleanup();
}

#[test]
fn heterogeneous_factory_serving_prefill_decode_and_cached_extend_match_full_forward() {
    compare_cached_and_full_forward(Device::Cpu);
}

#[test]
fn cuda_heterogeneous_factory_serving_prefill_decode_and_cached_extend_match_full_forward() {
    if tch::Cuda::is_available() {
        compare_cached_and_full_forward(Device::Cuda(0));
    }
}
