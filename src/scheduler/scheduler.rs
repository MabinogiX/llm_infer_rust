//! Scheduler request lifecycle and configuration handling.

use std::{
    collections::{BTreeSet, VecDeque},
    fs,
};

use serde_json::Value;

use crate::engine::kvcache::{CacheManager, NaiveCacheManager, RadixCacheManager};
use crate::engine::{
    Batch, BatchContext, BatchPhase, Engine, EngineError, SamplingParams, ServerArgs,
};

use super::{
    DecodeManager, FinishReason, OutputToken, PrefillManager, Request, RequestId, Result,
    SchedulerError, SequenceStatus,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    Radix,
    Naive,
}

pub struct Scheduler {
    engine: Engine,
    args: ServerArgs,
    cache_strategy: CacheStrategy,
    eos_token_ids: BTreeSet<i64>,
    next_uid: RequestId,
    prefill: PrefillManager,
    decode: DecodeManager,
    aborted: VecDeque<OutputToken>,
    last_step_error: Option<EngineError>,
}

impl Scheduler {
    pub fn new(engine: Engine) -> Result<Self> {
        Self::with_cache_strategy(engine, CacheStrategy::Radix)
    }

    pub fn with_cache_strategy(engine: Engine, cache_strategy: CacheStrategy) -> Result<Self> {
        let args = engine.server_args().clone();
        let eos_token_ids = load_eos_token_ids(&args.model_path)?;
        let pool = engine.shared_kv_cache_pool()?;
        let cache: Box<dyn CacheManager> = match cache_strategy {
            CacheStrategy::Radix => Box::new(RadixCacheManager::new(pool.clone(), args.page_size)?),
            CacheStrategy::Naive => Box::new(NaiveCacheManager::new(pool.clone())),
        };
        let batch_context = BatchContext::new(
            args.max_running_req,
            args.max_seq_len,
            args.page_size,
            engine.device(),
        )?;
        let prefill = PrefillManager::new(&args, pool, cache, batch_context);
        let decode = DecodeManager::new(&args, engine.device())?;
        Ok(Self {
            engine,
            args,
            cache_strategy,
            eos_token_ids,
            next_uid: 0,
            prefill,
            decode,
            aborted: VecDeque::new(),
            last_step_error: None,
        })
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    pub fn cache_strategy(&self) -> CacheStrategy {
        self.cache_strategy
    }

    pub fn eos_token_ids(&self) -> &BTreeSet<i64> {
        &self.eos_token_ids
    }

    pub fn pending_len(&self) -> usize {
        self.prefill.pending_len()
    }

    pub fn running_len(&self) -> usize {
        self.prefill.running_len()
    }

    pub fn last_step_error(&self) -> Option<&EngineError> {
        self.last_step_error.as_ref()
    }

    /// Assigns a UID and queues a request. Oversized prompts yield an abort
    /// result on the next `step` without entering the pending queue.
    pub fn add_request(
        &mut self,
        input_ids: Vec<i64>,
        sampling_params: SamplingParams,
    ) -> Result<RequestId> {
        let uid = self.next_uid;
        self.next_uid = uid
            .checked_add(1)
            .ok_or(SchedulerError::RequestIdExhausted)?;
        if input_ids.len() > self.args.max_seq_len {
            self.aborted
                .push_back(self.terminal_result(uid, FinishReason::Abort));
        } else {
            self.prefill.add_request(Request {
                uid,
                input_ids,
                sampling_params: sampling_params.normalized(),
                cached_len: 0,
                output_len: 0,
                cache_handle: None,
                status: SequenceStatus::Waiting,
            });
        }
        Ok(uid)
    }

    /// Removes a waiting or running request and releases its cache references.
    pub fn abort_request(&mut self, uid: RequestId) -> bool {
        self.prefill.abort(uid)
    }

    pub fn is_idle(&self) -> bool {
        self.prefill.pending_len() == 0
            && self.prefill.running_len() == 0
            && !self.prefill.has_aborted()
            && self.aborted.is_empty()
    }

    /// Runs prefill followed by decode for every request still running.
    pub fn step(&mut self) -> Result<Vec<OutputToken>> {
        self.last_step_error = None;
        let batch = self.prefill.schedule_prefill()?;
        let mut results: Vec<_> = self.aborted.drain(..).collect();
        if let Some(batch) = batch {
            self.run_phase(
                batch.request_ids,
                batch.model_batch,
                BatchPhase::Prefill,
                &mut results,
            )?;
        }
        results.extend(
            self.prefill
                .drain_aborted()
                .into_iter()
                .map(|uid| self.terminal_result(uid, FinishReason::Abort)),
        );
        if let Some(batch) = self
            .decode
            .schedule_decode(self.prefill.running_requests())?
        {
            self.run_phase(
                batch.request_ids,
                batch.model_batch,
                BatchPhase::Decode,
                &mut results,
            )?;
        }
        Ok(results)
    }

    fn run_phase(
        &mut self,
        request_ids: Vec<RequestId>,
        model_batch: Batch,
        phase: BatchPhase,
        results: &mut Vec<OutputToken>,
    ) -> Result<()> {
        let params = request_ids
            .iter()
            .map(|uid| {
                self.prefill
                    .running_request(*uid)
                    .expect("scheduled request is running")
                    .sampling_params
            })
            .collect::<Vec<_>>();
        let sampled = self
            .engine
            .forward(&model_batch)
            .and_then(|logits| self.engine.sample(&logits, &params));
        match sampled {
            Ok(tokens) => {
                let mut finished = Vec::new();
                for (uid, token_id) in request_ids.into_iter().zip(tokens) {
                    let request = self
                        .prefill
                        .running_request_mut(uid)
                        .expect("scheduled request is running");
                    request.append_token(token_id);
                    let reason = if self.eos_token_ids.contains(&token_id)
                        && !request.sampling_params.ignore_eos
                    {
                        Some(FinishReason::Stop)
                    } else if request.output_len >= request.sampling_params.max_tokens
                        || request.input_ids.len() >= self.args.max_seq_len
                    {
                        Some(FinishReason::Length)
                    } else {
                        None
                    };
                    if reason.is_some() {
                        request.status = SequenceStatus::Finished;
                        finished.push(uid);
                    }
                    results.push(OutputToken {
                        uid,
                        token_id,
                        finished: reason.is_some(),
                        finish_reason: reason,
                    });
                }
                self.prefill.remove_finished_batch(&finished)?;
            }
            Err(error) => {
                self.last_step_error = Some(error);
                match phase {
                    BatchPhase::Prefill => {
                        self.prefill.remove_failed_prefill_batch(&request_ids)?
                    }
                    BatchPhase::Decode => self.prefill.remove_finished_batch(&request_ids)?,
                }
                results.extend(
                    request_ids
                        .into_iter()
                        .map(|uid| self.terminal_result(uid, FinishReason::Error)),
                );
            }
        }
        Ok(())
    }

    fn terminal_result(&self, uid: RequestId, reason: FinishReason) -> OutputToken {
        OutputToken {
            uid,
            token_id: *self.eos_token_ids.first().expect("EOS set is never empty"),
            finished: true,
            finish_reason: Some(reason),
        }
    }
}

fn load_eos_token_ids(model_path: &std::path::Path) -> Result<BTreeSet<i64>> {
    for filename in [
        "generation_config.json",
        "tokenizer_config.json",
        "config.json",
    ] {
        let path = model_path.join(filename);
        let contents = match fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(SchedulerError::InvalidModelConfig {
                    path,
                    message: error.to_string(),
                });
            }
        };
        let config: Value = serde_json::from_str(&contents).map_err(|error| {
            SchedulerError::InvalidModelConfig {
                path: path.clone(),
                message: error.to_string(),
            }
        })?;
        if let Some(raw) = config.get("eos_token_id") {
            if !raw.is_null() {
                let ids = normalize_eos(raw);
                return Ok(if ids.is_empty() {
                    BTreeSet::from([0])
                } else {
                    ids
                });
            }
        }
        if filename == "config.json" {
            break;
        }
    }
    Ok(BTreeSet::from([0]))
}

fn normalize_eos(raw: &Value) -> BTreeSet<i64> {
    let raw = if let Some(object) = raw.as_object() {
        object
            .get("token_id")
            .or_else(|| object.get("id"))
            .unwrap_or(raw)
    } else {
        raw
    };
    match raw {
        Value::Number(value) => value.as_i64().into_iter().collect(),
        Value::Array(values) => values
            .iter()
            .filter_map(|value| {
                value
                    .as_i64()
                    .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
            })
            .collect(),
        _ => BTreeSet::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use tch::{Device, Tensor};

    use crate::engine::{
        AttentionMetadata, ModelArgs, ModelExecutor, ModelRunner, ModelRunnerError,
    };

    use super::*;

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn scheduler() -> (Scheduler, PathBuf) {
        scheduler_with_strategy_and_capacity(CacheStrategy::Radix, 1)
    }

    fn scheduler_with_strategy(strategy: CacheStrategy) -> (Scheduler, PathBuf) {
        scheduler_with_strategy_and_capacity(strategy, 1)
    }

    fn scheduler_with_strategy_and_capacity(
        strategy: CacheStrategy,
        max_running_req: usize,
    ) -> (Scheduler, PathBuf) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "sglang-rust-scheduler-{nonce}-{}",
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("config.json"), r#"{"eos_token_id": 2}"#).unwrap();
        fs::write(
            path.join("generation_config.json"),
            r#"{"eos_token_id": [7, 8]}"#,
        )
        .unwrap();
        let mut args = ServerArgs::new(&path);
        args.max_running_req = max_running_req;
        args.max_seq_len = 4;
        args.page_size = 2;
        let engine = Engine::new(
            args,
            ModelArgs {
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 1,
                max_position_embeddings: 4,
                ..Default::default()
            },
            0,
        )
        .unwrap();
        (
            Scheduler::with_cache_strategy(engine, strategy).unwrap(),
            path,
        )
    }

    #[test]
    fn oversized_prompt_emits_abort_and_uses_generation_config_eos() {
        let (mut scheduler, path) = scheduler();
        assert_eq!(scheduler.eos_token_ids(), &BTreeSet::from([7, 8]));
        let uid = scheduler
            .add_request(vec![1, 2, 3, 4, 5], SamplingParams::default())
            .unwrap();
        assert!(!scheduler.is_idle());
        assert_eq!(scheduler.pending_len(), 0);
        assert_eq!(
            scheduler.step().unwrap(),
            vec![OutputToken {
                uid,
                token_id: 7,
                finished: true,
                finish_reason: Some(FinishReason::Abort),
            }]
        );
        assert!(scheduler.is_idle());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn prefill_failure_rolls_back_and_keeps_other_abort_results() {
        let (mut scheduler, path) = scheduler();
        let uid = scheduler
            .add_request(vec![1], SamplingParams::default())
            .unwrap();
        let free_before = scheduler.engine().kv_cache_pool().unwrap().free_count();
        let rejected = scheduler
            .add_request(vec![1, 2, 3, 4, 5], SamplingParams::default())
            .unwrap();
        let results = scheduler.step().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].uid, rejected);
        assert_eq!(results[0].finish_reason, Some(FinishReason::Abort));
        assert_eq!(results[1].uid, uid);
        assert_eq!(results[1].finish_reason, Some(FinishReason::Error));
        assert!(matches!(
            scheduler.last_step_error(),
            Some(EngineError::ModelRunnerNotAttached)
        ));
        assert_eq!(
            scheduler.engine().kv_cache_pool().unwrap().free_count(),
            free_before
        );
        assert!(!scheduler.abort_request(uid));
        assert!(scheduler.is_idle());
        assert!(scheduler.step().unwrap().is_empty());
        fs::remove_dir_all(path).unwrap();
    }

    struct FixedTokenModel(i64);

    impl ModelExecutor for FixedTokenModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _positions: &Tensor,
            _attention_metadata: Option<&AttentionMetadata>,
            logits_indices: Option<&Tensor>,
        ) -> std::result::Result<Tensor, ModelRunnerError> {
            let rows = logits_indices
                .map(|indices| indices.size()[0])
                .unwrap_or_else(|| input_ids.size()[0]);
            let mut logits = vec![0f32; rows as usize * 9];
            for row in 0..rows as usize {
                logits[row * 9 + self.0 as usize] = 1.0;
            }
            Ok(Tensor::from_slice(&logits).view([rows, 9]))
        }
    }

    struct DecodeFailModel;

    impl ModelExecutor for DecodeFailModel {
        fn forward(
            &self,
            _input_ids: &Tensor,
            _positions: &Tensor,
            attention_metadata: Option<&AttentionMetadata>,
            logits_indices: Option<&Tensor>,
        ) -> std::result::Result<Tensor, ModelRunnerError> {
            if attention_metadata
                .is_some_and(|metadata| metadata.forward_mode == BatchPhase::Decode)
            {
                return Err(ModelRunnerError::Model("decode failed".to_owned()));
            }
            let rows = logits_indices
                .expect("prefill passes logits indices")
                .size()[0];
            let mut logits = vec![0f32; rows as usize * 9];
            for row in 0..rows as usize {
                logits[row * 9 + 3] = 1.0;
            }
            Ok(Tensor::from_slice(&logits).view([rows, 9]))
        }
    }

    struct FailSecondPrefillModel {
        prefill_calls: Cell<usize>,
    }

    impl ModelExecutor for FailSecondPrefillModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _positions: &Tensor,
            attention_metadata: Option<&AttentionMetadata>,
            logits_indices: Option<&Tensor>,
        ) -> std::result::Result<Tensor, ModelRunnerError> {
            let is_prefill = attention_metadata
                .is_some_and(|metadata| metadata.forward_mode == BatchPhase::Prefill);
            if is_prefill {
                let count = self.prefill_calls.get() + 1;
                self.prefill_calls.set(count);
                if count == 2 {
                    return Err(ModelRunnerError::Model("second prefill failed".to_owned()));
                }
            }
            let rows = logits_indices
                .map(|indices| indices.size()[0])
                .unwrap_or_else(|| input_ids.size()[0]);
            let mut logits = vec![0f32; rows as usize * 9];
            for row in 0..rows as usize {
                logits[row * 9 + 3] = 1.0;
            }
            Ok(Tensor::from_slice(&logits).view([rows, 9]))
        }
    }

    #[test]
    fn prefill_and_decode_run_in_the_same_step() {
        let (mut scheduler, path) = scheduler();
        scheduler
            .engine_mut()
            .attach_model_runner(ModelRunner::new(Box::new(FixedTokenModel(3)), Device::Cpu))
            .unwrap();
        let uid = scheduler
            .add_request(vec![1], SamplingParams::default())
            .unwrap();
        assert_eq!(
            scheduler.step().unwrap(),
            vec![
                OutputToken {
                    uid,
                    token_id: 3,
                    finished: false,
                    finish_reason: None
                },
                OutputToken {
                    uid,
                    token_id: 3,
                    finished: false,
                    finish_reason: None
                },
            ]
        );
        assert_eq!(scheduler.running_len(), 1);
        assert_eq!(
            scheduler.step().unwrap(),
            vec![OutputToken {
                uid,
                token_id: 3,
                finished: true,
                finish_reason: Some(FinishReason::Length)
            },]
        );
        assert!(scheduler.is_idle());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn decode_failure_preserves_prefill_output_and_finishes_request() {
        let (mut scheduler, path) = scheduler();
        scheduler
            .engine_mut()
            .attach_model_runner(ModelRunner::new(Box::new(DecodeFailModel), Device::Cpu))
            .unwrap();
        let uid = scheduler
            .add_request(vec![1], SamplingParams::default())
            .unwrap();
        assert_eq!(
            scheduler.step().unwrap(),
            vec![
                OutputToken {
                    uid,
                    token_id: 3,
                    finished: false,
                    finish_reason: None
                },
                OutputToken {
                    uid,
                    token_id: 7,
                    finished: true,
                    finish_reason: Some(FinishReason::Error)
                },
            ]
        );
        assert!(matches!(
            scheduler.last_step_error(),
            Some(EngineError::ModelRunner(_))
        ));
        assert!(scheduler.is_idle());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn prefill_failure_does_not_stop_existing_decode() {
        let (mut scheduler, path) = scheduler_with_strategy_and_capacity(CacheStrategy::Radix, 2);
        scheduler
            .engine_mut()
            .attach_model_runner(ModelRunner::new(
                Box::new(FailSecondPrefillModel {
                    prefill_calls: Cell::new(0),
                }),
                Device::Cpu,
            ))
            .unwrap();
        let first = scheduler
            .add_request(vec![1], SamplingParams::default())
            .unwrap();
        assert_eq!(scheduler.step().unwrap().len(), 2);
        let second = scheduler
            .add_request(vec![2], SamplingParams::default())
            .unwrap();
        let results = scheduler.step().unwrap();
        assert_eq!(
            results,
            vec![
                OutputToken {
                    uid: second,
                    token_id: 7,
                    finished: true,
                    finish_reason: Some(FinishReason::Error)
                },
                OutputToken {
                    uid: first,
                    token_id: 3,
                    finished: true,
                    finish_reason: Some(FinishReason::Length)
                },
            ]
        );
        assert!(scheduler.is_idle());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn prefill_finishes_on_eos_and_releases_running_slot() {
        let (mut scheduler, path) = scheduler();
        scheduler
            .engine_mut()
            .attach_model_runner(ModelRunner::new(Box::new(FixedTokenModel(7)), Device::Cpu))
            .unwrap();
        let uid = scheduler
            .add_request(vec![1], SamplingParams::default())
            .unwrap();
        assert_eq!(
            scheduler.step().unwrap(),
            vec![OutputToken {
                uid,
                token_id: 7,
                finished: true,
                finish_reason: Some(FinishReason::Stop),
            }]
        );
        assert!(scheduler.is_idle());
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn naive_prefill_returns_pages_when_request_finishes() {
        let (mut scheduler, path) = scheduler_with_strategy(CacheStrategy::Naive);
        let free_before = scheduler.engine().kv_cache_pool().unwrap().free_count();
        scheduler
            .engine_mut()
            .attach_model_runner(ModelRunner::new(Box::new(FixedTokenModel(7)), Device::Cpu))
            .unwrap();
        scheduler
            .add_request(vec![1], SamplingParams::default())
            .unwrap();
        assert_eq!(
            scheduler.step().unwrap()[0].finish_reason,
            Some(FinishReason::Stop)
        );
        assert_eq!(
            scheduler.engine().kv_cache_pool().unwrap().free_count(),
            free_before
        );
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn eos_normalization_handles_config_variants() {
        assert_eq!(
            normalize_eos(&serde_json::json!({"token_id": [1, "2", null]})),
            BTreeSet::from([1, 2])
        );
        assert!(normalize_eos(&serde_json::json!({"unexpected": 3})).is_empty());
    }
}
