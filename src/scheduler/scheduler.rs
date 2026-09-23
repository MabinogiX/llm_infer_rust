//! Scheduler request lifecycle and configuration handling.

use std::{
    collections::{BTreeSet, VecDeque},
    fs,
    path::PathBuf,
};

use serde_json::Value;

use crate::engine::{Engine, SamplingParams, ServerArgs};

use super::{
    FinishReason, OutputToken, Request, RequestId, Result, SchedulerError, SequenceStatus,
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
    pending: VecDeque<Request>,
    aborted: VecDeque<OutputToken>,
}

impl Scheduler {
    pub fn new(engine: Engine) -> Result<Self> {
        Self::with_cache_strategy(engine, CacheStrategy::Radix)
    }

    pub fn with_cache_strategy(engine: Engine, cache_strategy: CacheStrategy) -> Result<Self> {
        let args = engine.server_args().clone();
        let eos_token_ids = load_eos_token_ids(&args.model_path)?;
        Ok(Self {
            engine,
            args,
            cache_strategy,
            eos_token_ids,
            next_uid: 0,
            pending: VecDeque::new(),
            aborted: VecDeque::new(),
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
        self.pending.len()
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
            self.pending.push_back(Request {
                uid,
                input_ids,
                sampling_params: sampling_params.normalized(),
                cached_len: 0,
                output_len: 0,
                status: SequenceStatus::Waiting,
            });
        }
        Ok(uid)
    }

    /// Removes a waiting request. Running requests become abortable when the
    /// prefill manager is connected.
    pub fn abort_request(&mut self, uid: RequestId) -> bool {
        if let Some(index) = self.pending.iter().position(|request| request.uid == uid) {
            self.pending.remove(index);
            return true;
        }
        false
    }

    pub fn is_idle(&self) -> bool {
        self.pending.is_empty() && self.aborted.is_empty()
    }

    /// Delivers queued terminal results first. Once only waiting requests
    /// remain, returns an explicit error until prefill/decode is migrated.
    /// Waiting requests are retained on error.
    pub fn step(&mut self) -> Result<Vec<OutputToken>> {
        if !self.aborted.is_empty() {
            return Ok(self.aborted.drain(..).collect());
        }
        if !self.pending.is_empty() {
            return Err(SchedulerError::ExecutionNotAvailable);
        }
        Ok(Vec::new())
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
        fs,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::engine::ModelArgs;

    use super::*;

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn scheduler() -> (Scheduler, PathBuf) {
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
        args.max_running_req = 1;
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
        (Scheduler::new(engine).unwrap(), path)
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
    fn pending_request_survives_unavailable_step_and_can_be_aborted() {
        let (mut scheduler, path) = scheduler();
        let uid = scheduler
            .add_request(vec![1], SamplingParams::default())
            .unwrap();
        let rejected = scheduler
            .add_request(vec![1, 2, 3, 4, 5], SamplingParams::default())
            .unwrap();
        assert_eq!(scheduler.step().unwrap()[0].uid, rejected);
        assert!(matches!(
            scheduler.step(),
            Err(SchedulerError::ExecutionNotAvailable)
        ));
        assert_eq!(scheduler.pending_len(), 1);
        assert!(scheduler.abort_request(uid));
        assert!(!scheduler.abort_request(uid));
        assert!(scheduler.is_idle());
        assert!(scheduler.step().unwrap().is_empty());
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
