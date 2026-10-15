//! Scheduler request lifecycle and configuration handling.

use std::collections::{BTreeSet, VecDeque};

use crate::engine::kvcache::{CacheManager, NaiveCacheManager, RadixCacheManager};
use crate::engine::{
    Batch, BatchContext, BatchPhase, Engine, EngineError, SamplingParams, ServerArgs,
};
use crate::profiling::{StepProfiler, StepStage, time_schedule};
use tch::Tensor;

use super::{
    DecodeManager, FinishReason, OutputToken, PrefillManager, Request, RequestId, Result,
    SchedulerError, SequenceStatus,
    future_map::{CompletionInput, FutureMap, RelayPlan},
    prefill::PrefillChunk,
    request_manager::RequestManager,
    schedule_plan::SchedulePlan,
};
use crate::engine::MixedDecodeRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    Radix,
    Naive,
}

/// Completion boundary for an outstanding forward; the CPU sequence may lag
/// its device-side token relay by one invocation.
struct PendingBatch {
    fence: crate::engine::CompletionFence,
    phase: BatchPhase,
    request_ids: Vec<RequestId>,
    inputs: Vec<CompletionInput>,
    sample: crate::engine::PendingSample,
    chunk: Option<PrefillChunk>,
}

/// Own the detached sequence/handle until the last GPU invocation using it
/// finishes. Logical completion has already been delivered to the caller.
struct RetiredRequest {
    request: Request,
    publish: bool,
    fence: crate::engine::CompletionFence,
}

pub struct Scheduler {
    engine: Engine,
    args: ServerArgs,
    cache_strategy: CacheStrategy,
    eos_token_ids: BTreeSet<i64>,
    next_uid: RequestId,
    requests: RequestManager,
    prefill: PrefillManager,
    decode: DecodeManager,
    aborted: VecDeque<OutputToken>,
    last_step_error: Option<EngineError>,
    outstanding: Option<PendingBatch>,
    retired: VecDeque<RetiredRequest>,
    future_map: FutureMap,
}

impl Scheduler {
    pub fn new(engine: Engine, eos_token_ids: BTreeSet<i64>) -> Result<Self> {
        Self::with_cache_strategy(engine, CacheStrategy::Radix, eos_token_ids)
    }

    pub fn with_cache_strategy(
        engine: Engine,
        cache_strategy: CacheStrategy,
        eos_token_ids: BTreeSet<i64>,
    ) -> Result<Self> {
        let args = engine.server_args().clone();
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
        let requests = RequestManager::new(cache);
        let prefill = PrefillManager::new(&args, batch_context);
        let decode = DecodeManager::new(&args, engine.device())?;
        let future_map_rows = args.max_running_req;
        let future_map_device = engine.device();
        Ok(Self {
            engine,
            args,
            cache_strategy,
            eos_token_ids,
            next_uid: 0,
            requests,
            prefill,
            decode,
            aborted: VecDeque::new(),
            last_step_error: None,
            outstanding: None,
            retired: VecDeque::new(),
            future_map: FutureMap::new(future_map_rows, future_map_device),
        })
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        // Model/graph/KV replacement must not race an outstanding forward.
        let mut outputs = Vec::new();
        self.drain_outstanding(&mut outputs);
        self.reap_retired(true);
        self.future_map
            .release_inactive(self.requests.running_requests());
        self.aborted.extend(outputs);
        &mut self.engine
    }

    pub fn cache_strategy(&self) -> CacheStrategy {
        self.cache_strategy
    }

    pub fn eos_token_ids(&self) -> &BTreeSet<i64> {
        &self.eos_token_ids
    }

    pub fn pending_len(&self) -> usize {
        self.requests.pending_len()
    }

    pub fn running_len(&self) -> usize {
        self.requests.running_len()
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
            self.requests.add_request(Request {
                uid,
                input_ids,
                sampling_params: sampling_params.normalized(),
                cached_len: 0,
                prefill_cursor: 0,
                output_len: 0,
                cache_handle: None,
                status: SequenceStatus::Waiting,
            });
        }
        Ok(uid)
    }

    /// Cancel logically now; a handle still used by the outstanding invocation
    /// is detached and released only after its completion event.
    pub fn abort_request(&mut self, uid: RequestId) -> bool {
        if self.requests.running_request(uid).is_some() {
            self.retire_request(uid, false);
            true
        } else {
            self.requests.abort(uid)
        }
    }

    pub fn is_idle(&self) -> bool {
        self.outstanding.is_none()
            && self.retired.is_empty()
            && self.requests.pending_len() == 0
            && self.requests.running_len() == 0
            && !self.requests.has_aborted()
            && self.aborted.is_empty()
    }

    /// Runs prefill followed by decode for every request still running.
    pub fn step(&mut self) -> Result<Vec<OutputToken>> {
        if cfg!(has_cuda_graph)
            && self.engine.device().is_cuda()
            && !self.args.disable_cpu_gpu_overlap
            && !crate::logging::step_timing_enabled()
        {
            return self.step_overlap();
        }
        let mut outputs = Vec::new();
        self.drain_outstanding(&mut outputs);
        self.reap_retired(true);
        outputs.extend(self.step_serial()?);
        Ok(outputs)
    }

    fn step_serial(&mut self) -> Result<Vec<OutputToken>> {
        self.last_step_error = None;
        let (batch, schedule_us) = time_schedule(self.engine.device(), || {
            self.prefill.schedule_prefill(&mut self.requests)
        });
        let batch = batch?;
        let mut results: Vec<_> = self.aborted.drain(..).collect();
        if let Some(batch) = batch {
            let chunk = batch.chunk;
            self.run_phase(
                batch.request_ids,
                batch.model_batch,
                BatchPhase::Prefill,
                schedule_us,
                chunk,
                &mut results,
            )?;
            if chunk.is_some_and(|chunk| !chunk.is_last) {
                results.extend(
                    self.requests
                        .drain_aborted()
                        .into_iter()
                        .map(|uid| self.terminal_result(uid, FinishReason::Abort)),
                );
                return Ok(results);
            }
        }
        results.extend(
            self.requests
                .drain_aborted()
                .into_iter()
                .map(|uid| self.terminal_result(uid, FinishReason::Abort)),
        );
        let (decode_batch, schedule_us) = time_schedule(self.engine.device(), || {
            self.decode
                .schedule_decode(self.requests.running_requests())
        });
        let decode_batch = decode_batch?;
        if let Some(batch) = decode_batch {
            self.run_phase(
                batch.request_ids,
                batch.model_batch,
                BatchPhase::Decode,
                schedule_us,
                None,
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
        schedule_us: u128,
        chunk: Option<PrefillChunk>,
        results: &mut Vec<OutputToken>,
    ) -> Result<()> {
        let mut profiler = StepProfiler::new(self.engine.device(), schedule_us);
        let profile_requests = profiler.enabled().then(|| {
            request_ids
                .iter()
                .map(|uid| {
                    (
                        *uid,
                        self.requests
                            .running_request(*uid)
                            .expect("scheduled request is running")
                            .output_len
                            + 1,
                    )
                })
                .collect::<Vec<_>>()
        });
        let params = profiler.measure(StepStage::Params, || {
            request_ids
                .iter()
                .map(|uid| {
                    self.requests
                        .running_request(*uid)
                        .expect("scheduled request is running")
                        .sampling_params
                })
                .collect::<Vec<_>>()
        });
        let logits = profiler.measure(StepStage::Forward, || self.engine.forward(&model_batch));
        if chunk.is_some_and(|chunk| !chunk.is_last) {
            let forward_ok = logits.is_ok();
            match logits {
                Ok(_) => {
                    let chunk = chunk.expect("non-final chunk metadata");
                    if let Err(error) = self.requests.publish_prefix(chunk.uid, chunk.end) {
                        self.last_step_error = Some(EngineError::KVCache(error));
                        self.requests.remove_batch(&[chunk.uid]);
                    } else if let Some(request) = self.requests.running_request_mut(chunk.uid) {
                        request.prefill_cursor = chunk.end;
                    }
                }
                Err(error) => {
                    self.synchronize();
                    self.complete(request_ids, None, BatchPhase::Prefill, Err(error), results);
                }
            }
            profiler.log(
                BatchPhase::Prefill,
                profile_requests.as_deref().unwrap_or(&[]),
                forward_ok,
            );
            return Ok(());
        }
        let sampled = profiler.measure(StepStage::Sample, || {
            logits
                .and_then(|logits| self.engine.sample(&logits, &params))
                .and_then(|tokens| {
                    if tokens.len() == request_ids.len() {
                        Ok(tokens)
                    } else {
                        Err(EngineError::InvalidArgument(
                            "sampled token count differs from batch size".to_owned(),
                        ))
                    }
                })
        });
        let success = sampled.is_ok();
        if !success {
            self.synchronize();
        }
        profiler.measure(StepStage::Publish, || {
            self.complete(request_ids, None, phase, sampled, results);
        });
        profiler.log(phase, profile_requests.as_deref().unwrap_or(&[]), success);
        Ok(())
    }

    fn complete(
        &mut self,
        request_ids: Vec<RequestId>,
        inputs: Option<Vec<CompletionInput>>,
        phase: BatchPhase,
        sampled: std::result::Result<Vec<i64>, EngineError>,
        results: &mut Vec<OutputToken>,
    ) {
        match sampled {
            Ok(tokens) => {
                let mut finished = Vec::new();
                for (row, (uid, token_id)) in request_ids.into_iter().zip(tokens).enumerate() {
                    // Aborted rows, including their speculative sampled tokens,
                    // are ignored after the outstanding invocation is fenced.
                    if self.requests.running_request(uid).is_none() {
                        continue;
                    }
                    let input = inputs
                        .as_ref()
                        .map(|inputs| inputs[row])
                        .unwrap_or_else(|| self.completion_input(uid));
                    let reason = self.finish_reason(input, token_id);
                    if reason.is_some() && self.resource_fence(uid).is_some() {
                        self.requests.mark_written(uid, input.written_len);
                        self.requests
                            .running_request_mut(uid)
                            .unwrap()
                            .append_token(token_id);
                        self.retire_request(uid, true);
                        results.push(OutputToken {
                            uid,
                            token_id,
                            finished: true,
                            finish_reason: reason,
                        });
                        continue;
                    }
                    if phase == BatchPhase::Prefill {
                        if let Some(request) = self.requests.running_request_mut(uid) {
                            request.prefill_cursor = request.input_ids.len();
                        }
                    }
                    if phase == BatchPhase::Prefill || reason.is_some() {
                        if let Err(error) = self.requests.publish(uid) {
                            tracing::error!(uid, error = %error, "KV cache publish failed");
                            self.requests.remove_batch(&[uid]);
                            results.push(self.terminal_result(uid, FinishReason::Error));
                            continue;
                        }
                    } else {
                        self.requests.mark_written(uid, input.written_len);
                    }
                    let request = self.requests.running_request_mut(uid).unwrap();
                    request.append_token(token_id);
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
                self.requests.remove_batch(&finished);
            }
            Err(error) => {
                self.last_step_error = Some(error);
                for uid in request_ids {
                    if self.requests.running_request(uid).is_some() {
                        self.retire_request(uid, false);
                        results.push(self.terminal_result(uid, FinishReason::Error));
                    }
                }
            }
        }
    }

    fn resource_fence(&self, uid: RequestId) -> Option<crate::engine::CompletionFence> {
        self.outstanding
            .as_ref()
            .filter(|batch| batch.request_ids.contains(&uid))
            .map(|batch| batch.fence.clone())
    }

    fn retire_request(&mut self, uid: RequestId, publish: bool) {
        let fence = self.resource_fence(uid);
        let mut request = self
            .requests
            .retire(uid)
            .expect("retired request is running");
        request.status = SequenceStatus::Finished;
        if let Some(fence) = fence {
            self.retired.push_back(RetiredRequest {
                request,
                publish,
                fence,
            });
        } else {
            self.release_retired(request, publish);
        }
    }

    fn release_retired(&mut self, request: Request, publish: bool) {
        let uid = request.uid;
        if let Err(error) = self.requests.release_retired(request, publish) {
            // Logical terminal output was already emitted. Cache maintenance
            // failure is reported here, without emitting a second terminal.
            tracing::error!(uid, error = %error, "deferred KV cache publish failed");
            self.last_step_error = Some(EngineError::KVCache(error));
        }
    }

    fn reap_retired(&mut self, wait: bool) {
        let count = self.retired.len();
        for _ in 0..count {
            let retired = self.retired.pop_front().unwrap();
            let ready = if wait {
                retired.fence.wait().map(|()| true)
            } else {
                retired.fence.ready()
            };
            match ready {
                Ok(true) => self.release_retired(retired.request, retired.publish),
                Ok(false) => self.retired.push_back(retired),
                Err(error) => {
                    // A failed event query/readback is exceptional. Fence all
                    // device access before the error cleanup can release pages.
                    self.synchronize();
                    self.last_step_error = Some(EngineError::Sampling(error));
                    self.release_retired(retired.request, retired.publish);
                }
            }
        }
    }

    fn finish_reason(&self, input: CompletionInput, token_id: i64) -> Option<FinishReason> {
        if self.eos_token_ids.contains(&token_id) && !input.ignore_eos {
            Some(FinishReason::Stop)
        } else if input.length_finished {
            Some(FinishReason::Length)
        } else {
            None
        }
    }

    fn synchronize(&self) {
        if let tch::Device::Cuda(device) = self.engine.device() {
            tch::Cuda::synchronize(device as i64);
        }
    }

    fn launch_decode(
        &mut self,
        batch: super::decode::DecodeBatch,
        future: Option<&RelayPlan>,
    ) -> std::result::Result<PendingBatch, EngineError> {
        self.launch_batch(batch.request_ids, batch.model_batch, future, None)
    }

    fn launch_plan(
        &mut self,
        mut plan: SchedulePlan,
    ) -> std::result::Result<PendingBatch, EngineError> {
        if plan.phase == BatchPhase::Prefill {
            if let (Some(relay), Some(offset)) = (&plan.relay, plan.relay_offset) {
                let slots = Tensor::from_slice(
                    &relay
                        .slots()
                        .into_iter()
                        .map(|slot| slot as i64)
                        .collect::<Vec<_>>(),
                )
                .to_device(self.future_map.output_tokens().device());
                let decode_ids = self
                    .future_map
                    .output_tokens()
                    .index_select(0, &slots)
                    .view([-1]);
                let offset = i64::try_from(offset).map_err(|_| {
                    EngineError::InvalidArgument(
                        "mixed prefill token count exceeds i64".to_string(),
                    )
                })?;
                let prefill_ids = plan.model_batch.input_ids.narrow(0, 0, offset);
                plan.model_batch.input_ids = Tensor::cat(&[prefill_ids, decode_ids], 0);
            }
        }
        self.launch_batch(
            plan.request_ids,
            plan.model_batch,
            plan.relay.as_ref(),
            plan.chunk,
        )
    }

    /// Select the next batch after an outstanding invocation. This method
    /// only builds a plan; it does not launch GPU work or consume the previous
    /// result.
    fn plan_after(&mut self, previous: &PendingBatch) -> Result<Option<SchedulePlan>> {
        let previous_relay = if previous.phase == BatchPhase::Decode {
            Some(self.future_map.plan(
                &previous.request_ids,
                &previous.inputs,
                self.requests.running_requests(),
            )?)
        } else {
            None
        };
        if let Some(selection) = self
            .prefill
            .schedule_prefill_selection(&mut self.requests)?
        {
            if let Some(relay) = previous_relay {
                let decode_rows = relay
                    .rows()
                    .map(|row| {
                        let request = self
                            .requests
                            .running_request(row.uid)
                            .expect("relay request is running");
                        MixedDecodeRequest {
                            input_id: 0,
                            position: row.input.written_len.saturating_sub(1),
                            page_ids: request
                                .cache_handle
                                .as_ref()
                                .expect("relay request has KV handle")
                                .page_ids
                                .clone(),
                        }
                    })
                    .collect::<Vec<_>>();
                if !decode_rows.is_empty() {
                    let mixed = match self.prefill.prepare_mixed_prefill(
                        &self.requests,
                        &selection.request_ids,
                        &decode_rows,
                        selection.chunk,
                    ) {
                        Ok(batch) => batch,
                        Err(error) => {
                            self.prefill
                                .rollback_selection(&mut self.requests, &selection.request_ids);
                            return Err(error.into());
                        }
                    };
                    let mut request_ids = selection.request_ids.clone();
                    request_ids.extend(relay.rows().map(|row| row.uid));
                    return Ok(Some(SchedulePlan::mixed(
                        request_ids,
                        mixed,
                        relay,
                        selection.token_count,
                        selection.chunk,
                    )));
                }
            }
            let model_batch = match self.prefill.build_prefill(&self.requests, &selection) {
                Ok(batch) => batch,
                Err(error) => {
                    self.prefill
                        .rollback_selection(&mut self.requests, &selection.request_ids);
                    return Err(error.into());
                }
            };
            return Ok(Some(SchedulePlan::prefill(
                selection.request_ids,
                model_batch,
                selection.chunk,
            )));
        }

        // A prefill result must be completed before decode planning because
        // its sampled token is not stored in FutureMap. Throughput mode may
        // overlap consecutive prefills, but it still establishes a barrier
        // before switching back to decode.
        if previous.phase == BatchPhase::Prefill {
            return Ok(None);
        }

        let relay = previous_relay.expect("decode previous batch has relay plan");
        let Some(batch) = self.decode.schedule_relay(
            self.requests.running_requests(),
            &relay,
            self.future_map.output_tokens(),
        )?
        else {
            return Ok(None);
        };
        Ok(Some(SchedulePlan::decode(batch, Some(relay))))
    }

    fn launch_batch(
        &mut self,
        request_ids: Vec<RequestId>,
        model_batch: Batch,
        future: Option<&RelayPlan>,
        chunk: Option<PrefillChunk>,
    ) -> std::result::Result<PendingBatch, EngineError> {
        let params = request_ids
            .iter()
            .map(|uid| self.requests.running_request(*uid).unwrap().sampling_params)
            .collect::<Vec<_>>();
        let inputs = request_ids
            .iter()
            .map(|uid| {
                let request = self.requests.running_request(*uid).unwrap();
                let written_len = future
                    .and_then(|plan| plan.input(*uid))
                    .map(|input| input.written_len)
                    .unwrap_or(request.input_ids.len());
                let output_len = request
                    .output_len
                    .checked_add(1)
                    .and_then(|value| value.checked_add(written_len))
                    .and_then(|value| value.checked_sub(request.input_ids.len()))
                    .ok_or_else(|| {
                        EngineError::InvalidArgument(
                            "decode completion length exceeds usize bounds".to_owned(),
                        )
                    })?;
                let next_len = written_len.checked_add(1).ok_or_else(|| {
                    EngineError::InvalidArgument(
                        "decode written length exceeds usize bounds".to_owned(),
                    )
                })?;
                Ok(CompletionInput {
                    written_len,
                    length_finished: output_len >= request.sampling_params.max_tokens
                        || next_len >= self.args.max_seq_len,
                    ignore_eos: request.sampling_params.ignore_eos,
                })
            })
            .collect::<std::result::Result<Vec<_>, EngineError>>()?;
        let sample =
            self.engine
                .forward_with_read_fence(&model_batch)
                .and_then(|(logits, read_done)| {
                    let sample = self.engine.defer_sample(logits, &params)?;
                    if model_batch.phase == BatchPhase::Decode {
                        self.decode.set_read_boundary(read_done, sample.fence());
                    }
                    Ok(sample)
                });
        match sample {
            Ok(sample) => {
                if model_batch.phase == BatchPhase::Decode {
                    self.future_map
                        .publish(&request_ids, &sample.relay())
                        .map_err(|error| EngineError::InvalidArgument(error.to_string()))?;
                }
                Ok(PendingBatch {
                    fence: sample.fence(),
                    phase: model_batch.phase,
                    request_ids,
                    inputs,
                    sample,
                    chunk,
                })
            }
            Err(error) => {
                self.synchronize();
                Err(error)
            }
        }
    }

    fn read_pending(
        pending: PendingBatch,
    ) -> (
        BatchPhase,
        Vec<RequestId>,
        Vec<CompletionInput>,
        std::result::Result<Vec<i64>, EngineError>,
    ) {
        let PendingBatch {
            fence: _,
            phase,
            request_ids,
            inputs,
            sample,
            ..
        } = pending;
        let tokens = sample
            .finish()
            .map_err(EngineError::from)
            .and_then(|tokens| {
                if tokens.len() == request_ids.len() {
                    Ok(tokens)
                } else {
                    Err(EngineError::InvalidArgument(
                        "sampled token count differs from batch size".into(),
                    ))
                }
            });
        (phase, request_ids, inputs, tokens)
    }

    fn drain_outstanding(&mut self, results: &mut Vec<OutputToken>) {
        if let Some(pending) = self.outstanding.take() {
            let chunk = pending.chunk;
            let (phase, ids, inputs, tokens) = Self::read_pending(pending);
            if let Some(chunk) = chunk.filter(|chunk| !chunk.is_last) {
                match tokens {
                    Ok(tokens) => {
                        self.complete_mixed_chunk_rows(ids, inputs, tokens, phase, chunk, results)
                    }
                    Err(error) => self.complete(ids, Some(inputs), phase, Err(error), results),
                }
            } else {
                self.complete(ids, Some(inputs), phase, tokens, results);
            }
        }
        self.reap_retired(false);
    }

    fn complete_mixed_chunk_rows(
        &mut self,
        request_ids: Vec<RequestId>,
        inputs: Vec<CompletionInput>,
        tokens: Vec<i64>,
        phase: BatchPhase,
        chunk: PrefillChunk,
        results: &mut Vec<OutputToken>,
    ) {
        let mut other_ids = Vec::new();
        let mut other_inputs = Vec::new();
        let mut other_tokens = Vec::new();
        for ((uid, input), token) in request_ids.into_iter().zip(inputs).zip(tokens) {
            if uid != chunk.uid {
                other_ids.push(uid);
                other_inputs.push(input);
                other_tokens.push(token);
            }
        }
        if !other_ids.is_empty() {
            self.complete(
                other_ids,
                Some(other_inputs),
                phase,
                Ok(other_tokens),
                results,
            );
        }
        if let Err(error) = self.requests.publish_prefix(chunk.uid, chunk.end) {
            self.last_step_error = Some(EngineError::KVCache(error));
            self.requests.remove_batch(&[chunk.uid]);
        } else if let Some(request) = self.requests.running_request_mut(chunk.uid) {
            request.prefill_cursor = chunk.end;
        }
    }

    fn step_overlap(&mut self) -> Result<Vec<OutputToken>> {
        self.last_step_error = None;
        let mut results: Vec<_> = self.aborted.drain(..).collect();
        self.reap_retired(false);
        // A prefill's publish may canonicalize its pages. Wait for that
        // invocation and publish before planning decode for those same rows.
        // Incoming prefill alone does not require draining an old decode.
        if self.outstanding.as_ref().is_some_and(|batch| {
            batch.chunk.is_some()
                || (batch.phase == BatchPhase::Prefill
                    && (self.args.disable_consecutive_prefill_overlap
                        || self.requests.pending_len() == 0))
        }) {
            self.drain_outstanding(&mut results);
        }
        if let Some(previous) = self.outstanding.take() {
            let plan = match self.plan_after(&previous) {
                Ok(plan) => plan,
                Err(error) => {
                    self.outstanding = Some(previous);
                    self.drain_outstanding(&mut results);
                    self.aborted.extend(results);
                    return Err(error);
                }
            };
            let (next_ids, next_phase, next) = if let Some(plan) = plan {
                let ids = plan.request_ids.clone();
                let phase = plan.phase;
                (ids, phase, self.launch_plan(plan).map(Some))
            } else {
                (Vec::new(), BatchPhase::Decode, Ok(None))
            };
            let (phase, ids, inputs, tokens) = Self::read_pending(previous);
            match next {
                Err(error) => {
                    // Launch errors already fence queued GPU operations.
                    self.complete(ids, Some(inputs), phase, tokens, &mut results);
                    self.complete(next_ids, None, next_phase, Err(error), &mut results);
                }
                Ok(next) => {
                    // Install the last-use fence before logical completion can
                    // detach EOS/error rows. Survivor commits never wait on it.
                    self.outstanding = next;
                    self.complete(ids, Some(inputs), phase, tokens, &mut results);
                }
            }
            self.reap_retired(false);
            self.future_map
                .release_inactive(self.requests.running_requests());
            results.extend(
                self.requests
                    .drain_aborted()
                    .into_iter()
                    .map(|uid| self.terminal_result(uid, FinishReason::Abort)),
            );
            return Ok(results);
        }
        // No GPU invocation remains; any detached handle can now be reclaimed.
        self.reap_retired(true);
        self.future_map
            .release_inactive(self.requests.running_requests());
        let (batch, schedule_us) = time_schedule(self.engine.device(), || {
            self.prefill.schedule_prefill(&mut self.requests)
        });
        if let Some(batch) = batch? {
            let chunk = batch.chunk;
            self.run_phase(
                batch.request_ids,
                batch.model_batch,
                BatchPhase::Prefill,
                schedule_us,
                chunk,
                &mut results,
            )?;
            if chunk.is_some_and(|chunk| !chunk.is_last) {
                results.extend(
                    self.requests
                        .drain_aborted()
                        .into_iter()
                        .map(|uid| self.terminal_result(uid, FinishReason::Abort)),
                );
                return Ok(results);
            }
        }
        results.extend(
            self.requests
                .drain_aborted()
                .into_iter()
                .map(|uid| self.terminal_result(uid, FinishReason::Abort)),
        );
        if let Some(batch) = self
            .decode
            .schedule_pipeline(self.requests.running_requests())?
        {
            let ids = batch.request_ids.clone();
            match self.launch_decode(batch, None) {
                Ok(pending) => self.outstanding = Some(pending),
                Err(error) => {
                    self.complete(ids, None, BatchPhase::Decode, Err(error), &mut results)
                }
            }
        }
        Ok(results)
    }

    fn completion_input(&self, uid: RequestId) -> CompletionInput {
        let request = self
            .requests
            .running_request(uid)
            .expect("scheduled request is running");
        CompletionInput {
            written_len: request.input_ids.len(),
            length_finished: request.output_len + 1 >= request.sampling_params.max_tokens
                || request.input_ids.len() + 1 >= self.args.max_seq_len,
            ignore_eos: request.sampling_params.ignore_eos,
        }
    }

    // Abort/error events carry a placeholder token when the model has no EOS.
    // Consumers inspect finish_reason before decoding this event.
    fn terminal_result(&self, uid: RequestId, reason: FinishReason) -> OutputToken {
        OutputToken {
            uid,
            token_id: self.eos_token_ids.first().copied().unwrap_or(0),
            finished: true,
            finish_reason: Some(reason),
        }
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        // Fence before engine, graph catalog, and cache managers are dropped.
        if let Some(batch) = &self.outstanding {
            if batch.fence.wait().is_err() {
                self.synchronize();
            }
        }
        drop(self.outstanding.take());
        self.reap_retired(true);
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

    use crate::engine::kvcache::{AcquireOutcome, BaseCacheHandle, KVCacheError};
    use crate::engine::{ModelExecutor, ModelRunner, ModelRunnerError, RuntimeModelConfig};

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
        scheduler_with_options(strategy, max_running_req, 4)
    }

    fn scheduler_with_options(
        strategy: CacheStrategy,
        max_running_req: usize,
        max_seq_len: usize,
    ) -> (Scheduler, PathBuf) {
        scheduler_with_execution(strategy, max_running_req, max_seq_len, Device::Cpu, false)
    }

    fn scheduler_with_execution(
        strategy: CacheStrategy,
        max_running_req: usize,
        max_seq_len: usize,
        device: Device,
        disable_overlap: bool,
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
        fs::write(path.join("config.json"), "{}").unwrap();
        let mut args = ServerArgs::new(&path);
        args.max_running_req = max_running_req;
        args.max_seq_len = max_seq_len;
        args.page_size = 2;
        // These assertions exercise fixed batch shapes, independent of GPU auto-tuning.
        args.chunked_prefill_size_explicit = true;
        // Lifecycle tests need equal, small pools even when comparing two CUDA engines.
        args.max_total_tokens =
            Some(max_running_req * (max_seq_len / args.page_size + 1) * args.page_size);
        args.disable_cpu_gpu_overlap = disable_overlap;
        let engine = Engine::with_runtime(
            args,
            RuntimeModelConfig {
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 1,
                max_position_embeddings: max_seq_len,
                ..Default::default()
            },
            0,
            tch::Kind::Float,
            device,
        )
        .unwrap();
        (
            Scheduler::with_cache_strategy(engine, strategy, BTreeSet::from([7, 8])).unwrap(),
            path,
        )
    }

    #[test]
    fn oversized_prompt_emits_abort_and_uses_injected_eos() {
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
            batch: &crate::engine::ForwardBatch<'_>,
        ) -> std::result::Result<crate::engine::ForwardOutput, ModelRunnerError> {
            let input_ids = batch.input_ids;
            let _positions = batch.positions;
            let _attention_metadata = batch.attention;
            let logits_indices = batch.logits_indices();
            (|| -> std::result::Result<Tensor, ModelRunnerError> {
                let rows = logits_indices
                    .map(|indices| indices.size()[0])
                    .unwrap_or_else(|| input_ids.size()[0]);
                let mut logits = vec![0f32; rows as usize * 9];
                for row in 0..rows as usize {
                    logits[row * 9 + self.0 as usize] = 1.0;
                }
                Ok(Tensor::from_slice(&logits)
                    .view([rows, 9])
                    .to_device(batch.input_ids.device()))
            })()
            .map(crate::engine::ForwardOutput::new)
        }
    }

    struct DecodeFailModel;

    struct FailSecondPublish {
        inner: RadixCacheManager,
        calls: usize,
    }

    impl CacheManager for FailSecondPublish {
        fn acquire(
            &mut self,
            input_ids: &[i64],
            capacity_tokens: usize,
            budget: Option<usize>,
        ) -> crate::engine::kvcache::Result<AcquireOutcome> {
            self.inner.acquire(input_ids, capacity_tokens, budget)
        }

        fn publish(
            &mut self,
            handle: &mut BaseCacheHandle,
            written_ids: &[i64],
        ) -> crate::engine::kvcache::Result<()> {
            self.calls += 1;
            if self.calls == 2 {
                self.inner.publish(handle, written_ids)?;
                return Err(KVCacheError::InvalidArgument(
                    "injected failure after one request published".into(),
                ));
            }
            self.inner.publish(handle, written_ids)
        }

        fn release(&mut self, handle: &mut BaseCacheHandle) {
            self.inner.release(handle);
        }
    }

    impl ModelExecutor for DecodeFailModel {
        fn forward(
            &self,
            batch: &crate::engine::ForwardBatch<'_>,
        ) -> std::result::Result<crate::engine::ForwardOutput, ModelRunnerError> {
            let _input_ids = batch.input_ids;
            let _positions = batch.positions;
            let attention_metadata = batch.attention;
            let logits_indices = batch.logits_indices();
            (|| -> std::result::Result<Tensor, ModelRunnerError> {
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
                Ok(Tensor::from_slice(&logits)
                    .view([rows, 9])
                    .to_device(batch.input_ids.device()))
            })()
            .map(crate::engine::ForwardOutput::new)
        }
    }

    struct FailSecondPrefillModel {
        prefill_calls: Cell<usize>,
    }

    impl ModelExecutor for FailSecondPrefillModel {
        fn forward(
            &self,
            batch: &crate::engine::ForwardBatch<'_>,
        ) -> std::result::Result<crate::engine::ForwardOutput, ModelRunnerError> {
            let input_ids = batch.input_ids;
            let _positions = batch.positions;
            let attention_metadata = batch.attention;
            let logits_indices = batch.logits_indices();
            (|| -> std::result::Result<Tensor, ModelRunnerError> {
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
                Ok(Tensor::from_slice(&logits)
                    .view([rows, 9])
                    .to_device(batch.input_ids.device()))
            })()
            .map(crate::engine::ForwardOutput::new)
        }
    }

    #[cfg(all(has_flashinfer, has_cuda_graph))]
    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn cuda_completion_overlap_matches_serial_lifecycle_and_errors() {
        let device = Device::Cuda(0);
        for strategy in [CacheStrategy::Radix, CacheStrategy::Naive] {
            let mut pair = [
                scheduler_with_execution(strategy, 4, 16, device, false),
                scheduler_with_execution(strategy, 4, 16, device, true),
            ];
            for (scheduler, _) in &mut pair {
                scheduler
                    .engine_mut()
                    .attach_model_runner(ModelRunner::new(Box::new(FixedTokenModel(3)), device))
                    .unwrap();
                for (ids, max, temperature) in [
                    (vec![1, 2], 3, 0.),
                    (vec![1, 2], 1, 1.),
                    (vec![5, 6], 5, 0.),
                ] {
                    scheduler
                        .add_request(
                            ids,
                            SamplingParams {
                                max_tokens: max,
                                temperature,
                                top_k: 1,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                }
            }
            let mut traces = vec![];
            for (scheduler, _) in &mut pair {
                let mut trace = vec![scheduler.step().unwrap()];
                scheduler.drain_outstanding(&mut trace[0]);
                assert!(scheduler.abort_request(2));
                scheduler
                    .add_request(
                        vec![1, 2, 3],
                        SamplingParams {
                            max_tokens: 1,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                let mut outputs = scheduler.step().unwrap();
                scheduler.drain_outstanding(&mut outputs);
                trace.push(outputs);
                assert!(scheduler.is_idle());
                // EOS, including the distinct stop-vs-length precedence.
                scheduler
                    .engine_mut()
                    .attach_model_runner(ModelRunner::new(Box::new(FixedTokenModel(7)), device))
                    .unwrap();
                scheduler
                    .add_request(
                        vec![4],
                        SamplingParams {
                            max_tokens: 1,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                trace.push(scheduler.step().unwrap());
                assert_eq!(
                    trace.last().unwrap()[0].finish_reason,
                    Some(FinishReason::Stop)
                );
                // A failed decode preserves the preceding prefill token and
                // releases the failing request only after forward completes.
                scheduler
                    .engine_mut()
                    .attach_model_runner(ModelRunner::new(Box::new(DecodeFailModel), device))
                    .unwrap();
                scheduler
                    .add_request(
                        vec![6],
                        SamplingParams {
                            max_tokens: 5,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                trace.push(scheduler.step().unwrap());
                assert_eq!(
                    trace.last().unwrap().last().unwrap().finish_reason,
                    Some(FinishReason::Error)
                );
                assert!(scheduler.is_idle());
                traces.push((
                    trace,
                    scheduler.engine().kv_cache_pool().unwrap().free_count(),
                ));
            }
            assert_eq!(traces[0], traces[1]);
            for (_, path) in pair {
                fs::remove_dir_all(path).unwrap();
            }
        }
        // Publish can fail after installing a valid radix prefix. Both modes
        // must isolate that request without freeing the newly shared page.
        let mut outcomes = vec![];
        for disable in [false, true] {
            let (mut scheduler, path) =
                scheduler_with_execution(CacheStrategy::Radix, 2, 4, device, disable);
            scheduler
                .engine_mut()
                .attach_model_runner(ModelRunner::new(Box::new(FixedTokenModel(3)), device))
                .unwrap();
            let pool = scheduler.engine.shared_kv_cache_pool().unwrap();
            let cache = Box::new(FailSecondPublish {
                inner: RadixCacheManager::new(pool, 2).unwrap(),
                calls: 0,
            });
            scheduler.requests = RequestManager::new(cache);
            scheduler.prefill =
                PrefillManager::new(&scheduler.args, BatchContext::new(2, 4, 2, device).unwrap());
            for ids in [vec![1, 2], vec![3, 4]] {
                scheduler
                    .add_request(
                        ids,
                        SamplingParams {
                            max_tokens: 1,
                            ..Default::default()
                        },
                    )
                    .unwrap();
            }
            let results = scheduler.step().unwrap();
            assert_eq!(results[1].finish_reason, Some(FinishReason::Error));
            assert!(scheduler.is_idle());
            outcomes.push((
                results,
                scheduler.engine().kv_cache_pool().unwrap().free_count(),
            ));
            fs::remove_dir_all(path).unwrap();
        }
        assert_eq!(outcomes[0], outcomes[1]);
    }

    #[derive(Default)]
    struct RelayTokenModel {
        cache: Option<crate::engine::kvcache::ModelKvCache>,
    }

    impl ModelExecutor for RelayTokenModel {
        fn bind_state_cache(
            &mut self,
            cache: crate::engine::kvcache::ModelKvCache,
        ) -> std::result::Result<(), ModelRunnerError> {
            self.cache = Some(cache);
            Ok(())
        }

        fn forward(
            &self,
            batch: &crate::engine::ForwardBatch<'_>,
        ) -> std::result::Result<crate::engine::ForwardOutput, ModelRunnerError> {
            let ids = batch.input_ids.view([-1]);
            if let Some(cache) = &self.cache {
                let slots = batch
                    .attention
                    .unwrap()
                    .write_loc
                    .as_ref()
                    .unwrap()
                    .to_kind(tch::Kind::Int64);
                for layer in &cache.layers {
                    let shape = layer.k.size();
                    let values = ids
                        .to_kind(layer.k.kind())
                        .view([-1, 1, 1])
                        .expand([-1, shape[2], shape[3]], true);
                    let v_values = (&ids + batch.positions.view([-1]) * 10)
                        .to_kind(layer.v.kind())
                        .view([-1, 1, 1])
                        .expand([-1, shape[2], shape[3]], true);
                    let _ = layer
                        .k
                        .view([-1, shape[2], shape[3]])
                        .index_copy_(0, &slots, &values);
                    let _ = layer
                        .v
                        .view([-1, shape[2], shape[3]])
                        .index_copy_(0, &slots, &v_values);
                }
            }
            let ids = match batch.logits_indices() {
                Some(indices) => ids.index_select(0, indices),
                None => ids,
            };
            // Next token depends on the actual device input; a stale CPU token
            // or wrong relay row changes the entire generated sequence.
            let next = (ids + 1).remainder(9).unsqueeze(1);
            let logits = Tensor::zeros(
                [next.size()[0], 9],
                (tch::Kind::Float, batch.input_ids.device()),
            )
            .scatter_value(1, &next, 10.);
            Ok(crate::engine::ForwardOutput::new(logits))
        }
    }

    fn exercise_pipeline(device: Device) {
        for strategy in [CacheStrategy::Radix, CacheStrategy::Naive] {
            let mut outcomes = Vec::new();
            for pipeline in [false, true] {
                let (mut scheduler, path) =
                    scheduler_with_execution(strategy, 4, 32, device, !pipeline);
                scheduler
                    .engine_mut()
                    .attach_model_runner(ModelRunner::new(
                        Box::new(RelayTokenModel::default()),
                        device,
                    ))
                    .unwrap();
                for (ids, max_tokens, ignore_eos) in [
                    (vec![1, 2], 8, false),
                    (vec![5], 8, false),
                    (vec![0, 1], 8, true),
                ] {
                    scheduler
                        .add_request(
                            ids,
                            SamplingParams {
                                max_tokens,
                                ignore_eos,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                }
                let mut outputs = if pipeline {
                    scheduler.step_overlap().unwrap()
                } else {
                    scheduler.step_serial().unwrap()
                };
                if pipeline {
                    assert!(scheduler.outstanding.is_some());
                }
                let mut iterations = 0;
                while !scheduler.is_idle() {
                    outputs.extend(if pipeline {
                        scheduler.step_overlap().unwrap()
                    } else {
                        scheduler.step_serial().unwrap()
                    });
                    iterations += 1;
                    assert!(iterations < 32);
                }
                let mut rows = std::collections::BTreeMap::<_, Vec<_>>::new();
                for output in outputs {
                    rows.entry(output.uid)
                        .or_default()
                        .push((output.token_id, output.finish_reason));
                }
                assert_eq!(
                    rows[&0].iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                    vec![3, 4, 5, 6, 7]
                );
                assert_eq!(rows[&1], vec![(6, None), (7, Some(FinishReason::Stop))]);
                assert_eq!(rows[&2].len(), 8);
                assert_eq!(rows[&2].last().unwrap().1, Some(FinishReason::Length));
                // The cached prefix excludes the EOS token, despite the extra
                // speculative forward having consumed it before CPU EOS handling.
                scheduler
                    .add_request(
                        vec![1, 2, 3, 4, 5, 6, 7, 8],
                        SamplingParams {
                            max_tokens: 1,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                let batch = scheduler
                    .prefill
                    .schedule_prefill(&mut scheduler.requests)
                    .unwrap()
                    .unwrap();
                assert_eq!(batch.request_ids, vec![3]);
                if strategy == CacheStrategy::Radix {
                    assert_eq!(scheduler.requests.running_request(3).unwrap().cached_len, 6);
                    let request = scheduler.requests.running_request(3).unwrap();
                    let handle = request.cache_handle.as_ref().unwrap();
                    let cache = scheduler
                        .engine()
                        .kv_cache_pool()
                        .unwrap()
                        .get_layer_kv_cache(0)
                        .unwrap();
                    for position in 0..request.cached_len {
                        let page = handle.page_ids[position / 2] as i64;
                        let offset = (position % 2) as i64;
                        assert_eq!(
                            cache.k.get(page).get(offset).double_value(&[0, 0]),
                            request.input_ids[position] as f64
                        );
                        assert_eq!(
                            cache.v.get(page).get(offset).double_value(&[0, 0]),
                            (request.input_ids[position] + position as i64 * 10) as f64
                        );
                    }
                }
                scheduler.abort_request(3);
                outcomes.push((
                    rows,
                    scheduler.engine().kv_cache_pool().unwrap().free_count(),
                ));
                fs::remove_dir_all(path).unwrap();
            }
            assert_eq!(outcomes[0], outcomes[1]);
        }
        // Cancellation detaches pending rows; new prefill overlaps prior commit.
        let (mut scheduler, path) =
            scheduler_with_execution(CacheStrategy::Radix, 2, 16, device, false);
        scheduler
            .engine_mut()
            .attach_model_runner(ModelRunner::new(
                Box::new(RelayTokenModel::default()),
                device,
            ))
            .unwrap();
        scheduler
            .add_request(
                vec![1, 2],
                SamplingParams {
                    max_tokens: 8,
                    ..Default::default()
                },
            )
            .unwrap();
        scheduler
            .add_request(
                vec![0, 1],
                SamplingParams {
                    max_tokens: 5,
                    ignore_eos: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let mut survivor = scheduler
            .step_overlap()
            .unwrap()
            .into_iter()
            .filter(|output| output.uid == 1)
            .collect::<Vec<_>>();
        assert!(scheduler.outstanding.is_some());
        assert!(scheduler.abort_request(0));
        while !scheduler.is_idle() {
            let outputs = scheduler.step_overlap().unwrap();
            assert!(outputs.iter().all(|output| output.uid != 0));
            survivor.extend(outputs);
        }
        assert_eq!(
            survivor
                .iter()
                .map(|output| output.token_id)
                .collect::<Vec<_>>(),
            vec![2, 3, 4, 5, 6]
        );
        // Re-enter the pipeline and interrupt it with an incoming prefill.
        scheduler
            .add_request(
                vec![0, 1],
                SamplingParams {
                    max_tokens: 4,
                    ignore_eos: true,
                    ..Default::default()
                },
            )
            .unwrap();
        scheduler.step_overlap().unwrap();
        assert!(scheduler.outstanding.is_some());
        scheduler
            .add_request(
                vec![4],
                SamplingParams {
                    max_tokens: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        let outputs = scheduler.step_overlap().unwrap();
        assert!(outputs.iter().all(|output| output.uid != 0));
        while !scheduler.is_idle() {
            scheduler.step_overlap().unwrap();
        }
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn throughput_policy_builds_mixed_prefill_batch() {
        let (mut scheduler, path) =
            scheduler_with_execution(CacheStrategy::Radix, 2, 16, Device::Cpu, false);
        scheduler
            .engine_mut()
            .attach_model_runner(ModelRunner::new(
                Box::new(RelayTokenModel::default()),
                Device::Cpu,
            ))
            .unwrap();
        scheduler.args.disable_consecutive_prefill_overlap = false;
        scheduler
            .add_request(
                vec![1, 2],
                SamplingParams {
                    max_tokens: 4,
                    ..Default::default()
                },
            )
            .unwrap();
        let mut outputs = scheduler.step_overlap().unwrap();
        scheduler
            .add_request(
                vec![4, 5],
                SamplingParams {
                    max_tokens: 2,
                    ..Default::default()
                },
            )
            .unwrap();
        outputs.extend(scheduler.step_overlap().unwrap());
        let pending = scheduler.outstanding.as_ref().unwrap();
        assert_eq!(pending.phase, BatchPhase::Prefill);
        assert_eq!(pending.request_ids.len(), 2);
        assert_eq!(pending.sample.fence().ready().unwrap(), true);
        while !scheduler.is_idle() {
            outputs.extend(scheduler.step_overlap().unwrap());
        }
        let mut rows = std::collections::BTreeMap::<_, Vec<_>>::new();
        for output in outputs {
            rows.entry(output.uid)
                .or_default()
                .push((output.token_id, output.finish_reason));
        }
        assert_eq!(
            rows[&0].iter().map(|(token, _)| *token).collect::<Vec<_>>(),
            vec![3, 4, 5, 6]
        );
        assert_eq!(
            rows[&1].iter().map(|(token, _)| *token).collect::<Vec<_>>(),
            vec![6, 7]
        );
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn chunked_prefill_emits_only_final_chunk_and_matches_decode() {
        let (mut scheduler, path) =
            scheduler_with_execution(CacheStrategy::Radix, 1, 16, Device::Cpu, false);
        scheduler.prefill.set_chunked_prefill_size(2);
        scheduler
            .engine_mut()
            .attach_model_runner(ModelRunner::new(
                Box::new(RelayTokenModel::default()),
                Device::Cpu,
            ))
            .unwrap();
        scheduler
            .add_request(
                vec![1, 2, 3, 4, 5],
                SamplingParams {
                    max_tokens: 2,
                    ..Default::default()
                },
            )
            .unwrap();

        let mut outputs = Vec::new();
        let mut steps = 0;
        while !scheduler.is_idle() || steps == 0 {
            outputs.extend(scheduler.step_overlap().unwrap());
            steps += 1;
            assert!(steps < 16);
        }
        assert!(
            (3..=5).contains(&steps),
            "three prefill chunks plus overlap completion barriers, got {steps}"
        );
        assert_eq!(
            outputs
                .iter()
                .map(|output| output.token_id)
                .collect::<Vec<_>>(),
            vec![6, 7]
        );
        assert_eq!(outputs.iter().filter(|output| output.finished).count(), 1);
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn chunked_prefill_can_mix_with_an_outstanding_decode() {
        let (mut scheduler, path) =
            scheduler_with_execution(CacheStrategy::Radix, 2, 16, Device::Cpu, false);
        scheduler.prefill.set_chunked_prefill_size(2);
        scheduler
            .engine_mut()
            .attach_model_runner(ModelRunner::new(
                Box::new(RelayTokenModel::default()),
                Device::Cpu,
            ))
            .unwrap();
        scheduler
            .add_request(
                vec![1],
                SamplingParams {
                    max_tokens: 4,
                    ignore_eos: true,
                    ..Default::default()
                },
            )
            .unwrap();
        scheduler.step_overlap().unwrap();
        assert!(scheduler.outstanding.is_some());
        scheduler
            .add_request(
                vec![1, 2, 3, 4, 5],
                SamplingParams {
                    max_tokens: 2,
                    ignore_eos: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let mut outputs = Vec::new();
        let mut steps = 0;
        while !scheduler.is_idle() {
            outputs.extend(scheduler.step_overlap().unwrap());
            steps += 1;
            assert!(steps < 32);
        }
        assert!(outputs.iter().any(|output| output.uid == 0));
        assert!(outputs.iter().any(|output| output.uid == 1));
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn terminal_output_and_cancel_hold_pages_until_their_fence() {
        for cancel in [false, true] {
            let (mut scheduler, path) = scheduler_with_options(CacheStrategy::Radix, 1, 16);
            scheduler
                .engine_mut()
                .attach_model_runner(ModelRunner::new(Box::new(FixedTokenModel(3)), Device::Cpu))
                .unwrap();
            scheduler
                .add_request(
                    vec![1, 2],
                    SamplingParams {
                        max_tokens: 8,
                        ..Default::default()
                    },
                )
                .unwrap();
            scheduler.step_overlap().unwrap();
            let (fence, ready) = crate::engine::CompletionFence::controlled();
            scheduler.outstanding.as_mut().unwrap().fence = fence;
            let free = scheduler.engine().kv_cache_pool().unwrap().free_count();
            let private_pages = scheduler
                .requests
                .running_request(0)
                .unwrap()
                .cache_handle
                .as_ref()
                .unwrap()
                .num_pages()
                - 1;
            let mut outputs = Vec::new();
            if cancel {
                assert!(scheduler.abort_request(0));
            } else {
                let input = scheduler.completion_input(0);
                scheduler.complete(
                    vec![0],
                    Some(vec![input]),
                    BatchPhase::Decode,
                    Ok(vec![7]),
                    &mut outputs,
                );
                assert_eq!(outputs.len(), 1);
                assert_eq!(outputs[0].finish_reason, Some(FinishReason::Stop));
            }
            assert!(scheduler.requests.running_request(0).is_none());
            assert_eq!(scheduler.retired.len(), 1);
            assert!(!ready.get()); // No blocking wait happened at logical finish.
            scheduler.reap_retired(false);
            assert_eq!(
                scheduler.engine().kv_cache_pool().unwrap().free_count(),
                free
            );
            assert_eq!(scheduler.retired.len(), 1);
            assert!(!scheduler.is_idle());
            // Detachment opens a request slot, but the old pages remain owned.
            scheduler
                .add_request(
                    vec![1, 2, 4],
                    SamplingParams {
                        max_tokens: 1,
                        ..Default::default()
                    },
                )
                .unwrap();
            let batch = scheduler
                .prefill
                .schedule_prefill(&mut scheduler.requests)
                .unwrap()
                .unwrap();
            assert_eq!(batch.request_ids, vec![1]);
            assert_eq!(scheduler.requests.running_request(1).unwrap().cached_len, 2);
            drop(batch);
            let free = scheduler.engine().kv_cache_pool().unwrap().free_count();
            ready.set(true);
            scheduler.reap_retired(false);
            assert!(scheduler.retired.is_empty());
            assert_eq!(
                scheduler.engine().kv_cache_pool().unwrap().free_count(),
                free + private_pages
            );
            assert!(scheduler.abort_request(1));
            let mut discarded = Vec::new();
            scheduler.drain_outstanding(&mut discarded);
            assert!(discarded.is_empty());
            assert!(scheduler.is_idle());
            fs::remove_dir_all(path).unwrap();
        }
    }

    #[test]
    fn deferred_publish_failure_releases_private_pages_without_second_terminal() {
        let (mut scheduler, path) = scheduler_with_options(CacheStrategy::Radix, 1, 16);
        scheduler
            .engine_mut()
            .attach_model_runner(ModelRunner::new(Box::new(FixedTokenModel(3)), Device::Cpu))
            .unwrap();
        let pool = scheduler.engine.shared_kv_cache_pool().unwrap();
        scheduler.requests = RequestManager::new(Box::new(FailSecondPublish {
            inner: RadixCacheManager::new(pool, 2).unwrap(),
            calls: 0,
        }));
        scheduler.prefill = PrefillManager::new(
            &scheduler.args,
            BatchContext::new(1, 16, 2, Device::Cpu).unwrap(),
        );
        scheduler
            .add_request(
                vec![1, 2],
                SamplingParams {
                    max_tokens: 8,
                    ..Default::default()
                },
            )
            .unwrap();
        scheduler.step_serial().unwrap();
        let batch = scheduler
            .decode
            .schedule_decode(scheduler.requests.running_requests())
            .unwrap()
            .unwrap();
        let mut pending = scheduler.launch_decode(batch, None).unwrap();
        let (fence, ready) = crate::engine::CompletionFence::controlled();
        pending.fence = fence;
        scheduler.outstanding = Some(pending);
        let free = scheduler.engine().kv_cache_pool().unwrap().free_count();
        let input = scheduler.completion_input(0);
        let mut outputs = Vec::new();
        scheduler.complete(
            vec![0],
            Some(vec![input]),
            BatchPhase::Decode,
            Ok(vec![7]),
            &mut outputs,
        );
        assert_eq!(outputs[0].finish_reason, Some(FinishReason::Stop));
        assert!(scheduler.last_step_error().is_none());
        ready.set(true);
        scheduler.reap_retired(false);
        assert!(matches!(
            scheduler.last_step_error(),
            Some(EngineError::KVCache(_))
        ));
        // Deferred publish installed a second canonical page before failing.
        // Release returns the three remaining private pages exactly once.
        assert_eq!(
            scheduler.engine().kv_cache_pool().unwrap().free_count(),
            free + 3
        );
        scheduler.drain_outstanding(&mut outputs);
        assert_eq!(outputs.len(), 1);
        assert!(scheduler.is_idle());
        fs::remove_dir_all(path).unwrap();
    }

    fn exercise_eos_and_prefill_switch(device: Device) {
        for cancel_prefill in [false, true] {
            let (mut scheduler, path) =
                scheduler_with_execution(CacheStrategy::Radix, 4, 32, device, false);
            scheduler
                .engine_mut()
                .attach_model_runner(ModelRunner::new(
                    Box::new(RelayTokenModel::default()),
                    device,
                ))
                .unwrap();
            scheduler
                .add_request(
                    vec![5],
                    SamplingParams {
                        max_tokens: 8,
                        ..Default::default()
                    },
                )
                .unwrap();
            scheduler
                .add_request(
                    vec![0, 1],
                    SamplingParams {
                        max_tokens: 6,
                        ignore_eos: true,
                        ..Default::default()
                    },
                )
                .unwrap();
            let mut outputs = scheduler.step_overlap().unwrap();
            let eos = scheduler.step_overlap().unwrap();
            assert_eq!(
                eos.iter()
                    .map(|row| (row.uid, row.token_id))
                    .collect::<Vec<_>>(),
                vec![(0, 7), (1, 3)]
            );
            assert_eq!(eos[0].finish_reason, Some(FinishReason::Stop));
            // EOS does not drain the next decode or commit its survivor early.
            assert_eq!(
                scheduler.outstanding.as_ref().unwrap().phase,
                BatchPhase::Decode
            );
            assert!(scheduler.requests.running_request(0).is_none());
            assert_eq!(
                *scheduler
                    .requests
                    .running_request(1)
                    .unwrap()
                    .input_ids
                    .last()
                    .unwrap(),
                3
            );
            outputs.extend(eos);
            scheduler
                .add_request(
                    vec![4],
                    SamplingParams {
                        max_tokens: 2,
                        ..Default::default()
                    },
                )
                .unwrap();
            let switching = scheduler.step_overlap().unwrap();
            // New prefill is in flight, while the old decode result is returned.
            assert_eq!(
                switching
                    .iter()
                    .map(|row| (row.uid, row.token_id))
                    .collect::<Vec<_>>(),
                vec![(1, 4)]
            );
            let pending = scheduler.outstanding.as_ref().unwrap();
            assert_eq!(pending.phase, BatchPhase::Prefill);
            assert_eq!(pending.request_ids, vec![2, 1]);
            assert_eq!(scheduler.requests.running_request(2).unwrap().output_len, 0);
            outputs.extend(switching);
            if cancel_prefill {
                assert!(scheduler.abort_request(2));
            }
            while !scheduler.is_idle() {
                outputs.extend(scheduler.step_overlap().unwrap());
            }
            assert_eq!(
                outputs
                    .iter()
                    .filter(|row| row.uid == 0)
                    .map(|row| row.token_id)
                    .collect::<Vec<_>>(),
                vec![6, 7]
            );
            assert_eq!(
                outputs
                    .iter()
                    .filter(|row| row.uid == 1)
                    .map(|row| row.token_id)
                    .collect::<Vec<_>>(),
                vec![2, 3, 4, 5, 6, 7]
            );
            let new = outputs
                .iter()
                .filter(|row| row.uid == 2)
                .map(|row| row.token_id)
                .collect::<Vec<_>>();
            assert_eq!(new, if cancel_prefill { vec![] } else { vec![5, 6] });
            assert!(scheduler.retired.is_empty());
            fs::remove_dir_all(path).unwrap();
        }
    }

    #[test]
    fn eos_survivors_and_incoming_prefill_keep_the_pipeline() {
        exercise_eos_and_prefill_switch(Device::Cpu);
    }

    #[cfg(has_cuda_graph)]
    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn cuda_eos_survivors_and_incoming_prefill_keep_the_pipeline() {
        exercise_eos_and_prefill_switch(Device::Cuda(0));
    }

    struct FailLaterDecode {
        calls: Cell<usize>,
    }
    impl ModelExecutor for FailLaterDecode {
        fn forward(
            &self,
            batch: &crate::engine::ForwardBatch<'_>,
        ) -> std::result::Result<crate::engine::ForwardOutput, ModelRunnerError> {
            if batch.phase() == BatchPhase::Decode {
                self.calls.set(self.calls.get() + 1);
                if self.calls.get() == 2 {
                    return Err(ModelRunnerError::Model("second decode failed".into()));
                }
            }
            RelayTokenModel::default().forward(batch)
        }
    }

    #[test]
    fn pipeline_next_forward_error_preserves_completed_output() {
        let (mut scheduler, path) = scheduler_with_options(CacheStrategy::Radix, 1, 16);
        scheduler
            .engine_mut()
            .attach_model_runner(ModelRunner::new(
                Box::new(FailLaterDecode {
                    calls: Cell::new(0),
                }),
                Device::Cpu,
            ))
            .unwrap();
        scheduler
            .add_request(
                vec![1, 2],
                SamplingParams {
                    max_tokens: 8,
                    ..Default::default()
                },
            )
            .unwrap();
        let first = scheduler.step_overlap().unwrap();
        assert_eq!(first[0].token_id, 3);
        assert!(scheduler.outstanding.is_some());
        let outputs = scheduler.step_overlap().unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].token_id, 4);
        assert_eq!(outputs[0].finish_reason, None);
        assert_eq!(outputs[1].finish_reason, Some(FinishReason::Error));
        assert!(scheduler.is_idle());
        assert!(matches!(
            scheduler.last_step_error(),
            Some(EngineError::ModelRunner(_))
        ));
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn one_batch_pipeline_relays_tokens_and_fences_lifecycle() {
        exercise_pipeline(Device::Cpu);
    }

    #[cfg(has_cuda_graph)]
    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn cuda_one_batch_pipeline_relays_tokens_and_fences_lifecycle() {
        exercise_pipeline(Device::Cuda(0));
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
    fn generated_prefix_collision_reclaims_the_duplicate_page() {
        let (mut scheduler, path) = scheduler_with_options(CacheStrategy::Radix, 2, 6);
        scheduler
            .engine_mut()
            .attach_model_runner(ModelRunner::new(Box::new(FixedTokenModel(3)), Device::Cpu))
            .unwrap();
        let total_pages = scheduler.engine().kv_cache_pool().unwrap().layout.num_pages;
        let a = scheduler
            .add_request(
                vec![1, 2],
                SamplingParams {
                    max_tokens: 3,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(scheduler.step().unwrap().len(), 2);
        assert_eq!(scheduler.running_len(), 1);

        let b = scheduler
            .add_request(
                vec![1, 2, 3, 3, 4],
                SamplingParams {
                    max_tokens: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        let results = scheduler.step().unwrap();
        assert_eq!(
            results
                .iter()
                .map(|output| (output.uid, output.finish_reason))
                .collect::<Vec<_>>(),
            vec![
                (b, Some(FinishReason::Length)),
                (a, Some(FinishReason::Length))
            ]
        );
        assert!(scheduler.is_idle());
        // The two complete token pages are canonical. A's duplicate output
        // page and B's partial tail page were returned exactly once.
        assert_eq!(
            scheduler.engine().kv_cache_pool().unwrap().free_count(),
            total_pages - 2
        );
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn abort_running_request_releases_only_its_private_pages() {
        let (mut scheduler, path) = scheduler_with_options(CacheStrategy::Radix, 1, 6);
        scheduler
            .engine_mut()
            .attach_model_runner(ModelRunner::new(Box::new(FixedTokenModel(3)), Device::Cpu))
            .unwrap();
        let total_pages = scheduler.engine.kv_cache_pool().unwrap().layout.num_pages;
        let uid = scheduler
            .add_request(
                vec![1, 2],
                SamplingParams {
                    max_tokens: 4,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(scheduler.step().unwrap().len(), 2);
        assert!(scheduler.abort_request(uid));
        assert!(!scheduler.abort_request(uid));
        assert!(scheduler.is_idle());
        // The prompt page was published; generation pages stayed private.
        assert_eq!(
            scheduler.engine.kv_cache_pool().unwrap().free_count(),
            total_pages - 1
        );
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn publish_failure_isolated_to_one_request() {
        let (mut scheduler, path) = scheduler_with_options(CacheStrategy::Radix, 2, 4);
        scheduler
            .engine_mut()
            .attach_model_runner(ModelRunner::new(Box::new(FixedTokenModel(3)), Device::Cpu))
            .unwrap();
        let pool = scheduler.engine.shared_kv_cache_pool().unwrap();
        let total_pages = pool.borrow().layout.num_pages;
        let cache = Box::new(FailSecondPublish {
            inner: RadixCacheManager::new(pool, 2).unwrap(),
            calls: 0,
        });
        let context = BatchContext::new(2, 4, 2, Device::Cpu).unwrap();
        scheduler.requests = RequestManager::new(cache);
        scheduler.prefill = PrefillManager::new(&scheduler.args, context);
        let first = scheduler
            .add_request(
                vec![1, 2],
                SamplingParams {
                    max_tokens: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        let second = scheduler
            .add_request(
                vec![3, 4],
                SamplingParams {
                    max_tokens: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        let results = scheduler.step().unwrap();
        assert_eq!(
            results
                .iter()
                .map(|out| (out.uid, out.finish_reason))
                .collect::<Vec<_>>(),
            vec![
                (first, Some(FinishReason::Length)),
                (second, Some(FinishReason::Error))
            ]
        );
        assert!(scheduler.is_idle());
        // Both valid pages reached the tree before the injected error; the
        // failed request's release must not free its newly published page.
        assert_eq!(
            scheduler.engine.kv_cache_pool().unwrap().free_count(),
            total_pages - 2
        );
        fs::remove_dir_all(path).unwrap();
    }
}
