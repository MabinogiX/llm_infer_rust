//! Pending queue, KV allocation, and prefill batch selection.

use crate::engine::kvcache::AcquireOutcome;
use crate::engine::{Batch, BatchContext, BatchRequestRef, MixedDecodeRequest, ServerArgs};

use super::{RequestId, Result, request_manager::RequestManager};

pub struct PrefillBatch {
    pub request_ids: Vec<RequestId>,
    pub model_batch: Batch,
    pub(crate) chunk: Option<PrefillChunk>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillChunk {
    pub(crate) uid: RequestId,
    pub(crate) end: usize,
    pub(crate) is_last: bool,
}

pub(crate) struct PrefillSelection {
    pub(crate) request_ids: Vec<RequestId>,
    pub(crate) token_count: usize,
    pub(crate) chunk: Option<PrefillChunk>,
}

pub struct PrefillManager {
    batch_context: BatchContext,
    max_running_req: usize,
    max_seq_len: usize,
    max_prefill_tokens_per_batch: usize,
    chunked_prefill_size: usize,
}

impl PrefillManager {
    pub fn new(args: &ServerArgs, batch_context: BatchContext) -> Self {
        Self {
            batch_context,
            max_running_req: args.max_running_req,
            max_seq_len: args.max_seq_len,
            max_prefill_tokens_per_batch: args.max_prefill_tokens_per_batch,
            chunked_prefill_size: args.chunked_prefill_size,
        }
    }

    pub(crate) fn set_metadata_requirements(
        &mut self,
        requirements: crate::engine::MetadataRequirements,
    ) {
        self.batch_context = self.batch_context.with_metadata_requirements(requirements);
    }

    #[cfg(test)]
    pub(crate) fn set_chunked_prefill_size(&mut self, size: usize) {
        self.chunked_prefill_size = size;
    }

    pub(crate) fn schedule_prefill(
        &mut self,
        requests: &mut RequestManager,
    ) -> Result<Option<PrefillBatch>> {
        let Some(selection) = self.schedule_prefill_selection(requests)? else {
            return Ok(None);
        };
        let model_batch = match self.build_prefill(requests, &selection) {
            Ok(batch) => batch,
            Err(error) => {
                self.rollback_selection(requests, &selection.request_ids);
                return Err(error);
            }
        };
        Ok(Some(PrefillBatch {
            request_ids: selection.request_ids,
            model_batch,
            chunk: selection.chunk,
        }))
    }

    pub(crate) fn schedule_prefill_selection(
        &mut self,
        requests: &mut RequestManager,
    ) -> Result<Option<PrefillSelection>> {
        if let Some(selection) = self.select_continuation(requests)? {
            if selection.chunk.is_some_and(|chunk| !chunk.is_last) {
                return Ok(Some(selection));
            }
            let mut extra_ids = Vec::new();
            let extra = match self.select_requests(
                requests,
                &mut extra_ids,
                selection.token_count,
                false,
            ) {
                Ok(extra) => extra,
                Err(error) => {
                    self.rollback_selection(requests, &extra_ids);
                    return Err(error);
                }
            };
            let Some(extra) = extra else {
                return Ok(Some(selection));
            };
            let mut request_ids = selection.request_ids;
            request_ids.extend(extra.request_ids);
            return Ok(Some(PrefillSelection {
                request_ids,
                token_count: selection.token_count + extra.token_count,
                chunk: selection.chunk,
            }));
        }
        let mut selected = Vec::new();
        let result = self.select_requests(requests, &mut selected, 0, true);
        if result.is_err() {
            self.rollback_selection(requests, &selected);
        }
        result
    }

    pub(crate) fn rollback_selection(&self, requests: &mut RequestManager, selected: &[RequestId]) {
        let mut restored = Vec::new();
        for uid in selected {
            if let Some(request) = requests.remove_running(*uid) {
                restored.push(request);
            }
        }
        for request in restored.into_iter().rev() {
            requests.restore_waiting(request);
        }
    }

    pub(crate) fn prepare_mixed_prefill(
        &self,
        requests: &RequestManager,
        prefill_ids: &[RequestId],
        decode_rows: &[MixedDecodeRequest],
        chunk: Option<PrefillChunk>,
    ) -> Result<Batch> {
        let prefills = prefill_ids
            .iter()
            .map(|&uid| {
                let request = requests
                    .batch_request(uid)
                    .expect("scheduled prefill request is running");
                let chunk = chunk.filter(|chunk| chunk.uid == uid);
                BatchRequestRef {
                    cached_len: chunk.map_or(request.cached_len, |_| {
                        requests
                            .running_request(uid)
                            .unwrap()
                            .prefill_cursor()
                            .expect("分块请求必须处于 prefill 阶段")
                    }),
                    end_len: chunk.map(|chunk| chunk.end),
                    ..request
                }
            })
            .collect::<Vec<_>>();
        Ok(self
            .batch_context
            .prepare_mixed_prefill_borrowed(&prefills, decode_rows)?)
    }

    fn select_requests(
        &mut self,
        requests: &mut RequestManager,
        selected: &mut Vec<RequestId>,
        mut total_tokens: usize,
        allow_chunk: bool,
    ) -> Result<Option<PrefillSelection>> {
        // Chunk size is an aggregate prefill-token budget, not a limit on
        // request count. Cached prefixes do not consume either token budget.
        let token_limit = [self.max_prefill_tokens_per_batch, self.chunked_prefill_size]
            .into_iter()
            .filter(|limit| *limit > 0)
            .min();
        while requests.running_len() < self.max_running_req {
            let Some(request) = requests.pending_front() else {
                break;
            };
            if request.input_ids.is_empty() {
                let request = requests.pop_pending().expect("front exists");
                requests.push_aborted(request.uid);
                continue;
            }

            let remaining = token_limit.map(|limit| limit.saturating_sub(total_tokens));
            if remaining == Some(0) {
                break;
            }

            let capacity_tokens = request
                .input_ids
                .len()
                .saturating_add(request.sampling_params.max_tokens.saturating_sub(1))
                .min(self.max_seq_len);
            let prompt_len = request.input_ids.len();
            // A new chunk may use just the remaining budget. When finishing
            // an existing chunk, only admit whole requests: selection tracks
            // at most one chunk's completion boundary.
            let budget = if (allow_chunk && self.chunked_prefill_size > 0)
                || (selected.is_empty() && total_tokens == 0)
            {
                None
            } else {
                remaining
            };
            let handle = match requests.acquire_pending(capacity_tokens, budget)? {
                AcquireOutcome::Ready(handle) => handle,
                AcquireOutcome::Impossible => {
                    let request = requests.pop_pending().expect("front exists");
                    requests.push_aborted(request.uid);
                    continue;
                }
                AcquireOutcome::DeferredBudget | AcquireOutcome::DeferredMemory => break,
            };
            let matched_len = handle.cached_len;
            let uncached = prompt_len.checked_sub(matched_len).ok_or_else(|| {
                super::SchedulerError::InvalidDecode(
                    "cache matched length exceeds prompt length".to_owned(),
                )
            })?;
            let mut request = requests.pop_pending().expect("front exists");
            request.begin_prefill(handle);
            let chunk_len = if allow_chunk && self.chunked_prefill_size > 0 {
                uncached.min(remaining.expect("chunking has a positive token budget"))
            } else {
                uncached
            };
            let chunk = (chunk_len < uncached).then_some(PrefillChunk {
                uid: request.uid,
                end: matched_len + chunk_len,
                is_last: false,
            });
            selected.push(request.uid);
            requests.add_running(request);
            total_tokens += chunk_len;
            if chunk.is_some() {
                return Ok(Some(PrefillSelection {
                    request_ids: std::mem::take(selected),
                    token_count: total_tokens,
                    chunk,
                }));
            }
        }

        if selected.is_empty() {
            return Ok(None);
        }
        Ok(Some(PrefillSelection {
            request_ids: std::mem::take(selected),
            token_count: total_tokens,
            chunk: None,
        }))
    }

    fn select_continuation(
        &self,
        requests: &mut RequestManager,
    ) -> Result<Option<PrefillSelection>> {
        let Some(chunk_size) = (self.chunked_prefill_size > 0).then_some(self.chunked_prefill_size)
        else {
            return Ok(None);
        };
        let Some(request) = requests.running_requests().iter().find(|request| {
            request
                .prefill_cursor()
                .is_some_and(|cursor| cursor < request.prompt_len())
        }) else {
            return Ok(None);
        };
        let prompt_len = request.prompt_len();
        let cursor = request
            .prefill_cursor()
            .expect("续块请求必须处于 prefill 阶段");
        let budget_limit = if self.max_prefill_tokens_per_batch == 0 {
            chunk_size
        } else {
            self.max_prefill_tokens_per_batch
        };
        let end = cursor
            .saturating_add(chunk_size.min(budget_limit))
            .min(prompt_len);
        let selection = PrefillSelection {
            request_ids: vec![request.uid],
            token_count: end - cursor,
            chunk: Some(PrefillChunk {
                uid: request.uid,
                end,
                is_last: end == prompt_len,
            }),
        };
        Ok(Some(selection))
    }

    pub(crate) fn build_prefill(
        &self,
        requests: &RequestManager,
        selection: &PrefillSelection,
    ) -> Result<Batch> {
        let batch_requests = selection
            .request_ids
            .iter()
            .map(|uid| {
                let request = requests
                    .running_request(*uid)
                    .expect("selected request is running");
                let chunk = selection.chunk.filter(|chunk| chunk.uid == *uid);
                BatchRequestRef {
                    input_ids: &request.input_ids,
                    cached_len: chunk.map_or(request.cached_len, |_| {
                        request
                            .prefill_cursor()
                            .expect("分块请求必须处于 prefill 阶段")
                    }),
                    end_len: chunk.map(|chunk| chunk.end),
                    page_ids: &request
                        .cache_handle
                        .as_ref()
                        .expect("selected request has a handle")
                        .page_ids,
                }
            })
            .collect::<Vec<_>>();
        let model_batch = self
            .batch_context
            .prepare_mixed_prefill_borrowed(&batch_requests, &[])?;
        for uid in selection.request_ids.iter().copied() {
            let request = requests
                .running_request(uid)
                .expect("selected request is running");
            tracing::info!(
                request_id = uid,
                cached_tokens = request.cached_len,
                prompt_tokens = request.input_ids.len(),
                "prefill KV cache match"
            );
        }
        Ok(model_batch)
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use tch::Device;

    use crate::engine::SamplingParams;
    use crate::engine::kvcache::{KVCacheLayout, KVCachePool, RadixCacheManager};
    use crate::scheduler::{Request, SequenceStatus};

    use super::*;
    use crate::scheduler::request_manager::RequestManager;

    struct TestManager {
        requests: RequestManager,
        prefill: PrefillManager,
    }

    impl TestManager {
        fn add_request(&mut self, request: Request) {
            self.requests.add_request(request);
        }

        fn schedule_prefill(&mut self) -> Result<Option<PrefillBatch>> {
            self.prefill.schedule_prefill(&mut self.requests)
        }

        fn pending_len(&self) -> usize {
            self.requests.pending_len()
        }

        fn running_len(&self) -> usize {
            self.requests.running_len()
        }

        fn running_request(&self, uid: RequestId) -> Option<&Request> {
            self.requests.running_request(uid)
        }

        fn running_request_mut(&mut self, uid: RequestId) -> Option<&mut Request> {
            self.requests.running_request_mut(uid)
        }

        fn remove_batch(&mut self, ids: &[RequestId]) {
            self.requests.remove_batch(ids);
        }

        fn publish(&mut self, uid: RequestId) -> crate::engine::kvcache::Result<()> {
            self.requests.publish(uid)
        }

        fn drain_aborted(&mut self) -> Vec<RequestId> {
            self.requests.drain_aborted()
        }

        fn pending_ids(&self) -> Vec<RequestId> {
            self.requests.pending_ids()
        }
    }

    fn manager(
        num_pages: usize,
        max_running_req: usize,
        max_seq_len: usize,
    ) -> (Rc<RefCell<KVCachePool>>, TestManager) {
        let page_size = 2;
        let pool = Rc::new(RefCell::new(KVCachePool::without_tensor(
            KVCacheLayout::new(1, num_pages, page_size, 1, 1).unwrap(),
        )));
        let cache = Box::new(RadixCacheManager::new(pool.clone(), page_size).unwrap());
        let mut args = ServerArgs::new("unused-model-path");
        args.max_running_req = max_running_req;
        args.max_seq_len = max_seq_len;
        args.max_prefill_tokens_per_batch = max_seq_len;
        args.page_size = page_size;
        let context =
            BatchContext::new(max_running_req, max_seq_len, page_size, Device::Cpu).unwrap();
        let manager = TestManager {
            requests: RequestManager::new(cache),
            prefill: PrefillManager::new(&args, context),
        };
        (pool, manager)
    }

    fn request(uid: RequestId, input_ids: Vec<i64>, max_tokens: usize) -> Request {
        Request {
            uid,
            input_ids,
            sampling_params: SamplingParams {
                max_tokens,
                ..Default::default()
            },
            cached_len: 0,
            output_len: 0,
            cache_handle: None,
            status: SequenceStatus::Waiting,
        }
    }

    #[test]
    fn chunked_prefill_batches_short_requests() {
        let (_pool, mut manager) = manager(16, 4, 16);
        manager.prefill.set_chunked_prefill_size(8);
        manager.prefill.max_prefill_tokens_per_batch = 0;
        for uid in 0..4 {
            manager.add_request(request(uid, vec![uid as i64 + 1; 2], 1));
        }
        let batch = manager.schedule_prefill().unwrap().unwrap();
        assert_eq!(batch.request_ids, vec![0, 1, 2, 3]);
        assert_eq!(batch.model_batch.input_ids.size(), vec![8]);
        assert!(batch.chunk.is_none());
        assert_eq!(manager.pending_len(), 0);
    }

    #[test]
    fn chunked_prefill_uses_remaining_batch_budget_for_long_request() {
        let (_pool, mut manager) = manager(16, 4, 16);
        manager.prefill.set_chunked_prefill_size(8);
        manager.prefill.max_prefill_tokens_per_batch = 6;
        manager.add_request(request(0, vec![1, 2], 1));
        manager.add_request(request(1, vec![3; 9], 1));
        manager.add_request(request(2, vec![4, 5], 1));
        let batch = manager.schedule_prefill().unwrap().unwrap();
        assert_eq!(batch.request_ids, vec![0, 1]);
        assert_eq!(batch.model_batch.input_ids.size(), vec![6]);
        let chunk = batch.chunk.unwrap();
        assert_eq!((chunk.uid, chunk.end, chunk.is_last), (1, 4, false));
        assert_eq!(manager.pending_ids(), vec![2]);
    }

    #[test]
    fn chunked_prefill_budget_counts_uncached_tokens() {
        let (_pool, mut manager) = manager(16, 3, 16);
        manager.add_request(request(0, vec![1, 2, 3, 4], 1));
        manager.schedule_prefill().unwrap().unwrap();
        manager.publish(0).unwrap();
        manager.remove_batch(&[0]);
        manager.prefill.set_chunked_prefill_size(4);
        manager.add_request(request(1, vec![1, 2, 3, 4, 5, 6], 1));
        manager.add_request(request(2, vec![7, 8], 1));
        manager.add_request(request(3, vec![9], 1));
        let batch = manager.schedule_prefill().unwrap().unwrap();
        assert_eq!(batch.request_ids, vec![1, 2]);
        assert_eq!(batch.model_batch.input_ids.size(), vec![4]);
        assert_eq!(manager.running_request(1).unwrap().cached_len, 4);
        assert!(batch.chunk.is_none());
        assert_eq!(manager.pending_ids(), vec![3]);
    }

    #[test]
    fn final_chunk_only_admits_short_requests_within_remaining_budget() {
        let (_pool, mut manager) = manager(24, 4, 16);
        manager.prefill.set_chunked_prefill_size(4);
        manager.add_request(request(0, vec![1; 6], 1));
        let first = manager.schedule_prefill().unwrap().unwrap();
        assert_eq!(first.chunk.unwrap().end, 4);
        manager.requests.publish_prefix(0, 4).unwrap();
        manager.running_request_mut(0).unwrap().advance_prefill(4);
        manager.add_request(request(1, vec![2, 3], 1));
        manager.add_request(request(2, vec![4], 1));
        let batch = manager.schedule_prefill().unwrap().unwrap();
        assert_eq!(batch.request_ids, vec![0, 1]);
        assert_eq!(batch.model_batch.input_ids.size(), vec![4]);
        let chunk = batch.chunk.unwrap();
        assert_eq!((chunk.uid, chunk.end, chunk.is_last), (0, 6, true));
        assert_eq!(manager.pending_ids(), vec![2]);
    }

    #[test]
    fn final_chunk_defers_a_second_unfinished_request() {
        let (_pool, mut manager) = manager(24, 3, 16);
        manager.prefill.set_chunked_prefill_size(4);
        manager.add_request(request(0, vec![1; 5], 1));
        manager.schedule_prefill().unwrap().unwrap();
        manager.requests.publish_prefix(0, 4).unwrap();
        manager.running_request_mut(0).unwrap().advance_prefill(4);
        manager.add_request(request(1, vec![2; 7], 1));
        manager.add_request(request(2, vec![3], 1));
        let batch = manager.schedule_prefill().unwrap().unwrap();
        assert_eq!(batch.request_ids, vec![0]);
        assert_eq!(batch.model_batch.input_ids.size(), vec![1]);
        assert!(batch.chunk.unwrap().is_last);
        assert_eq!(manager.pending_ids(), vec![1, 2]);
        assert_eq!(manager.running_len(), 1);
    }

    #[test]
    fn respects_running_limit_and_keeps_fifo_request_pending() {
        let (_pool, mut manager) = manager(6, 1, 4);
        manager.add_request(request(0, vec![1, 2], 1));
        manager.add_request(request(1, vec![3, 4], 1));
        let first = manager.schedule_prefill().unwrap().unwrap();
        assert_eq!(first.request_ids, vec![0]);
        assert_eq!(first.model_batch.input_ids.size(), vec![2]);
        assert_eq!(manager.pending_len(), 1);
        assert!(manager.schedule_prefill().unwrap().is_none());
        manager.remove_batch(&[0]);
        assert_eq!(
            manager.schedule_prefill().unwrap().unwrap().request_ids,
            vec![1]
        );
    }

    #[test]
    fn respects_prefill_token_budget() {
        let (_pool, mut manager) = manager(8, 2, 4);
        manager.add_request(request(0, vec![1, 2, 3], 1));
        manager.add_request(request(1, vec![4, 5, 6], 1));
        assert_eq!(
            manager.schedule_prefill().unwrap().unwrap().request_ids,
            vec![0]
        );
        assert_eq!(manager.pending_len(), 1);
    }

    #[test]
    fn uses_independent_prefill_token_budget() {
        let (_pool, mut manager) = manager(8, 2, 8);
        manager.prefill.max_prefill_tokens_per_batch = 4;
        manager.add_request(request(0, vec![1, 2, 3], 1));
        manager.add_request(request(1, vec![4, 5, 6], 1));
        assert_eq!(
            manager.schedule_prefill().unwrap().unwrap().request_ids,
            vec![0]
        );
        assert_eq!(manager.pending_len(), 1);
    }

    #[test]
    fn aborts_request_that_can_never_fit_in_pool() {
        let (_pool, mut manager) = manager(1, 1, 4);
        manager.add_request(request(9, vec![1, 2, 3], 1));
        assert!(manager.schedule_prefill().unwrap().is_none());
        assert_eq!(manager.drain_aborted(), vec![9]);
        assert_eq!(manager.pending_len(), 0);
    }

    #[test]
    fn rematches_after_eviction_without_duplicate_page_ids() {
        let (_pool, mut manager) = manager(2, 1, 4);
        manager.add_request(request(0, vec![1, 2, 3], 1));
        manager.schedule_prefill().unwrap().unwrap();
        manager.remove_batch(&[0]);
        manager.add_request(request(1, vec![1, 2, 4], 1));
        let batch = manager.schedule_prefill().unwrap().unwrap();
        assert_eq!(batch.request_ids, vec![1]);
        let handle = manager
            .running_request(1)
            .unwrap()
            .cache_handle
            .as_ref()
            .unwrap();
        let unique: std::collections::BTreeSet<_> = handle.page_ids.iter().collect();
        assert_eq!(unique.len(), handle.page_ids.len());
    }

    #[test]
    fn completed_prefill_does_not_cache_unwritten_sampled_token() {
        let (_pool, mut manager) = manager(3, 1, 6);
        manager.add_request(request(0, vec![1, 2], 1));
        manager.schedule_prefill().unwrap().unwrap();
        manager.publish(0).unwrap();
        manager.running_request_mut(0).unwrap().append_token(7);
        manager.remove_batch(&[0]);

        manager.add_request(request(1, vec![1, 2, 7, 4], 1));
        manager.schedule_prefill().unwrap().unwrap();
        assert_eq!(manager.running_request(1).unwrap().cached_len, 2);
    }

    #[test]
    fn capacity_excludes_the_last_sampled_token() {
        let (pool, mut manager) = manager(1, 1, 3);
        manager.add_request(request(0, vec![1, 2], 1));
        assert_eq!(
            manager.schedule_prefill().unwrap().unwrap().request_ids,
            vec![0]
        );
        assert_eq!(pool.borrow().free_count(), 0);
        assert!(manager.drain_aborted().is_empty());
        manager.remove_batch(&[0]);
        assert_eq!(pool.borrow().free_count(), 1);
    }

    #[test]
    fn batch_preparation_error_restores_order_and_all_pages() {
        let page_size = 2;
        let pool = Rc::new(RefCell::new(KVCachePool::without_tensor(
            KVCacheLayout::new(1, 4, page_size, 1, 1).unwrap(),
        )));
        let cache = Box::new(RadixCacheManager::new(pool.clone(), page_size).unwrap());
        let mut args = ServerArgs::new("unused-model-path");
        args.max_running_req = 2;
        args.max_seq_len = 4;
        args.page_size = page_size;
        args.chunked_prefill_size = 4;
        // The narrower context rejects two selected requests after acquire.
        let context = BatchContext::new(1, 4, page_size, Device::Cpu).unwrap();
        let mut manager = TestManager {
            requests: RequestManager::new(cache),
            prefill: PrefillManager::new(&args, context),
        };
        manager.add_request(request(0, vec![1], 1));
        manager.add_request(request(1, vec![2], 1));
        assert!(manager.schedule_prefill().is_err());
        assert_eq!(manager.running_len(), 0);
        assert_eq!(pool.borrow().free_count(), 4);
        assert_eq!(manager.pending_ids(), vec![0, 1]);
        let request = manager.requests.pending_front().unwrap();
        assert_eq!(request.status, SequenceStatus::Waiting);
        assert_eq!(request.prefill_cursor(), None);
        assert!(request.cache_handle.is_none());
        // 回退后仍能正常重新准入，且从初始游标开始。
        manager.prefill.max_running_req = 1;
        let batch = manager.schedule_prefill().unwrap().unwrap();
        assert_eq!(batch.request_ids, vec![0]);
        assert_eq!(
            manager.running_request(0).unwrap().prefill_cursor(),
            Some(0)
        );
    }
}
