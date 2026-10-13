//! Pending queue, KV allocation, and prefill batch selection.

use crate::engine::kvcache::AcquireOutcome;
use crate::engine::{Batch, BatchContext, BatchRequest, MixedDecodeRequest, ServerArgs};

use super::{RequestId, Result, SequenceStatus, request_manager::RequestManager};

pub struct PrefillBatch {
    pub request_ids: Vec<RequestId>,
    pub model_batch: Batch,
}

pub struct PrefillManager {
    batch_context: BatchContext,
    max_running_req: usize,
    max_seq_len: usize,
    max_prefill_tokens_per_batch: usize,
}

impl PrefillManager {
    pub fn new(args: &ServerArgs, batch_context: BatchContext) -> Self {
        Self {
            batch_context,
            max_running_req: args.max_running_req,
            max_seq_len: args.max_seq_len,
            max_prefill_tokens_per_batch: args.max_prefill_tokens_per_batch,
        }
    }

    pub(crate) fn schedule_prefill(
        &mut self,
        requests: &mut RequestManager,
    ) -> Result<Option<PrefillBatch>> {
        let mut selected = Vec::new();
        let result = self.select_and_build(requests, &mut selected);
        if result.is_err() {
            // A preparation error must not make accepted requests disappear.
            // Undo allocations and put this step's selections back at the
            // head of the queue in their original order.
            let mut restored = Vec::new();
            for uid in &selected {
                if let Some(request) = requests.remove_running(*uid) {
                    restored.push(request);
                }
            }
            for request in restored.into_iter().rev() {
                requests.restore_waiting(request);
            }
        }
        result
    }

    pub(crate) fn prepare_mixed_prefill(
        &self,
        requests: &RequestManager,
        prefill_ids: &[RequestId],
        decode_rows: &[MixedDecodeRequest],
    ) -> Result<Batch> {
        let prefills = prefill_ids
            .iter()
            .map(|&uid| {
                requests
                    .batch_request(uid)
                    .expect("scheduled prefill request is running")
            })
            .collect::<Vec<_>>();
        Ok(self
            .batch_context
            .prepare_mixed_prefill(&prefills, decode_rows)?)
    }

    fn select_and_build(
        &mut self,
        requests: &mut RequestManager,
        selected: &mut Vec<RequestId>,
    ) -> Result<Option<PrefillBatch>> {
        let mut total_tokens = 0usize;
        while requests.running_len() < self.max_running_req {
            let Some(request) = requests.pending_front() else {
                break;
            };
            if request.input_ids.is_empty() {
                let request = requests.pop_pending().expect("front exists");
                requests.push_aborted(request.uid);
                continue;
            }

            let capacity_tokens = request
                .input_ids
                .len()
                .saturating_add(request.sampling_params.max_tokens.saturating_sub(1))
                .min(self.max_seq_len);
            let prompt_len = request.input_ids.len();
            let budget = if selected.is_empty() || self.max_prefill_tokens_per_batch == 0 {
                None
            } else {
                Some(
                    self.max_prefill_tokens_per_batch
                        .saturating_sub(total_tokens),
                )
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
            let uncached = prompt_len - matched_len;
            let mut request = requests.pop_pending().expect("front exists");
            request.cached_len = matched_len;
            request.cache_handle = Some(handle);
            request.status = SequenceStatus::Running;
            selected.push(request.uid);
            requests.add_running(request);
            total_tokens += uncached;
        }

        if selected.is_empty() {
            return Ok(None);
        }
        let batch_requests = selected
            .iter()
            .map(|uid| {
                let request = requests
                    .running_request(*uid)
                    .expect("selected request is running");
                BatchRequest {
                    input_ids: request.input_ids.clone(),
                    cached_len: request.cached_len,
                    page_ids: request
                        .cache_handle
                        .as_ref()
                        .expect("selected request has a handle")
                        .page_ids
                        .clone(),
                }
            })
            .collect::<Vec<_>>();
        let model_batch = self.batch_context.prepare_prefill(&batch_requests)?;
        for uid in selected.iter().copied() {
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
        Ok(Some(PrefillBatch {
            request_ids: std::mem::take(selected),
            model_batch,
        }))
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
    }
}
