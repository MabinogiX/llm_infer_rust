//! Pending queue, KV allocation, and prefill batch selection.

use std::collections::VecDeque;

use crate::engine::kvcache::{AcquireOutcome, CacheManager};
use crate::engine::{Batch, BatchContext, BatchRequest, ServerArgs};

use super::{Request, RequestId, Result, SequenceStatus};

pub struct PrefillBatch {
    pub request_ids: Vec<RequestId>,
    pub model_batch: Batch,
}

pub struct PrefillManager {
    pending: VecDeque<Request>,
    running: Vec<Request>,
    aborted: Vec<RequestId>,
    cache: Box<dyn CacheManager>,
    batch_context: BatchContext,
    max_running_req: usize,
    max_seq_len: usize,
}

impl PrefillManager {
    pub fn new(
        args: &ServerArgs,
        cache: Box<dyn CacheManager>,
        batch_context: BatchContext,
    ) -> Self {
        Self {
            pending: VecDeque::new(),
            running: Vec::new(),
            aborted: Vec::new(),
            cache,
            batch_context,
            max_running_req: args.max_running_req,
            max_seq_len: args.max_seq_len,
        }
    }

    pub fn add_request(&mut self, request: Request) {
        self.pending.push_back(request);
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    pub fn running_requests(&self) -> &[Request] {
        &self.running
    }

    pub fn has_aborted(&self) -> bool {
        !self.aborted.is_empty()
    }

    pub fn drain_aborted(&mut self) -> Vec<RequestId> {
        std::mem::take(&mut self.aborted)
    }

    pub fn running_request(&self, uid: RequestId) -> Option<&Request> {
        self.running.iter().find(|request| request.uid == uid)
    }

    pub fn running_request_mut(&mut self, uid: RequestId) -> Option<&mut Request> {
        self.running.iter_mut().find(|request| request.uid == uid)
    }

    pub fn schedule_prefill(&mut self) -> Result<Option<PrefillBatch>> {
        let mut selected = Vec::new();
        let result = self.select_and_build(&mut selected);
        if result.is_err() {
            // A preparation error must not make accepted requests disappear.
            // Undo allocations and put this step's selections back at the
            // head of the queue in their original order.
            let mut restored = Vec::new();
            for uid in &selected {
                if let Some(index) = self.running.iter().position(|request| request.uid == *uid) {
                    let mut request = self.running.remove(index);
                    if let Some(mut handle) = request.cache_handle.take() {
                        self.cache.release(&mut handle);
                    }
                    request.cached_len = 0;
                    request.status = SequenceStatus::Waiting;
                    restored.push(request);
                }
            }
            for request in restored.into_iter().rev() {
                self.pending.push_front(request);
            }
        }
        result
    }

    fn select_and_build(&mut self, selected: &mut Vec<RequestId>) -> Result<Option<PrefillBatch>> {
        let mut total_tokens = 0usize;
        while self.running.len() < self.max_running_req {
            let Some(request) = self.pending.front() else {
                break;
            };
            if request.input_ids.is_empty() {
                let request = self.pending.pop_front().expect("front exists");
                self.aborted.push(request.uid);
                continue;
            }

            let capacity_tokens = request
                .input_ids
                .len()
                .saturating_add(request.sampling_params.max_tokens.saturating_sub(1))
                .min(self.max_seq_len);
            let budget = (!selected.is_empty()).then_some(self.max_seq_len - total_tokens);
            let handle = match self
                .cache
                .acquire(&request.input_ids, capacity_tokens, budget)?
            {
                AcquireOutcome::Ready(handle) => handle,
                AcquireOutcome::Impossible => {
                    let request = self.pending.pop_front().expect("front exists");
                    self.aborted.push(request.uid);
                    continue;
                }
                AcquireOutcome::DeferredBudget | AcquireOutcome::DeferredMemory => break,
            };
            let matched_len = handle.cached_len;
            let uncached = request.input_ids.len() - matched_len;
            let mut request = self.pending.pop_front().expect("front exists");
            request.cached_len = matched_len;
            request.cache_handle = Some(handle);
            request.status = SequenceStatus::Running;
            selected.push(request.uid);
            self.running.push(request);
            total_tokens += uncached;
        }

        if selected.is_empty() {
            return Ok(None);
        }
        let requests = selected
            .iter()
            .map(|uid| {
                let request = self
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
        let model_batch = self.batch_context.prepare_prefill(&requests)?;
        for uid in selected.iter().copied() {
            let request = self
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

    pub fn abort(&mut self, uid: RequestId) -> bool {
        if let Some(index) = self.pending.iter().position(|request| request.uid == uid) {
            self.pending.remove(index);
            return true;
        }
        if let Some(index) = self.running.iter().position(|request| request.uid == uid) {
            let mut request = self.running.remove(index);
            if let Some(mut handle) = request.cache_handle.take() {
                self.cache.release(&mut handle);
            }
            return true;
        }
        false
    }

    pub fn remove_batch(&mut self, ids: &[RequestId]) {
        for uid in ids {
            if let Some(index) = self.running.iter().position(|request| request.uid == *uid) {
                let mut request = self.running.remove(index);
                if let Some(mut handle) = request.cache_handle.take() {
                    self.cache.release(&mut handle);
                }
            }
        }
    }

    pub fn publish(
        &mut self,
        uid: RequestId,
        written_ids: &[i64],
    ) -> crate::engine::kvcache::Result<()> {
        let request = self
            .running
            .iter_mut()
            .find(|request| request.uid == uid)
            .expect("scheduled request is running");
        let handle = request
            .cache_handle
            .as_mut()
            .expect("running request has KV handle");
        self.cache.publish(handle, written_ids)
    }

    pub fn mark_written(&mut self, uid: RequestId, written_len: usize) {
        let request = self
            .running_request_mut(uid)
            .expect("scheduled request is running");
        request
            .cache_handle
            .as_mut()
            .expect("running request has KV handle")
            .written_len = written_len;
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use tch::Device;

    use crate::engine::SamplingParams;
    use crate::engine::kvcache::{KVCacheLayout, KVCachePool, RadixCacheManager};

    use super::*;

    fn manager(
        num_pages: usize,
        max_running_req: usize,
        max_seq_len: usize,
    ) -> (Rc<RefCell<KVCachePool>>, PrefillManager) {
        let page_size = 2;
        let pool = Rc::new(RefCell::new(KVCachePool::without_tensor(
            KVCacheLayout::new(1, num_pages, page_size, 1, 1).unwrap(),
        )));
        let cache = Box::new(RadixCacheManager::new(pool.clone(), page_size).unwrap());
        let mut args = ServerArgs::new("unused-model-path");
        args.max_running_req = max_running_req;
        args.max_seq_len = max_seq_len;
        args.page_size = page_size;
        let context =
            BatchContext::new(max_running_req, max_seq_len, page_size, Device::Cpu).unwrap();
        let manager = PrefillManager::new(&args, cache, context);
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
        manager.publish(0, &[1, 2]).unwrap();
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
        let mut manager = PrefillManager::new(&args, cache, context);
        manager.add_request(request(0, vec![1], 1));
        manager.add_request(request(1, vec![2], 1));
        assert!(manager.schedule_prefill().is_err());
        assert_eq!(manager.running_len(), 0);
        assert_eq!(pool.borrow().free_count(), 4);
        assert_eq!(
            manager
                .pending
                .iter()
                .map(|req| req.uid)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }
}
