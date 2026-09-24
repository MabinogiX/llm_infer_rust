//! Pending queue, KV allocation, and prefill batch selection.

use std::{cell::RefCell, collections::VecDeque, rc::Rc};

use crate::engine::kvcache::{CacheManager, KVCachePool};
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
    pool: Rc<RefCell<KVCachePool>>,
    cache: Box<dyn CacheManager>,
    batch_context: BatchContext,
    max_running_req: usize,
    max_seq_len: usize,
    page_size: usize,
}

impl PrefillManager {
    pub fn new(
        args: &ServerArgs,
        pool: Rc<RefCell<KVCachePool>>,
        cache: Box<dyn CacheManager>,
        batch_context: BatchContext,
    ) -> Self {
        Self {
            pending: VecDeque::new(),
            running: Vec::new(),
            aborted: Vec::new(),
            pool,
            cache,
            batch_context,
            max_running_req: args.max_running_req,
            max_seq_len: args.max_seq_len,
            page_size: args.page_size,
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
                    let request = &self.running[index];
                    if let Some(handle) = &request.cache_handle {
                        self.cache.rollback_insert(&request.input_ids, handle)?;
                    }
                    let mut request = self.running.remove(index);
                    request.cache_handle = None;
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

            let (mut matched_len, mut shared_pages) =
                self.cache.match_prefix(&request.input_ids)?;
            let mut uncached = request.input_ids.len() - matched_len;
            if total_tokens.saturating_add(uncached) > self.max_seq_len && !selected.is_empty() {
                break;
            }
            let mut new_pages = self.pages_needed(request, matched_len);
            if new_pages > self.pool.borrow().layout.num_pages {
                let request = self.pending.pop_front().expect("front exists");
                self.aborted.push(request.uid);
                continue;
            }

            if self.pool.borrow().free_count() < new_pages {
                let shortfall = new_pages - self.pool.borrow().free_count();
                self.cache.evict(shortfall)?;
                // Pages matched before eviction may have been returned to the
                // pool. Match again before assigning any of them to this request.
                (matched_len, shared_pages) = self.cache.match_prefix(&request.input_ids)?;
                uncached = request.input_ids.len() - matched_len;
                new_pages = self.pages_needed(request, matched_len);
            }
            if new_pages > self.pool.borrow().layout.num_pages {
                let request = self.pending.pop_front().expect("front exists");
                self.aborted.push(request.uid);
                continue;
            }
            if total_tokens.saturating_add(uncached) > self.max_seq_len && !selected.is_empty() {
                break;
            }
            if self.pool.borrow().free_count() < new_pages {
                break;
            }

            let mut handle = self.pool.borrow_mut().alloc(new_pages)?;
            handle.page_ids.splice(0..0, shared_pages);
            handle.num_shared = matched_len / self.page_size;
            handle.cached_len = matched_len;
            let request = self.pending.front().expect("front exists");
            if let Err(error) = self.cache.insert(&request.input_ids, &handle) {
                self.pool
                    .borrow_mut()
                    .free_pages_by_id(handle.page_ids[handle.num_shared..].iter().copied());
                return Err(error.into());
            }
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
                    cache_handle: request.cache_handle.clone(),
                }
            })
            .collect::<Vec<_>>();
        let model_batch = self.batch_context.prepare_prefill(&requests)?;
        Ok(Some(PrefillBatch {
            request_ids: std::mem::take(selected),
            model_batch,
        }))
    }

    fn pages_needed(&self, request: &Request, matched_len: usize) -> usize {
        let upper = request
            .input_ids
            .len()
            .saturating_add(request.sampling_params.max_tokens)
            .min(self.max_seq_len);
        upper
            .saturating_sub(matched_len)
            .div_ceil(self.page_size)
            .max(1)
    }

    pub fn abort(&mut self, uid: RequestId) -> bool {
        if let Some(index) = self.pending.iter().position(|request| request.uid == uid) {
            self.pending.remove(index);
            return true;
        }
        if let Some(index) = self.running.iter().position(|request| request.uid == uid) {
            let request = &self.running[index];
            if let Some(handle) = &request.cache_handle {
                if self
                    .cache
                    .remove(request.written_input_ids(), handle)
                    .is_err()
                {
                    return false;
                }
            }
            self.running.remove(index);
            return true;
        }
        false
    }

    pub fn remove_finished_batch(&mut self, ids: &[RequestId]) -> Result<()> {
        for uid in ids {
            if let Some(index) = self.running.iter().position(|request| request.uid == *uid) {
                let request = &self.running[index];
                if let Some(handle) = &request.cache_handle {
                    self.cache.remove(request.written_input_ids(), handle)?;
                }
                self.running.remove(index);
            }
        }
        Ok(())
    }

    pub fn remove_failed_prefill_batch(&mut self, ids: &[RequestId]) -> Result<()> {
        for uid in ids {
            if let Some(index) = self.running.iter().position(|request| request.uid == *uid) {
                let request = &self.running[index];
                if let Some(handle) = &request.cache_handle {
                    self.cache.rollback_insert(&request.input_ids, handle)?;
                }
                self.running.remove(index);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tch::Device;

    use crate::engine::SamplingParams;
    use crate::engine::kvcache::{KVCacheLayout, RadixCacheManager};

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
        let manager = PrefillManager::new(&args, pool.clone(), cache, context);
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
        manager.remove_finished_batch(&[0]).unwrap();
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
        manager.remove_finished_batch(&[0]).unwrap();
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
        manager.add_request(request(0, vec![1], 1));
        manager.schedule_prefill().unwrap().unwrap();
        manager.running_request_mut(0).unwrap().append_token(7);
        manager.remove_finished_batch(&[0]).unwrap();

        manager.add_request(request(1, vec![1, 7, 4], 1));
        manager.schedule_prefill().unwrap().unwrap();
        assert_eq!(manager.running_request(1).unwrap().cached_len, 0);
    }
}
