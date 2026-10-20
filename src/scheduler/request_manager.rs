//! Scheduler-wide request state and KV handle ownership.

use std::collections::VecDeque;

use crate::engine::BatchRequestRef;
use crate::engine::kvcache::{AcquireOutcome, CacheManager};

use super::{Request, RequestId, Result};

pub(crate) struct RequestManager {
    pending: VecDeque<Request>,
    running: Vec<Request>,
    aborted: Vec<RequestId>,
    cache: Box<dyn CacheManager>,
}

impl RequestManager {
    pub(crate) fn new(cache: Box<dyn CacheManager>) -> Self {
        Self {
            pending: VecDeque::new(),
            running: Vec::new(),
            aborted: Vec::new(),
            cache,
        }
    }

    pub(crate) fn add_request(&mut self, request: Request) {
        self.pending.push_back(request);
    }

    pub(crate) fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub(crate) fn running_len(&self) -> usize {
        self.running.len()
    }

    pub(crate) fn running_requests(&self) -> &[Request] {
        &self.running
    }

    pub(crate) fn has_aborted(&self) -> bool {
        !self.aborted.is_empty()
    }

    pub(crate) fn drain_aborted(&mut self) -> Vec<RequestId> {
        std::mem::take(&mut self.aborted)
    }

    pub(crate) fn running_request(&self, uid: RequestId) -> Option<&Request> {
        self.running.iter().find(|request| request.uid == uid)
    }

    pub(crate) fn running_request_mut(&mut self, uid: RequestId) -> Option<&mut Request> {
        self.running.iter_mut().find(|request| request.uid == uid)
    }

    pub(crate) fn batch_request(&self, uid: RequestId) -> Option<BatchRequestRef<'_>> {
        let request = self.running_request(uid)?;
        Some(BatchRequestRef {
            input_ids: &request.input_ids,
            cached_len: request.cached_len,
            end_len: None,
            page_ids: &request.cache_handle.as_ref()?.page_ids,
        })
    }

    pub(crate) fn pop_pending(&mut self) -> Option<Request> {
        self.pending.pop_front()
    }

    pub(crate) fn pending_front(&self) -> Option<&Request> {
        self.pending.front()
    }

    #[cfg(test)]
    pub(crate) fn pending_ids(&self) -> Vec<RequestId> {
        self.pending.iter().map(|request| request.uid).collect()
    }

    pub(crate) fn push_pending_front(&mut self, request: Request) {
        self.pending.push_front(request);
    }

    pub(crate) fn push_aborted(&mut self, uid: RequestId) {
        self.aborted.push(uid);
    }

    pub(crate) fn acquire_pending(
        &mut self,
        capacity_tokens: usize,
        budget: Option<usize>,
    ) -> Result<AcquireOutcome> {
        let (pending, cache) = (&self.pending, &mut self.cache);
        let request = pending.front().expect("pending request exists");
        Ok(cache.acquire(&request.input_ids, capacity_tokens, budget)?)
    }

    pub(crate) fn add_running(&mut self, request: Request) {
        self.running.push(request);
    }

    pub(crate) fn remove_running(&mut self, uid: RequestId) -> Option<Request> {
        self.running
            .iter()
            .position(|request| request.uid == uid)
            .map(|index| self.running.remove(index))
    }

    pub(crate) fn release_handle(&mut self, request: &mut Request) {
        if let Some(mut handle) = request.cache_handle.take() {
            self.cache.release(&mut handle);
        }
    }

    pub(crate) fn abort(&mut self, uid: RequestId) -> bool {
        if let Some(index) = self.pending.iter().position(|request| request.uid == uid) {
            self.pending.remove(index);
            return true;
        }
        if let Some(mut request) = self.remove_running(uid) {
            self.release_handle(&mut request);
            return true;
        }
        false
    }

    /// Detach logical request state while its handle still protects every page.
    pub(crate) fn retire(&mut self, uid: RequestId) -> Option<Request> {
        self.remove_running(uid)
    }

    /// Called only after the last invocation using this handle completes.
    pub(crate) fn release_retired(
        &mut self,
        mut request: Request,
        publish: bool,
    ) -> crate::engine::kvcache::Result<()> {
        let result = if publish {
            let Some(handle) = request.cache_handle.as_mut() else {
                return Ok(());
            };
            self.cache
                .publish(handle, &request.input_ids[..handle.written_len])
        } else {
            Ok(())
        };
        self.release_handle(&mut request);
        result
    }

    pub(crate) fn remove_batch(&mut self, ids: &[RequestId]) {
        for uid in ids {
            if let Some(mut request) = self.remove_running(*uid) {
                self.release_handle(&mut request);
            }
        }
    }

    /// Publish the current sequence after its KV write completes and before
    /// appending the sampled token, which has not yet entered KV.
    pub(crate) fn publish(&mut self, uid: RequestId) -> crate::engine::kvcache::Result<()> {
        let (running, cache) = (&mut self.running, &mut self.cache);
        let request = running
            .iter_mut()
            .find(|request| request.uid == uid)
            .expect("scheduled request is running");
        let handle = request
            .cache_handle
            .as_mut()
            .expect("running request has KV handle");
        cache.publish(handle, &request.input_ids)
    }

    pub(crate) fn publish_prefix(
        &mut self,
        uid: RequestId,
        written_len: usize,
    ) -> crate::engine::kvcache::Result<()> {
        let (running, cache) = (&mut self.running, &mut self.cache);
        let request = running
            .iter_mut()
            .find(|request| request.uid == uid)
            .expect("scheduled request is running");
        if written_len > request.input_ids.len() {
            return Err(crate::engine::kvcache::KVCacheError::InvalidArgument(
                "written prefix exceeds request input".to_owned(),
            ));
        }
        let handle = request
            .cache_handle
            .as_mut()
            .expect("running request has KV handle");
        cache.publish(handle, &request.input_ids[..written_len])
    }

    pub(crate) fn mark_written(&mut self, uid: RequestId, written_len: usize) {
        let request = self
            .running_request_mut(uid)
            .expect("scheduled request is running");
        request
            .cache_handle
            .as_mut()
            .expect("running request has KV handle")
            .written_len = written_len;
    }

    pub(crate) fn restore_waiting(&mut self, mut request: Request) {
        self.release_handle(&mut request);
        request.restore_waiting();
        self.push_pending_front(request);
    }
}
