//! Cache manager without prefix sharing.

use std::{cell::RefCell, rc::Rc};

use super::{AcquireOutcome, BaseCacheHandle, CacheManager, KVCacheError, KVCachePool, Result};

pub struct NaiveCacheManager {
    pool: Rc<RefCell<KVCachePool>>,
    state_pool: Option<super::SharedRecurrentPool>,
}

impl NaiveCacheManager {
    pub fn new(pool: Rc<RefCell<KVCachePool>>) -> Self {
        Self {
            pool,
            state_pool: None,
        }
    }
}

impl NaiveCacheManager {
    pub fn with_recurrent_state(mut self, pool: Option<super::SharedRecurrentPool>) -> Self {
        self.state_pool = pool;
        self
    }
}
impl CacheManager for NaiveCacheManager {
    fn acquire(
        &mut self,
        input_ids: &[i64],
        capacity_tokens: usize,
        budget: Option<usize>,
    ) -> Result<AcquireOutcome> {
        if input_ids.is_empty() || capacity_tokens < input_ids.len() {
            return Err(KVCacheError::InvalidArgument(
                "nonempty prompt must fit the KV capacity".to_owned(),
            ));
        }
        let page_size = self.pool.borrow().layout.page_size;
        let pages = capacity_tokens.div_ceil(page_size);
        if pages > self.pool.borrow().layout.num_pages {
            return Ok(AcquireOutcome::Impossible);
        }
        if budget.is_some_and(|limit| input_ids.len() > limit) {
            return Ok(AcquireOutcome::DeferredBudget);
        }
        if pages > self.pool.borrow().free_count() {
            return Ok(AcquireOutcome::DeferredMemory);
        }
        let mut handle = self.pool.borrow_mut().alloc(pages)?;
        if let Some(pool) = &self.state_pool {
            let slot = pool.borrow_mut().alloc_request();
            let Some(slot) = slot else {
                self.pool.borrow_mut().free(&mut handle);
                return Ok(AcquireOutcome::DeferredMemory);
            };
            handle.state_slot = Some(slot);
        }
        Ok(AcquireOutcome::Ready(handle))
    }

    fn publish(&mut self, handle: &mut BaseCacheHandle, written_ids: &[i64]) -> Result<()> {
        if written_ids.len() < handle.written_len
            || written_ids.len() > handle.page_ids.len() * self.pool.borrow().layout.page_size
        {
            return Err(KVCacheError::InvalidArgument(
                "written tokens do not fit the KV handle".to_owned(),
            ));
        }
        handle.written_len = written_ids.len();
        Ok(())
    }

    fn release(&mut self, handle: &mut BaseCacheHandle) {
        if let Some(slot) = handle.state_slot.take() {
            self.state_pool
                .as_ref()
                .unwrap()
                .borrow_mut()
                .free_request(slot)
                .expect("请求状态槽所有权错误");
        }
        self.pool.borrow_mut().free(handle);
    }
}
