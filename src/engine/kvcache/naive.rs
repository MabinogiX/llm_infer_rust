//! Cache manager without prefix sharing.

use std::{cell::RefCell, rc::Rc};

use super::{AcquireOutcome, BaseCacheHandle, CacheManager, KVCacheError, KVCachePool, Result};

pub struct NaiveCacheManager {
    pool: Rc<RefCell<KVCachePool>>,
}

impl NaiveCacheManager {
    pub fn new(pool: Rc<RefCell<KVCachePool>>) -> Self {
        Self { pool }
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
        Ok(AcquireOutcome::Ready(self.pool.borrow_mut().alloc(pages)?))
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
        self.pool.borrow_mut().free(handle);
    }
}
