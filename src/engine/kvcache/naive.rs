//! Cache manager without prefix sharing.

use std::{cell::RefCell, rc::Rc};

use super::{BaseCacheHandle, CacheManager, KVCachePool, Result};

pub struct NaiveCacheManager {
    pool: Rc<RefCell<KVCachePool>>,
}

impl NaiveCacheManager {
    pub fn new(pool: Rc<RefCell<KVCachePool>>) -> Self {
        Self { pool }
    }
}

impl CacheManager for NaiveCacheManager {
    fn match_prefix(&self, _input_ids: &[i64]) -> Result<(usize, Vec<usize>)> {
        Ok((0, Vec::new()))
    }

    fn insert(&mut self, _input_ids: &[i64], _handle: &BaseCacheHandle) -> Result<()> {
        Ok(())
    }

    fn evict(&mut self, _num_pages: usize) -> Result<Vec<usize>> {
        Ok(Vec::new())
    }

    fn remove(&mut self, _input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()> {
        self.pool
            .borrow_mut()
            .free_pages_by_id(handle.page_ids.iter().copied());
        Ok(())
    }

    fn rollback_insert(&mut self, input_ids: &[i64], handle: &BaseCacheHandle) -> Result<()> {
        self.remove(input_ids, handle)
    }
}
