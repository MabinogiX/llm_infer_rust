//! Page-granular prefix cache. Only successful forwards publish KV pages.

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use super::{AcquireOutcome, BaseCacheHandle, CacheManager, KVCacheError, KVCachePool, Result};
use super::{CheckpointId, SharedRecurrentPool};
use crate::engine::kvcache::pool::PageOwner;

type NodeId = usize;

/// One complete token page and its canonical physical KV page.
#[derive(Debug)]
pub struct RadixNode {
    tokens: Vec<i64>,
    page_id: usize,
    parent: Option<NodeId>,
    children: HashMap<Vec<i64>, NodeId>,
    pin_count: usize,
    depth: usize,
    checkpoint: Option<CheckpointId>,
    checkpoint_used: u64,
}

impl RadixNode {
    fn root() -> Self {
        Self {
            tokens: Vec::new(),
            page_id: usize::MAX,
            parent: None,
            children: HashMap::new(),
            pin_count: 0,
            depth: 0,
            checkpoint: None,
            checkpoint_used: 0,
        }
    }

    fn child(tokens: Vec<i64>, page_id: usize, parent: NodeId, depth: usize) -> Self {
        Self {
            tokens,
            page_id,
            parent: Some(parent),
            children: HashMap::new(),
            pin_count: 1,
            depth,
            checkpoint: None,
            checkpoint_used: 0,
        }
    }
}

/// The scheduler owns this manager exclusively; the pool is shared with Engine.
pub struct RadixCacheManager {
    pool: Rc<RefCell<KVCachePool>>,
    page_size: usize,
    nodes: Vec<Option<RadixNode>>,
    free_node_ids: Vec<NodeId>,
    state_pool: Option<SharedRecurrentPool>,
    checkpoint_clock: u64,
}

impl RadixCacheManager {
    pub fn new(pool: Rc<RefCell<KVCachePool>>, page_size: usize) -> Result<Self> {
        if page_size == 0 || page_size != pool.borrow().layout.page_size {
            return Err(KVCacheError::InvalidArgument(
                "radix page size must match the KV pool".to_owned(),
            ));
        }
        Ok(Self {
            pool,
            page_size,
            nodes: vec![Some(RadixNode::root())],
            free_node_ids: Vec::new(),
            state_pool: None,
            checkpoint_clock: 0,
        })
    }

    pub fn with_recurrent_state(mut self, pool: Option<SharedRecurrentPool>) -> Self {
        self.state_pool = pool;
        self
    }
    fn touch_checkpoint(&mut self, id: NodeId) {
        self.checkpoint_clock = self
            .checkpoint_clock
            .checked_add(1)
            .expect("检查点时钟溢出");
        self.node_mut(id).checkpoint_used = self.checkpoint_clock;
    }
    /// 内部节点只释放状态；未被请求引用的叶子同时回收无状态的 KV 尾部。
    fn evict_checkpoint(&mut self) -> Result<bool> {
        let victim = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| {
                let node = node.as_ref()?;
                if node.children.is_empty() && node.pin_count > 0 {
                    return None;
                }
                node.checkpoint.map(|_| (id, node.checkpoint_used))
            })
            .min_by_key(|(_, used)| *used)
            .map(|(id, _)| id);
        let Some(id) = victim else {
            return Ok(false);
        };
        let checkpoint = self.node(id).checkpoint.unwrap();
        self.state_pool
            .as_ref()
            .unwrap()
            .borrow_mut()
            .free_checkpoint(checkpoint)?;
        self.node_mut(id).checkpoint = None;
        self.prune_unusable_tail(id);
        tracing::debug!(node_id = id, "LRU 淘汰循环状态检查点并回收无用叶子");
        Ok(true)
    }
    /// 先取得状态所有权再发布 KV，避免状态池耗尽后留下不可恢复的尾部。
    fn prepare_checkpoint(
        &mut self,
        handle: &BaseCacheHandle,
        written_ids: &[i64],
    ) -> Result<(usize, Option<(CheckpointId, usize)>)> {
        let previous = handle.published_pages * self.page_size;
        let (Some(pool), Some(slot)) = (self.state_pool.clone(), handle.state_slot) else {
            return Ok((written_ids.len() / self.page_size * self.page_size, None));
        };
        let Some(position) = pool.borrow().staged_position(slot)? else {
            return Ok((previous, None));
        };
        // The overlap pipeline may already have queued one more decode when
        // the preceding sample ends the request. Its state is complete only
        // after the last-use fence, but its extra input is not part of the
        // confirmed prefix. Never attach that speculative state to radix.
        if position % self.page_size == 0 && position == written_ids.len().saturating_add(1) {
            pool.borrow_mut().discard_snapshot(slot)?;
            return Ok((previous, None));
        }
        if position == 0 || position > written_ids.len() || position % self.page_size != 0 {
            return Err(KVCacheError::InvalidArgument(
                "检查点必须位于已计算的页边界".into(),
            ));
        }
        let path = self.kv_match_path(&written_ids[..position]);
        if path.len() * self.page_size == position
            && self.node(*path.last().unwrap()).checkpoint.is_some()
        {
            pool.borrow_mut().discard_snapshot(slot)?;
            return Ok((previous.max(position), None));
        }
        if pool.borrow().checkpoint_capacity() == 0 {
            pool.borrow_mut().discard_snapshot(slot)?;
            return Ok((previous, None));
        }
        let mut saved = pool.borrow_mut().save_checkpoint(slot, position)?;
        if saved.is_none() && self.evict_checkpoint()? {
            saved = pool.borrow_mut().save_checkpoint(slot, position)?;
        }
        Ok(match saved {
            Some(checkpoint) => (previous.max(position), Some((checkpoint, position))),
            None => {
                pool.borrow_mut().discard_snapshot(slot)?;
                (previous, None)
            }
        })
    }

    fn remove_leaf(&mut self, id: NodeId) -> usize {
        let node = self.nodes[id].take().expect("有效叶子");
        assert!(node.children.is_empty() && node.pin_count == 0);
        if let Some(checkpoint) = node.checkpoint {
            self.state_pool
                .as_ref()
                .expect("检查点状态池")
                .borrow_mut()
                .free_checkpoint(checkpoint)
                .expect("radix 检查点必须有效");
        }
        self.node_mut(node.parent.expect("非根节点"))
            .children
            .remove(&node.tokens);
        self.free_node_ids.push(id);
        self.pool.borrow_mut().free_pages_by_id([node.page_id]);
        node.page_id
    }

    fn prune_unusable_tail(&mut self, mut id: NodeId) -> Vec<usize> {
        let mut freed = Vec::new();
        while id != 0 {
            let Some(node) = self.nodes[id].as_ref() else {
                break;
            };
            if !node.children.is_empty() || node.pin_count != 0 || node.checkpoint.is_some() {
                break;
            }
            let parent = node.parent.unwrap();
            freed.push(self.remove_leaf(id));
            id = parent;
        }
        freed
    }
    pub fn checkpoint_count(&self) -> usize {
        self.nodes
            .iter()
            .flatten()
            .filter(|n| n.checkpoint.is_some())
            .count()
    }

    fn node(&self, id: NodeId) -> &RadixNode {
        self.nodes[id].as_ref().expect("live radix node")
    }

    fn node_mut(&mut self, id: NodeId) -> &mut RadixNode {
        self.nodes[id].as_mut().expect("live radix node")
    }

    fn kv_match_path(&self, input_ids: &[i64]) -> Vec<NodeId> {
        let max_pages = input_ids.len() / self.page_size;
        let mut path = Vec::new();
        let mut parent = 0;
        for index in 0..max_pages {
            let tokens = &input_ids[index * self.page_size..(index + 1) * self.page_size];
            let Some(&child) = self.node(parent).children.get(tokens) else {
                break;
            };
            path.push(child);
            parent = child;
        }
        path
    }

    fn match_path(&self, input_ids: &[i64]) -> Vec<NodeId> {
        // 至少重算一个 prompt token，取得 logits。
        let mut path = self.kv_match_path(&input_ids[..input_ids.len().saturating_sub(1)]);
        if self.state_pool.is_some() {
            let reusable = path
                .iter()
                .rposition(|&id| self.node(id).checkpoint.is_some())
                .map_or(0, |index| index + 1);
            path.truncate(reusable);
        }
        path
    }

    /// Read-only inspection; actual reuse must go through `acquire` to pin pages.
    pub fn match_prefix(&self, input_ids: &[i64]) -> (usize, Vec<usize>) {
        let path = self.match_path(input_ids);
        (
            path.len() * self.page_size,
            path.iter().map(|&id| self.node(id).page_id).collect(),
        )
    }

    fn insert_page(&mut self, parent: NodeId, tokens: Vec<i64>, page_id: usize) -> NodeId {
        let id = self.free_node_ids.pop().unwrap_or(self.nodes.len());
        let depth = self.node(parent).depth + 1;
        let node = RadixNode::child(tokens.clone(), page_id, parent, depth);
        if id == self.nodes.len() {
            self.nodes.push(Some(node));
        } else {
            self.nodes[id] = Some(node);
        }
        self.node_mut(parent).children.insert(tokens, id);
        id
    }

    /// Evict only unpinned leaves; their ancestors remain until no child needs them.
    pub fn evict(&mut self, num_pages: usize) -> Vec<usize> {
        let mut freed = Vec::new();
        while freed.len() < num_pages {
            let leaf = self
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(id, node)| {
                    let node = node.as_ref()?;
                    (id != 0 && node.children.is_empty() && node.pin_count == 0)
                        .then_some((id, node.depth))
                })
                .max_by_key(|(_, depth)| *depth)
                .map(|(id, _)| id);
            let Some(id) = leaf else { break };
            let parent = self.node(id).parent.unwrap();
            freed.push(self.remove_leaf(id));
            if self.state_pool.is_some() {
                freed.extend(self.prune_unusable_tail(parent));
            }
        }
        freed
    }
}

impl RadixCacheManager {
    fn acquire_kv(
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
        let total_pages = capacity_tokens.div_ceil(self.page_size);
        if total_pages > self.pool.borrow().layout.num_pages {
            return Ok(AcquireOutcome::Impossible);
        }
        let path = self.match_path(input_ids);
        let cached_len = path.len() * self.page_size;
        if budget.is_some_and(|limit| input_ids.len() - cached_len > limit) {
            return Ok(AcquireOutcome::DeferredBudget);
        }

        for &id in &path {
            self.node_mut(id).pin_count += 1;
        }
        let private_count = total_pages - path.len();
        let free = self.pool.borrow().free_count();
        if free < private_count {
            self.evict(private_count - free);
        }
        if self.pool.borrow().free_count() < private_count {
            for &id in &path {
                self.node_mut(id).pin_count -= 1;
            }
            return Ok(AcquireOutcome::DeferredMemory);
        }

        let allocation_result = { self.pool.borrow_mut().alloc(private_count) };
        let allocation = match allocation_result {
            Ok(allocation) => allocation,
            Err(error) => {
                for &id in &path {
                    self.node_mut(id).pin_count -= 1;
                }
                return Err(error);
            }
        };
        let mut page_ids = path
            .iter()
            .map(|&id| self.node(id).page_id)
            .collect::<Vec<_>>();
        page_ids.extend(allocation.page_ids);
        let mut owners = path
            .iter()
            .map(|&id| PageOwner::Tree(id))
            .collect::<Vec<_>>();
        owners.extend(allocation.owners);
        Ok(AcquireOutcome::Ready(BaseCacheHandle {
            state_slot: None,
            page_ids,
            cached_len,
            owners,
            published_pages: path.len(),
            written_len: cached_len,
        }))
    }
}
impl CacheManager for RadixCacheManager {
    fn acquire(
        &mut self,
        input_ids: &[i64],
        capacity_tokens: usize,
        budget: Option<usize>,
    ) -> Result<AcquireOutcome> {
        // 分配私有页可能触发 KV 淘汰，先记住原始 KV 命中的分支位置。
        let full_kv_len = if self.state_pool.is_some() {
            self.kv_match_path(&input_ids[..input_ids.len().saturating_sub(1)])
                .len()
                * self.page_size
        } else {
            0
        };
        let outcome = self.acquire_kv(input_ids, capacity_tokens, budget)?;
        let AcquireOutcome::Ready(mut handle) = outcome else {
            return Ok(outcome);
        };
        let Some(pool) = self.state_pool.clone() else {
            return Ok(AcquireOutcome::Ready(handle));
        };
        let allocation = pool.borrow_mut().alloc_request();
        let Some(slot) = allocation else {
            self.release(&mut handle);
            return Ok(AcquireOutcome::DeferredMemory);
        };
        handle.state_slot = Some(slot);
        if handle.cached_len > 0 {
            let PageOwner::Tree(node) = handle.owners[handle.cached_len / self.page_size - 1]
            else {
                unreachable!()
            };
            let checkpoint = self.node(node).checkpoint.expect("匹配必须包含循环检查点");
            let restore = pool
                .borrow_mut()
                .restore(checkpoint, slot, handle.cached_len);
            if let Err(error) = restore {
                self.release(&mut handle);
                return Err(error);
            }
            self.touch_checkpoint(node);
            tracing::info!(
                cached_tokens = handle.cached_len,
                state_slot = slot.index,
                "恢复 KV 与循环状态前缀"
            );
        }
        let branching = (full_kv_len > handle.cached_len).then_some(full_kv_len);
        pool.borrow_mut().set_branching_seqlen(slot, branching)?;
        if let Some(branching_seqlen) = branching {
            tracing::info!(
                cached_tokens = handle.cached_len,
                full_kv_hit_length = full_kv_len,
                mamba_branching_seqlen = branching_seqlen,
                "记录循环状态分支补齐位置"
            );
        }
        Ok(AcquireOutcome::Ready(handle))
    }
    fn publish(&mut self, handle: &mut BaseCacheHandle, written_ids: &[i64]) -> Result<()> {
        if handle.page_ids.len() != handle.owners.len()
            || written_ids.len() < handle.written_len
            || written_ids.len() > handle.page_ids.len() * self.page_size
        {
            return Err(KVCacheError::InvalidArgument(
                "written tokens do not fit the KV handle".to_owned(),
            ));
        }
        let full_pages = written_ids.len() / self.page_size;
        for index in 0..full_pages {
            let PageOwner::Tree(id) = handle.owners[index] else {
                break;
            };
            let tokens = &written_ids[index * self.page_size..(index + 1) * self.page_size];
            if self.node(id).tokens != tokens {
                return Err(KVCacheError::InvalidArgument(
                    "published prefix differs from the handle".to_owned(),
                ));
            }
        }
        for index in handle.published_pages..full_pages {
            if handle.owners[index] != PageOwner::Private {
                return Err(KVCacheError::InvalidArgument(
                    "unpublished page is not private".to_owned(),
                ));
            }
        }
        let (publish_len, checkpoint) = self.prepare_checkpoint(handle, written_ids)?;
        let result = (|| {
            let full_pages = publish_len / self.page_size;
            for index in handle.published_pages..full_pages {
                let parent = if index == 0 {
                    0
                } else if let PageOwner::Tree(id) = handle.owners[index - 1] {
                    id
                } else {
                    return Err(KVCacheError::InvalidArgument(
                        "published page path is discontinuous".to_owned(),
                    ));
                };
                let tokens =
                    written_ids[index * self.page_size..(index + 1) * self.page_size].to_vec();
                let private_page = handle.page_ids[index];
                if let Some(&canonical) = self.node(parent).children.get(tokens.as_slice()) {
                    let canonical_page = self.node(canonical).page_id;
                    if private_page == canonical_page {
                        return Err(KVCacheError::InvalidArgument(
                            "private page already belongs to radix".to_owned(),
                        ));
                    }
                    self.node_mut(canonical).pin_count += 1;
                    handle.page_ids[index] = canonical_page;
                    handle.owners[index] = PageOwner::Tree(canonical);
                    self.pool.borrow_mut().free_pages_by_id([private_page]);
                } else {
                    let id = self.insert_page(parent, tokens, private_page);
                    handle.owners[index] = PageOwner::Tree(id);
                }
                handle.published_pages += 1;
            }
            handle.written_len = written_ids.len();
            if let Some((checkpoint, position)) = checkpoint {
                let PageOwner::Tree(node) = handle.owners[position / self.page_size - 1] else {
                    unreachable!()
                };
                self.node_mut(node).checkpoint = Some(checkpoint);
                self.touch_checkpoint(node);
                tracing::debug!(prefix_len = position, node_id = node, "发布循环状态检查点");
            }
            Ok(())
        })();
        if result.is_err() {
            if let Some((checkpoint, _)) = checkpoint {
                self.state_pool
                    .as_ref()
                    .unwrap()
                    .borrow_mut()
                    .free_checkpoint(checkpoint)?;
            }
        }
        result
    }

    fn release(&mut self, handle: &mut BaseCacheHandle) {
        if let Some(slot) = handle.state_slot.take() {
            self.state_pool
                .as_ref()
                .expect("请求循环状态池")
                .borrow_mut()
                .free_request(slot)
                .expect("请求拥有的状态槽必须有效");
        }
        let page_ids = std::mem::take(&mut handle.page_ids);
        let owners = std::mem::take(&mut handle.owners);
        assert_eq!(page_ids.len(), owners.len(), "KV handle ownership mismatch");
        let mut private = Vec::new();
        let mut tree = Vec::new();
        for (page_id, owner) in page_ids.into_iter().zip(owners) {
            match owner {
                PageOwner::Private => private.push(page_id),
                PageOwner::Tree(id) => {
                    let node = self.node_mut(id);
                    assert!(node.pin_count > 0, "radix pin underflow");
                    node.pin_count -= 1;
                    tree.push(id);
                }
            }
        }
        if self.state_pool.is_some() {
            for id in tree.into_iter().rev() {
                if self.nodes[id].is_some() {
                    self.prune_unusable_tail(id);
                }
            }
        }
        self.pool.borrow_mut().free_pages_by_id(private);
        handle.cached_len = 0;
        handle.published_pages = 0;
        handle.written_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::engine::kvcache::KVCacheLayout;

    fn cache(num_pages: usize) -> (Rc<RefCell<KVCachePool>>, RadixCacheManager) {
        let pool = Rc::new(RefCell::new(KVCachePool::without_tensor(
            KVCacheLayout::new(1, num_pages, 2, 1, 1).unwrap(),
        )));
        let cache = RadixCacheManager::new(pool.clone(), 2).unwrap();
        (pool, cache)
    }

    fn ready(outcome: AcquireOutcome) -> BaseCacheHandle {
        match outcome {
            AcquireOutcome::Ready(handle) => handle,
            _ => panic!("request should acquire pages"),
        }
    }

    fn assert_partition(cache: &RadixCacheManager, handles: &[&BaseCacheHandle]) {
        let pool = cache.pool.borrow();
        let mut all = HashSet::new();
        for &page in pool.free_page_ids() {
            assert!(all.insert(page), "duplicate free page {page}");
        }
        for node in cache.nodes.iter().skip(1).flatten() {
            assert!(all.insert(node.page_id), "tree page is already owned");
        }
        for handle in handles {
            for (&page, owner) in handle.page_ids.iter().zip(&handle.owners) {
                match owner {
                    PageOwner::Private => {
                        assert!(all.insert(page), "private page is already owned");
                    }
                    PageOwner::Tree(id) => {
                        assert_eq!(cache.node(*id).page_id, page);
                        assert!(cache.node(*id).pin_count > 0);
                        assert!(!pool.free_page_ids().contains(&page));
                    }
                }
            }
        }
        assert_eq!(all.len(), pool.layout.num_pages);
    }

    #[test]
    fn cold_batch_keeps_private_pages_until_serial_publication() {
        let (pool, mut cache) = cache(6);
        let mut a = ready(cache.acquire(&[1, 2, 3, 4], 4, None).unwrap());
        let mut b = ready(cache.acquire(&[1, 2, 3, 4], 4, None).unwrap());
        assert_ne!(a.page_ids, b.page_ids);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5]), (0, vec![]));
        assert_partition(&cache, &[&a, &b]);

        cache.publish(&mut a, &[1, 2, 3, 4]).unwrap();
        cache.publish(&mut b, &[1, 2, 3, 4]).unwrap();
        assert_eq!(a.page_ids, b.page_ids);
        assert_eq!(pool.borrow().free_count(), 4);
        assert_partition(&cache, &[&a, &b]);
        cache.release(&mut a);
        cache.release(&mut b);
        cache.release(&mut b);
        assert_eq!(cache.evict(2).len(), 2);
        assert_eq!(pool.borrow().free_count(), 6);
    }

    #[test]
    fn generated_prefix_merges_with_later_prompt_without_leaking_a_page() {
        let (pool, mut cache) = cache(6);
        let mut a = ready(cache.acquire(&[1, 2], 4, None).unwrap());
        cache.publish(&mut a, &[1, 2]).unwrap();
        let a_generated_page = a.page_ids[1];

        let mut b = ready(cache.acquire(&[1, 2, 3, 4, 6], 5, None).unwrap());
        assert_eq!(b.cached_len, 2);
        assert_ne!(b.page_ids[1], a_generated_page);
        cache.publish(&mut b, &[1, 2, 3, 4, 6]).unwrap();
        cache.publish(&mut a, &[1, 2, 3, 4]).unwrap();
        assert_eq!(a.page_ids[1], b.page_ids[1]);
        assert!(pool.borrow().free_page_ids().contains(&a_generated_page));
        assert_partition(&cache, &[&a, &b]);

        cache.release(&mut a);
        assert_partition(&cache, &[&b]);
        cache.release(&mut b);
        cache.evict(6);
        assert_eq!(pool.borrow().free_count(), 6);
    }

    #[test]
    fn completed_generation_is_reused_by_a_later_prompt() {
        let (pool, mut cache) = cache(4);
        let mut a = ready(cache.acquire(&[1, 2], 4, None).unwrap());
        cache.publish(&mut a, &[1, 2]).unwrap();
        cache.publish(&mut a, &[1, 2, 3, 4]).unwrap();
        cache.release(&mut a);

        let mut b = ready(cache.acquire(&[1, 2, 3, 4, 5], 5, None).unwrap());
        assert_eq!(b.cached_len, 4);
        assert_eq!(b.published_pages, 2);
        assert_partition(&cache, &[&b]);
        cache.release(&mut b);
        cache.evict(4);
        assert_eq!(pool.borrow().free_count(), 4);
    }

    #[test]
    fn partial_and_unwritten_tokens_are_never_published() {
        let (_, mut cache) = cache(5);
        let mut a = ready(cache.acquire(&[1, 2, 3], 3, None).unwrap());
        cache.publish(&mut a, &[1, 2, 3]).unwrap();
        assert_eq!(a.published_pages, 1);
        assert_eq!(
            cache.match_prefix(&[1, 2, 3, 4, 5]),
            (2, vec![a.page_ids[0]])
        );
        cache.release(&mut a);
        assert_eq!(cache.evict(5).len(), 1);
    }

    #[test]
    fn budget_deferral_does_not_pin_or_evict() {
        let (pool, mut cache) = cache(2);
        let mut first = ready(cache.acquire(&[1, 2], 2, None).unwrap());
        cache.publish(&mut first, &[1, 2]).unwrap();
        cache.release(&mut first);
        let free = pool.borrow().free_count();
        assert!(matches!(
            cache.acquire(&[1, 2, 3], 3, Some(0)).unwrap(),
            AcquireOutcome::DeferredBudget
        ));
        assert_eq!(pool.borrow().free_count(), free);
        assert_eq!(cache.node(1).pin_count, 0);
    }

    #[test]
    fn matched_page_stays_pinned_during_eviction_and_capacity_is_total() {
        let (pool, mut cache) = cache(2);
        let mut first = ready(cache.acquire(&[1, 2, 3, 4], 4, None).unwrap());
        cache.publish(&mut first, &[1, 2, 3, 4]).unwrap();
        cache.release(&mut first);
        assert!(matches!(
            cache.acquire(&[1, 2, 3, 4, 5], 5, None).unwrap(),
            AcquireOutcome::Impossible
        ));

        let mut second = ready(cache.acquire(&[1, 2, 9], 3, None).unwrap());
        assert_eq!(second.cached_len, 2);
        assert_ne!(second.page_ids[0], second.page_ids[1]);
        assert_partition(&cache, &[&second]);
        cache.release(&mut second);
        cache.evict(2);
        assert_eq!(pool.borrow().free_count(), 2);
    }

    #[test]
    fn evicted_node_slot_is_reused_only_after_release() {
        let (_, mut cache) = cache(3);
        let mut first = ready(cache.acquire(&[1, 2], 2, None).unwrap());
        cache.publish(&mut first, &[1, 2]).unwrap();
        let id = match first.owners[0] {
            PageOwner::Tree(id) => id,
            _ => unreachable!(),
        };
        assert!(cache.evict(1).is_empty());
        cache.release(&mut first);
        assert_eq!(cache.evict(1).len(), 1);
        let mut second = ready(cache.acquire(&[3, 4], 2, None).unwrap());
        cache.publish(&mut second, &[3, 4]).unwrap();
        assert_eq!(second.owners[0], PageOwner::Tree(id));
        cache.release(&mut second);
    }
}

#[cfg(test)]
mod recurrent_tests {
    use super::*;
    use crate::engine::kvcache::{KVCacheLayout, LinearState, RecurrentStatePool, RequestState};
    use tch::{Device, Kind, Tensor};
    fn cache(requests: usize, checkpoints: usize) -> (RadixCacheManager, SharedRecurrentPool) {
        let kv = Rc::new(RefCell::new(KVCachePool::without_tensor(
            KVCacheLayout::new(1, 24, 2, 1, 1).unwrap(),
        )));
        let states = Rc::new(RefCell::new(RecurrentStatePool::new(
            requests,
            checkpoints,
            || RequestState {
                position: 0,
                layers: vec![LinearState {
                    conv: Tensor::zeros([1, 3], (Kind::Float, Device::Cpu)),
                    recurrent: Tensor::zeros([1, 2, 2], (Kind::Float, Device::Cpu)),
                }],
            },
        )));
        (
            RadixCacheManager::new(kv, 2)
                .unwrap()
                .with_recurrent_state(Some(states.clone())),
            states,
        )
    }
    fn ready(cache: &mut RadixCacheManager, ids: &[i64]) -> BaseCacheHandle {
        let AcquireOutcome::Ready(handle) = cache.acquire(ids, ids.len() + 4, None).unwrap() else {
            panic!("应分配成功")
        };
        handle
    }
    fn publish(
        cache: &mut RadixCacheManager,
        pool: &SharedRecurrentPool,
        handle: &mut BaseCacheHandle,
        ids: &[i64],
        position: usize,
        value: f64,
    ) {
        let slot = handle.state_slot.unwrap();
        let mut states = pool.borrow_mut();
        let captured = states.begin_snapshot(slot).unwrap();
        let _ = states.current_mut(slot).unwrap().layers[0]
            .recurrent
            .fill_(value + 1.);
        if captured {
            let (_, staged) = states.snapshot_states_mut(slot).unwrap();
            let _ = staged.layers[0].recurrent.fill_(value);
            let _ = staged.layers[0].conv.fill_(value);
        }
        states
            .commit_forward(slot, ids.len(), captured.then_some(position))
            .unwrap();
        drop(states);
        cache.publish(handle, ids).unwrap();
    }
    #[test]
    fn speculative_snapshot_is_not_published_past_confirmed_tokens() {
        let (mut cache, pool) = cache(1, 4);
        let ids = [1, 2, 3, 4, 5, 6];
        let mut handle = ready(&mut cache, &ids);
        publish(&mut cache, &pool, &mut handle, &ids[..3], 2, 7.);
        let slot = handle.state_slot.unwrap();
        assert!(pool.borrow_mut().begin_snapshot(slot).unwrap());
        pool.borrow_mut().commit_forward(slot, 6, Some(6)).unwrap();
        // EOS was sampled by the preceding invocation; position 5 is the
        // exclusive confirmed end. The queued invocation captured position 6.
        cache.publish(&mut handle, &ids[..5]).unwrap();
        assert_eq!(handle.published_pages, 1);
        assert_eq!(handle.written_len, 5);
        assert_eq!(pool.borrow().staged_position(slot).unwrap(), None);
        cache.release(&mut handle);
        let mut warm = ready(&mut cache, &ids);
        assert_eq!(warm.cached_len, 2);
        cache.release(&mut warm);
        cache.evict(24);
        assert_eq!(pool.borrow().usage(), (0, 0));
    }

    #[test]
    fn restored_requests_own_distinct_slots_and_checkpoint_is_immutable() {
        let (mut cache, pool) = cache(3, 2);
        let ids = [1, 2, 3, 4, 5];
        let mut a = ready(&mut cache, &ids);
        publish(&mut cache, &pool, &mut a, &ids, 4, 7.);
        cache.release(&mut a);
        let mut a = ready(&mut cache, &ids);
        let mut b = ready(&mut cache, &ids);
        assert_eq!(a.cached_len, 4);
        assert_eq!(b.cached_len, 4);
        assert_eq!(a.page_ids[..2], b.page_ids[..2]);
        assert_ne!(a.state_slot, b.state_slot);
        let _ = pool
            .borrow_mut()
            .current_mut(a.state_slot.unwrap())
            .unwrap()
            .layers[0]
            .recurrent
            .fill_(99.);
        assert_eq!(
            pool.borrow_mut()
                .current_mut(b.state_slot.unwrap())
                .unwrap()
                .layers[0]
                .recurrent
                .min()
                .double_value(&[]),
            7.
        );
        let mut c = ready(&mut cache, &ids);
        assert_eq!(
            pool.borrow_mut()
                .current_mut(c.state_slot.unwrap())
                .unwrap()
                .layers[0]
                .recurrent
                .min()
                .double_value(&[]),
            7.
        );
        cache.release(&mut a);
        cache.release(&mut b);
        cache.release(&mut c);
        assert_eq!(pool.borrow().usage(), (0, 1));
        cache.evict(24);
        assert_eq!(pool.borrow().usage(), (0, 0));
    }
    #[test]
    fn checkpoint_lru_reclaims_unpinned_leaf_and_preserves_active_copies() {
        let (mut cache, pool) = cache(2, 2);
        let a = [1, 2, 3, 4, 5];
        let b = [6, 7, 8, 9, 10];
        let c = [11, 12, 13, 14, 15];
        let mut h = ready(&mut cache, &a);
        publish(&mut cache, &pool, &mut h, &a, 4, 1.);
        cache.release(&mut h);
        let mut h = ready(&mut cache, &b);
        publish(&mut cache, &pool, &mut h, &b, 4, 2.);
        cache.release(&mut h);
        let mut active = ready(&mut cache, &a); // A 成为最近使用，B 应先被淘汰。
        let mut h = ready(&mut cache, &c);
        publish(&mut cache, &pool, &mut h, &c, 4, 3.);
        cache.release(&mut h);
        assert_eq!(cache.match_prefix(&b).0, 0);
        assert_eq!(cache.match_prefix(&a).0, 4);
        assert_eq!(cache.checkpoint_count(), 2);
        assert_eq!(cache.nodes.iter().flatten().count(), 5); // B 的无引用叶子与无状态祖先已回收。
        cache.evict_checkpoint().unwrap(); // A 是仍被引用的叶子，跳过 A、淘汰 C。
        assert_eq!(cache.match_prefix(&a).0, 4);
        assert_eq!(cache.match_prefix(&c).0, 0);
        assert!(!cache.evict_checkpoint().unwrap()); // 唯一余下的检查点仍被活跃请求保护。
        assert_eq!(
            pool.borrow_mut()
                .current_mut(active.state_slot.unwrap())
                .unwrap()
                .layers[0]
                .recurrent
                .min()
                .double_value(&[]),
            1.
        );
        cache.release(&mut active);
        cache.evict(24);
        assert_eq!(pool.borrow().usage(), (0, 0));
    }
    #[test]
    fn missing_deep_checkpoint_falls_back_to_shallow_state_not_longer_kv() {
        let (mut cache, pool) = cache(1, 2);
        let mut h = ready(&mut cache, &[1, 2, 3, 4, 5, 6, 7]);
        publish(&mut cache, &pool, &mut h, &[1, 2, 3], 2, 2.);
        publish(&mut cache, &pool, &mut h, &[1, 2, 3, 4, 5, 6, 7], 4, 4.);
        cache.release(&mut h);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7]).0, 4);
        let node = cache
            .nodes
            .iter()
            .position(|n| n.as_ref().is_some_and(|n| n.depth == 2))
            .unwrap();
        let checkpoint = cache.node_mut(node).checkpoint.take().unwrap();
        pool.borrow_mut().free_checkpoint(checkpoint).unwrap();
        let mut h = ready(&mut cache, &[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(h.cached_len, 2);
        assert_eq!(
            pool.borrow_mut()
                .current_mut(h.state_slot.unwrap())
                .unwrap()
                .position,
            2
        );
        cache.release(&mut h);
    }
    #[test]
    fn exhausted_request_pool_rolls_back_kv_and_generation_prevents_use_after_cancel() {
        let (mut cache, pool) = cache(1, 1);
        let mut a = ready(&mut cache, &[1, 2, 3]);
        let old = a.state_slot.unwrap();
        let before = cache.pool.borrow().free_count();
        assert!(matches!(
            cache.acquire(&[4, 5, 6], 7, None).unwrap(),
            AcquireOutcome::DeferredMemory
        ));
        assert_eq!(cache.pool.borrow().free_count(), before);
        assert_eq!(pool.borrow().usage(), (1, 0));
        cache.release(&mut a);
        let mut b = ready(&mut cache, &[4, 5, 6]);
        assert_ne!(b.state_slot.unwrap(), old);
        assert!(pool.borrow_mut().current_mut(old).is_err());
        cache.release(&mut b);
        assert_eq!(cache.pool.borrow().free_count(), 24);
    }
    #[test]
    fn publishing_stops_at_state_boundary_and_releases_private_tail() {
        let (mut cache, pool) = cache(1, 2);
        let ids = [1, 2, 3, 4, 5, 6, 7];
        let mut h = ready(&mut cache, &ids);
        publish(&mut cache, &pool, &mut h, &ids, 4, 4.);
        assert_eq!(h.published_pages, 2);
        assert_eq!(h.written_len, 7);
        assert!(
            h.owners[2..]
                .iter()
                .all(|owner| *owner == PageOwner::Private)
        );
        assert_eq!(cache.kv_match_path(&ids).len(), 2);
        cache.release(&mut h);
        assert_eq!(cache.pool.borrow().free_count(), 22);
        assert_eq!(cache.match_prefix(&ids).0, 4);
        cache.evict(24);
        assert_eq!(cache.pool.borrow().free_count(), 24);
        assert_eq!(pool.borrow().usage(), (0, 0));
    }

    #[test]
    fn zero_capacity_and_pinned_leaf_exhaustion_publish_no_orphan_kv() {
        for capacity in [0, 1] {
            let (mut cache, pool) = cache(2, capacity);
            let mut active = ready(&mut cache, &[1, 2, 3]);
            publish(&mut cache, &pool, &mut active, &[1, 2, 3], 2, 1.);
            let mut second = ready(&mut cache, &[4, 5, 6, 7, 8]);
            publish(&mut cache, &pool, &mut second, &[4, 5, 6, 7, 8], 4, 2.);
            assert_eq!(second.published_pages, 0);
            assert!(
                pool.borrow_mut()
                    .snapshot_states_mut(second.state_slot.unwrap())
                    .is_err()
            );
            assert_eq!(cache.kv_match_path(&[4, 5, 6, 7, 8]).len(), 0);
            cache.release(&mut second);
            cache.release(&mut active);
            assert_eq!(pool.borrow().usage(), (0, capacity));
            cache.evict(24);
            assert_eq!(cache.pool.borrow().free_count(), 24);
        }
    }

    #[test]
    fn internal_state_eviction_preserves_deeper_checkpoint_and_repairs_branch() {
        let (mut cache, pool) = cache(2, 3);
        let ids = [1, 2, 3, 4, 5, 6, 7];
        let mut h = ready(&mut cache, &ids);
        publish(&mut cache, &pool, &mut h, &ids[..3], 2, 2.);
        publish(&mut cache, &pool, &mut h, &ids, 6, 6.);
        cache.release(&mut h);
        assert!(cache.evict_checkpoint().unwrap()); // 内部状态 2 被淘汰，KV 路径保留。
        assert_eq!(cache.kv_match_path(&ids).len(), 3);
        assert_eq!(cache.match_prefix(&ids).0, 6);
        let branch = [1, 2, 3, 4, 9, 10, 11];
        let mut h = ready(&mut cache, &branch);
        assert_eq!(h.cached_len, 0);
        assert_eq!(
            pool.borrow()
                .branching_seqlen(h.state_slot.unwrap())
                .unwrap(),
            Some(4)
        );
        publish(&mut cache, &pool, &mut h, &branch, 4, 4.);
        cache.release(&mut h);
        assert_eq!(cache.match_prefix(&branch).0, 4);
        assert_eq!(cache.match_prefix(&ids).0, 6);
        cache.evict(24);
        assert_eq!(cache.pool.borrow().free_count(), 24);
        assert_eq!(pool.borrow().usage(), (0, 0));
    }

    #[test]
    fn evicting_pinned_internal_checkpoint_keeps_path_until_request_release() {
        let (mut cache, pool) = cache(2, 2);
        let ids = [1, 2, 3, 4, 5];
        let mut h = ready(&mut cache, &ids);
        publish(&mut cache, &pool, &mut h, &ids[..3], 2, 2.);
        publish(&mut cache, &pool, &mut h, &ids, 4, 4.);
        cache.release(&mut h);
        let mut active = ready(&mut cache, &ids);
        assert!(cache.evict_checkpoint().unwrap()); // 内部节点可独立释放状态。
        assert_eq!(cache.kv_match_path(&ids).len(), 2);
        assert_eq!(cache.match_prefix(&ids).0, 4);
        assert_eq!(
            pool.borrow_mut()
                .current_mut(active.state_slot.unwrap())
                .unwrap()
                .position,
            4
        );
        assert!(!cache.evict_checkpoint().unwrap()); // pin 住的叶子不级联淘汰。
        cache.release(&mut active);
        assert!(cache.evict_checkpoint().unwrap());
        assert_eq!(cache.nodes.iter().flatten().count(), 1);
        assert_eq!(cache.pool.borrow().free_count(), 24);
    }

    #[test]
    fn duplicate_checkpoint_drops_request_snapshot_and_keeps_canonical_state() {
        let (mut cache, pool) = cache(2, 2);
        let ids = [1, 2, 3, 4, 5];
        let mut a = ready(&mut cache, &ids);
        let mut b = ready(&mut cache, &ids); // 在 A 发布前取得相同的冷请求。
        publish(&mut cache, &pool, &mut a, &ids, 4, 7.);
        publish(&mut cache, &pool, &mut b, &ids, 4, 99.);
        assert_eq!(cache.checkpoint_count(), 1);
        assert!(
            pool.borrow_mut()
                .snapshot_states_mut(a.state_slot.unwrap())
                .is_err()
        );
        assert!(
            pool.borrow_mut()
                .snapshot_states_mut(b.state_slot.unwrap())
                .is_err()
        );
        cache.release(&mut a);
        cache.release(&mut b);
        let mut restored = ready(&mut cache, &ids);
        assert_eq!(
            pool.borrow_mut()
                .current_mut(restored.state_slot.unwrap())
                .unwrap()
                .layers[0]
                .recurrent
                .min()
                .double_value(&[]),
            7.
        );
        cache.release(&mut restored);
        cache.evict(24);
        assert_eq!(pool.borrow().usage(), (0, 0));
    }

    #[test]
    fn kv_without_successful_snapshot_is_not_a_hybrid_cache_hit() {
        let (mut cache, pool) = cache(1, 1);
        let mut h = ready(&mut cache, &[1, 2, 3, 4, 5]);
        cache.publish(&mut h, &[1, 2, 3, 4, 5]).unwrap();
        cache.release(&mut h);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5]).0, 0);
        assert_eq!(pool.borrow().usage(), (0, 0));
        assert_eq!(cache.nodes.iter().flatten().count(), 1);
        assert_eq!(cache.pool.borrow().free_count(), 24);
    }
}
