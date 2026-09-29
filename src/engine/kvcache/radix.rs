//! Page-granular prefix cache. Only successful forwards publish KV pages.

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use super::{AcquireOutcome, BaseCacheHandle, CacheManager, KVCacheError, KVCachePool, Result};
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
        }
    }
}

/// The scheduler owns this manager exclusively; the pool is shared with Engine.
pub struct RadixCacheManager {
    pool: Rc<RefCell<KVCachePool>>,
    page_size: usize,
    nodes: Vec<Option<RadixNode>>,
    free_node_ids: Vec<NodeId>,
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
        })
    }

    fn node(&self, id: NodeId) -> &RadixNode {
        self.nodes[id].as_ref().expect("live radix node")
    }

    fn node_mut(&mut self, id: NodeId) -> &mut RadixNode {
        self.nodes[id].as_mut().expect("live radix node")
    }

    fn match_path(&self, input_ids: &[i64]) -> Vec<NodeId> {
        // Prefill must recompute at least one prompt token for its logits.
        let max_pages = input_ids.len().saturating_sub(1) / self.page_size;
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
            let node = self.nodes[id].take().expect("selected live leaf");
            let parent = node.parent.expect("leaf is not root");
            self.node_mut(parent).children.remove(&node.tokens);
            self.free_node_ids.push(id);
            self.pool.borrow_mut().free_pages_by_id([node.page_id]);
            freed.push(node.page_id);
        }
        freed
    }
}

impl CacheManager for RadixCacheManager {
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
            page_ids,
            cached_len,
            owners,
            published_pages: path.len(),
            written_len: cached_len,
        }))
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
            let parent = if index == 0 {
                0
            } else if let PageOwner::Tree(id) = handle.owners[index - 1] {
                id
            } else {
                return Err(KVCacheError::InvalidArgument(
                    "published page path is discontinuous".to_owned(),
                ));
            };
            let tokens = written_ids[index * self.page_size..(index + 1) * self.page_size].to_vec();
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
        Ok(())
    }

    fn release(&mut self, handle: &mut BaseCacheHandle) {
        let page_ids = std::mem::take(&mut handle.page_ids);
        let owners = std::mem::take(&mut handle.owners);
        assert_eq!(page_ids.len(), owners.len(), "KV handle ownership mismatch");
        let mut private = Vec::new();
        for (page_id, owner) in page_ids.into_iter().zip(owners) {
            match owner {
                PageOwner::Private => private.push(page_id),
                PageOwner::Tree(id) => {
                    let node = self.node_mut(id);
                    assert!(node.pin_count > 0, "radix pin underflow");
                    node.pin_count -= 1;
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
