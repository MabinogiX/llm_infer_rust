//! 请求当前状态、可移交历史状态和 radix 检查点共用状态槽池。
use super::{KVCacheError, Result};
use std::{cell::RefCell, rc::Rc};
use tch::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateSlotId {
    pub(crate) index: usize,
    generation: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckpointId {
    index: usize,
    generation: u64,
}
pub type SharedRecurrentPool = Rc<RefCell<RecurrentStatePool>>;

pub struct LinearState {
    pub conv: Tensor,
    pub recurrent: Tensor,
}
pub struct RequestState {
    pub position: i64,
    pub layers: Vec<LinearState>,
}
impl RequestState {
    fn clear(&mut self) {
        self.position = 0;
        for layer in &mut self.layers {
            let _ = layer.conv.zero_();
            let _ = layer.recurrent.zero_();
        }
    }
    fn copy_from(&mut self, source: &Self) {
        self.position = source.position;
        for (dst, src) in self.layers.iter_mut().zip(&source.layers) {
            dst.conv.copy_(&src.conv);
            dst.recurrent.copy_(&src.recurrent);
        }
    }
}
struct RequestEntry {
    generation: u64,
    allocated: bool,
    current: Option<usize>,
    snapshot: Option<CheckpointId>,
    staged_position: Option<usize>,
    mamba_branching_seqlen: Option<usize>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    Free,
    Current(StateSlotId),
    Tracking(StateSlotId),
    Radix,
}
struct StateEntry {
    generation: u64,
    owner: Owner,
    readers: usize,
    state: RequestState,
}
/// 请求独占 current 和尚未发布的历史槽；发布仅转移历史槽所有权，不复制张量。
/// 已发布状态不可写，恢复时复制到请求 current。所有设备操作使用同一执行流。
pub struct RecurrentStatePool {
    requests: Vec<RequestEntry>,
    slots: Vec<StateEntry>,
    checkpoint_capacity: usize,
}
fn invalid(message: &str) -> KVCacheError {
    KVCacheError::InvalidArgument(message.into())
}
impl RecurrentStatePool {
    pub fn empty() -> Self {
        Self {
            requests: Vec::new(),
            slots: Vec::new(),
            checkpoint_capacity: 0,
        }
    }
    /// `request_capacity` 限制同时持有状态的请求数。
    /// `checkpoint_capacity` 限制 radix 持有的历史状态数，不包含 current 和待发布快照。
    /// 开启历史缓存时总槽数为 2 × request_capacity + checkpoint_capacity；否则仅预留 current。
    pub fn new(
        request_capacity: usize,
        checkpoint_capacity: usize,
        mut make: impl FnMut() -> RequestState,
    ) -> Self {
        // 保留 current + 每请求一个待发布历史状态 + radix 上限的峰值容量。
        // 未开启前缀状态缓存时只预留 current，不分配历史状态空间。
        let storage_capacity = if checkpoint_capacity == 0 {
            request_capacity
        } else {
            2 * request_capacity + checkpoint_capacity
        };
        Self {
            requests: (0..request_capacity)
                .map(|_| RequestEntry {
                    generation: 0,
                    allocated: false,
                    current: None,
                    snapshot: None,
                    staged_position: None,
                    mamba_branching_seqlen: None,
                })
                .collect(),
            slots: (0..storage_capacity)
                .map(|_| StateEntry {
                    generation: 0,
                    owner: Owner::Free,
                    readers: 0,
                    state: make(),
                })
                .collect(),
            checkpoint_capacity,
        }
    }
    fn alloc_slot(&mut self, owner: Owner) -> Option<CheckpointId> {
        let (index, entry) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, e)| e.owner == Owner::Free)?;
        entry.generation = entry.generation.checked_add(1).expect("状态槽代际溢出");
        entry.owner = owner;
        entry.readers = 0;
        Some(CheckpointId {
            index,
            generation: entry.generation,
        })
    }
    pub fn alloc_request(&mut self) -> Option<StateSlotId> {
        let index = self.requests.iter().position(|e| !e.allocated)?;
        let generation = self.requests[index]
            .generation
            .checked_add(1)
            .expect("请求槽代际溢出");
        let id = StateSlotId { index, generation };
        let current = self.alloc_slot(Owner::Current(id))?.index;
        self.slots[current].state.clear();
        let entry = &mut self.requests[index];
        entry.generation = generation;
        entry.allocated = true;
        entry.current = Some(current);
        entry.snapshot = None;
        entry.staged_position = None;
        entry.mamba_branching_seqlen = None;
        Some(id)
    }
    fn request(&self, id: StateSlotId) -> Result<&RequestEntry> {
        self.requests
            .get(id.index)
            .filter(|e| e.allocated && e.generation == id.generation)
            .ok_or_else(|| invalid("无效或已释放的请求状态槽"))
    }
    fn request_mut(&mut self, id: StateSlotId) -> Result<&mut RequestEntry> {
        self.requests
            .get_mut(id.index)
            .filter(|e| e.allocated && e.generation == id.generation)
            .ok_or_else(|| invalid("无效或已释放的请求状态槽"))
    }
    fn owned_slot(&self, id: CheckpointId, owner: Owner) -> Result<&StateEntry> {
        self.slots
            .get(id.index)
            .filter(|e| e.owner == owner && e.generation == id.generation)
            .ok_or_else(|| invalid("状态槽代际或所有权不匹配"))
    }
    pub fn discard_snapshot(&mut self, id: StateSlotId) -> Result<()> {
        let snapshot = self.request(id)?.snapshot;
        if let Some(snapshot) = snapshot {
            self.owned_slot(snapshot, Owner::Tracking(id))?;
            self.slots[snapshot.index].owner = Owner::Free;
        }
        let entry = self.request_mut(id)?;
        entry.snapshot = None;
        entry.staged_position = None;
        Ok(())
    }
    pub fn free_request(&mut self, id: StateSlotId) -> Result<()> {
        self.discard_snapshot(id)?;
        let current = self.request(id)?.current.unwrap();
        self.slots[current].owner = Owner::Free;
        let entry = self.request_mut(id)?;
        entry.allocated = false;
        entry.current = None;
        entry.mamba_branching_seqlen = None;
        Ok(())
    }
    pub fn current_mut(&mut self, id: StateSlotId) -> Result<&mut RequestState> {
        let current = self.request(id)?.current.unwrap();
        Ok(&mut self.slots[current].state)
    }
    fn pair_mut(&mut self, a: usize, b: usize) -> (&mut RequestState, &mut RequestState) {
        assert_ne!(a, b, "当前状态与历史状态不得别名");
        if a < b {
            let (left, right) = self.slots.split_at_mut(b);
            (&mut left[a].state, &mut right[0].state)
        } else {
            let (left, right) = self.slots.split_at_mut(a);
            (&mut right[0].state, &mut left[b].state)
        }
    }
    pub fn snapshot_states_mut(
        &mut self,
        id: StateSlotId,
    ) -> Result<(&mut RequestState, &mut RequestState)> {
        let entry = self.request(id)?;
        let current = entry.current.unwrap();
        let snapshot = entry
            .snapshot
            .ok_or_else(|| invalid("请求尚未申请历史状态槽"))?;
        self.owned_slot(snapshot, Owner::Tracking(id))?;
        Ok(self.pair_mut(current, snapshot.index))
    }
    /// 仅完整 forward 成功后提交状态；失败时历史槽仍不可发布，由请求清理回收。
    pub fn commit_forward(
        &mut self,
        id: StateSlotId,
        end: usize,
        staged: Option<usize>,
    ) -> Result<()> {
        let current = self.request(id)?.current.unwrap();
        if let Some(position) = staged {
            if position == 0 || position > end {
                return Err(invalid("历史状态位置超出本次 forward"));
            }
            let snapshot = self
                .request(id)?
                .snapshot
                .ok_or_else(|| invalid("没有待提交的历史状态槽"))?;
            self.owned_slot(snapshot, Owner::Tracking(id))?;
            self.slots[snapshot.index].state.position = position as i64;
        }
        self.slots[current].state.position = end as i64;
        let entry = self.request_mut(id)?;
        if let Some(position) = staged {
            entry.staged_position = Some(position);
            if entry
                .mamba_branching_seqlen
                .is_some_and(|target| position >= target)
            {
                entry.mamba_branching_seqlen = None;
            }
        }
        Ok(())
    }
    /// 复用请求仍拥有的历史槽，或从统一空闲池申请；已移交的槽绝不再写。
    pub fn begin_snapshot(&mut self, id: StateSlotId) -> Result<bool> {
        self.request_mut(id)?.staged_position = None;
        if self.checkpoint_capacity == 0 {
            return Ok(false);
        }
        if let Some(snapshot) = self.request(id)?.snapshot {
            self.owned_slot(snapshot, Owner::Tracking(id))?;
            return Ok(true);
        }
        let Some(snapshot) = self.alloc_slot(Owner::Tracking(id)) else {
            return Ok(false);
        };
        self.request_mut(id)?.snapshot = Some(snapshot);
        Ok(true)
    }
    pub fn staged_position(&self, id: StateSlotId) -> Result<Option<usize>> {
        Ok(self.request(id)?.staged_position)
    }
    pub fn set_branching_seqlen(&mut self, id: StateSlotId, position: Option<usize>) -> Result<()> {
        self.request_mut(id)?.mamba_branching_seqlen = position;
        Ok(())
    }
    pub fn branching_seqlen(&self, id: StateSlotId) -> Result<Option<usize>> {
        Ok(self.request(id)?.mamba_branching_seqlen)
    }
    pub fn checkpoint_capacity(&self) -> usize {
        self.checkpoint_capacity
    }
    pub fn storage_capacity(&self) -> usize {
        self.slots.len()
    }
    /// 将已提交的历史槽移交 radix；没有任何状态张量复制或新槽分配。
    pub fn save_checkpoint(
        &mut self,
        id: StateSlotId,
        position: usize,
    ) -> Result<Option<CheckpointId>> {
        let source = self.request(id)?;
        if source.staged_position != Some(position) {
            return Err(invalid("检查点长度与已提交状态不一致"));
        }
        let snapshot = source
            .snapshot
            .ok_or_else(|| invalid("没有可移交的历史状态"))?;
        self.owned_slot(snapshot, Owner::Tracking(id))?;
        if self.usage().1 >= self.checkpoint_capacity {
            return Ok(None);
        }
        self.slots[snapshot.index].owner = Owner::Radix;
        let source = self.request_mut(id)?;
        source.snapshot = None;
        source.staged_position = None;
        Ok(Some(snapshot))
    }
    pub fn restore(
        &mut self,
        checkpoint: CheckpointId,
        request: StateSlotId,
        position: usize,
    ) -> Result<()> {
        let current = self.request(request)?.current.unwrap();
        if self.owned_slot(checkpoint, Owner::Radix)?.state.position != position as i64 {
            return Err(invalid("检查点与 KV 前缀长度不一致"));
        }
        self.discard_snapshot(request)?;
        self.slots[checkpoint.index].readers += 1;
        let (target, source) = self.pair_mut(current, checkpoint.index);
        target.copy_from(source);
        self.slots[checkpoint.index].readers -= 1;
        Ok(())
    }
    pub fn free_checkpoint(&mut self, id: CheckpointId) -> Result<()> {
        if self.owned_slot(id, Owner::Radix)?.readers != 0 {
            return Err(invalid("检查点仍在恢复中"));
        }
        self.slots[id.index].owner = Owner::Free;
        Ok(())
    }
    pub fn usage(&self) -> (usize, usize) {
        (
            self.requests.iter().filter(|e| e.allocated).count(),
            self.slots
                .iter()
                .filter(|e| e.owner == Owner::Radix)
                .count(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};
    fn make() -> RequestState {
        RequestState {
            position: 0,
            layers: vec![LinearState {
                conv: Tensor::zeros([2, 3], (Kind::Float, Device::Cpu)),
                recurrent: Tensor::zeros([2, 2, 2], (Kind::Float, Device::Cpu)),
            }],
        }
    }
    fn capture(
        pool: &mut RecurrentStatePool,
        id: StateSlotId,
        position: usize,
        value: f64,
    ) -> usize {
        assert!(pool.begin_snapshot(id).unwrap());
        let (_, state) = pool.snapshot_states_mut(id).unwrap();
        let _ = state.layers[0].conv.fill_(value);
        let _ = state.layers[0].recurrent.fill_(value);
        let pointer = state.layers[0].recurrent.data_ptr() as usize;
        pool.commit_forward(id, position + 1, Some(position))
            .unwrap();
        pointer
    }
    fn free_slots(pool: &RecurrentStatePool) -> usize {
        pool.slots.iter().filter(|e| e.owner == Owner::Free).count()
    }
    #[test]
    fn donation_preserves_tensor_address_and_restores_private_current() {
        let mut pool = RecurrentStatePool::new(2, 2, make);
        let a = pool.alloc_request().unwrap();
        let b = pool.alloc_request().unwrap();
        let pointer = capture(&mut pool, a, 4, 7.);
        let checkpoint = pool.save_checkpoint(a, 4).unwrap().unwrap();
        assert_eq!(
            pool.slots[checkpoint.index].state.layers[0]
                .recurrent
                .data_ptr() as usize,
            pointer
        );
        assert_eq!(pool.staged_position(a).unwrap(), None);
        assert!(pool.snapshot_states_mut(a).is_err());
        assert!(pool.save_checkpoint(a, 4).is_err()); // 所有权已移交，不能二次发布。
        let _ = pool.current_mut(a).unwrap().layers[0].recurrent.fill_(99.);
        let replacement_pointer = capture(&mut pool, a, 6, 88.);
        assert_ne!(pointer, replacement_pointer); // 下一次捕获不能覆盖 radix 所有的槽。
        assert!(pool.restore(checkpoint, b, 5).is_err());
        pool.restore(checkpoint, b, 4).unwrap();
        assert_eq!(pool.current_mut(b).unwrap().position, 4);
        assert_eq!(
            pool.current_mut(b).unwrap().layers[0]
                .recurrent
                .min()
                .double_value(&[]),
            7.
        );
        assert_ne!(
            pool.current_mut(b).unwrap().layers[0].recurrent.data_ptr() as usize,
            pointer
        );
        pool.free_request(a).unwrap(); // 归还 current 与尚未发布的替换槽。
        assert_eq!(pool.usage(), (1, 1));
        pool.free_checkpoint(checkpoint).unwrap();
        assert!(pool.restore(checkpoint, b, 4).is_err());
        assert_eq!(
            pool.current_mut(b).unwrap().layers[0]
                .conv
                .min()
                .double_value(&[]),
            7.
        );
        pool.free_request(b).unwrap();
        assert_eq!(free_slots(&pool), pool.storage_capacity());
    }
    #[test]
    fn uncommitted_snapshots_and_stale_generations_are_rejected() {
        let mut pool = RecurrentStatePool::new(1, 1, make);
        let a = pool.alloc_request().unwrap();
        assert!(pool.save_checkpoint(a, 4).is_err());
        capture(&mut pool, a, 4, 3.);
        assert!(pool.save_checkpoint(a, 5).is_err());
        assert!(pool.begin_snapshot(a).unwrap());
        assert_eq!(pool.staged_position(a).unwrap(), None);
        assert!(pool.save_checkpoint(a, 4).is_err()); // 部分 forward 不能发布旧位置标记。
        pool.commit_forward(a, 7, Some(6)).unwrap();
        let old = pool.save_checkpoint(a, 6).unwrap().unwrap();
        pool.slots[old.index].readers = 1;
        assert!(pool.free_checkpoint(old).is_err());
        pool.slots[old.index].readers = 0;
        pool.free_checkpoint(old).unwrap();
        capture(&mut pool, a, 8, 5.);
        let new = pool.save_checkpoint(a, 8).unwrap().unwrap();
        assert_eq!(old.index, new.index);
        assert_ne!(old, new);
        assert!(pool.free_checkpoint(old).is_err());
        pool.free_request(a).unwrap();
        assert!(pool.free_request(a).is_err());
        let reused = pool.alloc_request().unwrap();
        assert_ne!(a, reused);
        assert!(pool.current_mut(a).is_err());
        assert_eq!(
            pool.current_mut(reused).unwrap().layers[0]
                .recurrent
                .sum(Kind::Float)
                .double_value(&[]),
            0.
        );
        pool.free_checkpoint(new).unwrap();
        pool.free_request(reused).unwrap();
        assert_eq!(free_slots(&pool), pool.storage_capacity());
    }
    #[test]
    fn full_checkpoint_budget_retains_snapshot_for_retry_and_discards_unadopted_slot() {
        let mut pool = RecurrentStatePool::new(1, 1, make);
        let a = pool.alloc_request().unwrap();
        capture(&mut pool, a, 4, 4.);
        let first = pool.save_checkpoint(a, 4).unwrap().unwrap();
        capture(&mut pool, a, 6, 6.);
        let before = free_slots(&pool);
        assert!(pool.save_checkpoint(a, 6).unwrap().is_none());
        assert_eq!(pool.staged_position(a).unwrap(), Some(6)); // 淘汰旧 checkpoint 后可重试。
        pool.discard_snapshot(a).unwrap();
        assert_eq!(free_slots(&pool), before + 1);
        assert!(pool.snapshot_states_mut(a).is_err());
        capture(&mut pool, a, 8, 8.);
        pool.free_checkpoint(first).unwrap();
        let next = pool.save_checkpoint(a, 8).unwrap().unwrap();
        pool.free_request(a).unwrap();
        assert_eq!(pool.usage(), (0, 1));
        pool.free_checkpoint(next).unwrap();
        assert_eq!(free_slots(&pool), pool.storage_capacity());
    }
    #[test]
    fn disabled_checkpoints_allocate_only_current_and_no_history_tensors() {
        let mut tensors = 0;
        let mut pool = RecurrentStatePool::new(3, 0, || {
            tensors += 1;
            make()
        });
        assert_eq!(tensors, 3);
        assert_eq!(pool.storage_capacity(), 3);
        let a = pool.alloc_request().unwrap();
        assert!(!pool.begin_snapshot(a).unwrap());
        assert!(pool.snapshot_states_mut(a).is_err());
        pool.commit_forward(a, 10, None).unwrap();
        assert_eq!(pool.current_mut(a).unwrap().position, 10);
        pool.free_request(a).unwrap();
        assert_eq!(free_slots(&pool), 3);
    }
    #[test]
    fn restore_and_cancel_reclaim_pending_or_failed_history_slots() {
        let mut pool = RecurrentStatePool::new(2, 1, make);
        let a = pool.alloc_request().unwrap();
        let b = pool.alloc_request().unwrap();
        capture(&mut pool, a, 4, 4.);
        let checkpoint = pool.save_checkpoint(a, 4).unwrap().unwrap();
        capture(&mut pool, b, 6, 6.);
        let before = free_slots(&pool);
        pool.restore(checkpoint, b, 4).unwrap();
        assert_eq!(free_slots(&pool), before + 1);
        assert_eq!(pool.staged_position(b).unwrap(), None);
        assert!(pool.begin_snapshot(b).unwrap());
        let _ = pool.snapshot_states_mut(b).unwrap().1.layers[0]
            .conv
            .fill_(999.);
        assert!(pool.save_checkpoint(b, 4).is_err()); // 模拟未完成 forward。
        pool.free_request(b).unwrap();
        pool.free_request(a).unwrap();
        pool.free_checkpoint(checkpoint).unwrap();
        assert_eq!(free_slots(&pool), pool.storage_capacity());
    }
    #[test]
    fn repeated_donation_eviction_and_request_reuse_do_not_grow_or_leak_storage() {
        let mut tensors = 0;
        let mut pool = RecurrentStatePool::new(2, 2, || {
            tensors += 1;
            make()
        });
        assert_eq!(tensors, 6);
        for i in 1..100 {
            let a = pool.alloc_request().unwrap();
            let b = pool.alloc_request().unwrap();
            capture(&mut pool, a, 4, i as f64);
            let checkpoint = pool.save_checkpoint(a, 4).unwrap().unwrap();
            pool.restore(checkpoint, b, 4).unwrap();
            assert_eq!(
                pool.current_mut(b).unwrap().layers[0]
                    .recurrent
                    .min()
                    .double_value(&[]),
                i as f64
            );
            pool.free_checkpoint(checkpoint).unwrap();
            pool.free_request(a).unwrap();
            pool.free_request(b).unwrap();
            assert_eq!(free_slots(&pool), 6);
        }
        assert_eq!(pool.storage_capacity(), tensors);
    }
}
