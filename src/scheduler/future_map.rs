//! Stable per-request relay state for the decode overlap pipeline.
//!
//! The original SGLang scheduler indexes its FutureMap by a long-lived
//! request-pool row.  Keeping that indirection separate from the order of a
//! particular batch avoids scanning the previous batch's request IDs whenever
//! requests finish or a prefill is inserted.

use std::collections::{HashMap, HashSet};

use tch::{Device, Kind, Tensor};

use super::{Request, RequestId, Result, SchedulerError};

#[derive(Clone, Copy)]
pub(crate) struct CompletionInput {
    pub(crate) written_len: usize,
    pub(crate) length_finished: bool,
    pub(crate) ignore_eos: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct RelayRow {
    pub(crate) uid: RequestId,
    pub(crate) slot: usize,
    pub(crate) input: CompletionInput,
}

pub(crate) struct RelayPlan {
    rows: Vec<RelayRow>,
    by_uid: HashMap<RequestId, RelayRow>,
}

impl RelayPlan {
    pub(crate) fn rows(&self) -> impl Iterator<Item = RelayRow> + '_ {
        self.rows.iter().copied()
    }

    pub(crate) fn slots(&self) -> Vec<usize> {
        self.rows.iter().map(|row| row.slot).collect()
    }

    pub(crate) fn input(&self, uid: RequestId) -> Option<CompletionInput> {
        self.by_uid.get(&uid).map(|row| row.input)
    }

    pub(crate) fn row(&self, uid: RequestId) -> Option<RelayRow> {
        self.by_uid.get(&uid).copied()
    }
}

pub(crate) struct FutureMap {
    slots: Vec<Option<RequestId>>,
    by_uid: HashMap<RequestId, usize>,
    output_tokens: Tensor,
}

impl FutureMap {
    pub(crate) fn new(max_rows: usize, device: Device) -> Self {
        Self {
            slots: vec![None; max_rows],
            by_uid: HashMap::new(),
            output_tokens: Tensor::zeros([max_rows as i64], (Kind::Int64, device)),
        }
    }

    /// Device buffer indexed by stable request slots.  The next decode batch
    /// gathers from this buffer on the execution stream after `publish`.
    pub(crate) fn output_tokens(&self) -> &Tensor {
        &self.output_tokens
    }

    /// Publish sampled IDs into stable request slots.  The copy is queued on
    /// the current execution stream, immediately after sampling, so it does
    /// not require a CPU readback or a synchronizing event.
    pub(crate) fn publish(&mut self, request_ids: &[RequestId], ids: &Tensor) -> Result<()> {
        if request_ids.len() != ids.size().first().copied().unwrap_or_default() as usize {
            return Err(SchedulerError::InvalidDecode(
                "future map publish size differs from sampled IDs".to_owned(),
            ));
        }
        let slots = request_ids
            .iter()
            .map(|&uid| self.slot_for(uid))
            .collect::<Result<Vec<_>>>()?;
        if slots.is_empty() {
            return Ok(());
        }
        let index = Tensor::from_slice(&slots.iter().map(|&slot| slot as i64).collect::<Vec<_>>())
            .to_device(self.output_tokens.device());
        let _ = self.output_tokens.index_copy_(0, &index, ids);
        Ok(())
    }

    /// Build the next decode plan from the previous batch's CPU completion
    /// metadata.  Request order is taken from `running`, while token values
    /// are read from stable slots on the device.
    pub(crate) fn plan(
        &mut self,
        previous_ids: &[RequestId],
        inputs: &[CompletionInput],
        running: &[Request],
    ) -> Result<RelayPlan> {
        if previous_ids.len() != inputs.len() {
            return Err(SchedulerError::InvalidDecode(
                "future map input size differs from previous request IDs".to_owned(),
            ));
        }
        let previous = previous_ids
            .iter()
            .copied()
            .zip(inputs.iter().copied())
            .collect::<HashMap<_, _>>();
        let mut rows = Vec::with_capacity(running.len());
        for request in running.iter().filter(|request| !request.is_finished()) {
            let Some(&input) = previous.get(&request.uid) else {
                continue;
            };
            if input.length_finished {
                continue;
            }
            let slot = self.by_uid.get(&request.uid).copied().ok_or_else(|| {
                SchedulerError::InvalidDecode(format!(
                    "future map has no relay slot for request {}",
                    request.uid
                ))
            })?;
            rows.push(RelayRow {
                uid: request.uid,
                slot,
                input: CompletionInput {
                    written_len: input.written_len + 1,
                    ..input
                },
            });
        }
        let by_uid = rows.iter().copied().map(|row| (row.uid, row)).collect();
        Ok(RelayPlan { rows, by_uid })
    }

    /// Release slots whose requests are no longer running.  This is called
    /// only after the previous batch has been read and the next batch has been
    /// installed, so a reused row cannot race an outstanding GPU write.
    pub(crate) fn release_inactive(&mut self, running: &[Request]) {
        let active = running
            .iter()
            .map(|request| request.uid)
            .collect::<HashSet<_>>();
        let stale = self
            .by_uid
            .iter()
            .filter_map(|(&uid, &slot)| (!active.contains(&uid)).then_some((uid, slot)))
            .collect::<Vec<_>>();
        for (uid, slot) in stale {
            self.by_uid.remove(&uid);
            self.slots[slot] = None;
        }
    }

    fn slot_for(&mut self, uid: RequestId) -> Result<usize> {
        if let Some(&slot) = self.by_uid.get(&uid) {
            return Ok(slot);
        }
        let slot = self.slots.iter().position(Option::is_none).ok_or_else(|| {
            SchedulerError::InvalidDecode("future map relay slots exhausted".to_owned())
        })?;
        self.slots[slot] = Some(uid);
        self.by_uid.insert(uid, slot);
        Ok(slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_slots_stable_when_batch_order_changes() {
        let mut map = FutureMap::new(3, Device::Cpu);
        let ids = Tensor::from_slice(&[11_i64, 22]);
        map.publish(&[7, 9], &ids).unwrap();
        let inputs = vec![
            CompletionInput {
                written_len: 4,
                length_finished: false,
                ignore_eos: false,
            },
            CompletionInput {
                written_len: 5,
                length_finished: false,
                ignore_eos: false,
            },
        ];
        let running = vec![
            Request {
                uid: 9,
                input_ids: vec![1],
                sampling_params: Default::default(),
                cached_len: 0,
                output_len: 0,
                cache_handle: None,
                status: super::super::SequenceStatus::Running,
            },
            Request {
                uid: 7,
                input_ids: vec![1],
                sampling_params: Default::default(),
                cached_len: 0,
                output_len: 0,
                cache_handle: None,
                status: super::super::SequenceStatus::Running,
            },
        ];
        let plan = map.plan(&[7, 9], &inputs, &running).unwrap();
        let row_9 = plan.row(9).unwrap();
        let row_7 = plan.row(7).unwrap();
        assert_eq!(row_9.uid, 9);
        assert_eq!(row_7.uid, 7);
        assert_ne!(row_9.slot, row_7.slot);
        assert_eq!(row_9.input.written_len, 6);
    }
}
