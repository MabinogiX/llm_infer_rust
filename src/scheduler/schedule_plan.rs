//! Scheduler decisions separated from GPU launch and result completion.

use crate::engine::{Batch, BatchPhase};

use super::{DecodeBatch, RequestId};
use super::{future_map::RelayPlan, prefill::PrefillChunk};

pub(crate) struct SchedulePlan {
    pub(crate) phase: BatchPhase,
    pub(crate) request_ids: Vec<RequestId>,
    pub(crate) model_batch: Batch,
    /// Present only when this decode consumes the previous decode's device
    /// token buffer. Initial decode and prefill use batch-owned inputs.
    pub(crate) relay: Option<RelayPlan>,
    pub(crate) relay_offset: Option<usize>,
    pub(crate) chunk: Option<PrefillChunk>,
}

impl SchedulePlan {
    pub(crate) fn prefill(
        request_ids: Vec<RequestId>,
        model_batch: Batch,
        chunk: Option<PrefillChunk>,
    ) -> Self {
        Self {
            phase: BatchPhase::Prefill,
            request_ids,
            model_batch,
            relay: None,
            relay_offset: None,
            chunk,
        }
    }

    pub(crate) fn decode(batch: DecodeBatch, relay: Option<RelayPlan>) -> Self {
        Self {
            phase: BatchPhase::Decode,
            request_ids: batch.request_ids,
            model_batch: batch.model_batch,
            relay,
            relay_offset: None,
            chunk: None,
        }
    }

    pub(crate) fn mixed(
        request_ids: Vec<RequestId>,
        model_batch: Batch,
        relay: RelayPlan,
        relay_offset: usize,
        chunk: Option<PrefillChunk>,
    ) -> Self {
        Self {
            phase: BatchPhase::Prefill,
            request_ids,
            model_batch,
            relay: Some(relay),
            relay_offset: Some(relay_offset),
            chunk,
        }
    }
}
