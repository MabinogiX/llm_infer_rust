//! Scheduler decisions separated from GPU launch and result completion.

use crate::engine::{Batch, BatchPhase};

use super::{DecodeBatch, RequestId};
use super::{future_map::RelayPlan, prefill::PrefillChunk};

pub(crate) enum SchedulePlan {
    Prefill {
        request_ids: Vec<RequestId>,
        model_batch: Batch,
        chunk: Option<PrefillChunk>,
    },
    Decode {
        batch: DecodeBatch,
        relay: Option<RelayPlan>,
    },
    Mixed {
        request_ids: Vec<RequestId>,
        model_batch: Batch,
        relay: RelayPlan,
        prefill_tokens: usize,
        chunk: Option<PrefillChunk>,
    },
}

impl SchedulePlan {
    pub(crate) fn request_ids(&self) -> &[RequestId] {
        match self {
            Self::Prefill { request_ids, .. } | Self::Mixed { request_ids, .. } => request_ids,
            Self::Decode { batch, .. } => &batch.request_ids,
        }
    }

    pub(crate) fn phase(&self) -> BatchPhase {
        match self {
            Self::Prefill { .. } | Self::Mixed { .. } => BatchPhase::Prefill,
            Self::Decode { .. } => BatchPhase::Decode,
        }
    }

    pub(crate) fn prefill(
        request_ids: Vec<RequestId>,
        model_batch: Batch,
        chunk: Option<PrefillChunk>,
    ) -> Self {
        Self::Prefill {
            request_ids,
            model_batch,
            chunk,
        }
    }

    pub(crate) fn decode(batch: DecodeBatch, relay: Option<RelayPlan>) -> Self {
        Self::Decode { batch, relay }
    }

    pub(crate) fn mixed(
        request_ids: Vec<RequestId>,
        model_batch: Batch,
        relay: RelayPlan,
        prefill_tokens: usize,
        chunk: Option<PrefillChunk>,
    ) -> Self {
        Self::Mixed {
            request_ids,
            model_batch,
            relay,
            prefill_tokens,
            chunk,
        }
    }
}
