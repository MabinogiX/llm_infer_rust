//! Request and output values exposed by the scheduler.

use crate::engine::SamplingParams;
use crate::engine::kvcache::BaseCacheHandle;

pub type RequestId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Abort,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputToken {
    pub uid: RequestId,
    /// The token is only meaningful when `finish_reason` is not Abort or Error.
    pub token_id: i64,
    pub finished: bool,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub uid: RequestId,
    pub input_ids: Vec<i64>,
    pub sampling_params: SamplingParams,
    pub cached_len: usize,
    pub output_len: usize,
    pub cache_handle: Option<BaseCacheHandle>,
    pub status: SequenceStatus,
}

impl Request {
    pub fn uncached_len(&self) -> usize {
        self.input_ids.len().saturating_sub(self.cached_len)
    }

    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished
    }

    pub fn append_token(&mut self, token_id: i64) {
        self.input_ids.push(token_id);
        self.output_len += 1;
    }

    /// The most recently sampled token has not entered the model's KV cache
    /// until a following decode forward consumes it.
    pub fn written_input_ids(&self) -> &[i64] {
        let unwritten = usize::from(self.output_len > 0);
        &self.input_ids[..self.input_ids.len().saturating_sub(unwritten)]
    }
}
