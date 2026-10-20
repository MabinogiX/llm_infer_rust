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
    /// 尚未完成 prefill；游标仅在 KV 提交成功后推进。
    Prefilling {
        cursor: usize,
    },
    Decoding,
    Finished,
}

#[derive(Debug, PartialEq)]
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

    pub fn is_decoding(&self) -> bool {
        self.status == SequenceStatus::Decoding
    }

    pub fn prefill_cursor(&self) -> Option<usize> {
        match self.status {
            SequenceStatus::Prefilling { cursor } => Some(cursor),
            _ => None,
        }
    }

    pub(crate) fn prompt_len(&self) -> usize {
        self.input_ids.len() - self.output_len
    }

    pub(crate) fn begin_prefill(&mut self, handle: BaseCacheHandle) {
        assert_eq!(self.status, SequenceStatus::Waiting);
        assert!(handle.cached_len <= self.prompt_len());
        self.cached_len = handle.cached_len;
        self.status = SequenceStatus::Prefilling {
            cursor: handle.cached_len,
        };
        self.cache_handle = Some(handle);
    }

    pub(crate) fn advance_prefill(&mut self, end: usize) {
        let cursor = self.prefill_cursor().expect("请求必须处于 prefill 阶段");
        assert!(cursor <= end && end <= self.prompt_len());
        self.status = SequenceStatus::Prefilling { cursor: end };
    }

    pub(crate) fn finish_prefill(&mut self) {
        assert!(self.prefill_cursor().is_some());
        self.status = SequenceStatus::Decoding;
    }

    pub(crate) fn restore_waiting(&mut self) {
        assert!(self.cache_handle.is_none());
        assert_eq!(self.output_len, 0);
        self.cached_len = 0;
        self.status = SequenceStatus::Waiting;
    }

    pub(crate) fn finish(&mut self) {
        self.status = SequenceStatus::Finished;
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
