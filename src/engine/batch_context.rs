//! Scheduler batch preparation for prefill execution.

use std::fmt;

use tch::{Device, Tensor};

use super::{AttentionMetadata, Batch, BatchPhase};

/// Scheduler-side request data needed to derive one prefill [`Batch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchRequest {
    pub input_ids: Vec<i64>,
    pub cached_len: usize,
    /// Non-owning snapshot used only to construct this forward's tensors.
    pub page_ids: Vec<usize>,
}

impl BatchRequest {
    pub fn new(input_ids: Vec<i64>) -> Self {
        Self {
            input_ids,
            cached_len: 0,
            page_ids: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub enum BatchContextError {
    InvalidConfiguration(&'static str),
    EmptyBatch,
    CachedLengthExceedsInput { cached_len: usize, input_len: usize },
    EmptyUncachedRequest,
    IntegerOverflow(&'static str),
}

impl fmt::Display for BatchContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => write!(f, "invalid BatchContext: {message}"),
            Self::EmptyBatch => write!(f, "prefill Batch 不能没有请求"),
            Self::CachedLengthExceedsInput {
                cached_len,
                input_len,
            } => write!(
                f,
                "cached_len ({cached_len}) 不能大于 input_ids 长度 ({input_len})"
            ),
            Self::EmptyUncachedRequest => write!(f, "prefill 请求必须至少包含一个未缓存 token"),
            Self::IntegerOverflow(field) => {
                write!(f, "{field} exceeds supported tensor index range")
            }
        }
    }
}

impl std::error::Error for BatchContextError {}

pub type Result<T> = std::result::Result<T, BatchContextError>;

/// Builds prefill tensors from scheduler request and page-table metadata.
#[derive(Debug, Clone, Copy)]
pub struct BatchContext {
    max_running_req: usize,
    max_seq_len: usize,
    page_size: usize,
    device: Device,
}

impl BatchContext {
    pub fn new(
        max_running_req: usize,
        max_seq_len: usize,
        page_size: usize,
        device: Device,
    ) -> Result<Self> {
        if max_running_req == 0 || max_seq_len == 0 || page_size == 0 {
            return Err(BatchContextError::InvalidConfiguration(
                "max_running_req, max_seq_len, and page_size must be greater than zero",
            ));
        }
        Ok(Self {
            max_running_req,
            max_seq_len,
            page_size,
            device,
        })
    }

    /// Prepares a flattened prefill batch and its page-table attention metadata.
    pub fn prepare_prefill(&self, requests: &[BatchRequest]) -> Result<Batch> {
        if requests.is_empty() {
            return Err(BatchContextError::EmptyBatch);
        }
        if requests.len() > self.max_running_req {
            return Err(BatchContextError::InvalidConfiguration(
                "request count exceeds max_running_req",
            ));
        }

        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut write_loc = Vec::new();
        let mut prefix_lens = Vec::with_capacity(requests.len());
        let mut sequence_lengths = Vec::with_capacity(requests.len());
        let mut logits_indices = Vec::with_capacity(requests.len());
        let mut offset = 0usize;

        for request in requests {
            if request.cached_len > request.input_ids.len() {
                return Err(BatchContextError::CachedLengthExceedsInput {
                    cached_len: request.cached_len,
                    input_len: request.input_ids.len(),
                });
            }
            if request.input_ids.len() > self.max_seq_len {
                return Err(BatchContextError::InvalidConfiguration(
                    "request input exceeds max_seq_len",
                ));
            }

            let uncached = &request.input_ids[request.cached_len..];
            if uncached.is_empty() {
                return Err(BatchContextError::EmptyUncachedRequest);
            }
            input_ids.extend_from_slice(uncached);
            positions.extend(
                (request.cached_len..request.input_ids.len())
                    .map(|position| {
                        i64::try_from(position)
                            .map_err(|_| BatchContextError::IntegerOverflow("position"))
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
            write_loc.extend(self.build_write_locations(request)?);
            prefix_lens.push(to_i32(request.cached_len, "cached_len")?);
            sequence_lengths.push(to_i32(uncached.len(), "sequence length")?);
            logits_indices.push(to_i64(
                offset
                    .checked_add(uncached.len() - 1)
                    .ok_or(BatchContextError::IntegerOverflow("logits index"))?,
                "logits index",
            )?);
            offset = offset
                .checked_add(uncached.len())
                .ok_or(BatchContextError::IntegerOverflow("total token count"))?;
        }

        let cumulative_lengths = cumulative_lengths(&sequence_lengths)?;
        let block_table = self.build_block_table(requests)?;
        let req_to_token = self.build_req_to_token(requests)?;

        Ok(Batch::prefill(
            Tensor::from_slice(&input_ids).to_device(self.device),
            Tensor::from_slice(&positions).to_device(self.device),
            Some(AttentionMetadata {
                forward_mode: BatchPhase::Prefill,
                write_loc: Some(Tensor::from_slice(&write_loc).to_device(self.device)),
                cu_seqlens_q: Some(Tensor::from_slice(&cumulative_lengths).to_device(self.device)),
                prefix_lens: Some(Tensor::from_slice(&prefix_lens).to_device(self.device)),
                block_table: Some(block_table),
                req_to_token: Some(req_to_token),
                cache_seqlens: None,
                max_seqlen: sequence_lengths.iter().map(|&length| length as usize).max(),
            }),
            Tensor::from_slice(&logits_indices).to_device(self.device),
        ))
    }

    fn build_write_locations(&self, request: &BatchRequest) -> Result<Vec<i32>> {
        let mut locations = Vec::with_capacity(request.input_ids.len() - request.cached_len);
        for position in request.cached_len..request.input_ids.len() {
            let location = request
                .page_ids
                .get(position / self.page_size)
                .map(|&page_id| cache_location(page_id, position % self.page_size, self.page_size))
                .transpose()?
                .unwrap_or(-1);
            locations.push(location);
        }
        Ok(locations)
    }

    fn build_block_table(&self, requests: &[BatchRequest]) -> Result<Tensor> {
        let max_blocks = self.max_seq_len.div_ceil(self.page_size);
        let row_len = to_i64(max_blocks, "max blocks")?;
        let mut values = vec![-1i32; requests.len() * max_blocks];
        for (row, request) in requests.iter().enumerate() {
            for (column, &page_id) in request.page_ids.iter().take(max_blocks).enumerate() {
                values[row * max_blocks + column] = to_i32(page_id, "page id")?;
            }
        }
        Ok(Tensor::from_slice(&values)
            .view([to_i64(requests.len(), "request count")?, row_len])
            .to_device(self.device))
    }

    fn build_req_to_token(&self, requests: &[BatchRequest]) -> Result<Tensor> {
        let mut values = vec![-1i32; requests.len() * self.max_seq_len];
        for (row, request) in requests.iter().enumerate() {
            if request.page_ids.is_empty() {
                continue;
            }
            let page_capacity = request.page_ids.len().saturating_mul(self.page_size);
            let filled = request
                .input_ids
                .len()
                .min(self.max_seq_len)
                .min(page_capacity);
            for position in 0..filled {
                values[row * self.max_seq_len + position] = cache_location(
                    request.page_ids[position / self.page_size],
                    position % self.page_size,
                    self.page_size,
                )?;
            }
        }
        Ok(Tensor::from_slice(&values)
            .view([
                to_i64(requests.len(), "request count")?,
                to_i64(self.max_seq_len, "max sequence length")?,
            ])
            .to_device(self.device))
    }
}

fn cumulative_lengths(lengths: &[i32]) -> Result<Vec<i32>> {
    let mut cumulative = Vec::with_capacity(lengths.len() + 1);
    cumulative.push(0i32);
    for &length in lengths {
        let next = cumulative
            .last()
            .expect("cumulative always has an initial zero")
            .checked_add(length)
            .ok_or(BatchContextError::IntegerOverflow(
                "cumulative sequence length",
            ))?;
        cumulative.push(next);
    }
    Ok(cumulative)
}

fn cache_location(page_id: usize, offset: usize, page_size: usize) -> Result<i32> {
    let location = page_id
        .checked_mul(page_size)
        .and_then(|base| base.checked_add(offset))
        .ok_or(BatchContextError::IntegerOverflow("KV cache location"))?;
    to_i32(location, "KV cache location")
}

fn to_i32(value: usize, field: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| BatchContextError::IntegerOverflow(field))
}

fn to_i64(value: usize, field: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_| BatchContextError::IntegerOverflow(field))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepares_prefill_tensors_and_page_tables() {
        let context = BatchContext::new(2, 8, 2, Device::Cpu).unwrap();
        let requests = [
            BatchRequest {
                input_ids: vec![10, 11, 12],
                cached_len: 2,
                page_ids: vec![5, 3],
            },
            BatchRequest {
                input_ids: vec![20, 21],
                cached_len: 0,
                page_ids: vec![7],
            },
        ];

        let batch = context.prepare_prefill(&requests).unwrap();
        let metadata = batch.attention_metadata.as_ref().unwrap();
        assert_eq!(
            Vec::<i64>::try_from(&batch.input_ids).unwrap(),
            vec![12, 20, 21]
        );
        assert_eq!(
            Vec::<i64>::try_from(&batch.positions).unwrap(),
            vec![2, 0, 1]
        );
        assert_eq!(
            Vec::<i32>::try_from(metadata.write_loc.as_ref().unwrap()).unwrap(),
            vec![6, 14, 15]
        );
        assert_eq!(
            Vec::<i64>::try_from(&batch.logits_indices.unwrap()).unwrap(),
            vec![0, 2]
        );
        assert_eq!(
            Vec::<Vec<i32>>::try_from(metadata.req_to_token.as_ref().unwrap()).unwrap(),
            vec![
                vec![10, 11, 6, -1, -1, -1, -1, -1],
                vec![14, 15, -1, -1, -1, -1, -1, -1]
            ]
        );
    }

    #[test]
    fn rejects_requests_without_uncached_tokens() {
        let context = BatchContext::new(1, 4, 2, Device::Cpu).unwrap();
        let request = BatchRequest {
            input_ids: vec![1],
            cached_len: 1,
            page_ids: Vec::new(),
        };
        assert!(matches!(
            context.prepare_prefill(&[request]),
            Err(BatchContextError::EmptyUncachedRequest)
        ));
    }
}
