//! Scheduler batch preparation for prefill execution.

use std::fmt;

use tch::{Device, Tensor};

use super::{AttentionMetadata, Batch, BatchPhase, MetadataRequirements};

/// Scheduler-side request data needed to derive one prefill [`Batch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchRequest {
    pub input_ids: Vec<i64>,
    pub cached_len: usize,
    /// Exclusive absolute prompt end for a chunk; `None` means full prompt.
    pub end_len: Option<usize>,
    /// Non-owning snapshot used only to construct this forward's tensors.
    pub page_ids: Vec<usize>,
}

/// Borrowed scheduler input; prompt and page snapshots need no intermediate copies.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BatchRequestRef<'a> {
    pub input_ids: &'a [i64],
    pub cached_len: usize,
    pub end_len: Option<usize>,
    pub page_ids: &'a [usize],
}

impl<'a> From<&'a BatchRequest> for BatchRequestRef<'a> {
    fn from(request: &'a BatchRequest) -> Self {
        Self {
            input_ids: &request.input_ids,
            cached_len: request.cached_len,
            end_len: request.end_len,
            page_ids: &request.page_ids,
        }
    }
}

/// One existing decode request embedded into a mixed prefill batch. The
/// query has length one; `position` is the uncached sequence position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedDecodeRequest {
    pub input_id: i64,
    pub position: usize,
    pub page_ids: Vec<usize>,
}

impl BatchRequest {
    pub fn new(input_ids: Vec<i64>) -> Self {
        Self {
            input_ids,
            cached_len: 0,
            end_len: None,
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
    MissingWritablePage { position: usize },
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
            Self::MissingWritablePage { position } => {
                write!(f, "prefill 位置 {position} 没有可写的 KV page")
            }
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
    metadata_requirements: MetadataRequirements,
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
            metadata_requirements: MetadataRequirements {
                block_table: true,
                req_to_token: true,
            },
        })
    }

    /// Select the table layouts required by the model's eager and graph paths.
    pub fn with_metadata_requirements(mut self, requirements: MetadataRequirements) -> Self {
        self.metadata_requirements = requirements;
        self
    }

    /// Prepares a flattened prefill batch and its page-table attention metadata.
    pub fn prepare_prefill(&self, requests: &[BatchRequest]) -> Result<Batch> {
        self.prepare_mixed_prefill(requests, &[])
    }

    /// Builds prefill rows followed by one-token decode rows. The caller may
    /// replace decode placeholder IDs with a device relay after construction.
    pub fn prepare_mixed_prefill(
        &self,
        prefills: &[BatchRequest],
        decodes: &[MixedDecodeRequest],
    ) -> Result<Batch> {
        let prefills: Vec<_> = prefills.iter().map(BatchRequestRef::from).collect();
        self.prepare_mixed_prefill_borrowed(&prefills, decodes)
    }

    pub(crate) fn prepare_mixed_prefill_borrowed(
        &self,
        prefills: &[BatchRequestRef<'_>],
        decodes: &[MixedDecodeRequest],
    ) -> Result<Batch> {
        if prefills.is_empty() && decodes.is_empty() {
            return Err(BatchContextError::EmptyBatch);
        }
        if prefills.len() + decodes.len() > self.max_running_req {
            return Err(BatchContextError::InvalidConfiguration(
                "request count exceeds max_running_req",
            ));
        }
        let mut rows = Vec::with_capacity(prefills.len() + decodes.len());
        for request in prefills {
            rows.push(self.prefill_row(*request)?);
        }
        for request in decodes {
            if request.position >= self.max_seq_len {
                return Err(BatchContextError::InvalidConfiguration(
                    "decode position exceeds max_seq_len",
                ));
            }
            rows.push(MixedRow {
                input_ids: std::slice::from_ref(&request.input_id),
                prefix_len: request.position,
                total_len: request.position + 1,
                page_ids: &request.page_ids,
            });
        }
        self.prepare_rows(&rows)
    }

    fn prefill_row<'a>(&self, request: BatchRequestRef<'a>) -> Result<MixedRow<'a>> {
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
        let end_len = request.end_len.unwrap_or(request.input_ids.len());
        if end_len < request.cached_len || end_len > request.input_ids.len() {
            return Err(BatchContextError::InvalidConfiguration(
                "prefill end_len must be within cached_len..=input length",
            ));
        }
        let input_ids = &request.input_ids[request.cached_len..end_len];
        if input_ids.is_empty() {
            return Err(BatchContextError::EmptyUncachedRequest);
        }
        Ok(MixedRow {
            input_ids,
            prefix_len: request.cached_len,
            total_len: end_len,
            page_ids: request.page_ids,
        })
    }

    fn prepare_rows(&self, rows: &[MixedRow<'_>]) -> Result<Batch> {
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut write_loc = Vec::new();
        let mut prefix_lens = Vec::with_capacity(rows.len());
        let mut sequence_lengths = Vec::with_capacity(rows.len());
        let mut logits_indices = Vec::with_capacity(rows.len());
        let mut offset = 0usize;

        for row in rows {
            input_ids.extend_from_slice(row.input_ids);
            for position in row.prefix_len..row.total_len {
                positions.push(to_i64(position, "position")?);
            }
            write_loc.extend(self.build_write_locations(row)?);
            prefix_lens.push(to_i32(row.prefix_len, "cached_len")?);
            sequence_lengths.push(to_i32(row.input_ids.len(), "sequence length")?);
            logits_indices.push(to_i64(
                offset
                    .checked_add(row.input_ids.len() - 1)
                    .ok_or(BatchContextError::IntegerOverflow("logits index"))?,
                "logits index",
            )?);
            offset = offset
                .checked_add(row.input_ids.len())
                .ok_or(BatchContextError::IntegerOverflow("total token count"))?;
        }

        let cumulative_lengths = cumulative_lengths(&sequence_lengths)?;
        let block_table = self
            .metadata_requirements
            .block_table
            .then(|| self.build_block_table(rows))
            .transpose()?;
        let req_to_token = self
            .metadata_requirements
            .req_to_token
            .then(|| self.build_req_to_token(rows))
            .transpose()?;

        Ok(Batch::prefill(
            Tensor::from_slice(&input_ids).to_device(self.device),
            Tensor::from_slice(&positions).to_device(self.device),
            Some(AttentionMetadata {
                state_slots: None,
                forward_mode: BatchPhase::Prefill,
                decode_host: None,
                write_loc: Some(Tensor::from_slice(&write_loc).to_device(self.device)),
                cu_seqlens_q: Some(Tensor::from_slice(&cumulative_lengths).to_device(self.device)),
                prefix_lens: Some(Tensor::from_slice(&prefix_lens).to_device(self.device)),
                block_table,
                req_to_token,
                cache_seqlens: None,
                max_seqlen: sequence_lengths.iter().map(|&length| length as usize).max(),
            }),
            Tensor::from_slice(&logits_indices).to_device(self.device),
        ))
    }

    fn build_write_locations(&self, row: &MixedRow<'_>) -> Result<Vec<i32>> {
        let required_pages = row.total_len.div_ceil(self.page_size);
        if row.page_ids.len() < required_pages {
            return Err(BatchContextError::MissingWritablePage {
                position: row
                    .prefix_len
                    .max(row.page_ids.len().saturating_mul(self.page_size)),
            });
        }
        let mut locations = Vec::with_capacity(row.input_ids.len());
        for position in row.prefix_len..row.total_len {
            let page_id = row.page_ids[position / self.page_size];
            let location = cache_location(page_id, position % self.page_size, self.page_size)?;
            locations.push(location);
        }
        Ok(locations)
    }

    fn build_block_table(&self, rows: &[MixedRow<'_>]) -> Result<Tensor> {
        let max_blocks = self.max_seq_len.div_ceil(self.page_size);
        let row_len = to_i64(max_blocks, "max blocks")?;
        let mut values = vec![-1i32; rows.len() * max_blocks];
        for (row, request) in rows.iter().enumerate() {
            for (column, &page_id) in request.page_ids.iter().take(max_blocks).enumerate() {
                values[row * max_blocks + column] = to_i32(page_id, "page id")?;
            }
        }
        Ok(Tensor::from_slice(&values)
            .view([to_i64(rows.len(), "request count")?, row_len])
            .to_device(self.device))
    }

    fn build_req_to_token(&self, rows: &[MixedRow<'_>]) -> Result<Tensor> {
        let mut values = vec![-1i32; rows.len() * self.max_seq_len];
        for (row, request) in rows.iter().enumerate() {
            if request.page_ids.is_empty() {
                continue;
            }
            let page_capacity = request.page_ids.len().saturating_mul(self.page_size);
            let filled = request.total_len.min(self.max_seq_len).min(page_capacity);
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
                to_i64(rows.len(), "request count")?,
                to_i64(self.max_seq_len, "max sequence length")?,
            ])
            .to_device(self.device))
    }
}

struct MixedRow<'a> {
    input_ids: &'a [i64],
    prefix_len: usize,
    total_len: usize,
    page_ids: &'a [usize],
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
    fn borrowed_chunk_rows_share_sources_but_own_execution_tensors() {
        let context = BatchContext::new(2, 8, 2, Device::Cpu).unwrap();
        let mut prompt = vec![10, 11, 12, 13, 14, 15];
        let pages = vec![5, 3, 7];
        let request = BatchRequestRef {
            input_ids: &prompt,
            cached_len: 2,
            end_len: Some(4),
            page_ids: &pages,
        };
        let row = context.prefill_row(request).unwrap();
        assert_eq!(row.input_ids.as_ptr(), prompt[2..].as_ptr());
        assert_eq!(row.page_ids.as_ptr(), pages.as_ptr());
        let batch = context
            .prepare_mixed_prefill_borrowed(&[request], &[])
            .unwrap();
        prompt[2] = 99;
        assert_eq!(
            Vec::<i64>::try_from(&batch.input_ids).unwrap(),
            vec![12, 13]
        );
        assert_eq!(Vec::<i64>::try_from(&batch.positions).unwrap(), vec![2, 3]);
        let meta = batch.attention_metadata.unwrap();
        assert_eq!(
            Vec::<i32>::try_from(meta.write_loc.as_ref().unwrap()).unwrap(),
            vec![6, 7]
        );
        assert_eq!(
            Vec::<i32>::try_from(meta.cu_seqlens_q.as_ref().unwrap()).unwrap(),
            vec![0, 2]
        );
    }

    #[test]
    fn table_requirements_apply_to_plain_and_mixed_prefill() {
        for requirements in [
            MetadataRequirements {
                block_table: true,
                req_to_token: false,
            },
            MetadataRequirements {
                block_table: false,
                req_to_token: true,
            },
        ] {
            let context = BatchContext::new(2, 8, 2, Device::Cpu)
                .unwrap()
                .with_metadata_requirements(requirements);
            let requests = [BatchRequest {
                input_ids: vec![10, 11, 12],
                cached_len: 1,
                end_len: Some(2),
                page_ids: vec![2, 3],
            }];
            let plain = context.prepare_prefill(&requests).unwrap();
            let mixed = context
                .prepare_mixed_prefill(
                    &requests,
                    &[MixedDecodeRequest {
                        input_id: 20,
                        position: 2,
                        page_ids: vec![4, 5],
                    }],
                )
                .unwrap();
            for batch in [&plain, &mixed] {
                let meta = batch.attention_metadata.as_ref().unwrap();
                assert_eq!(meta.block_table.is_some(), requirements.block_table);
                assert_eq!(meta.req_to_token.is_some(), requirements.req_to_token);
                crate::engine::ForwardBatch::from_scheduler(batch)
                    .unwrap()
                    .validate(Device::Cpu)
                    .unwrap();
            }
            assert_eq!(Vec::<i64>::try_from(&plain.input_ids).unwrap(), vec![11]);
            assert_eq!(
                Vec::<i64>::try_from(&mixed.input_ids).unwrap(),
                vec![11, 20]
            );
            assert_eq!(
                Vec::<i32>::try_from(
                    mixed
                        .attention_metadata
                        .as_ref()
                        .unwrap()
                        .write_loc
                        .as_ref()
                        .unwrap()
                )
                .unwrap(),
                vec![5, 10]
            );
        }
    }

    #[test]
    fn plain_and_mixed_prefill_share_chunk_validation() {
        let context = BatchContext::new(2, 8, 2, Device::Cpu).unwrap();
        for (cached_len, end_len) in [(3, Some(2)), (1, Some(4)), (2, Some(2))] {
            let requests = [BatchRequest {
                input_ids: vec![10, 11, 12],
                cached_len,
                end_len,
                page_ids: vec![1, 2],
            }];
            let plain = context.prepare_prefill(&requests).unwrap_err();
            let mixed = context.prepare_mixed_prefill(&requests, &[]).unwrap_err();
            assert_eq!(plain.to_string(), mixed.to_string());
        }
    }

    #[test]
    fn prepares_prefill_tensors_and_page_tables() {
        let context = BatchContext::new(2, 8, 2, Device::Cpu).unwrap();
        let requests = [
            BatchRequest {
                input_ids: vec![10, 11, 12],
                cached_len: 2,
                end_len: None,
                page_ids: vec![5, 3],
            },
            BatchRequest {
                input_ids: vec![20, 21],
                cached_len: 0,
                end_len: None,
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
    fn prepares_mixed_prefill_with_one_token_decode_rows() {
        let context = BatchContext::new(4, 8, 2, Device::Cpu).unwrap();
        let batch = context
            .prepare_mixed_prefill(
                &[BatchRequest {
                    input_ids: vec![1, 2],
                    cached_len: 0,
                    end_len: None,
                    page_ids: vec![0],
                }],
                &[MixedDecodeRequest {
                    input_id: 0,
                    position: 2,
                    page_ids: vec![1, 2],
                }],
            )
            .unwrap();
        assert_eq!(batch.phase, BatchPhase::Prefill);
        assert_eq!(batch.input_ids.size(), vec![3]);
        assert_eq!(batch.positions.size(), vec![3]);
        let metadata = batch.attention_metadata.as_ref().unwrap();
        let bounds = metadata.cu_seqlens_q.as_ref().unwrap();
        let prefixes = metadata.prefix_lens.as_ref().unwrap();
        assert_eq!(Vec::<i32>::try_from(bounds).unwrap(), vec![0, 2, 3]);
        assert_eq!(Vec::<i32>::try_from(prefixes).unwrap(), vec![0, 2]);
        assert_eq!(
            Vec::<i32>::try_from(metadata.write_loc.as_ref().unwrap()).unwrap(),
            vec![0, 1, 4]
        );
        assert_eq!(
            Vec::<i64>::try_from(&batch.logits_indices.unwrap()).unwrap(),
            vec![1, 2]
        );
        assert_eq!(
            Vec::<Vec<i32>>::try_from(metadata.block_table.as_ref().unwrap()).unwrap(),
            vec![vec![0, -1, -1, -1], vec![1, 2, -1, -1]]
        );
        assert_eq!(
            Vec::<Vec<i32>>::try_from(metadata.req_to_token.as_ref().unwrap()).unwrap(),
            vec![
                vec![0, 1, -1, -1, -1, -1, -1, -1],
                vec![2, 3, 4, -1, -1, -1, -1, -1]
            ]
        );
    }

    #[test]
    fn rejects_oversized_mixed_prefill() {
        let context = BatchContext::new(1, 2, 2, Device::Cpu).unwrap();
        let result = context.prepare_mixed_prefill(
            &[BatchRequest {
                input_ids: vec![1, 2, 3],
                cached_len: 0,
                end_len: None,
                page_ids: vec![0, 1],
            }],
            &[],
        );
        assert!(matches!(
            result,
            Err(BatchContextError::InvalidConfiguration(
                "request input exceeds max_seq_len"
            ))
        ));
    }

    #[test]
    fn rejects_requests_without_uncached_tokens() {
        let context = BatchContext::new(1, 4, 2, Device::Cpu).unwrap();
        let request = BatchRequest {
            input_ids: vec![1],
            cached_len: 1,
            end_len: None,
            page_ids: Vec::new(),
        };
        assert!(matches!(
            context.prepare_prefill(&[request]),
            Err(BatchContextError::EmptyUncachedRequest)
        ));
    }

    #[test]
    fn rejects_prefill_without_a_writable_page() {
        let context = BatchContext::new(1, 4, 2, Device::Cpu).unwrap();
        let request = BatchRequest {
            input_ids: vec![1, 2, 3],
            cached_len: 0,
            end_len: None,
            page_ids: vec![0],
        };
        assert!(matches!(
            context.prepare_prefill(&[request]),
            Err(BatchContextError::MissingWritablePage { position: 2 })
        ));
        let cached_request = BatchRequest {
            input_ids: vec![1, 2, 3],
            cached_len: 2,
            end_len: None,
            page_ids: vec![0],
        };
        assert!(matches!(
            context.prepare_prefill(&[cached_request]),
            Err(BatchContextError::MissingWritablePage { position: 2 })
        ));
    }
}
