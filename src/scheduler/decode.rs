//! Builds one-token decode batches from running requests and their KV page tables.

use tch::{Device, Kind, Tensor};

use crate::engine::{AttentionMetadata, Batch, BatchPhase, ServerArgs};

use super::{Request, RequestId, Result, SchedulerError};

pub struct DecodeBatch {
    pub request_ids: Vec<RequestId>,
    pub model_batch: Batch,
}

pub struct DecodeManager {
    max_running_req: usize,
    max_seq_len: usize,
    page_size: usize,
    max_blocks: usize,
    device: Device,
    input_ids_buf: Tensor,
    positions_buf: Tensor,
    write_loc_buf: Tensor,
    cache_seqlens_buf: Tensor,
    block_table_buf: Tensor,
    req_to_token_buf: Tensor,
    row_mappings: Vec<RowMapping>,
    #[cfg(test)]
    mapped_tokens: usize,
}

#[derive(Default)]
struct RowMapping {
    uid: Option<RequestId>,
    pages: Vec<usize>,
    len: usize,
}

impl DecodeManager {
    pub fn new(args: &ServerArgs, device: Device) -> Result<Self> {
        if args.max_running_req == 0 || args.max_seq_len == 0 || args.page_size == 0 {
            return Err(SchedulerError::InvalidDecode(
                "max_running_req, max_seq_len, and page_size must be positive".to_owned(),
            ));
        }
        let rows = to_i64(args.max_running_req, "max_running_req")?;
        let blocks = to_i64(args.max_seq_len.div_ceil(args.page_size), "max_blocks")?;
        let sequence = to_i64(args.max_seq_len, "max_seq_len")?;
        Ok(Self {
            max_running_req: args.max_running_req,
            max_seq_len: args.max_seq_len,
            page_size: args.page_size,
            max_blocks: args.max_seq_len.div_ceil(args.page_size),
            device,
            input_ids_buf: Tensor::zeros([rows, 1], (Kind::Int64, device)),
            positions_buf: Tensor::zeros([rows, 1], (Kind::Int64, device)),
            write_loc_buf: Tensor::full([rows], -1, (Kind::Int, device)),
            cache_seqlens_buf: Tensor::zeros([rows], (Kind::Int, device)),
            block_table_buf: Tensor::full([rows, blocks], -1, (Kind::Int, device)),
            req_to_token_buf: Tensor::full([rows, sequence], -1, (Kind::Int, device)),
            row_mappings: (0..args.max_running_req)
                .map(|_| RowMapping::default())
                .collect(),
            #[cfg(test)]
            mapped_tokens: 0,
        })
    }

    /// Returned tensors view reusable buffers and must be consumed before the
    /// next call to `schedule_decode`.
    pub fn schedule_decode(&mut self, running: &[Request]) -> Result<Option<DecodeBatch>> {
        let active: Vec<_> = running
            .iter()
            .filter(|request| !request.is_finished())
            .collect();
        if active.is_empty() {
            return Ok(None);
        }
        if active.len() > self.max_running_req {
            return Err(SchedulerError::InvalidDecode(
                "request count exceeds max_running_req".to_owned(),
            ));
        }

        let rows = active.len();
        let mut request_ids = Vec::with_capacity(rows);
        let mut input_ids = Vec::with_capacity(rows);
        let mut positions = Vec::with_capacity(rows);
        let mut write_loc = Vec::with_capacity(rows);
        let mut cache_seqlens = Vec::with_capacity(rows);
        let mut max_seqlen = 0;
        let rows_i64 = to_i64(rows, "request count")?;

        for (row, request) in active.into_iter().enumerate() {
            let len = request.input_ids.len();
            if len == 0 || len > self.max_seq_len {
                return Err(SchedulerError::InvalidDecode(format!(
                    "request {} length {len} is outside 1..={}",
                    request.uid, self.max_seq_len
                )));
            }
            max_seqlen = max_seqlen.max(len);
            request_ids.push(request.uid);
            input_ids.push(request.input_ids[len - 1]);
            positions.push(to_i64(len - 1, "position")?);
            cache_seqlens.push(to_i32(len, "cache sequence length")?);

            let handle = request.cache_handle.as_ref();
            let location = handle
                .and_then(|handle| handle.page_ids.get((len - 1) / self.page_size))
                .map(|&page| self.cache_location(page, (len - 1) % self.page_size))
                .transpose()?
                .unwrap_or(-1);
            if location < 0 {
                return Err(SchedulerError::InvalidDecode(format!(
                    "request {} has no writable KV page for position {}",
                    request.uid,
                    len - 1
                )));
            }
            write_loc.push(location);

            if let Some(handle) = handle {
                let pages = &handle.page_ids[..handle.page_ids.len().min(self.max_blocks)];
                let previous = &self.row_mappings[row];
                let reset = previous.uid != Some(request.uid);
                let changed_page = if reset {
                    0
                } else {
                    previous
                        .pages
                        .iter()
                        .zip(pages)
                        .position(|(old, new)| old != new)
                        .unwrap_or(previous.pages.len().min(pages.len()))
                };
                let page_values = pages[changed_page..]
                    .iter()
                    .map(|&page| to_i32(page, "page ID"))
                    .collect::<Result<Vec<_>>>()?;
                let start = if reset {
                    0
                } else {
                    previous.len.min(changed_page * self.page_size).min(len)
                };
                let mut locations = Vec::with_capacity(len - start);
                for position in start..len {
                    locations.push(self.cache_location(
                        pages[position / self.page_size],
                        position % self.page_size,
                    )?);
                }
                if reset {
                    let _ = self.block_table_buf.get(row as i64).fill_(-1);
                    let _ = self.req_to_token_buf.get(row as i64).fill_(-1);
                } else {
                    if pages.len() < previous.pages.len() {
                        let _ = self
                            .block_table_buf
                            .get(row as i64)
                            .narrow(
                                0,
                                pages.len() as i64,
                                (previous.pages.len() - pages.len()) as i64,
                            )
                            .fill_(-1);
                    }
                    if len < previous.len {
                        let _ = self
                            .req_to_token_buf
                            .get(row as i64)
                            .narrow(0, len as i64, (previous.len - len) as i64)
                            .fill_(-1);
                    }
                }
                if !page_values.is_empty() {
                    self.block_table_buf
                        .get(row as i64)
                        .narrow(0, changed_page as i64, page_values.len() as i64)
                        .copy_(&Tensor::from_slice(&page_values).to_device(self.device));
                }
                if !locations.is_empty() {
                    self.req_to_token_buf
                        .get(row as i64)
                        .narrow(0, start as i64, locations.len() as i64)
                        .copy_(&Tensor::from_slice(&locations).to_device(self.device));
                }
                #[cfg(test)]
                {
                    self.mapped_tokens += locations.len();
                }
                let previous = &mut self.row_mappings[row];
                previous.uid = Some(request.uid);
                if previous.pages != pages {
                    previous.pages.clear();
                    previous.pages.extend_from_slice(pages);
                }
                previous.len = len;
            }
        }

        self.input_ids_buf.narrow(0, 0, rows_i64).copy_(
            &Tensor::from_slice(&input_ids)
                .view([rows_i64, 1])
                .to_device(self.device),
        );
        self.positions_buf.narrow(0, 0, rows_i64).copy_(
            &Tensor::from_slice(&positions)
                .view([rows_i64, 1])
                .to_device(self.device),
        );
        self.write_loc_buf
            .narrow(0, 0, rows_i64)
            .copy_(&Tensor::from_slice(&write_loc).to_device(self.device));
        self.cache_seqlens_buf
            .narrow(0, 0, rows_i64)
            .copy_(&Tensor::from_slice(&cache_seqlens).to_device(self.device));
        let model_batch = Batch::decode(
            self.input_ids_buf.narrow(0, 0, rows_i64),
            self.positions_buf.narrow(0, 0, rows_i64),
            Some(AttentionMetadata {
                forward_mode: BatchPhase::Decode,
                write_loc: Some(self.write_loc_buf.narrow(0, 0, rows_i64)),
                cu_seqlens_q: None,
                prefix_lens: None,
                block_table: Some(self.block_table_buf.narrow(0, 0, rows_i64)),
                req_to_token: Some(self.req_to_token_buf.narrow(0, 0, rows_i64)),
                cache_seqlens: Some(self.cache_seqlens_buf.narrow(0, 0, rows_i64)),
                max_seqlen: Some(max_seqlen),
            }),
        );
        Ok(Some(DecodeBatch {
            request_ids,
            model_batch,
        }))
    }

    fn cache_location(&self, page: usize, offset: usize) -> Result<i32> {
        let slot = page
            .checked_mul(self.page_size)
            .and_then(|base| base.checked_add(offset))
            .ok_or_else(|| SchedulerError::InvalidDecode("KV slot overflow".to_owned()))?;
        to_i32(slot, "KV slot")
    }
}

fn to_i32(value: usize, field: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| SchedulerError::InvalidDecode(format!("{field} exceeds i32")))
}

fn to_i64(value: usize, field: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| SchedulerError::InvalidDecode(format!("{field} exceeds i64")))
}

#[cfg(test)]
mod tests {
    use crate::engine::SamplingParams;
    use crate::engine::kvcache::BaseCacheHandle;
    use crate::scheduler::SequenceStatus;

    use super::*;

    fn manager() -> DecodeManager {
        let mut args = ServerArgs::new("unused-model-path");
        args.max_running_req = 2;
        args.max_seq_len = 6;
        args.page_size = 2;
        DecodeManager::new(&args, Device::Cpu).unwrap()
    }

    fn running(uid: RequestId, input_ids: Vec<i64>, pages: Vec<usize>) -> Request {
        Request {
            uid,
            input_ids,
            sampling_params: SamplingParams::default(),
            cached_len: 0,
            output_len: 1,
            cache_handle: Some(BaseCacheHandle {
                page_ids: pages,
                ..Default::default()
            }),
            status: SequenceStatus::Running,
        }
    }

    #[test]
    fn builds_decode_tensors_and_page_metadata() {
        let batch = manager()
            .schedule_decode(&[
                running(4, vec![11, 12, 13], vec![5, 6]),
                running(7, vec![21, 22], vec![9]),
            ])
            .unwrap()
            .unwrap();
        assert_eq!(batch.request_ids, vec![4, 7]);
        assert_eq!(batch.model_batch.phase, BatchPhase::Decode);
        assert_eq!(batch.model_batch.input_ids.size(), vec![2, 1]);
        assert_eq!(
            Vec::<i64>::try_from(&batch.model_batch.input_ids.view([-1])).unwrap(),
            vec![13, 22]
        );
        assert_eq!(
            Vec::<i64>::try_from(&batch.model_batch.positions.view([-1])).unwrap(),
            vec![2, 1]
        );
        let metadata = batch.model_batch.attention_metadata.unwrap();
        assert_eq!(metadata.forward_mode, BatchPhase::Decode);
        assert_eq!(
            Vec::<i32>::try_from(metadata.write_loc.as_ref().unwrap()).unwrap(),
            vec![12, 19]
        );
        assert_eq!(
            Vec::<i32>::try_from(metadata.cache_seqlens.as_ref().unwrap()).unwrap(),
            vec![3, 2]
        );
        assert_eq!(metadata.max_seqlen, Some(3));
        assert_eq!(
            Vec::<i32>::try_from(&metadata.block_table.unwrap().view([-1])).unwrap(),
            vec![5, 6, -1, 9, -1, -1]
        );
        assert_eq!(
            Vec::<i32>::try_from(&metadata.req_to_token.unwrap().view([-1])).unwrap(),
            vec![10, 11, 12, -1, -1, -1, 18, 19, -1, -1, -1, -1]
        );
    }

    #[test]
    fn empty_and_finished_requests_produce_no_batch() {
        let mut manager = manager();
        assert!(manager.schedule_decode(&[]).unwrap().is_none());
        let mut request = running(1, vec![2], vec![0]);
        request.status = SequenceStatus::Finished;
        assert!(manager.schedule_decode(&[request]).unwrap().is_none());
    }

    #[test]
    fn reused_buffers_clear_stale_page_entries() {
        let mut manager = manager();
        let first = manager
            .schedule_decode(&[running(1, vec![1, 2, 3, 4], vec![5, 6])])
            .unwrap()
            .unwrap();
        drop(first);

        let second = manager
            .schedule_decode(&[running(2, vec![9], vec![1])])
            .unwrap()
            .unwrap();
        let metadata = second.model_batch.attention_metadata.unwrap();
        assert_eq!(
            Vec::<i32>::try_from(&metadata.block_table.unwrap().view([-1])).unwrap(),
            vec![1, -1, -1]
        );
        assert_eq!(
            Vec::<i32>::try_from(&metadata.req_to_token.unwrap().view([-1])).unwrap(),
            vec![2, -1, -1, -1, -1, -1]
        );
    }

    #[test]
    fn out_of_range_sequence_is_rejected() {
        let request = running(1, vec![1, 2, 3, 4, 5, 6, 7], vec![0, 1, 2, 3]);
        assert!(matches!(
            manager().schedule_decode(&[request]),
            Err(SchedulerError::InvalidDecode(_))
        ));
    }

    #[test]
    fn growing_decode_only_maps_new_tokens_and_handles_page_changes() {
        let mut manager = manager();
        let mut request = running(1, vec![1, 2], vec![5, 6, 7]);
        drop(
            manager
                .schedule_decode(std::slice::from_ref(&request))
                .unwrap(),
        );
        assert_eq!(manager.mapped_tokens, 2);
        request.input_ids.push(3);
        drop(
            manager
                .schedule_decode(std::slice::from_ref(&request))
                .unwrap(),
        );
        assert_eq!(manager.mapped_tokens, 3);
        request.input_ids.push(4);
        drop(
            manager
                .schedule_decode(std::slice::from_ref(&request))
                .unwrap(),
        );
        assert_eq!(manager.mapped_tokens, 4);
        request.cache_handle.as_mut().unwrap().page_ids[0] = 9;
        drop(
            manager
                .schedule_decode(std::slice::from_ref(&request))
                .unwrap(),
        );
        assert_eq!(manager.mapped_tokens, 8);
        assert_eq!(
            Vec::<i32>::try_from(&manager.req_to_token_buf.get(0)).unwrap(),
            vec![18, 19, 12, 13, -1, -1]
        );
        request.input_ids.truncate(1);
        request.cache_handle.as_mut().unwrap().page_ids.truncate(1);
        drop(
            manager
                .schedule_decode(std::slice::from_ref(&request))
                .unwrap(),
        );
        assert_eq!(
            Vec::<i32>::try_from(&manager.req_to_token_buf.get(0)).unwrap(),
            vec![18, -1, -1, -1, -1, -1]
        );
        assert_eq!(
            Vec::<i32>::try_from(&manager.block_table_buf.get(0)).unwrap(),
            vec![9, -1, -1]
        );
    }

    #[test]
    fn compacting_rows_and_reusing_pages_preserves_request_mappings() {
        let mut manager = manager();
        let a = running(1, vec![1, 2], vec![5]);
        let mut b = running(2, vec![3, 4], vec![6]);
        drop(manager.schedule_decode(&[a, b]).unwrap());
        b = running(2, vec![3, 4, 5], vec![6, 7]);
        drop(manager.schedule_decode(std::slice::from_ref(&b)).unwrap());
        assert_eq!(
            Vec::<i32>::try_from(&manager.req_to_token_buf.get(0)).unwrap(),
            vec![12, 13, 14, -1, -1, -1]
        );
        let c = running(3, vec![9], vec![6]);
        drop(manager.schedule_decode(&[c, b]).unwrap());
        assert_eq!(
            Vec::<i32>::try_from(&manager.req_to_token_buf.get(0)).unwrap(),
            vec![12, -1, -1, -1, -1, -1]
        );
        assert_eq!(
            Vec::<i32>::try_from(&manager.req_to_token_buf.get(1)).unwrap(),
            vec![12, 13, 14, -1, -1, -1]
        );
    }
}
