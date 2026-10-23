//! Builds one-token decode batches from running requests and their KV page tables.

use std::sync::atomic::{AtomicU64, Ordering};
use tch::{Device, Kind, Tensor};

// Revisions remain unique across DecodeManager instances and buffer lifetimes.
static NEXT_TABLE_VERSION: AtomicU64 = AtomicU64::new(1);

use crate::engine::metadata_transfer::{MetadataUpload, UploadGuard};
use crate::engine::{
    AttentionMetadata, Batch, BatchPhase, CompletionFence, DecodeHostMetadata, InputReadFence,
    MetadataRequirements, ServerArgs,
};

use super::future_map::RelayPlan;
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
    allow_upload: bool,
    upload: Option<MetadataUpload>,
    read_done: Option<InputReadFence>,
    full_done: Option<CompletionFence>,
    relay_rows_buf: Tensor,
    input_ids_buf: Tensor,
    positions_buf: Tensor,
    write_loc_buf: Tensor,
    cache_seqlens_buf: Tensor,
    block_table_buf: Option<Tensor>,
    req_to_token_buf: Option<Tensor>,
    row_mappings: Vec<RowMapping>,
    #[cfg(test)]
    mapped_tokens: usize,
}

#[derive(Default)]
struct RowMapping {
    uid: Option<RequestId>,
    pages: Vec<usize>,
    len: usize,
    table_version: u64,
}

impl DecodeManager {
    pub fn new(args: &ServerArgs, device: Device) -> Result<Self> {
        Self::with_metadata_requirements(args, device, MetadataRequirements::default())
    }

    pub(crate) fn with_metadata_requirements(
        args: &ServerArgs,
        device: Device,
        requirements: MetadataRequirements,
    ) -> Result<Self> {
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
            allow_upload: cfg!(has_cuda_graph)
                && device.is_cuda()
                && !args.disable_cpu_gpu_overlap
                && !args.disable_async_metadata
                && !crate::logging::step_timing_enabled(),
            upload: None,
            read_done: None,
            full_done: None,
            relay_rows_buf: Tensor::zeros([rows], (Kind::Int64, device)),
            input_ids_buf: Tensor::zeros([rows, 1], (Kind::Int64, device)),
            positions_buf: Tensor::zeros([rows, 1], (Kind::Int64, device)),
            write_loc_buf: Tensor::full([rows], -1, (Kind::Int, device)),
            cache_seqlens_buf: Tensor::zeros([rows], (Kind::Int, device)),
            block_table_buf: requirements
                .block_table
                .then(|| Tensor::full([rows, blocks], -1, (Kind::Int, device))),
            req_to_token_buf: requirements
                .req_to_token
                .then(|| Tensor::full([rows, sequence], -1, (Kind::Int, device))),
            row_mappings: (0..args.max_running_req)
                .map(|_| RowMapping::default())
                .collect(),
            #[cfg(test)]
            mapped_tokens: 0,
        })
    }

    /// Requirements can change after Engine mutation, once the scheduler has
    /// drained outstanding work. Invalidate incremental mappings on any change.
    pub(crate) fn set_metadata_requirements(&mut self, requirements: MetadataRequirements) {
        if self.block_table_buf.is_some() == requirements.block_table
            && self.req_to_token_buf.is_some() == requirements.req_to_token
        {
            return;
        }
        self.block_table_buf = requirements.block_table.then(|| {
            Tensor::full(
                [self.max_running_req as i64, self.max_blocks as i64],
                -1,
                (Kind::Int, self.device),
            )
        });
        self.req_to_token_buf = requirements.req_to_token.then(|| {
            Tensor::full(
                [self.max_running_req as i64, self.max_seq_len as i64],
                -1,
                (Kind::Int, self.device),
            )
        });
        self.row_mappings
            .iter_mut()
            .for_each(|row| *row = RowMapping::default());
        // The replacement tensors were initialized on the execution stream.
        // First pipeline use must establish a new initialization read boundary
        // before uploading on its separate stream.
        self.upload = None;
        self.read_done = None;
        self.full_done = None;
    }

    /// Returned tensors view reusable buffers and must be consumed before the
    /// next call to `schedule_decode`.
    pub fn schedule_decode(&mut self, running: &[Request]) -> Result<Option<DecodeBatch>> {
        // Public callers retain the original same-stream buffer-order contract.
        self.build(running, None, false)
    }

    pub(crate) fn schedule_pipeline(&mut self, running: &[Request]) -> Result<Option<DecodeBatch>> {
        self.build(running, None, true)
    }

    pub(crate) fn set_read_boundary(
        &mut self,
        early: Option<InputReadFence>,
        full: CompletionFence,
    ) {
        self.full_done = early.is_none().then_some(full);
        self.read_done = early;
    }

    /// Plan one future token per selected request without observing its value
    /// on the CPU. Future lengths belong to the pending forward, not input_ids.
    pub(crate) fn schedule_relay(
        &mut self,
        running: &[Request],
        future: &RelayPlan,
        ids: &Tensor,
    ) -> Result<Option<DecodeBatch>> {
        self.build(running, Some((future, ids)), true)
    }

    fn build(
        &mut self,
        running: &[Request],
        future: Option<(&RelayPlan, &Tensor)>,
        async_metadata: bool,
    ) -> Result<Option<DecodeBatch>> {
        let active: Vec<_> = running
            .iter()
            .filter(|request| request.is_decoding())
            .filter(|request| future.is_none_or(|(plan, _)| plan.row(request.uid).is_some()))
            .collect();
        if active.is_empty() {
            return Ok(None);
        }
        if active.len() > self.max_running_req {
            return Err(SchedulerError::InvalidDecode(
                "request count exceeds max_running_req".to_owned(),
            ));
        }

        if async_metadata && self.allow_upload && self.upload.is_none() {
            // First use also fences buffer initialization or any public
            // same-stream invocation preceding tracked pipeline execution.
            if self.read_done.is_none() && self.full_done.is_none() {
                self.read_done = Some(
                    InputReadFence::record(self.device)
                        .map_err(crate::engine::EngineError::from)?,
                );
            }
            self.upload =
                Some(MetadataUpload::new(self.device).map_err(crate::engine::EngineError::from)?);
        }
        let upload = if async_metadata {
            self.upload
                .as_ref()
                .map(|upload| upload.begin(self.read_done.as_ref(), self.full_done.as_ref()))
                .transpose()
                .map_err(crate::engine::EngineError::from)?
        } else {
            None
        };
        let state_slots = active
            .iter()
            .map(|r| r.cache_handle.as_ref().and_then(|h| h.state_slot))
            .collect();
        let rows = active.len();
        let mut request_ids = Vec::with_capacity(rows);
        let mut input_ids = Vec::with_capacity(rows);
        let mut relay_rows = Vec::with_capacity(rows);
        let mut positions = Vec::with_capacity(rows);
        let mut write_loc = Vec::with_capacity(rows);
        let mut cache_seqlens = Vec::with_capacity(rows);
        let mut max_seqlen = 0;
        let rows_i64 = to_i64(rows, "request count")?;

        for (row, request) in active.into_iter().enumerate() {
            let len = if let Some((future, _)) = future {
                let row = future.row(request.uid).expect("filtered relay request");
                relay_rows.push(row.slot as i64);
                row.input.written_len
            } else {
                request.input_ids.len()
            };
            if len == 0 || len > self.max_seq_len {
                return Err(SchedulerError::InvalidDecode(format!(
                    "request {} length {len} is outside 1..={}",
                    request.uid, self.max_seq_len
                )));
            }
            max_seqlen = max_seqlen.max(len);
            request_ids.push(request.uid);
            if future.is_none() {
                input_ids.push(request.input_ids[len - 1]);
            }
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
                if let Some(table) = &self.block_table_buf {
                    if reset {
                        let _ = table.get(row as i64).fill_(-1);
                    } else if pages.len() < previous.pages.len() {
                        let _ = table
                            .get(row as i64)
                            .narrow(
                                0,
                                pages.len() as i64,
                                (previous.pages.len() - pages.len()) as i64,
                            )
                            .fill_(-1);
                    }
                    let page_values = pages[changed_page..]
                        .iter()
                        .map(|&page| to_i32(page, "page ID"))
                        .collect::<Result<Vec<_>>>()?;
                    if !page_values.is_empty() {
                        copy_host(
                            &mut table.get(row as i64).narrow(
                                0,
                                changed_page as i64,
                                page_values.len() as i64,
                            ),
                            &Tensor::from_slice(&page_values),
                            upload.as_ref(),
                        )?;
                    }
                }
                if let Some(table) = &self.req_to_token_buf {
                    let start = if reset {
                        0
                    } else {
                        previous.len.min(changed_page * self.page_size).min(len)
                    };
                    let locations = (start..len)
                        .map(|position| {
                            self.cache_location(
                                pages[position / self.page_size],
                                position % self.page_size,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    if reset {
                        let _ = table.get(row as i64).fill_(-1);
                    } else if len < previous.len {
                        let _ = table
                            .get(row as i64)
                            .narrow(0, len as i64, (previous.len - len) as i64)
                            .fill_(-1);
                    }
                    if !locations.is_empty() {
                        copy_host(
                            &mut table.get(row as i64).narrow(
                                0,
                                start as i64,
                                locations.len() as i64,
                            ),
                            &Tensor::from_slice(&locations),
                            upload.as_ref(),
                        )?;
                    }
                    #[cfg(test)]
                    {
                        self.mapped_tokens += locations.len();
                    }
                }
                let previous = &mut self.row_mappings[row];
                if reset || previous.pages != pages {
                    previous.table_version = NEXT_TABLE_VERSION.fetch_add(1, Ordering::Relaxed);
                }
                previous.uid = Some(request.uid);
                if previous.pages != pages {
                    previous.pages.clear();
                    previous.pages.extend_from_slice(pages);
                }
                previous.len = len;
            }
        }

        if future.is_some() {
            copy_host(
                &mut self.relay_rows_buf.narrow(0, 0, rows_i64),
                &Tensor::from_slice(&relay_rows),
                upload.as_ref(),
            )?;
        } else {
            copy_host(
                &mut self.input_ids_buf.narrow(0, 0, rows_i64),
                &Tensor::from_slice(&input_ids).view([rows_i64, 1]),
                upload.as_ref(),
            )?;
        }
        copy_host(
            &mut self.positions_buf.narrow(0, 0, rows_i64),
            &Tensor::from_slice(&positions).view([rows_i64, 1]),
            upload.as_ref(),
        )?;
        copy_host(
            &mut self.write_loc_buf.narrow(0, 0, rows_i64),
            &Tensor::from_slice(&write_loc),
            upload.as_ref(),
        )?;
        copy_host(
            &mut self.cache_seqlens_buf.narrow(0, 0, rows_i64),
            &Tensor::from_slice(&cache_seqlens),
            upload.as_ref(),
        )?;
        if let Some(upload) = upload {
            upload.finish().map_err(crate::engine::EngineError::from)?;
        }
        // Future IDs are produced on the execution stream after sampling.
        // Never gather them on the upload stream at the earlier read boundary.
        if let Some((_, ids)) = future {
            let input = ids
                .index_select(0, &self.relay_rows_buf.narrow(0, 0, rows_i64))
                .view([rows_i64, 1]);
            self.input_ids_buf.narrow(0, 0, rows_i64).copy_(&input);
        }
        let model_batch = Batch::decode(
            self.input_ids_buf.narrow(0, 0, rows_i64),
            self.positions_buf.narrow(0, 0, rows_i64),
            Some(AttentionMetadata {
                state_slots,
                forward_mode: BatchPhase::Decode,
                write_loc: Some(self.write_loc_buf.narrow(0, 0, rows_i64)),
                cu_seqlens_q: None,
                prefix_lens: None,
                block_table: self
                    .block_table_buf
                    .as_ref()
                    .map(|table| table.narrow(0, 0, rows_i64)),
                req_to_token: self
                    .req_to_token_buf
                    .as_ref()
                    .map(|table| table.narrow(0, 0, rows_i64)),
                cache_seqlens: Some(self.cache_seqlens_buf.narrow(0, 0, rows_i64)),
                decode_host: Some(DecodeHostMetadata {
                    cache_seqlens,
                    block_table_versions: Some(
                        self.row_mappings[..rows]
                            .iter()
                            .map(|mapping| mapping.table_version)
                            .collect(),
                    ),
                }),
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

fn copy_host(target: &mut Tensor, source: &Tensor, upload: Option<&UploadGuard<'_>>) -> Result<()> {
    if let Some(upload) = upload {
        upload
            .copy(target, source)
            .map_err(crate::engine::EngineError::from)?;
    } else {
        target.copy_(&source.to_device(target.device()));
    }
    Ok(())
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
                state_slot: None,
                page_ids: pages,
                ..Default::default()
            }),
            status: SequenceStatus::Decoding,
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
    fn table_requirements_preserve_incremental_mappings_and_rebuild_on_change() {
        exercise_table_requirements(Device::Cpu);
    }

    #[cfg(has_cuda_graph)]
    #[test]
    #[ignore = "requires CUDA; run scripts/run-rust-tests.sh cuda"]
    fn cuda_table_requirements_preserve_upload_order_on_change() {
        assert!(tch::Cuda::is_available());
        exercise_table_requirements(Device::Cuda(0));
    }

    fn exercise_table_requirements(device: Device) {
        let mut args = ServerArgs::new("unused");
        args.max_running_req = 2;
        args.max_seq_len = 6;
        args.page_size = 2;
        let mut manager = DecodeManager::new(&args, device).unwrap();
        let mut request = running(1, vec![1, 2], vec![5, 6, 7]);
        for requirements in [
            MetadataRequirements {
                block_table: true,
                req_to_token: false,
            },
            MetadataRequirements {
                block_table: false,
                req_to_token: true,
            },
            MetadataRequirements::default(),
        ] {
            manager.set_metadata_requirements(requirements);
            let mapped_before = manager.mapped_tokens;
            let mut last_version = None;
            // The same UID/pages must be fully written after a layout switch;
            // growth only maps new tokens, while a changed page invalidates them.
            for change in 0..3 {
                if change == 1 {
                    request.input_ids.push(3);
                } else if change == 2 {
                    request.cache_handle.as_mut().unwrap().page_ids[0] += 1;
                }
                let batch = manager
                    .schedule_pipeline(std::slice::from_ref(&request))
                    .unwrap()
                    .unwrap();
                let metadata = batch.model_batch.attention_metadata.as_ref().unwrap();
                assert_eq!(metadata.block_table.is_some(), requirements.block_table);
                assert_eq!(metadata.req_to_token.is_some(), requirements.req_to_token);
                crate::engine::ForwardBatch::from_scheduler(&batch.model_batch)
                    .unwrap()
                    .validate(device)
                    .unwrap();
                let pages = &request.cache_handle.as_ref().unwrap().page_ids;
                if let Some(table) = &metadata.block_table {
                    assert_eq!(
                        Vec::<i32>::try_from(&table.to_device(Device::Cpu).view([-1])).unwrap(),
                        pages.iter().map(|&page| page as i32).collect::<Vec<_>>()
                    );
                }
                if let Some(table) = &metadata.req_to_token {
                    let expected = (0..6)
                        .map(|position| {
                            if position < request.input_ids.len() {
                                (pages[position / 2] * 2 + position % 2) as i32
                            } else {
                                -1
                            }
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        Vec::<i32>::try_from(&table.to_device(Device::Cpu).view([-1])).unwrap(),
                        expected
                    );
                }
                let version = metadata
                    .decode_host
                    .as_ref()
                    .unwrap()
                    .block_table_versions
                    .as_ref()
                    .unwrap()[0];
                if let Some(previous) = last_version {
                    if change == 1 {
                        assert_eq!(version, previous);
                    } else {
                        assert_ne!(version, previous);
                    }
                }
                last_version = Some(version);
            }
            if !requirements.req_to_token {
                assert_eq!(manager.mapped_tokens, mapped_before);
                assert!(manager.req_to_token_buf.is_none());
            }
            request.input_ids.truncate(2);
        }
    }

    #[test]
    fn empty_and_finished_requests_produce_no_batch() {
        let mut manager = manager();
        assert!(manager.schedule_decode(&[]).unwrap().is_none());
        let mut request = running(1, vec![2], vec![0]);
        request.finish();
        assert!(manager.schedule_decode(&[request]).unwrap().is_none());
    }

    #[test]
    fn decode_admits_only_decoding_requests_in_serial_and_pipeline() {
        let mut manager = manager();
        let mut requests = Vec::new();
        for (uid, status) in [
            SequenceStatus::Waiting,
            SequenceStatus::Prefilling { cursor: 0 },
            // 即使游标已到 prompt 尾部，仍需等待最后一次 forward 完成。
            SequenceStatus::Prefilling { cursor: 2 },
            SequenceStatus::Finished,
            SequenceStatus::Decoding,
        ]
        .into_iter()
        .enumerate()
        {
            let mut request = running(uid as u64, vec![1, 2], vec![0]);
            request.status = status;
            if !request.is_decoding() {
                request.output_len = 0;
            }
            requests.push(request);
        }
        for pipeline in [false, true] {
            let batch = if pipeline {
                manager.schedule_pipeline(&requests)
            } else {
                manager.schedule_decode(&requests)
            }
            .unwrap()
            .unwrap();
            assert_eq!(batch.request_ids, vec![4]);
        }
        requests.pop();
        assert!(manager.schedule_decode(&requests).unwrap().is_none());
        assert!(manager.schedule_pipeline(&requests).unwrap().is_none());
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
        let initial_version = manager.row_mappings[0].table_version;
        request.input_ids.push(3);
        drop(
            manager
                .schedule_decode(std::slice::from_ref(&request))
                .unwrap(),
        );
        assert_eq!(manager.mapped_tokens, 3);
        assert_eq!(manager.row_mappings[0].table_version, initial_version);
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
        assert_ne!(manager.row_mappings[0].table_version, initial_version);
        assert_eq!(
            Vec::<i32>::try_from(&manager.req_to_token_buf.as_ref().unwrap().get(0)).unwrap(),
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
            Vec::<i32>::try_from(&manager.req_to_token_buf.as_ref().unwrap().get(0)).unwrap(),
            vec![18, -1, -1, -1, -1, -1]
        );
        assert_eq!(
            Vec::<i32>::try_from(&manager.block_table_buf.as_ref().unwrap().get(0)).unwrap(),
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
            Vec::<i32>::try_from(&manager.req_to_token_buf.as_ref().unwrap().get(0)).unwrap(),
            vec![12, 13, 14, -1, -1, -1]
        );
        let c = running(3, vec![9], vec![6]);
        drop(manager.schedule_decode(&[c, b]).unwrap());
        assert_eq!(
            Vec::<i32>::try_from(&manager.req_to_token_buf.as_ref().unwrap().get(0)).unwrap(),
            vec![12, -1, -1, -1, -1, -1]
        );
        assert_eq!(
            Vec::<i32>::try_from(&manager.req_to_token_buf.as_ref().unwrap().get(1)).unwrap(),
            vec![12, 13, 14, -1, -1, -1]
        );
    }
}
