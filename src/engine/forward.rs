//! Validated model-facing execution inputs, distinct from scheduler-owned batches.
use super::{AttentionMetadata, Batch, BatchPhase, ModelRunnerError};
use tch::{Device, Kind, Tensor};
type Result<T> = std::result::Result<T, ModelRunnerError>;

#[derive(Clone, Copy)]
pub enum ForwardMode<'a> {
    Prefill { logits_indices: &'a Tensor },
    Decode,
}

#[derive(Clone, Copy)]
pub struct ForwardBatch<'a> {
    pub input_ids: &'a Tensor,
    pub positions: &'a Tensor,
    pub attention: Option<&'a AttentionMetadata>,
    pub mode: ForwardMode<'a>,
}

impl<'a> ForwardBatch<'a> {
    pub fn from_scheduler(batch: &'a Batch) -> Result<Self> {
        let mode = match batch.phase {
            BatchPhase::Prefill => ForwardMode::Prefill {
                logits_indices: batch
                    .logits_indices
                    .as_ref()
                    .ok_or(ModelRunnerError::MissingPrefillLogitsIndices)?,
            },
            BatchPhase::Decode => {
                if batch.logits_indices.is_some() {
                    return Err(invalid("decode must not have prefill logits indices"));
                }
                ForwardMode::Decode
            }
        };
        Ok(Self {
            input_ids: &batch.input_ids,
            positions: &batch.positions,
            attention: batch.attention_metadata.as_ref(),
            mode,
        })
    }
    pub fn phase(self) -> BatchPhase {
        match self.mode {
            ForwardMode::Prefill { .. } => BatchPhase::Prefill,
            ForwardMode::Decode => BatchPhase::Decode,
        }
    }
    pub fn logits_indices(self) -> Option<&'a Tensor> {
        match self.mode {
            ForwardMode::Prefill { logits_indices } => Some(logits_indices),
            ForwardMode::Decode => None,
        }
    }
    pub fn tokens(self) -> usize {
        self.input_ids.numel()
    }
    pub fn validate(self, device: Device) -> Result<()> {
        if self.tokens() != self.positions.numel() {
            return Err(ModelRunnerError::InputPositionLengthMismatch {
                input_ids: self.tokens(),
                positions: self.positions.numel(),
            });
        }
        if self.tokens() == 0 {
            return Err(invalid("forward requires nonempty tokens"));
        }
        for tensor in [self.input_ids, self.positions] {
            check_tensor(tensor, device, Kind::Int64)?;
            if tensor.dim() != 1 && (tensor.dim() != 2 || tensor.size()[1] != 1) {
                return Err(invalid(
                    "token and position tensors must be [tokens] or [tokens, 1]",
                ));
            }
        }
        if let Some(indices) = self.logits_indices() {
            check_tensor(indices, device, Kind::Int64)?;
            if indices.dim() != 1 || indices.numel() == 0 {
                return Err(invalid("prefill logits indices must be a nonempty vector"));
            }
        }
        let Some(meta) = self.attention else {
            return Ok(());
        };
        if meta.forward_mode != self.phase() {
            return Err(invalid("attention phase disagrees with forward mode"));
        }
        let writes = meta
            .write_loc
            .as_ref()
            .ok_or_else(|| invalid("cached forward requires write locations"))?;
        check_vector(writes, device, self.tokens())?;
        let rows = match self.mode {
            ForwardMode::Decode => {
                if meta.cu_seqlens_q.is_some() || meta.prefix_lens.is_some() {
                    return Err(invalid("decode cannot carry prefill boundaries"));
                }
                let lengths = meta
                    .cache_seqlens
                    .as_ref()
                    .ok_or_else(|| invalid("decode requires cache lengths"))?;
                check_vector(lengths, device, self.tokens())?;
                if let Some(host) = &meta.decode_host {
                    if host.cache_seqlens.len() != self.tokens()
                        || host.cache_seqlens.iter().any(|&length| length <= 0)
                        || host
                            .block_table_versions
                            .as_ref()
                            .is_some_and(|versions| versions.len() != self.tokens())
                    {
                        return Err(invalid("invalid decode host metadata"));
                    }
                }
                self.tokens()
            }
            ForwardMode::Prefill { .. } => {
                if meta.cache_seqlens.is_some() || meta.decode_host.is_some() {
                    return Err(invalid("prefill cannot carry decode cache lengths"));
                }
                let bounds = meta
                    .cu_seqlens_q
                    .as_ref()
                    .ok_or_else(|| invalid("prefill requires cumulative query lengths"))?;
                if bounds.dim() != 1 || bounds.numel() < 2 {
                    return Err(invalid("invalid prefill query boundaries"));
                }
                check_tensor(bounds, device, Kind::Int)?;
                let rows = bounds.numel() - 1;
                let prefix = meta
                    .prefix_lens
                    .as_ref()
                    .ok_or_else(|| invalid("prefill requires prefix lengths"))?;
                check_vector(prefix, device, rows)?;
                if self.logits_indices().unwrap().numel() != rows {
                    return Err(invalid("prefill needs one logits index per request"));
                }
                rows
            }
        };
        if let Some(slots) = &meta.state_slots {
            let unique: std::collections::HashSet<_> = slots.iter().copied().collect();
            if slots.len() != rows || unique.len() != rows {
                return Err(invalid("循环状态槽必须与请求一一对应且互不重复"));
            }
        }
        if meta.block_table.is_none() && meta.req_to_token.is_none() {
            return Err(invalid("cached forward requires a page or token table"));
        }
        for table in [&meta.block_table, &meta.req_to_token]
            .into_iter()
            .flatten()
        {
            check_tensor(table, device, Kind::Int)?;
            if table.dim() != 2 || table.size()[0] != rows as i64 || table.size()[1] == 0 {
                return Err(invalid(
                    "cache table row count disagrees with forward batch",
                ));
            }
        }
        if meta.max_seqlen.is_none_or(|n| n == 0) {
            return Err(invalid(
                "cached forward requires positive max sequence length",
            ));
        }
        Ok(())
    }
}

pub struct ForwardOutput {
    pub logits: Tensor,
}
impl ForwardOutput {
    pub fn new(logits: Tensor) -> Self {
        Self { logits }
    }
}

fn invalid(message: &str) -> ModelRunnerError {
    ModelRunnerError::Model(message.into())
}
fn check_tensor(t: &Tensor, device: Device, kind: Kind) -> Result<()> {
    if t.device() != device {
        return Err(ModelRunnerError::TensorOnWrongDevice {
            expected: device,
            actual: t.device(),
        });
    }
    if t.kind() != kind {
        return Err(invalid(&format!(
            "forward tensor expected {kind:?}, received {:?}",
            t.kind()
        )));
    }
    Ok(())
}
fn check_vector(t: &Tensor, device: Device, len: usize) -> Result<()> {
    check_tensor(t, device, Kind::Int)?;
    if t.dim() != 1 || t.numel() != len {
        return Err(invalid(
            "metadata vector length disagrees with forward batch",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn decode() -> Batch {
        Batch::decode(
            Tensor::ones([2, 1], (Kind::Int64, Device::Cpu)),
            Tensor::zeros([2, 1], (Kind::Int64, Device::Cpu)),
            Some(AttentionMetadata {
                state_slots: None,
                forward_mode: BatchPhase::Decode,
                decode_host: None,
                write_loc: Some(Tensor::from_slice(&[2i32, 4])),
                cache_seqlens: Some(Tensor::from_slice(&[1i32, 2])),
                block_table: Some(Tensor::ones([2, 2], (Kind::Int, Device::Cpu))),
                req_to_token: None,
                cu_seqlens_q: None,
                prefix_lens: None,
                max_seqlen: Some(2),
            }),
        )
    }
    #[test]
    fn rejects_conflicting_modes_dtypes_and_metadata_shapes() {
        let batch = decode();
        ForwardBatch::from_scheduler(&batch)
            .unwrap()
            .validate(Device::Cpu)
            .unwrap();
        let mut batch = decode();
        batch.logits_indices = Some(Tensor::from_slice(&[0i64]));
        assert!(ForwardBatch::from_scheduler(&batch).is_err());
        let mut batch = decode();
        batch.attention_metadata.as_mut().unwrap().forward_mode = BatchPhase::Prefill;
        assert!(
            ForwardBatch::from_scheduler(&batch)
                .unwrap()
                .validate(Device::Cpu)
                .is_err()
        );
        let mut batch = decode();
        batch.attention_metadata.as_mut().unwrap().block_table =
            Some(Tensor::ones([1, 2], (Kind::Int, Device::Cpu)));
        assert!(
            ForwardBatch::from_scheduler(&batch)
                .unwrap()
                .validate(Device::Cpu)
                .is_err()
        );
        let mut batch = decode();
        batch.attention_metadata.as_mut().unwrap().write_loc =
            Some(Tensor::ones([2], (Kind::Int64, Device::Cpu)));
        assert!(
            ForwardBatch::from_scheduler(&batch)
                .unwrap()
                .validate(Device::Cpu)
                .is_err()
        );
        let mut batch = decode();
        batch.input_ids = Tensor::ones([1, 2], (Kind::Int64, Device::Cpu));
        assert!(
            ForwardBatch::from_scheduler(&batch)
                .unwrap()
                .validate(Device::Cpu)
                .is_err()
        );
    }
    #[test]
    fn validates_decode_host_snapshot_without_device_readback() {
        for (lengths, versions, valid) in [
            (vec![1, 2], None, true),
            (vec![1], None, false),
            (vec![0, 2], None, false),
            (vec![1, 2], Some(vec![1]), false),
            (vec![1, 2], Some(vec![1, 2]), true),
        ] {
            let mut batch = decode();
            batch.attention_metadata.as_mut().unwrap().decode_host =
                Some(super::super::DecodeHostMetadata {
                    cache_seqlens: lengths,
                    block_table_versions: versions,
                });
            assert_eq!(
                ForwardBatch::from_scheduler(&batch)
                    .unwrap()
                    .validate(Device::Cpu)
                    .is_ok(),
                valid
            );
        }
    }

    #[test]
    fn prefill_requires_logit_rows_and_cache_boundaries_to_agree() {
        let batch = Batch::prefill(
            Tensor::ones([3], (Kind::Int64, Device::Cpu)),
            Tensor::zeros([3], (Kind::Int64, Device::Cpu)),
            Some(AttentionMetadata {
                state_slots: None,
                forward_mode: BatchPhase::Prefill,
                decode_host: None,
                write_loc: Some(Tensor::from_slice(&[2i32, 3, 4])),
                cache_seqlens: None,
                block_table: None,
                req_to_token: Some(Tensor::ones([2, 4], (Kind::Int, Device::Cpu))),
                cu_seqlens_q: Some(Tensor::from_slice(&[0i32, 1, 3])),
                prefix_lens: Some(Tensor::from_slice(&[0i32, 0])),
                max_seqlen: Some(2),
            }),
            Tensor::from_slice(&[0i64, 2]),
        );
        ForwardBatch::from_scheduler(&batch)
            .unwrap()
            .validate(Device::Cpu)
            .unwrap();
        let mut batch = batch;
        batch.logits_indices = Some(Tensor::from_slice(&[2i64]));
        assert!(
            ForwardBatch::from_scheduler(&batch)
                .unwrap()
                .validate(Device::Cpu)
                .is_err()
        );
    }
}

#[cfg(test)]
mod recurrent_slot_tests {
    use super::*;
    use crate::engine::kvcache::{RecurrentStatePool, RequestState};
    #[test]
    fn duplicate_or_wrong_count_state_slots_fail_before_forward() {
        let mut pool = RecurrentStatePool::new(2, 0, || RequestState {
            position: 0,
            layers: Vec::new(),
        });
        let a = pool.alloc_request().unwrap();
        let b = pool.alloc_request().unwrap();
        let make = |slots| crate::engine::AttentionMetadata {
            state_slots: Some(slots),
            forward_mode: BatchPhase::Decode,
            decode_host: None,
            write_loc: Some(Tensor::from_slice(&[2i32, 4])),
            cu_seqlens_q: None,
            prefix_lens: None,
            block_table: Some(Tensor::ones([2, 1], (Kind::Int, Device::Cpu))),
            req_to_token: None,
            cache_seqlens: Some(Tensor::from_slice(&[1i32, 1])),
            max_seqlen: Some(1),
        };
        for (slots, valid) in [(vec![a, b], true), (vec![a, a], false), (vec![a], false)] {
            let batch = Batch::decode(
                Tensor::from_slice(&[1i64, 2]),
                Tensor::from_slice(&[0i64, 0]),
                Some(make(slots)),
            );
            assert_eq!(
                ForwardBatch::from_scheduler(&batch)
                    .unwrap()
                    .validate(Device::Cpu)
                    .is_ok(),
                valid
            );
        }
    }
}
