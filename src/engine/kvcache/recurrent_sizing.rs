//! 根据同一份剩余缓存预算，规划线性状态和分页 KV；不分配任何张量。
use super::{KVCacheError, Result};

#[derive(Debug, Clone, Copy)]
pub struct RecurrentCacheOptions {
    pub memory_ratio: f64,
    pub max_slots: Option<usize>,
    pub max_checkpoints: Option<usize>,
}
impl RecurrentCacheOptions {
    pub fn validate(self) -> Result<()> {
        if !self.memory_ratio.is_finite() || self.memory_ratio <= 0.0 {
            return Err(KVCacheError::InvalidArgument(
                "mamba_full_memory_ratio 必须为有限正数".into(),
            ));
        }
        if self.max_slots == Some(0) {
            return Err(KVCacheError::InvalidArgument(
                "max_mamba_cache_size 必须大于零".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecurrentCachePlan {
    pub state_slots: usize,
    pub checkpoint_slots: usize,
    pub state_bytes: usize,
    pub kv_budget_bytes: usize,
}
impl RecurrentCachePlan {
    pub fn new(
        budget: usize,
        bytes_per_state: usize,
        requests: usize,
        min_kv_bytes: usize,
        options: RecurrentCacheOptions,
        cpu_fallback: bool,
    ) -> Result<Self> {
        options.validate()?;
        if bytes_per_state == 0 || requests == 0 {
            return Err(KVCacheError::InvalidArgument(
                "线性状态大小及并发数必须大于零".into(),
            ));
        }
        let disabled = options.max_checkpoints == Some(0);
        let reserved = requests
            .checked_mul(if disabled { 1 } else { 2 })
            .ok_or_else(|| KVCacheError::InvalidArgument("线性状态并发预留溢出".into()))?;
        let minimum = reserved
            .checked_add(usize::from(!disabled))
            .ok_or_else(|| KVCacheError::InvalidArgument("线性状态容量溢出".into()))?;
        let mut slots = if disabled {
            reserved
        } else {
            let fraction = options.memory_ratio / (1.0 + options.memory_ratio);
            let target = (budget as f64 * fraction / bytes_per_state as f64).floor() as usize;
            // CPU 没有显存遥测。限制小型测试模型的槽数量，避免数十万微小张量。
            if cpu_fallback {
                target.min(64usize.max(minimum))
            } else {
                target
            }
        };
        if let Some(limit) = options.max_slots {
            slots = slots.min(limit);
        }
        if let Some(limit) = options.max_checkpoints {
            slots = slots.min(reserved.saturating_add(limit));
        }
        if slots < minimum {
            return Err(KVCacheError::InvalidArgument(format!(
                "线性状态预算不足：可分配 {slots} 槽，但 max_running_req={requests} 至少需要 {minimum} 槽；请降低并发、提高 mamba_full_memory_ratio 或放宽显式容量上限"
            )));
        }
        let state_bytes = slots
            .checked_mul(bytes_per_state)
            .ok_or_else(|| KVCacheError::InvalidArgument("线性状态字节数溢出".into()))?;
        let kv_budget_bytes = budget
            .checked_sub(state_bytes)
            .filter(|&remaining| remaining >= min_kv_bytes)
            .ok_or(KVCacheError::InsufficientBudget {
                required_bytes: state_bytes.saturating_add(min_kv_bytes),
                available_bytes: budget,
            })?;
        Ok(Self {
            state_slots: slots,
            checkpoint_slots: if disabled { 0 } else { slots - reserved },
            state_bytes,
            kv_budget_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options() -> RecurrentCacheOptions {
        RecurrentCacheOptions {
            memory_ratio: 0.9,
            max_slots: None,
            max_checkpoints: None,
        }
    }
    #[test]
    fn historical_capacity_tracks_budget_instead_of_running_request_limit() {
        let small = RecurrentCachePlan::new(10_000, 100, 1, 200, options(), false).unwrap();
        let large = RecurrentCachePlan::new(20_000, 100, 1, 200, options(), false).unwrap();
        assert_eq!(small.state_slots, 47);
        assert_eq!(small.checkpoint_slots, 45);
        assert!(large.checkpoint_slots > small.checkpoint_slots);
        let concurrent = RecurrentCachePlan::new(10_000, 100, 8, 200, options(), false).unwrap();
        assert_eq!(small.state_slots, concurrent.state_slots);
        assert_eq!(concurrent.checkpoint_slots, 31);
        assert_eq!(small.state_bytes + small.kv_budget_bytes, 10_000);
    }
    #[test]
    fn explicit_caps_return_unused_state_budget_to_kv() {
        let capped = RecurrentCachePlan::new(
            10_000,
            100,
            1,
            200,
            RecurrentCacheOptions {
                max_slots: Some(10),
                ..options()
            },
            false,
        )
        .unwrap();
        assert_eq!(capped.checkpoint_slots, 8);
        assert_eq!(capped.kv_budget_bytes, 9_000);
        let checkpoint_cap = RecurrentCachePlan::new(
            10_000,
            100,
            1,
            200,
            RecurrentCacheOptions {
                max_checkpoints: Some(4),
                ..options()
            },
            false,
        )
        .unwrap();
        assert_eq!(checkpoint_cap.state_slots, 6);
        assert_eq!(checkpoint_cap.checkpoint_slots, 4);
        let huge_cap = RecurrentCachePlan::new(
            10_000,
            100,
            1,
            200,
            RecurrentCacheOptions {
                max_slots: Some(1000),
                max_checkpoints: Some(1000),
                ..options()
            },
            false,
        )
        .unwrap();
        assert_eq!(huge_cap.state_slots, 47); // 上限不能迫使分配超出显存比例的容量。
    }
    #[test]
    fn disabled_history_only_budgets_current_and_all_limits_still_apply() {
        let disabled = RecurrentCacheOptions {
            max_checkpoints: Some(0),
            ..options()
        };
        let plan = RecurrentCachePlan::new(1000, 100, 3, 200, disabled, false).unwrap();
        assert_eq!(plan.state_slots, 3);
        assert_eq!(plan.checkpoint_slots, 0);
        assert_eq!(plan.kv_budget_bytes, 700);
        assert!(
            RecurrentCachePlan::new(
                1000,
                100,
                3,
                200,
                RecurrentCacheOptions {
                    max_slots: Some(2),
                    ..disabled
                },
                false
            )
            .is_err()
        );
    }
    #[test]
    fn rejects_invalid_options_insufficient_peak_capacity_and_starved_kv() {
        for ratio in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(
                RecurrentCacheOptions {
                    memory_ratio: ratio,
                    ..options()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            RecurrentCacheOptions {
                max_slots: Some(0),
                ..options()
            }
            .validate()
            .is_err()
        );
        assert!(RecurrentCachePlan::new(1000, 100, 3, 200, options(), false).is_err());
        assert!(
            RecurrentCachePlan::new(
                1000,
                100,
                1,
                200,
                RecurrentCacheOptions {
                    memory_ratio: 100.0,
                    ..options()
                },
                false
            )
            .is_err()
        );
        assert!(RecurrentCachePlan::new(1000, 100, usize::MAX, 200, options(), false).is_err());
    }
    #[test]
    fn cpu_fallback_bounds_tiny_tensor_count_without_changing_gpu_policy() {
        let cpu = RecurrentCachePlan::new(512 * 1024 * 1024, 100, 2, 200, options(), true).unwrap();
        assert_eq!(cpu.state_slots, 64);
        let gpu =
            RecurrentCachePlan::new(512 * 1024 * 1024, 100, 2, 200, options(), false).unwrap();
        assert!(gpu.state_slots > 64);
    }
}
