use std::process::Command;

use tch::{Device, Kind};

use super::{KVCacheError, KVCachePool, ModelCacheSpec, Result};

/// The CPU fallback budget used by mini-sglang when accelerator memory metrics
/// are unavailable.
pub const CPU_KV_CACHE_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KVCacheServerConfig {
    pub page_size: usize,
    pub max_running_req: usize,
    pub max_seq_len: usize,
    pub memory_ratio: f64,
    pub max_total_tokens: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KVCacheModelConfig {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KVCacheAllocationConfig {
    pub server: KVCacheServerConfig,
    pub model: KVCacheModelConfig,
}

/// Computes the pool size and asks libtorch to allocate the backing tensor.
pub struct KVCacheAllocator {
    config: KVCacheAllocationConfig,
    spec: ModelCacheSpec,
}

impl KVCacheAllocator {
    pub fn new(config: KVCacheAllocationConfig) -> Result<Self> {
        let server = config.server;
        let model = config.model;
        if server.page_size == 0
            || server.max_running_req == 0
            || server.max_seq_len == 0
            || model.num_layers == 0
            || model.num_kv_heads == 0
            || model.head_dim == 0
        {
            return Err(KVCacheError::InvalidArgument(
                "KV-cache allocation dimensions must be greater than zero".to_owned(),
            ));
        }
        if !(0.0..=1.0).contains(&server.memory_ratio) {
            return Err(KVCacheError::InvalidArgument(
                "memory_ratio must be in the range 0.0..=1.0".to_owned(),
            ));
        }
        validate_token_cap(server)?;
        let spec = ModelCacheSpec::uniform(model.num_layers, model.num_kv_heads, model.head_dim)?;
        Ok(Self { config, spec })
    }

    pub fn with_spec(config: KVCacheAllocationConfig, spec: ModelCacheSpec) -> Result<Self> {
        let server = config.server;
        if server.page_size == 0 || server.max_running_req == 0 || server.max_seq_len == 0 {
            return Err(KVCacheError::InvalidArgument(
                "cache allocation dimensions must be positive".into(),
            ));
        }
        if !(0.0..=1.0).contains(&server.memory_ratio) {
            return Err(KVCacheError::InvalidArgument(
                "memory_ratio must be in the range 0.0..=1.0".into(),
            ));
        }
        validate_token_cap(server)?;
        if spec.layers().len() != config.model.num_layers {
            return Err(KVCacheError::InvalidArgument(
                "cache layer count must match runtime model layer count".into(),
            ));
        }
        spec.bytes_per_page(server.page_size, 1)?;
        Ok(Self { config, spec })
    }

    pub fn available_memory(&self, device: Device) -> Result<usize> {
        match device {
            Device::Cpu => Ok(CPU_KV_CACHE_BYTES),
            Device::Cuda(index) => {
                let (free, total) = cuda_memory_info(index)?;
                let available = cuda_cache_budget(free, total, self.config.server.memory_ratio);
                if available == 0 {
                    return Err(KVCacheError::MemoryQuery(format!(
                        "模型加载后没有可用于 KV cache 的显存（total={total}, free={free}, memory_ratio={}）",
                        self.config.server.memory_ratio
                    )));
                }
                Ok(available)
            }
            _ => Err(KVCacheError::NotImplemented(
                "加速器空闲显存查询（尚未迁移）",
            )),
        }
    }

    /// Number of usable pages that fit the memory budget, excluding the sink.
    pub fn num_pages(
        &self,
        available: usize,
        num_kv_heads_per_rank: usize,
        dtype_itemsize: usize,
    ) -> Result<usize> {
        if num_kv_heads_per_rank == 0 || dtype_itemsize == 0 {
            return Err(KVCacheError::InvalidArgument(
                "num_kv_heads_per_rank and dtype_itemsize must be greater than zero".to_owned(),
            ));
        }

        // Legacy uniform sizing entry point, retained for existing callers.
        let spec = ModelCacheSpec::uniform(
            self.config.model.num_layers,
            num_kv_heads_per_rank,
            self.config.model.head_dim,
        )?;
        self.num_pages_for_spec(available, &spec, dtype_itemsize)
    }

    /// Budget includes the sink page in every group; never exceed it by forcing
    /// a minimum page count when there is insufficient memory.
    pub fn num_pages_for_spec(
        &self,
        available: usize,
        spec: &ModelCacheSpec,
        itemsize: usize,
    ) -> Result<usize> {
        let server = self.config.server;
        let bytes_per_page = spec.bytes_per_page(server.page_size, itemsize)?;
        let pages_that_fit = (available / bytes_per_page).saturating_sub(1);
        if pages_that_fit == 0 {
            return Err(KVCacheError::InsufficientBudget {
                required_bytes: bytes_per_page.saturating_mul(2),
                available_bytes: available,
            });
        }
        // The shared pool also retains radix prefixes from completed requests.
        // Active-request limits must not cap its capacity on the GPU.
        Ok(server.max_total_tokens.map_or(pages_that_fit, |tokens| {
            pages_that_fit.min(tokens / server.page_size)
        }))
    }

    pub fn allocate(&self, kind: Kind, device: Device, tp_size: usize) -> Result<KVCachePool> {
        let available = self.available_memory(device)?;
        self.allocate_with_budget(kind, device, tp_size, available)
    }

    /// 使用已扣除线性状态的预算；CPU fallback 同样不能重复花费这笔空间。
    pub fn allocate_with_budget(
        &self,
        kind: Kind,
        device: Device,
        tp_size: usize,
        available: usize,
    ) -> Result<KVCachePool> {
        let spec = self.spec.per_rank(tp_size)?;
        let itemsize = kind.elt_size_in_bytes();
        let mut num_pages = self.num_pages_for_spec(available, &spec, itemsize)?;
        let server = self.config.server;
        // CPU has a fixed fallback budget rather than accelerator telemetry.
        // Preserve bounded allocations for small CPU servers and test fixtures.
        if device == Device::Cpu {
            let pages_per_request = server.max_seq_len / server.page_size + 1;
            num_pages = num_pages.min(server.max_running_req.saturating_mul(pages_per_request));
        }
        let allocated_bytes = (num_pages + 1) * spec.bytes_per_page(server.page_size, itemsize)?;
        let pool = KVCachePool::with_spec(spec, num_pages, server.page_size, kind, device)?;
        tracing::info!(
            ?device,
            memory_ratio = server.memory_ratio,
            max_total_tokens = server.max_total_tokens,
            budget_bytes = available,
            allocated_bytes,
            usable_pages = num_pages,
            page_size = server.page_size,
            token_capacity = num_pages.saturating_mul(server.page_size),
            "allocated KV cache pool"
        );
        Ok(pool)
    }
}

fn validate_token_cap(server: KVCacheServerConfig) -> Result<()> {
    if server
        .max_total_tokens
        .is_some_and(|tokens| tokens < server.page_size)
    {
        return Err(KVCacheError::InvalidArgument(
            "max_total_tokens must hold at least one KV page".into(),
        ));
    }
    Ok(())
}

fn cuda_cache_budget(free: usize, total: usize, memory_ratio: f64) -> usize {
    let used = total.saturating_sub(free);
    let budget = (total as f64 * memory_ratio) as usize;
    budget.saturating_sub(used).min(free)
}

fn cuda_memory_info(index: usize) -> Result<(usize, usize)> {
    // run-server.sh puts the same PyTorch environment used by libtorch on PATH.
    // Querying through torch preserves CUDA_VISIBLE_DEVICES index remapping.
    let output = Command::new("python")
        .args([
            "-c",
            "import sys, torch; free, total = torch.cuda.mem_get_info(int(sys.argv[1])); print(f'{free} {total}')",
            &index.to_string(),
        ])
        .output()
        .map_err(|error| {
            KVCacheError::MemoryQuery(format!(
                "无法运行 python/PyTorch：{error}；请使用 scripts/run-server.sh 启动"
            ))
        })?;
    if !output.status.success() {
        return Err(KVCacheError::MemoryQuery(format!(
            "PyTorch 返回状态 {}：{}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    parse_memory_info(&output.stdout)
}

fn parse_memory_info(output: &[u8]) -> Result<(usize, usize)> {
    let output = std::str::from_utf8(output)
        .map_err(|error| KVCacheError::MemoryQuery(error.to_string()))?;
    let mut numbers = output.split_whitespace();
    let free = numbers.next().and_then(|value| value.parse::<usize>().ok());
    let total = numbers.next().and_then(|value| value.parse::<usize>().ok());
    match (free, total, numbers.next()) {
        (Some(free), Some(total), None) if free <= total => Ok((free, total)),
        _ => Err(KVCacheError::MemoryQuery(format!(
            "PyTorch 返回了无效的显存数据: {output:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocator() -> KVCacheAllocator {
        KVCacheAllocator::new(KVCacheAllocationConfig {
            server: KVCacheServerConfig {
                page_size: 16,
                max_running_req: 8,
                max_seq_len: 128,
                memory_ratio: 0.9,
                max_total_tokens: None,
            },
            model: KVCacheModelConfig {
                num_layers: 2,
                num_kv_heads: 8,
                head_dim: 64,
            },
        })
        .unwrap()
    }

    #[test]
    fn explicit_remaining_budget_is_not_replaced_by_cpu_fallback() {
        let allocator = allocator();
        let bytes = allocator.spec.bytes_per_page(16, 4).unwrap();
        let pool = allocator
            .allocate_with_budget(Kind::Float, Device::Cpu, 1, bytes * 4)
            .unwrap();
        assert_eq!(pool.layout.num_pages, 3); // 第四页为 sink，不能重用 512 MiB fallback。
        assert!(
            allocator
                .allocate_with_budget(Kind::Float, Device::Cpu, 1, bytes)
                .is_err()
        );
    }

    #[test]
    fn heterogeneous_budget_includes_all_groups_and_reserved_pages() {
        use super::super::PagedKvSpec;
        let spec = ModelCacheSpec::new(vec![
            PagedKvSpec {
                num_kv_heads: 2,
                head_dim: 16,
            },
            PagedKvSpec {
                num_kv_heads: 1,
                head_dim: 64,
            },
        ])
        .unwrap();
        let bytes = spec.bytes_per_page(16, 4).unwrap();
        let allocator = KVCacheAllocator::with_spec(allocator().config, spec.clone()).unwrap();
        assert_eq!(
            allocator.num_pages_for_spec(bytes * 4, &spec, 4).unwrap(),
            3
        );
        assert!(
            allocator
                .num_pages_for_spec(bytes * 2 - 1, &spec, 4)
                .is_err()
        );
        assert_eq!(
            allocator.num_pages_for_spec(bytes * 2, &spec, 4).unwrap(),
            1
        );
        assert!(
            KVCacheAllocator::with_spec(
                allocator.config,
                ModelCacheSpec::uniform(3, 1, 4).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn sizes_pages_by_memory_budget_independently_of_scheduler_limits() {
        let budget = 1024 * 1024 * 1024;
        let allocator = allocator();
        let bytes = allocator.spec.bytes_per_page(16, 4).unwrap();
        assert_eq!(
            allocator.num_pages(budget, 8, 4).unwrap(),
            budget / bytes - 1
        );

        let mut config = allocator.config;
        config.server.max_running_req = 1;
        config.server.max_seq_len = 16;
        assert_eq!(
            KVCacheAllocator::new(config)
                .unwrap()
                .num_pages(budget, 8, 4)
                .unwrap(),
            budget / bytes - 1
        );
    }

    #[test]
    fn single_qwen3_request_can_use_full_gpu_cache_budget() {
        let allocator = KVCacheAllocator::new(KVCacheAllocationConfig {
            server: KVCacheServerConfig {
                page_size: 16,
                max_running_req: 1,
                max_seq_len: 40960,
                memory_ratio: 0.9,
                max_total_tokens: None,
            },
            model: KVCacheModelConfig {
                num_layers: 28,
                num_kv_heads: 8,
                head_dim: 128,
            },
        })
        .unwrap();
        let budget = 64 * 1024 * 1024 * 1024usize;
        let bytes = allocator.spec.bytes_per_page(16, 2).unwrap();
        let pages = allocator.num_pages(budget, 8, 2).unwrap();
        assert_eq!(pages, budget / bytes - 1);
        assert!((pages + 1) * bytes <= budget);
        assert!(budget - (pages + 1) * bytes < bytes);
    }

    #[test]
    fn cpu_pool_keeps_small_scheduler_demand_cap() {
        let pool = allocator().allocate(Kind::Float, Device::Cpu, 1).unwrap();
        assert_eq!(pool.layout.num_pages, 72);
    }

    #[test]
    fn explicit_token_cap_rounds_down_without_exceeding_memory_budget() {
        let mut config = allocator().config;
        config.server.max_total_tokens = Some(47);
        let allocator = KVCacheAllocator::new(config).unwrap();
        let bytes = allocator.spec.bytes_per_page(16, 4).unwrap();
        assert_eq!(allocator.num_pages(bytes * 100, 8, 4).unwrap(), 2);
        assert_eq!(allocator.num_pages(bytes * 2, 8, 4).unwrap(), 1);
        assert!(allocator.num_pages(bytes, 8, 4).is_err());
        for tokens in [0, 15] {
            config.server.max_total_tokens = Some(tokens);
            assert!(KVCacheAllocator::new(config).is_err());
            assert!(KVCacheAllocator::with_spec(config, allocator.spec.clone()).is_err());
        }
    }

    #[test]
    fn rejects_budget_smaller_than_sink_and_one_real_page() {
        assert!(allocator().num_pages(1, 8, 4).is_err());
    }

    #[test]
    fn cuda_budget_accounts_for_model_memory_and_reserve() {
        assert_eq!(cuda_cache_budget(600, 1000, 0.9), 500);
        assert_eq!(cuda_cache_budget(100, 1000, 0.9), 0);
        assert_eq!(cuda_cache_budget(600, 1000, 1.0), 600);
        assert_eq!(parse_memory_info(b"600 1000\n").unwrap(), (600, 1000));
        assert!(parse_memory_info(b"1100 1000\n").is_err());
    }
}
