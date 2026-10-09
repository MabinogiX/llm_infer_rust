# Qwen3 decode 性能优化结果

日期：2026-10-07。远程项目：`sglang-test:/sjtu/yaosikai/llm_infer_rust`。

已完成 FlashInfer workspace 复用、decode KV 映射增量更新、残差加法与 RMSNorm 融合。本地源码和远程项目均已更新，远程 release 二进制已重新构建。

同一张 A100 GPU 1 上，不使用 CUDA Graph 的最终单请求测试中，32、128、1024 tokens 的完整请求延迟分别降低约 20.4%、21.7%、22.8%。12 次请求的生成内容和 token 用量与修改前完全一致。

## 实现变化

### FlashInfer workspace 复用

attention backend 持有一个持久化 native plan，包含 device float workspace、device int workspace 和 pinned CPU int workspace。每次 forward 更新 DecodePlan，所有 decoder 层共享该次规划；容量足够时不再分配和释放 workspace。容量不足时按需求扩容，backend 销毁时释放。

Rust 通过共享所有权保留规划，并在前一批仍持有它时拒绝覆盖，避免活跃 batch 使用被改写的规划。C++ 桥接校验设备、dtype 和模型几何，并在对应 CUDA 设备上释放 workspace。

每步分页 metadata 的构造、`masked_select` 和 `indptr` 的 GPU→CPU 拷贝仍然存在。当前依赖该阻塞拷贝保证同一 CUDA stream 上的前序操作完成后再更新 workspace；本次没有消除这一同步点。

涉及文件：`src/models/attention/backend.rs`、`flashinfer_bridge.cpp`、`flashinfer_plan_run.cu`。

### KV 映射增量更新

DecodeManager 为每个 batch row 保存请求 ID、已映射长度和页号快照。稳定请求只生成、上传新增 token 的位置，页号没有变化时不再重新上传 block table。

请求换行或 row 被新请求复用时重新初始化；页号变化时更新受影响的后缀；序列或页表缩短时清理失效区间。保留页号快照检查，因此这里消除的是历史 token 映射的逐步重建，而不是全部 CPU 扫描工作。

涉及文件：`src/scheduler/decode.rs`。

### 原位 fused add RMSNorm

Qwen3 decoder 采用独立 residual 流，MLP 输出的残差加法延迟到下一层输入归一化，最后一层延迟到最终归一化。CUDA BF16/FP16 路径在一个 kernel 内更新 residual 并归一化 activation，复用已有张量。

残差和先舍入到 BF16/FP16，再进入 FP32 RMS reduction，保持原实现的计算顺序。28 层 Qwen3 0.6B 不再单独启动原来的 56 次 residual-add kernel。CPU 路径保留普通张量计算。

涉及文件：`src/models/qwen3/model.rs`、`ops.rs`、`fused_ops_bridge.cpp`、`fused_ops.cu`。

### 多请求 QKV 布局修复

并发验证发现，batch 大于 1 时，从 packed QKV 切出的 Q 是非连续视图，现有 native FlashInfer 接口要求连续 query。适配器现在通过 `contiguous()` 满足该要求；batch 为 1 时无需数据复制。新增 CUDA 回归测试使用实际的 packed QKV 非连续布局。

## 性能验证

修改前后在同一张 A100-SXM4-80GB GPU 1 上依次运行实例，参数均为 BF16、FlashInfer、`max-running-req=1`、`max-seq-len=40960`、`page-size=16`、`cuda-graph-bs=0`、`SGLANG_PROFILE_STEPS=0`。

固定 prompt 为 `Explain why the sky is blue in detail.`，两边均报告 18 个 prompt tokens。请求使用 `max_completion_tokens`、`temperature=0`、`ignore_eos=true`、非流式输出。每个输出长度请求 4 次，排除第一次，取其余 3 次客户端完整 HTTP 请求延迟的中位数。

| 输出长度 | 修改前 | 最终版本 | 延迟降低 |
| --- | --- | --- | --- |
| 32 tokens | 0.175019s | 0.139339s | 20.39% |
| 128 tokens | 0.655140s | 0.512903s | 21.71% |
| 1024 tokens | 5.262442s | 4.061340s | 22.82% |

先前预验证轮的优化版本中位数分别为 0.150410s、0.554241s、4.444570s，相对同一基线降低约 14%–16%。两轮测量存在运行波动，最终轮未穿插重测旧版本；上述结果是本机短 prompt、热请求下的观察值，不分别归因于某一项优化，也不能直接外推到长上下文或高并发吞吐。

原始数据保存在本地 `benchmark/results/decode-optimizations-2026-10-07/`：`before.json`、`after-initial.json`、`after.json`。`measure.py` 保存测量脚本，以输出 JSON 路径作为第一个参数，测试端口固定为 8002。该目录按仓库现有规则被 Git 忽略。

## 正确性验证

- 本地完整 Rust 测试：99 个通过。
- 远程 release 完整 Rust 测试：102 个通过，包含真实 CUDA 测试。
- workspace 测试：在 batch 1/2、不同序列长度和跨 page 边界变化下，检查同一 plan 复用、预热后分配次数不增加，并与 SDPA attention 结果对照。
- KV 映射测试：稳定增长只映射新增 token，覆盖页号变更、序列回退、请求换行和物理页复用。
- 融合算子测试：BF16、FP16，token 数 1/3/65、宽度 13/1024；残差和归一化结果均与原 CUDA 计算逐值一致，张量地址保持不变。
- 最终版本完整 API 测试：11 个通过，运行时 `max-running-req=2`；新增 12 个并发请求，覆盖不同 prompt 长度、输出上限和 decode row 复用。
- 最终性能测试的 12 次响应：生成 message 和 token usage 与修改前完全一致。

测试实例已停止。远程原有 `tmp.sh` 可以直接启动重新构建的版本。
