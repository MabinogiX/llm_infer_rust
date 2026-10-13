# 原版 SGLang 调度逻辑与 Rust 实现可迁移设计

## 结论

当前 Rust 实现已经具备原版 SGLang overlap 调度最核心的执行骨架：一批 outstanding 结果、GPU token relay、输入读取边界、completion fence，以及逻辑结束和 KV 回收分离。因此不应整体照搬 Python scheduler，而应优先迁移调度策略和状态边界。

建议的迁移顺序是：

1. 稳定 relay row 映射，避免按 UID 在线性查找；
2. 把调度选择抽象成显式的 batch plan；
3. 增加连续 prefill 的 TTFT/吞吐策略开关；
4. 增加独立的 prefill token budget；
5. 在上述基础稳定后，验证 mixed chunk prefill 的 graph 命中和真实吞吐；
6. 只有需要 speculative decoding 或 grammar 时，才引入对应 barrier。

## 原版 SGLang 的调度模型

### overlap 主循环

原版 `event_loop_overlap` 使用 `result_queue` 保存已经发射、但结果尚未完成 CPU 处理的 batch。每轮的顺序是：

1. 接收请求；
2. 根据 `running_batch` 和 `last_batch` 生成下一批；
3. 发射当前 batch；
4. 将当前 batch 的结果加入 `result_queue`；
5. 处理上一批结果；
6. 执行依赖上一批结果的采样、grammar 或 speculative 状态更新。

正常稳态下，队列通常只保留一批 outstanding 结果；`result_queue` 的形式允许在特殊 worker 或同步边界下暂时保留更多状态。

参考：

- 原版 [`event_loop_overlap`](/Users/dp/code/sglang/python/sglang/srt/managers/scheduler.py:1938)
- 当前 Rust [`step_overlap`](../src/scheduler/scheduler.rs:540)

### 调度选择顺序

原版 `get_next_batch_to_run` 会先把上一轮完成的 prefill 合并到 running batch，再决定本轮执行：

- 新的 prefill；
- running decode；
- mixed chunk prefill；
- 暂缓 prefill 或 idle。

因此原版可以形成：

```text
decode → prefill → prefill → decode
```

prefill 并不是简单地“有请求就立即排空 decode”，而是由 token budget、KV 容量、优先级、chunk 状态和调度策略共同决定。

参考原版 [`get_next_batch_to_run`](/Users/dp/code/sglang/python/sglang/srt/managers/scheduler.py:3655)。

### FutureMap 与 GPU relay

原版 `FutureMap` 是按 request pool index 索引的长期 GPU buffer。它不仅保存下一 token，还可以保存：

- 新 sequence length；
- speculative 额外输入；
- grammar 或 confidence relay 数据；
- 必要时的 pinned host mirror。

forward 入口通过 `resolve_forward_inputs` 从这些 buffer 中读取，forward 结束后通过 `publish` 写入下一轮状态。这样下一轮 decode 不需要先从 CPU 取得上一轮 token。

参考：

- [`resolve_forward_inputs`](/Users/dp/code/sglang/python/sglang/srt/managers/overlap_utils.py:87)
- [`FutureMap`](/Users/dp/code/sglang/python/sglang/srt/managers/overlap_utils.py:248)
- [`publish` / `stash`](/Users/dp/code/sglang/python/sglang/srt/managers/overlap_utils.py:513)

### 连续 prefill 的 overlap 边界

原版提供 `is_disable_overlap_for_batch`。当连续两个 batch 都是 prefill 时，可以主动处理完上一批再发射下一批，以改善第一个 prefill 的 TTFT；吞吐优先时也可以保留 overlap。

该判断还承担依赖同步职责，例如 grammar 状态必须先推进、DP rank 必须做出一致的 overlap 决策。

参考原版 [`is_disable_overlap_for_batch`](/Users/dp/code/sglang/python/sglang/srt/managers/scheduler.py:2012)。

### Prefill batch admission

原版 `PrefillAdder` 在构造 prefill batch 时综合考虑：

- uncached token 总量；
- 最大 running request 数；
- KV page 可用量；
- cache 命中长度；
- chunked prefill；
- priority 和 preemption；
- LoRA、pooling mode 等 batch 兼容性。

这使得短请求可以在长 prompt 阻塞时继续进入，而不是简单按 FIFO 等待。

参考原版 [`get_new_batch_prefill`](/Users/dp/code/sglang/python/sglang/srt/managers/scheduler.py:3825) 附近的 `PrefillAdder`。

## 当前 Rust 实现对照

当前 Rust 已经实现了以下核心机制：

| 原版概念 | Rust 对应实现 | 状态 |
| --- | --- | --- |
| `result_queue` / 上一批结果 | `PendingBatch` / `outstanding` | 已具备 |
| GPU token relay | `PendingSample::relay`、`schedule_relay` | 已具备 |
| 结果完成事件 | `CompletionFence` | 已具备 |
| 输入读取边界 | `InputReadFence` | 已具备 |
| EOS/取消后的资源延迟释放 | `RetiredRequest` | 已具备 |
| FutureMap 的稳定 request slot | `FutureMap` 的稳定 slot 与 device token buffer | 已实现普通 decode 子集 |
| running batch 与 last batch 计划 | `SchedulePlan` + `plan_after` | 已实现 overlap 子集 |
| 连续 prefill 策略 | `disable_consecutive_prefill_overlap` | 已实现 |
| token budget / priority / preemption | 独立 token budget；priority/preemption 尚未实现 | 部分实现 |
| mixed chunk prefill | prefill-shaped `SchedulePlan`，包含 prefill rows 和 decode rows | 已实现基础版本 |

当前 Rust 的关键路径：

- [`Scheduler::step_overlap`](../src/scheduler/scheduler.rs:540)
- [`DecodeManager::schedule_relay`](../src/scheduler/decode.rs:105)
- [`PrefillManager::schedule_prefill`](../src/scheduler/prefill.rs:74)

请求队列和 KV handle ownership 已从 `PrefillManager` 拆出到
[`RequestManager`](../src/scheduler/request_manager.rs)。`PrefillManager` 现在只负责按
prefill budget 选择请求并构造 `Batch`；decode、EOS、取消和 retirement 直接依赖
`RequestManager`。这与原版 scheduler 将 waiting/running 状态和 prefill admission
分开管理的职责边界一致。

## 建议迁移项

### 1. 稳定 relay row 映射

本阶段已将 `schedule_relay` 从 `future: Vec<(RequestId, usize)>` 改为 `FutureMap`。FutureMap 为每个 request 分配稳定 slot，并将 sampled IDs 发布到稳定的 device buffer；下一批只按 request row 读取 slot，不再对上一批 request IDs 逐行线性查找。

当前映射为：

```text
RequestId → stable decode row
stable row → relay token / predicted length / page metadata
```

请求结束、prefill 插队或 batch 重排时，只更新 row ownership，不重新扫描整个 future vector。该实现覆盖普通 decode overlap；sequence length 和 speculative 额外 payload 仍沿用现有 CPU completion metadata，后续可在同一 FutureMap 上扩展。

### 2. 显式 batch plan

此前 `step_overlap` 同时负责：

- 选择下一批；
- 发射 GPU；
- 读取上一批结果；
- 处理 EOS；
- 安排 KV retirement。

当前已增加以下内部计划对象：

```text
SchedulePlan {
    phase: Prefill | Decode | Mixed | Idle,
    request_ids,
    relay_source,
    resource_barrier,
}
```

`plan_after` 只决定“下一批是什么以及依赖什么”，`launch_plan` 负责发射，`read_pending/complete` 负责上一批结果处理。结果处理阶段再决定请求是否逻辑结束和 KV 是否进入 retirement，继续保持“结果发送”和“物理回收”分离。

### 3. 连续 prefill 策略开关

当前已增加显式配置：

```text
consecutive_prefill_policy = DrainForTTFT | OverlapForThroughput
```

Rust CLI 对应 `--enable-consecutive-prefill-overlap`。默认关闭，表示
`DrainForTTFT`；打开后表示 `OverlapForThroughput`。吞吐模式只允许连续 prefill
之间重叠，切回 decode 前仍会等待最后一个 prefill 结果完成。

该策略只能控制 overlap 边界，不能替代同一请求的 KV publish、页表变化和 decode metadata 依赖屏障。

### 4. 独立 prefill token budget

当前已增加独立的：

```text
max_prefill_tokens_per_batch
```

CLI 对应 `--max-prefill-tokens-per-batch`，按 uncached token 数累加，达到预算后停止继续吸收 pending request；传入 `0` 表示不启用独立 token 上限。默认值为 8192。这样可以控制长 prompt 对 decode 和短请求的影响，也为后续 priority/preemption 留出明确入口。

### 5. Priority 与有限抢占

在 token budget 稳定后，可以增加最小化的优先级策略：

- pending request 按优先级和等待时间排序；
- 只在 KV 或 token budget 不足时抢占；
- 被抢占请求回到 pending，保留可复用的 prefix cache；
- 不抢占仍被 outstanding GPU batch 使用的 request，必须遵守现有 completion fence。

## 暂不建议迁移项

以下原版能力依赖大量外围状态，当前没有明确需求时不应直接搬入：

- speculative decoding 全套 FutureMap payload；
- grammar FSM barrier；
- DP attention 的全局同步；
- chunked prefill 的完整状态机；
- disaggregation、LoRA、HiCache、beam search；
- 原版 `ScheduleBatch` 的大范围可变字段快照。

这些能力可以在需要时以独立 feature 引入。提前搬运会把 Python scheduler 的扩展复杂度引入当前已经较小的 Rust 核心路径。

## 分阶段实施建议

### 阶段一：relay 与 plan

FutureMap 的稳定 relay slot、普通 decode 的 `SchedulePlan`、连续 prefill policy、独立 token budget 和基础 mixed chunk prefill 已完成。下一步仍需完善 priority/preemption，并验证 mixed batch 的 CUDA graph 命中和真实模型吞吐。

验收重点：

- decode→decode 数值结果不变；
- decode→prefill→prefill→decode 的 KV、logits、保留输出不变；
- EOS、取消和 request reorder 不会读错 relay row；
- batch size 变化不产生额外完整 input_ids 复制。

### 阶段二：prefill 策略

加入连续 prefill policy 和 `max_prefill_tokens_per_batch`，分别测量：

- 首个 prefill 的 TTFT；
- 连续 prefill 吞吐；
- decode TPOT；
- 混合长短 prompt 的公平性；
- KV page 峰值占用。

### 阶段三：mixed chunk prefill

当前基础版本将 mixed batch 表达为一个 prefill-shaped forward：新请求使用正常 query length，存活 decode 请求使用 query length=1；两者通过同一组 `cu_seqlens_q`、`prefix_lens`、page table 和 `logits_indices` 进入 attention。decode token 仍从 FutureMap device buffer relay，CPU 只负责 row/page metadata。

该版本只覆盖普通 decode→prefill 混合，不引入 speculative、DP、LoRA 或 chunked request 状态机；切回 decode 前仍建立 prefill completion barrier。后续需要继续验证 CUDA graph 命中、不同 prefix 长度和长短 prompt 混合吞吐。

### 阶段四：speculative/grammar

如果未来需要 speculative decoding 或 grammar，再按 FutureMap 的 payload 类型逐项增加 relay 和 barrier；每种 payload 都必须明确它保护的是 token、sequence length、grammar state 还是 KV 生命周期，不能复用同一个 fence 类型。

## 总结

当前 Rust 实现不缺少原版 overlap 的基本执行模型，真正的差距在调度计划、relay 索引稳定性和 prefill admission 策略。优先迁移稳定 row 映射、显式 batch plan、连续 prefill policy 和 token budget，可以获得原版调度逻辑的主要收益，同时保持现有 KV retirement 与 input-read fence 的清晰边界。
