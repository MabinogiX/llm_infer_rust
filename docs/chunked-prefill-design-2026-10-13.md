# Chunked prefill 设计方案（2026-10-13）

本文研究原版 SGLang 的 chunked prefill，并结合当前 Rust scheduler 的 `RequestManager`、`PrefillManager`、`SchedulePlan`、`FutureMap` 和 KV retirement 设计，给出一套分阶段实现方案。目标是限制单次 prefill 的激活和调度开销，让长 prompt 可以被拆成多个 forward，同时保持 KV、采样结果和 CPU/GPU overlap 的生命周期正确。

## 结论

当前已实现“单 active continuation、可恢复的 chunk prefill”，并支持 continuation 与 outstanding decode 的基础 mixed forward；更复杂的 waiting 多请求公平性和 backend-specific gating 仍按后续阶段处理。中间 chunk 不能产生用户可见 token，也不能触发普通 prefill 的采样/finish 逻辑。只有最后一个 chunk 才进入采样和正常 decode 流水线。

推荐的核心状态是：

- `prompt_len`：原始 prompt 总长度，不因 chunk 改变；
- `prefix_len`：已经被 KV cache 匹配并可读的前缀长度；
- `chunk_end`：本次 forward 要写入的绝对结束位置；
- `is_chunked`：本次 batch 是否包含未完成 chunk；
- `is_last_chunk`：本次 chunk 是否覆盖到 prompt 末尾。

`chunk_end` 是调度状态，不应通过修改 `input_ids` 来隐式表达。一个 chunk 完成后，保留同一个 `Request` 和 KV handle，令 `prefix_len = chunk_end`，再调度下一 chunk；不要把请求重新当作新请求做 radix 匹配或重新分配 KV。

## 原版 SGLang 的关键机制

### 1. chunk 大小是计算预算，不是请求数限制

原版通过 `--chunked-prefill-size` 指定单个 chunk 的最大 token 数；显式 `-1`（Rust 中也接受显式 `0`）时关闭；未指定时按 GPU 显存自动推导。chunked prefill 启用时要求 chunk size 能被 page size 整除，避免页级 KV scratch 和 chunk 边界不一致（原版 `sglang/python/sglang/srt/arg_groups/validation_hook.py:159-165`）。当前 Rust 同时提供 `--max-prefill-tokens-per-batch`（批次总未缓存 token 预算）和 `--chunked-prefill-size`（单请求 chunk 上限），两者取最小值。

原版 admission 先计算 `extend_len = prompt_len - prefix_len`，再结合剩余 token budget、KV 内存和 tile budget 决定本轮长度（`sglang/python/sglang/srt/managers/schedule_policy.py:1430-1509`）。首个 chunk 将 `extend_end = prefix_len + chunk_len`，并把 `max_new_tokens` 记为 0；只有最后一个 chunk 才预留新 token 并允许结束该请求（同文件 `1217-1265`、`1493-1504`）。

### 2. continuation 优先于 waiting queue

原版 scheduler 在构造下一批时先处理已有的 `chunked_req`，调用 `init_next_round_input()` 后再按剩余预算生成下一 chunk（`sglang/python/sglang/srt/managers/scheduler.py:3937-3945`）。这保证长 prompt 不会因为新请求持续到达而永久停顿。

continuation 只在上一 chunk 已经安全完成后重新进入 batch。原版用单独的 `chunked_req` 指针表示“停在两个 batch 之间的请求”，并在 idle 判断、容量判断、取消和回收路径中单独处理（`scheduler.py:1323-1327`、`3500-3594`、`is_fully_idle` 相关逻辑）。

### 3. 未完成 chunk 不产生正常输出

原版 admission 对中间 chunk 设置 `max_new_tokens = 0`，并在 batch 上记录 `chunked_req`、`contains_last_prefill_chunk` 等状态（`schedule_policy.py:1500-1509`；`schedule_batch.py:2337-2348`）。这使结果处理可以区分：

- 中间 chunk：只提交 KV 进度、更新 chunk 状态，不发送 token；
- 最后 chunk：执行一次正常 prefill sampling，发送首 token，进入 decode。

因此不能复用当前 Rust `complete(..., phase = Prefill)` 的默认语义，因为当前逻辑会在每个 prefill 结果后 publish 并追加采样 token。

### 4. mixed chunk 是可选的 batch 组合

原版先生成新的 extend batch，再在满足条件时把 running decode 拼到同一个 batch：不能有 return-logprob/input-embeds/beam 等不兼容形态，running batch 会转换为 decode 形态，chunk prefill 行仍保持 extend 形态（`sglang/python/sglang/srt/managers/scheduler.py:4137-4169`）。`ScheduleBatch` 明确保存 `decoding_reqs`、`chunked_req` 和 chunk 是否结束等元数据（`schedule_batch.py:2337-2348`）。

这说明 mixed chunk 不是“把普通 prefill 的 input_ids 与 decode input_ids 拼起来”这么简单：结果处理必须知道哪些行需要采样、哪些行只写 KV；metadata、logits 行和 KV 写入位置必须按两种行分别解释。

### 5. 取消必须延迟到 chunk 边界

原版收到取消时只记录 `_pending_chunked_abort_req`，不在当前 forward 中途拆除 chunk 请求；下一轮调度前清理，已经发射的 chunk 由结果处理路径排空后再释放（`sglang/python/sglang/srt/managers/scheduler.py:3549-3594`）。这与当前项目的 completion fence/retirement 契约一致：逻辑取消可以立即生效，但 KV handle 只能在最后一个使用它的 GPU 事件完成后释放。

## 当前 Rust 项目的差距

当前 `Request` 只有 `input_ids`、`cached_len`、`output_len` 和 `cache_handle`（[src/scheduler/request.rs](/Users/dp/code/sglang-rust/src/scheduler/request.rs)）。`PrefillManager` 会把一个请求从 `cached_len` 一直构造成完整 uncached row（[src/scheduler/prefill.rs](/Users/dp/code/sglang-rust/src/scheduler/prefill.rs)），`BatchContext::prepare_prefill` 也默认每个 row 直到 prompt 末尾（[src/engine/batch_context.rs](/Users/dp/code/sglang-rust/src/engine/batch_context.rs)）。

当前 `SchedulePlan` 已经能表达 prefill、decode 和 mixed，但 mixed 的 relay 只针对“新请求 prefill + 已有 decode”，没有“同一个请求下一 chunk”的 continuation 元数据（[src/scheduler/schedule_plan.rs](/Users/dp/code/sglang-rust/src/scheduler/schedule_plan.rs)）。此外，`Scheduler::complete` 将 prefill 结果统一按普通 prefill 处理，因此不能直接把中间 chunk 接入现有路径（[src/scheduler/scheduler.rs](/Users/dp/code/sglang-rust/src/scheduler/scheduler.rs)）。

## 推荐实现

### 阶段一：非 mixed、单 continuation 请求

新增配置：

```text
--chunked-prefill-size N
```

约定 `N == 0` 表示关闭；启用时要求 `N > 0` 且 `N % page_size == 0`。第一阶段只允许一个未完成 chunk request，同时不与已有 decode 混 batch；这样可以先验证状态、KV 和结果边界。

建议新增轻量类型，而不是继续堆叠布尔字段：

```rust
struct PrefillProgress {
    prompt_len: usize,
    prefix_len: usize,
    chunk_end: usize,
    is_last_chunk: bool,
}
```

`Request` 持有 `Option<PrefillProgress>`。`PrefillSelection` 改为携带每个 row 的 `extend_start`、`extend_end` 和 `is_last_chunk`；`BatchContext` 新增按绝对区间构造 row 的接口，逻辑等价于当前 `cached_len..input_ids.len()`，但终点改为 `extend_end`。

调度顺序：

1. 若存在 continuation，先计算 `remaining = prompt_len - prefix_len`；
2. `chunk_len = min(remaining, chunked_prefill_size, max_prefill_tokens_per_batch)`；
3. 非最后 chunk 不为 `max_tokens` 预留输出 token；
4. 只为本 chunk 构造 positions、write locations、logits row 和 KV metadata；
5. forward 完成后，如果不是最后 chunk，只更新 `prefix_len = chunk_end`，不 append token、不 publish 用户输出；
6. 最后 chunk 才执行正常 prefill completion，publish 已写 KV、采样首 token，并清除 `PrefillProgress`。

如果预算小于一个 page，不能静默生成零长度 chunk。应返回“本轮不调度”并保留 continuation；若空 batch 也无法容纳一个完整 chunk，则返回明确的不可调度错误，避免 livelock。

### 阶段二：与 waiting prefill 共批

当阶段一的 continuation 回归稳定后，可允许 continuation 与新的 waiting prefill 共存，但仍先保持同一 batch 内全部为 prefill rows。预算按实际 uncached token 总数计算：continuation 的 chunk token、新请求 uncached token、必要的 page overhead 分别计入。

同一请求不能同时出现在 continuation 和 waiting selection 中；选择成功后必须一次性从 waiting 移除，并在 batch 构造失败时恢复原顺序。KV handle 继续由 `RequestManager` 持有，不能通过重新 `acquire_pending` 产生第二个 handle。

### 阶段三：接入现有 mixed decode

最后才接入当前 `prepare_mixed_prefill`。建议把 `SchedulePlan` 改为强类型 variant：

```rust
enum SchedulePlan {
    Prefill(PrefillPlan),
    Decode(DecodePlan),
    Mixed(MixedPlan),
}
```

`MixedPlan` 至少包含：

- prefill rows 的 `request_ids` 和每行 `extend_end`；
- decode relay rows；
- `prefill_token_count`；
- 每行是否为 chunk continuation、是否为最后 chunk。

mixed forward 的结果处理必须按 row mask 分流：

- decode rows：沿用 FutureMap relay 和 completion fence；
- 新 prefill 最后 chunk：publish、采样和输出；
- 中间 chunk：只提交 KV 进度，不发送 token，不进入 FutureMap decode relay。

若任意 row 需要 return-logprob、input embeds、beam 或不支持 chunk metadata，应回退为纯 prefill/decode batch。原版的 mixed chunk 兼容性检查（`scheduler.py:4137-4148`）应作为 Rust 的显式 admission 条件，而不是让 backend 在运行时猜测。

## KV cache 与 publish 契约

chunk continuation 的 KV 规则必须先定清楚：

1. 同一请求沿用同一 `cache_handle`，不重复分配；
2. 每个 chunk 只写自己的 `[prefix_len, chunk_end)` 区间；
3. 中间 chunk 完成后推进已计算边界；下一 chunk 必须读取更新后的 page table，不能继续使用旧的 metadata 快照；
4. 可以发布已经写满的完整页，但 partial page 和仍可能被 GPU 使用的页必须由同一 `cache_handle` pin；
5. 最后 chunk 完成并且没有后续 GPU 使用者后，再执行最终 canonical publish 和 handle release；
6. 取消、forward 错误和 event 错误走现有 retirement 队列，不能因为“没有用户输出”就立即释放 KV。

当前项目已经把逻辑结束与物理回收分离（见 [docs/cpu-gpu-overlap-retirement-2026-10-12.md](/Users/dp/code/sglang-rust/docs/cpu-gpu-overlap-retirement-2026-10-12.md)），chunked prefill 应复用这个契约，而不是创建第二套回收路径。

## CUDA graph 与 overlap

chunk size 应成为 prefill graph admission 的稳定 shape 上界，而不是为每个剩余长度创建一张 graph。推荐：

- graph bucket 以 `chunked_prefill_size` 和 request count 为上界；
- 最后一个不足整 chunk 的 batch 使用已有 prefill eager/静态 metadata 路径，或使用 padding 后的 graph；
- decode graph 继续由 `FutureMap` relay 驱动，chunk continuation 不写 FutureMap；
- 阶段一先禁止“旧 decode + chunk continuation”混 batch，避免同时验证两套 fence；
- 阶段三 mixed chunk 才允许复用当前“先发射 next，再完成 previous”的 overlap，但必须为 chunk request 保留独立的 completion boundary。

这也解释了为什么不建议把 chunked prefill 实现为简单的 `input_ids.narrow`：graph、logits index、KV write locations 和 completion semantics 都必须同时切到 chunk 视图。

## 调度公平性与配置交互

- continuation 优先，防止长 prompt 饥饿；但应设置最大连续 chunk 次数或 token 时间阈值，周期性允许 waiting request 插入；
- `max_prefill_tokens_per_batch` 是总预算，`chunked_prefill_size` 是单请求 chunk 上限，两者取最小值；
- `max_running_req` 限制 request rows，不限制单个 prompt 的 chunk 次数；
- `max_seq_len` 校验使用原始 prompt 总长和最终生成长度，不能只校验当前 chunk；
- `disable_consecutive_prefill_overlap` 只控制 prefill→decode 切换，不应关闭同一请求 continuation 的必要屏障；
- mixed chunk 与 speculative、return-logprob、beam、input embeds 的兼容性应先拒绝并回退；
- 多模态 prompt 若 token 与 embedding 不能按 chunk 独立切分，应默认关闭 chunked prefill，沿用原版 Transformers multimodal 的禁用策略（`scheduler.py:1311-1320`）。

## 验收计划

阶段一至少加入以下回归：

1. 一个长 prompt 被切成 3 个 chunk，输出 token 与不切 chunk 的 eager 路径完全一致；
2. 每个 chunk 的 `positions`、`write_loc`、`prefix_lens`、`logits_indices` 和 page table 正确；
3. 中间 chunk 不产生 `OutputToken`，最后 chunk 只产生一个首 token；
4. chunk 中途取消，GPU fence 完成前 KV 页仍被 pin，完成后恰好释放；
5. `decode → chunk1 → chunk2 → final chunk → decode` 的 KV、logits、relay 和保留输出与 serial 一致；
6. chunk size 小于 page size、预算不足、prompt 恰好整除、最后 chunk 短于 chunk size 等边界。

阶段三再加入 CUDA 专项：

- full chunk graph、short final chunk eager/padded graph；
- mixed chunk + decode 的 stream/event 顺序；
- EOS/length finish 只作用于 decode rows 或 final chunk rows；
- graph replay 后下一 chunk 的 KV page table 不使用旧 tensor。

## 不建议首期实现的内容

首期不引入动态 chunk sizer、PP microbatch、host cache load-back、disaggregated prefill transfer、Mamba/GDN 特殊状态和 speculative mixed chunk。这些功能在原版 scheduler 中都有独立的预算、状态或 pipeline 约束，直接合入会让当前 Rust 的状态机难以验证。等基础 continuation 的数值和 KV 回归稳定后，再按依赖逐项接入。
