# 统一模型执行接口与 CUDA graph 管理

日期：2026-10-10。本文记录路线图 3.4 的本次实现。

## 执行接口

Scheduler 继续构造持有 tensor 的 `Batch`；Runner 在执行入口转换为借用这些 tensor 的 `ForwardBatch`，不复制输入数据。模型只实现 `ModelExecutor::forward(&ForwardBatch) -> Result<ForwardOutput>`，输出目前只包含 logits。

`ForwardMode::Prefill` 必须携带 logits 行选择，`ForwardMode::Decode` 不携带该字段。模式从一个枚举派生，不再由模型调用者分别传入多个可任意组合的参数。现有 backend 的 `AttentionMetadata.forward_mode` 暂时保留，入口检查它与执行模式一致；该字段不成为第二个可独立选择的执行模式。

Runner 在 eager 与 graph 路径前统一验证 token／position 数量、非空输入、tensor rank、dtype 与设备，以及缓存元数据的形状约定。缓存 prefill 要求 query boundaries、prefix lengths 和每请求 logits indices 对齐；decode 要求每请求 cache length，不允许携带 prefill boundaries。所有 cached forward 都要求写入位置和页表或 token 表。验证发生在静态缓冲区复制及 backend planning 之前。

入口检查 tensor 描述与字段组合，不在每步新增 GPU→CPU 同步来扫描 token IDs、positions、页号或累积长度的数值。数值仍由 Scheduler 的请求／页表构造约定和既有 backend／kernel 检查负责。Engine 在 Scheduler 对接处取 `ForwardOutput.logits`，现有采样与 HTTP 输出路径保持不变。

`DecodeGraphRunner` 与 `SegmentedGraphRunner` 均接收已经校验的 `ForwardBatch`，返回 `Result<Option<ForwardOutput>>`；`None` 表示回退 eager。只有 `ModelRunner` 访问 Scheduler 的 `Batch`，并在选择执行路径前完成一次统一校验。独立的 `run_model` eager／capture 入口仍自行校验。两种 graph 执行器保留各自的回放编排，共用 `NativeCudaGraph`。

## 模式与能力声明

`ModelExecutor::graph_capabilities` 默认返回两种模式均不支持。`GraphCapabilities` 分别描述 full decode graph 和 segmented prefill graph 的 `GraphSupport`。支持项携带最大请求数、token 数、上下文约束与 padding 约定；拒绝项携带原因。

模型根据已经解析的计算配置、实际 device／dtype、已绑定状态和 backend 能力声明支持。Qwen3 只在 CUDA、BF16／FP16 与完整 KV 绑定后声明 graph 能力。Full decode 还要求保留 KV slot 0 和经过验证的 attention capture 路径；现有 FlashInfer 支持此路径，`pt`／`fa` 不声明 full decode graph 支持。它们的 attention 可以作为分段 prefill 的图外操作执行。

参数禁用、CPU、同步 profiling、能力不满足、捕获失败以及超出桶目录的情况均保持 eager 路径。启动记录能力拒绝和捕获失败的原因，动态形状回退记录 debug 原因。支持异构缓存不自动授予 graph 能力；新模型仍需验证具体组合。FlashInfer 现有 `head_dim=128` 内核限制保持不变。

## 通用分段 graph 接口

Runner 拥有 `SegmentedGraphRunner`，负责启动时桶目录、固定输入缓冲、native graph 捕获、回放、失败桶和释放。模型提供 `PrefillGraphProgram`：

- `create_inputs`：提供静态 token 输入和各段输入 tensor 的形状。
- `run_segment`：定义某段纯 GPU、静态形状的计算；同时接收上一段的固定输出，允许直接消费其地址。Runner 持有这些 tensor 和 native graph，直到整个桶失效。
- `prepare_replay`：按真实请求准备 backend 计划，返回一次 invocation 的 `PrefillGraphReplay`。

Replay 的 `run_eager` 在相邻段之间执行 attention 等图外操作，直接写入下一段的固定输入；`finish` 按真实请求选择输出行并产生独立拥有存储的 logits。所有段的固定输入都具有 bucket-sized 的首 token 轴，其余形状和 tensor 含义由模型定义。通用管理器在捕获前检查固定输入的 token 轴和设备；模型／backend 的写入接口验证输出 shape、dtype、device 和布局。模型负责清零 padding 行，不向 padding token 发布 KV。

Qwen3 只保留模型计算结构：首段 embedding／norm／QKV；attention 为 eager seam；后续段执行 output projection／MLP，组合下一层 norm／QKV，最后一段完成最终 norm。模型可自行组合相邻计算以减少 launch；通用管理器不访问 decoder layers，也不假定段数等于层数。一个三段算术测试程序使用不同数量和形状的输入，验证该接口不依赖 Transformer 布局；它不是新注册的生产模型。

## Prefill 元数据与固定输出优化

`Attention::prepare` 在每个实际批次读取一次 query boundaries 与 prefix lengths，所有层复用这一快照。`pt` 的因果 mask、缓存 gather 索引及位置范围检查信息在批次准备时构造；`fa` 的 padding／输出索引、缓存 gather 索引和 attention mask 同样只构造一次。每层仍根据自己的 K/V cache gather 实际内容，并验证缓存索引上界，不缓存层相关 K/V 值。FlashInfer 的 native planning 复用 Rust 已读的 host boundaries／prefixes，避免额外读取同一份 GPU 元数据；paged plan 的 indptr 构造仍在图外完成。

Qwen3 捕获下一段时直接使用上一段的 residual tensor 地址，不再分配或复制独立 residual bridge。每个 attention 边界仅保留一个 bucket-sized 输出缓冲区；真实 token 的 attention 写入该缓冲，余下行清零。各段输出由整个桶共同持有，失效释放前同步设备，防止被下一段引用的 allocation 提前销毁。

`AttentionBatch::forward_into` 提供固定地址输出契约。FlashInfer 直接将调用方输出指针传给已有 native attention kernel，避免中间输出分配和 D2D 写回；没有修改 kernel 的算法或布局。多请求 SDPA 使用 `index_select_out` 把解包后的结果直接写入静态输出；单请求 SDPA 和 `pt` 保留必要的结果写回，因为现有 LibTorch 接口不提供 attention kernel 的 `out` 参数。不将这些兼容路径称为完全零拷贝。

## 捕获、回放与生命周期

1. 权重加载和缓存绑定后，Runner 在启动时捕获；保留原有 prefill 桶策略与从大到小捕获顺序，不在请求路径懒捕获。
2. Decode graph 的静态缓冲区及计划仍由`DecodeGraphRunner` 持有。`DecodeGraphState::prepare_replay` 明确只能在图外调用，在每次 native replay 前刷新 backend 计划。
3. Decode padding 使用保留的零 KV page／slot 0；segmented prefill 只将真实 token 交给 eager seam，固定缓冲区中的额外 token 行不发布 KV。
4. 有效批次的 cache-table 宽度与静态 decode graph 不同时，复制前回退 eager；非法 metadata 直接返回错误。缺失／失败／超出限制的 prefill 桶也回退 eager，保留已捕获的其他桶。
5. 权重、缓存绑定和 reserved-slot 设置变化之前，Runner 同时失效两种 graph。重复 clear 安全。模型不再自己持有 prefill native graphs。
6. Capture 安装采用局部所有权：先构建 decode／prefill 管理器，整体初始化失败时局部资源自动释放，不留下半安装状态。单桶捕获失败是可记录的 eager 回退，不阻止其他桶使用。
7. 两种管理器释放前同步设备，确保排队计算不再访问 native graph allocation pool 和计划缓冲；graph 在模型、权重和缓存视图之前释放。

## 共享 graph pool：阶段一

共享范围为同一模型实例、设备及本次捕获目录。`NativeGraphCaptureContext` 持有 pool handle、统一 capture stream 和既定 execution stream；decode 与 segmented prefill 的全部 native graph 共享该 context，保留现有 owning 输入／输出及 residual 原地复用关系。两种模式均按大桶到小桶捕获。没有加入跨模型全局池、非 owning tensor 或共享 attention 大缓冲。

graph capture stream 用于 allocator 有效复用，execution stream 用于回放与图外操作的串行顺序，两者职责不同。执行入口在静态输入复制和 backend planning 之前验证当前 execution stream；native replay 同样拒绝其他 stream。若没有实际捕获到 graph，不安装该执行限制。完整 forward 可以任意选择 bucket，连续 prefill 或任意切换 prefill/decode；单个 segmented forward 中各段与 eager seam 保持固定顺序。KV、请求状态与独立 logits 不依赖 graph 临时池继续保存。

native bridge 保存 caller stream 并在成功结束、闭包错误、未开始捕获的 warmup 错误及释放时恢复。Rust closure 返回错误但 native capture 正常结束时可继续捕获其他桶；native begin/end 失败则标记整个 context 无效，立即终止目录初始化，局部 graph 释放，不安装损坏目录。clear／权重或缓存重绑定先释放两种 runner，再释放 context；新目录取得新 pool handle。native graph 的 Rust/C++ 所有权引用均保证 context 不早于 graph 销毁。

pool handle 只是唯一 ID，不是 allocator 保活引用。Context 创建前先 lazy-init LibTorch CUDA allocator；对 device pool 调用 `createOrIncrefPool`，销毁时对称 `releasePool`。部署版 LibTorch 同时给 graph 保留相同 ID 的 pinned-host pool，因此 context 通过 host allocator 的 `begin_allocate_to_pool`／`end_allocate_to_pool` 获取引用，filter 始终返回 false，不接管任何 host allocation；销毁时对称 `release_pool`。各 graph 仍由 LibTorch 增减自己的两份引用。这样首次 capture 在生成成功 graph 之前被丢弃，也不会把任一 pool 的 use_count 降到零。新增 CUDA 回归先复现 device/host allocator 的断言，再验证保活后的闭包失败／后续 capture 路径。

启动日志增加捕获前后 device-wide allocator `allocated`／`reserved`、增量与捕获耗时，便于真实模型验收；这些不是仅 graph pool 的统计，也不是同步重置后的峰值。完整约束、原版来源、隔离实验及后续阶段见 [共享 pool 评估与分阶段方案](shared-cuda-graph-pool-assessment-2026-10-10.md)。

## 验证与范围

CPU 测试覆盖模式、dtype、形状和元数据拒绝，桶目录、批次元数据快照复用、固定输出地址及写入范围；既有 Scheduler 和异构缓存集成测试迁移到结构化接口。

CUDA 测试覆盖通用三段程序的成功、失败、缺失、超限桶，跨段固定输出消费、padding 与保留输出，权重／缓存重绑定失效；原有 decode 动态页号／长度／padding 与数值测试增加静态表宽不匹配时的回退检查。Qwen3 分段 prefill 对照 eager logits 与 K/V，覆盖 BF16／FP16、`pt`／`fa`／FlashInfer、不同长度、多请求、缓存前缀、missing bucket 和 clear／recapture。FlashInfer 的固定输出测试同时验证返回 tensor 沿用调用方地址，且不覆盖有效 token 范围以外的存储。

真实模型验收继续使用远端 `/sjtu/yaosikai/llm_infer_rust`、GPU 1 与 Qwen3-0.6B，检查 HTTP、eager／graph 和已有输出基线。最终统计及日志路径记录在路线图的本次实施记录。

本次没有改变内核接口或数学计算，没有引入新生产模型、跨进程执行或 speculative decoding；第二个实际模型验证仍待后续接入。未进行性能测量，不把此次重构作为推理提速证据。
