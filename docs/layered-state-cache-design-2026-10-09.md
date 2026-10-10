# 分层状态缓存：通用分页 KV 设计与实现

日期：2026-10-09。

## 范围

本次实现覆盖采用 full-history attention 的普通 Transformer 分页 K/V，包括 MHA／GQA／MQA，以及同一模型不同层的 KV heads、head dimension 不同的情况。Dense／MoE FFN 不改变这部分缓存语义；本次没有新增 MoE 或其他生产模型的计算实现。

不把 recurrent／线性 attention／SSM、MLA 压缩状态、滑窗独立淘汰、量化 KV、多 dtype 缓存组表示为普通分页 K/V，也不为它们加入占位分支。当前 Qwen3 的配置检查继续拒绝未实现的计算类型。未来新增模型必须同时实现其计算、状态生命周期及 backend 校验；有 `ModelCacheSpec` 并不等于可以加载任意 Hugging Face 模型。

## 接口与所有权

`ModelFactory::cache_spec` 是必需的模型声明接口，返回按模型层顺序排列的 `ModelCacheSpec`。每层用 `PagedKvSpec { num_kv_heads, head_dim }` 声明 full-history K/V 几何。描述在启动时解析，forward 不重新读取配置。

`KVCachePool` 按相同几何合并层，即使它们不相邻也可以属于同组。每组拥有一个连续 buffer，布局为 `[2, group_layers, real_pages + 1, page_size, kv_heads, head_dim]`；内部映射把模型层编号转换为组编号与组内层编号。不同组允许不同 heads／head dimension。所有组共用当前模型的 dtype、device、page size 与逻辑 page IDs。

请求和 radix tree 继续持有一张页表。一页的分配、pin、发布和释放涵盖全部缓存组，不需要调度器维护多张页表，也不会出现某组释放成功、另一组失败的问题。该约定的前提是所有组保存相同 token 历史，不能用于独立淘汰的滑窗或可变递归状态。

模型通过 `ModelExecutor::bind_state_cache(ModelKvCache)` 接收层局部的 `LayerKvCache { k, v }`，每个 tensor 为 `[pages + 1, page_size, kv_heads, head_dim]`。这些是浅视图，不复制 buffer，也不把 allocator 暴露给模型。Qwen3 校验全部视图的层数、形状、dtype 和 device 后再绑定。模型根据自己的计算配置选择每层 backend；pool 不引入特定模型名称分支。

`KVCachePageLayout` 只保留所有组共有的层数、页数与 page size，不提供误导性的全局 heads／head dimension。`get_all_kv_cache` 仅保留为单组池的检查／测试入口，异构池调用明确报错；生产模型执行不使用它。`KVCacheLayout`、`Engine::with_runtime` 与 RuntimeModelConfig 的统一 KV 几何保留为已有构造调用的便利入口，生产加载以 factory 的逐层声明为准。`Engine::with_cache_spec` 可显式构造异构池，`build_model` 检查模型声明与已有 pool 一致。

## 资源与生命周期

- 显存预算：对每层累加 `2 × page_size × kv_heads_per_rank × head_dim × dtype_bytes`，扣除所有层合计的一页保留空间，再按调度需求限制真实页数。不足以容纳保留页和至少一页真实 KV 时明确失败，不强制超预算分配。CPU 路径沿用既有 512 MiB 上限；CUDA 在权重加载后查询可用显存。
- 请求创建：一次分配逻辑页，所有组同步获得该请求的空间。现有调度器预分配容量的策略保持不变；本次没有增加动态扩容和 chunked prefill 调度。
- 前缀发布：现有 cache manager 在 forward 成功后发布已写入的完整页。只有全部层完成计算的前缀才能被其他请求命中，执行失败时不发布未完成状态。
- 分支：已发布前缀只读共享，后缀页由各请求独占；不需要复制 full-history KV。修改共享前缀的调用不属于本接口允许的用法。
- 结束／取消／执行错误：现有 request handle 释放私有页并解除 radix pin，一次操作覆盖所有组；重复 release 不重复回收。
- Graph padding：每组物理 page 0 初始化为零，永不进入可分配页列表。实际 attention 写入跳过 slot 0，padding 不污染共享前缀。
- Graph 地址与销毁：pool 分配后不搬迁；runner 在权重重绑定、状态重绑定前清理 decode 与模型 prefill graphs。Engine 清理先释放 graph，再释放模型的 tensor views，最后释放 pool。分配中途失败由 Rust／Tensor 所有权回收已经分配的组。

异构池不自动授予 graph 能力。执行器仍需显式声明并通过自己的 backend 验证；本次生产 graph 回归验证 Qwen3。FlashInfer 现有 head_dim=128 的内核限制保持有效。

## 验证

测试经过真实接口，不只检查分组数量：

1. 以三个层 A／B／A、两种不同几何验证逐层视图和稳定地址，写入前缀后创建分支，检查各组前缀数值不被后缀和 padding 改写；取消、重复释放、pin 与淘汰后恢复全部页。
2. 显存预算验证所有层与保留页都计入，检查最小预算、维度与形状溢出、层数不一致和错误 TP 几何。
3. 一个测试用异构 factory 通过真实 safetensors 加载入口、Engine 分配与模型绑定执行 prefill、decode、cached extend，并与无缓存完整 causal attention 的数值对照。测试 checkpoint 只是标记 tensor，不代表新增生产模型；测试分别覆盖 CPU 与可用 CUDA。
4. 原有 Qwen3、调度、CUDA kernel、decode graph、segmented prefill graph 回归；远端真实 Qwen3-0.6B 的 HTTP 回归与 eager／graph／既有基线输出对照。

最终验收次数和远端日志路径记录在路线图的本次实施记录中。本次没有性能测量，不把接口重构作为推理提速证据。
