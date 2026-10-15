# CPU/GPU overlap：逻辑结束与 KV 回收分离

KV retirement 契约继续有效。后续加入 [调度输入读取边界与 metadata 上传](cpu-gpu-overlap-input-read-boundary-2026-10-12.md)，其 early fence 不替代本文的 KV completion fence；本文性能数字属于新增上传优化前版本。

## 实现范围

本阶段替换之前 EOS 时整批排空、incoming prefill 时无条件排空 decode 的策略。沿用一批 outstanding GPU 结果、同一 execution stream、GPU token relay 和独立 logits；不引入并发 GPU forward。序列复制优化保持不变。

## EOS、取消和错误

处理上一批结果时，若某请求结束且当前 outstanding 批次仍包含其 UID：

1. 更新该轮 committed written_len，追加终止 token，立即生成唯一的终止 OutputToken。
2. 从运行队列移出请求。之后 decode 行选择不再包含它，也不会提交已多发射一轮的采样结果。
3. 将整个 Request 移入 retirement 队列，保留原序列、KV handle、私有页以及共享 radix 前缀的 pin。移入不复制 input_ids。
4. 持有最后一次使用该 UID 的批次 completion fence。每次 step 用 CUDA event query 非阻塞检查；事件完成后才 publish committed prefix 并释放 handle。

存活请求正常提交上一轮结果，保留当前 outstanding decode，无需因为其他行 EOS 等待下一轮。已知长度结束的行没有被继续发射时，可以立即发布/释放，因为该行不再有 GPU 使用者。

取消同样立即脱离运行队列并延迟释放，不等待整批 GPU；取消不会发布未提交的新前缀。采样结果错误采用相同资源持有规则。GPU 发射错误属于异常路径，仍先同步设备再回收，避免部分发射的 forward 留下访问。

延迟终止发布只使用 handle.written_len 范围的序列，排除 EOS 等尚未提交 KV 的 token，即使多发射的 decode 已经把它写入私有页。若延迟 publish 失败，release 仍执行，已安装的 canonical 页保留，剩余私有页回收；记录 tracing 错误和 last_step_error，不给已结束请求发送第二个终止结果。同步路径在终止前发布失败仍返回原来的 Error 结果。

## 为什么可以按 UID 保护

当前 radix 实现只发布完整页，运行请求的未来写入位于它独占的私有页。每个 cache handle 对自己引用的 tree 页持有 pin：逻辑移出 running 时不能丢掉 handle；保留它可阻止 cache.acquire 的 eviction 回收 GPU 仍引用的页。

其他请求通过自己的 handle pin 保护共同的只读前缀。旧请求发布/释放只改变旧 handle 的页表和私有页，不重写其他请求的页表。因此，GPU 当前批次不包含结束 UID 时，无需等无关请求。该论证依赖现有“共享完整页只读、私有页独占、UID 不复用”约束；未来共享可写页、迁移 KV 或改变 allocator 语义时需重新审查。

## Prefill 切换

有 outstanding decode 时，先尝试 schedule_prefill。已有 decode 和 retirement 队列仍持有各自 handle，所以前缀查找、分配和 eviction 不能回收其在用页；新 prefill 使用独立 BatchContext buffers，在同一 execution stream 上排在旧 decode 后执行。

若能形成新 prefill，先发射它，再等旧 decode 的 readback event 并返回旧结果。当前 outstanding 成为 prefill，避免为了“有新请求到来”先把 decode 流水线排空。若请求槽满、预算或内存不足而未形成 prefill，继续 decode relay，存活请求仍前进。

Prefill 自身完成后，需要先 publish 再为这些 UID 准备 decode：radix canonicalization 可能替换其页 ID、释放重复私有页，不能让同一请求的下一轮持旧页表发射。这是页表依赖屏障。连续 prefill 也先提交已完成的 prefill 结果；mixed prefill 复用同一 prefill-shaped Batch，在该屏障之外不增加额外的 KV 回收语义。

长 prefill 的 GPU 计算仍会推迟后续 decode，因为 GPU forward 串行。消除的是额外的 CPU 排空与结果提交等待，不承诺其他请求的 TPOT 完全不受 prefill 工作影响。现有 pageable metadata H2D 和 backend 规划也可能同步，完整 host shadow 尚未实施。

## 完成事件与退出

CompletionFence 共享 SampleReadback 的 CUDA event 及 native source/pinned destination 生命周期，clone 不创建新事件。ready 是非阻塞 query，wait 用于 Engine 可变访问、退出或没有 GPU 工作剩余的回收边界。Scheduler 的 is_idle 包含 retirement 队列，防止逻辑结束后还持有资源却提前停止驱动。

Scheduler 销毁显式等待 outstanding 的 fence，之后才处理剩余 retirement 和销毁 Engine；不能仅 drop 一个 Rc 并假定它会等待，因为 retirement 队列也可能持有该 Rc。正常 EOS、取消、incoming prefill 不执行设备级同步；异常发射/事件错误仍保留保守同步。

## 验收

CPU 139 项通过（1 项 ignored）、CUDA 23 项（22 lib + 1 integration）、Python 基准依赖回归 2 项通过，格式与 diff 检查通过。日志为 `logs/overlap-retirement-{cpu,cuda,build,benchmark}-20261012.log`。

新增测试覆盖：事件未完成时终止结果已生成、KV 页保持持有、逻辑结束打开请求槽、共享前缀仍可被其他请求引用、事件完成后恰好释放；延迟 publish 部分成功后失败的页回收和单次终止；EOS 与存活请求继续流水线；decode→异步 prefill 的先发射后提交；取消 in-flight prefill 的结果丢弃。既有同一 runner logits/KV/保留输出回归继续执行。

## 真实模型 A/B

同一新二进制，以 `--disable-cpu-gpu-overlap` 开关对照，同一 Qwen3-0.6B / BF16 / FlashInfer 0.6.18 / A100 GPU 1 / KV 容量与 graph 桶。三轮轮换顺序，总计 816 个计时请求，408 对生成内容 SHA-256 全部一致，usage 长度检查通过。下表采用三轮统计中位数。

| case | 同步 TTFT ms | 流水线 TTFT ms | 同步 output token/s | 流水线 output token/s | 吞吐变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold64 | 3.28 | 3.25 | 291.2 | 293.2 | +0.71% |
| cold512 | 7.61 | 7.49 | 128.2 | 130.6 | +1.86% |
| cold2048 | 20.68 | 20.81 | 47.9 | 47.8 | -0.20% |
| fallback4096 | 43.47 | 43.59 | 22.9 | 22.8 | -0.22% |
| generate64c1 | 6.05 | 4.22 | 420.7 | 425.2 | +1.07% |
| generate512c4 | 21.97 | 19.31 | 1211.0 | 1217.8 | +0.56% |
| generate1024c4 | 35.85 | 33.11 | 966.2 | 972.9 | +0.69% |
| prefix1024 | 7.85 | 5.83 | 390.9 | 394.4 | +0.90% |

三个生成场景吞吐变化约 +0.56%～+1.07%，短请求有正有负；三轮数据不足以证明显著或稳定提速。TTFT 改善包括更早发送 prefill/旧 decode 结果的收益，不能解释成 GPU forward 本身更快。两模式 ready/steady GPU 驻留近似均为 20821/20849 MiB。

基准的生成请求忽略 EOS，不能仅靠这 408 对输出验证 EOS 边界；EOS、in-flight 取消、事件前持页和 prefill 切换由专门的 CPU/CUDA 回归验证。性能数据对照的是当前二进制的同步路径，没有把前一版跨批流水线作为独立性能对照。

原始结果目录：`benchmark/results/overlap-retirement-2026-10-12/`。二进制 SHA-256：`e50e7a288a37be3424252226e0fdae251d2c022f2c3d1e69f15b7b9ac6f15cd1`。实现、文档、日志和结果已同步本地/远端；不新增 HTTP 专项测试计数。

```sh
cd /llm_infer_rust
.venv/bin/python scripts/benchmark-shared-graph-pool.py \
  --rounds 3 --modes rust-serial rust \
  --output-dir benchmark/results/overlap-retirement-2026-10-12 --resume
```
