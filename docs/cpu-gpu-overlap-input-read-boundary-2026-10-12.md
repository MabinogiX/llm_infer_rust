# CPU/GPU overlap：调度输入的读取边界与异步 metadata 上传

## 安全前提与范围

可以在当前实现中使用更早的读取事件，但必须限定它保护的资源。它不能作为 KV 页释放或共享 graph pool activation 复用的证明，也不能只在原单 stream 上提前打点就认为获得 overlap。

本阶段仅对成功命中的 captured decode graph 给出精确边界：DecodeGraphRunner 已将 input_ids、positions、write_loc、cache_seqlens、block_table，以及 backend 需要的 req_to_token 复制到 graph 独立持有的 buffers。复制完成后，后续 padding、attention 规划和 graph replay 读取的是 private buffers。InputReadFence 在这些 source copies 之后记录，保护的是 DecodeManager staging buffers 的复用。

Eager、图目录缺桶、形状不匹配等 fallback 不给出这个 early fence，使用完整 forward/sampling/D2H completion fence。Prefill 没有声明早期读取边界；同一请求 prefill 发布到下一次 decode 的页表依赖仍保持。KV retirement 始终使用独立的 CompletionFence，两个 Rust 类型不可互换。

## 执行顺序

新增 metadata upload stream，只负责 DecodeManager buffers 的更新，GPU forward 和共享 graph pool replay 仍只使用既定 execution stream。

```text
execution: snapshot B -> input_read_B -> graph B -> sample B -> D2H B
                                                           -> relay clamp
                                                           -> wait upload_ready_C
                                                           -> gather C IDs -> snapshot C -> graph C
upload:                  wait input_read_B -> metadata C -> upload_ready_C
KV retire:                                         query full_completion_B -> publish/free
```

CPU 可在 B 的 GPU 计算期间准备下一轮 metadata 并提交上传。Upload stream 等待 input_read_B 后才覆盖旧 staging arrays；execution stream 等 upload_ready_C 后才消费这些新数组。这里两个 wait 都是 GPU stream 依赖，正常路径没有 CPU event synchronize。

未来 sampled IDs 由 execution stream 产生。其 clamp、index_select 和 input_ids 写回继续留在 execution stream，不能在较早的 upload stream 上读取尚未生成的 token。Relay row indices 是独立 staging buffer，同样受 input-read 边界和 upload-ready 依赖保护。

## 生命周期和接口

MetadataUpload 把 CPU 向量复制到 pinned CPU tensors，发射 non-blocking H2D，并按每次上传的完成事件保留这些 source tensors。原来的临时 CPU Tensor 可以立即销毁，但 pinned source 在 H2D 完成前不复用或释放。开始后续上传时非阻塞回收已完成 jobs；对象销毁等待 upload stream，之后才释放 source buffers。

UploadGuard 在正常结束或提前退出时恢复原 execution stream，并建立 upload-ready 依赖。模型 forward、采样和 graph replay 不在 upload scope 内调用；既有 NativeGraphCaptureContext 的 execution-stream 检查保持有效。

第一次 pipeline 上传用 execution stream 上的初始化事件保护既有 buffer 初始化/此前同 stream 的工作。公开的 DecodeManager.schedule_decode 保持原来同 stream 的消费约定；仅 scheduler 的 pipeline 入口启用异步上传。

`--disable-async-metadata` 仅关闭早期读取事件与异步上传，保留跨批 decode、GPU relay 和按事件延迟 KV 回收，可用于独立 A/B；`--disable-cpu-gpu-overlap` 仍关闭整个流水线。CPU 和 diagnostic profiling 不启用 upload stream。启动日志记录两个开关。

## 尚未改变的依赖

本文阶段的 FlashInfer graph.prepare_replay 仍从 private GPU metadata 取得 CPU 规划信息。本阶段测量只包含 CPU metadata 准备和 H2D 上传的重叠。2026-10-15 后续实现为调度 decode batch 增加 CPU 长度快照与固定分页索引缓冲，消除 captured decode 正常路径的规划 D2H；Pinned workspace 复用仅等待此前规划上传完成事件。缺少 host shadow 的调用保留原规划 fallback。

长 prefill 的 GPU 工作依然会占用同一 execution stream；早期事件不允许并发执行两轮共用 activation pool 的模型 forward。对新增字段或 backend，不应把本事件扩大解释为“所有共享资源读取完成”。

## 验收

CPU 130 项（129 lib + 1 integration）、CUDA 23 项（22 lib + 1 integration）、Python 基准依赖回归 2 项通过；格式和 diff 检查通过。日志为 `logs/overlap-read-boundary-{cpu,cuda,build,benchmark}-20261012.log`。

新增 upload 回归连续 64 次上传，立即销毁原 CPU source，交替显式 finish 与 guard drop，验证 stream 恢复、跨 stream buffer 重用顺序和所有保留快照。加强已有 FlashInfer graph 回归：在 early event 后覆盖原 input_ids/positions/metadata，核对 logits、整个 K/V、动态页/长度、padding 与保留输出；缺桶路径仍 fallback。既有 EOS、取消、异步 prefill、延迟回收与同 runner 模式切换数值回归继续通过。

## 独立 A/B

同一新二进制、Qwen3-0.6B / BF16 / FlashInfer 0.6.18 / A100 GPU 1 / 同 KV 容量与 graph 桶。三个模式分别为：`rust-no-metadata`（保留流水线，仅关闭本阶段优化）、`rust-serial`（全同步）、`rust`（默认启用）。三轮轮换顺序，共 1224 个计时请求；默认模式分别对照两个 baseline 的 816 对输出 SHA-256 全部相同，usage 长度检查通过。

下表对照的是仅关闭 metadata 优化的模式，各值取三轮统计中位数。

| case | 关闭 TTFT ms | 开启 TTFT ms | 关闭 output token/s | 开启 output token/s | 本阶段吞吐变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold64 | 3.21 | 3.30 | 299.6 | 290.6 | -3.01% |
| cold512 | 7.47 | 7.50 | 131.8 | 130.7 | -0.84% |
| cold2048 | 20.62 | 20.77 | 48.1 | 47.8 | -0.58% |
| fallback4096 | 43.72 | 43.54 | 22.8 | 22.8 | +0.23% |
| generate64c1 | 4.17 | 4.17 | 421.0 | 476.5 | +13.18% |
| generate512c4 | 19.38 | 19.36 | 1218.3 | 1355.7 | +11.28% |
| generate1024c4 | 33.04 | 32.99 | 976.4 | 1057.7 | +8.32% |
| prefix1024 | 5.85 | 5.98 | 395.4 | 433.6 | +9.65% |

三个生成场景本阶段中位数吞吐提升约 8.3%～13.2%，缓存生成约 9.7%；各场景三轮逐轮变化均为正。结果针对当前模型、A100 和 c1/c4 的 workload，不能外推到其他模型或大 batch。组合中同时包含 early fence、独立 stream、pinned/non-blocking H2D，不能将全部收益归因于提前记录事件本身。单 token prefill 请求不经过 decode metadata 上传，短请求的正负变化属于本次测量波动。

三个模式 ready/steady GPU 驻留近似均为 20821/20849 MiB。生成基准忽略 EOS，相关生命周期由专门 CPU/CUDA 回归验证。

原始目录：`benchmark/results/overlap-read-boundary-2026-10-12/`；服务二进制 SHA-256：`3fcb1387ad270241deaa534858e41b5171bb7eb764f8dee9cb1bbb84c215488c`。实现、文档、日志和结果本地/远端同步；不新增 HTTP 专项测试计数。

```sh
cd /llm_infer_rust
.venv/bin/python scripts/benchmark-shared-graph-pool.py \
  --rounds 3 --modes rust-no-metadata rust-serial rust \
  --output-dir benchmark/results/overlap-read-boundary-2026-10-12 --resume
```
