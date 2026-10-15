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

## 2026-10-15 captured decode 规划与元数据优化

本轮默认启用 CUDA Graph 和 CPU/GPU overlap，按规划、元数据复制、小算子的顺序改动，保留逐阶段二进制进行三轮轮换 A/B；以下数字不与上文的旧 eager 基准直接比较。

### 实现与生命周期

- `DecodeHostMetadata` 保存每批不可变的 CPU cache lengths，captured decode 用它计算分页 indptr。单个 CUDA kernel 将页索引写入固定容量缓冲并更新末页长度，替换动态 `masked_select` 和规划用的 D2H。缺少 host snapshot 时保留旧路径。
- 每个 FlashInfer plan 持有 pinned indptr 和上传完成事件。CPU 改写 pinned source 与 planner workspace 前等待上次规划上传完成；GPU workspace 更新在原执行 stream 上排在此前模型之后。该事件不替代 InputReadFence 或 KV CompletionFence。
- DecodeManager 在请求换行或整行页号变化时生成唯一版本号。每个图桶保存自己的页表版本，只复制相邻的脏行；padding 覆盖行和无版本 fallback 会使副本失效。稳定 token 增长不重新复制已分配的页表。
- FutureMap 缓存不可变的 GPU scatter indices，仅在 batch 槽位或顺序变化时上传，避免采样后逐 token 的阻塞 H2D。旧 index tensor 与其使用者均在同一执行 stream 上有序退役。
- SiLU × up 保留 FP32 激活运算、16-byte 向量访存和小 batch 分散 CTA 布局。2048/3072/4096/8192 等宽度使用编译期常量寻址，消除运行时整数除法。原版单 token 单 block 布局在本机候选测试中略慢，因此未保留。Norm、RoPE 的计算语义保持原样。

### 三轮交替 A/B

同一张 A100-SXM4-80GB GPU 1，Qwen3-0.6B BF16，PyTorch 2.13.0+cu130、FlashInfer 0.6.18，固定 81-token completions 输入，忽略 EOS、贪心、流式。服务 `max-running-req=4`、`max-seq-len=40960`，默认 graph、overlap、显存预算；每场景预热后测 6 组，五个阶段运行三轮，每轮轮换启动顺序。并发 4 为每组同时发送 4 个请求。表中为三轮统计的中位数，无 profiler。

| 场景 | 优化前 TPOT ms | 最终 TPOT ms | TPOT 降低 | 优化前输出 tok/s | 最终输出 tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 并发 1、输出 32 | 2.250 | 1.887 | 16.1% | 423.9 | 514.4 |
| 并发 1、输出 128 | 2.362 | 1.945 | 17.7% | 418.4 | 509.8 |
| 并发 4、输出 128 | 2.895 | 2.417 | 16.5% | 1330.3 | 1582.6 |

串行 128-token TPOT 的阶段中位数依次为：优化前 2.362、host planner 2.131、页表增量复制 2.125、SiLU 特化 2.121、relay 索引缓存 1.945 ms。主要可测收益来自规划 D2H 与 relay 阻塞 H2D 的消除；页表复制和 SiLU 的独立端到端变化在 1% 以内，不应宣称显著吞吐收益。所有阶段共 540 个计时请求的 token 长度与生成文本 SHA-256 检查通过。最终固定场景 ITL 为 1.964 ms，接近此前原版约 1.90 ms 的测量；这不代表所有负载都追平原版。

最终结果保存在 `benchmark/results/decode-opt-2026-10-15-final/`：`alternating/results.json` 为逐请求结果，`summary.json` 为汇总，`manifest.json` 为环境、参数与各阶段二进制 SHA-256。复测入口为 `scripts/benchmark-decode-stages.py`，其 `--binary LABEL=PATH` 可指定阶段二进制。构建时确认修改过的源文件时间戳触发 Cargo 重编译，并记录实际执行的五个独立二进制哈希。性能阈值属于本机诊断，不能作为跨硬件 CI 断言。早期候选结果另存于不带 `-final` 的目录，不作为最终版本验收数据。

正确性回归：CPU 145 个单元测试和 1 个集成测试、CUDA 24 个单元测试和 1 个集成测试、12 项 HTTP 用例及 512-token 长输出通过。覆盖包括 host snapshot 校验、跨页长度和 graph padding、fallback 与连续回放、源缓冲早期覆盖、页表版本与 relay 索引缓存失效，以及 BF16/FP16 SiLU 的 decode/prefill 和尾部宽度。

### 原始 workload 与 trace 复核

沿用用户的 random ISL/OSL=64/32、每组 100 请求、客户端并发 1/8/32，服务端 `max-running-req=1`。全部请求成功，每组仍报告 3200 输出 tokens、重分词 3100 tokens。单并发 mean TPOT 从 2.176 降至 1.830 ms，mean ITL 从 2.247 降至 1.891 ms，输出吞吐从 417.6 增至 505.4 tok/s。最终客户端并发 8/32 的输出吞吐分别为 518.2/515.9 tok/s。原始数据见 `smoke-comparison.json`；baseline 与最终版本使用相同工作负载，三轮内部 A/B 为主要性能证据。

原先失败的 `scripts/probe-decode-latency.py --samples 6 --max-tpot-ms 2.0` 复测通过：128-token 输出 TPOT 为 1.930 ms，128 个非空文本事件，结果见 `probe-final.json`。

相同短窗口 Nsight node trace 中，完整 decode graph 仍为 339 个 kernel，图执行跨度约 1.946→1.944 ms，图间空隙中位数约 1.168→0.102 ms。10 KiB 页表复制的次数从约每 decode 一次降到每请求一次；小额规划 D2H 只在请求边界/fallback 观察到。SiLU 的逐图 kernel 累计中位数为 66.9→64.1 μs，收益很小，与三轮端到端结果一致。trace 会增加开销，CPU 同步等待时间包含 GPU 计算，不能与 GPU 图耗时相加或直接推算上线收益。汇总为 `trace-comparison.json`；原始 trace 保留在远程 `/tmp/decode-opt-verified-rust.nsys-rep` 和 `.sqlite`。
