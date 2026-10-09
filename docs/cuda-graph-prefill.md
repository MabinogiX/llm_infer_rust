# Qwen3 prefill 分段 CUDA Graph

日期：2026-10-07。项目：`sglang-test:/sjtu/yaosikai/llm_infer_rust`。

Prefill 已增加按 token 桶复用的分段 CUDA Graph。图捕获 attention 之外的 transformer 计算，FlashInfer attention、分页规划、KV 写入和动态 logits 输出留在图外。Qwen3-0.6B 的开关对照中，短 prefill 请求中位延迟约降低 25%，六组缓存前缀用例约降低 7%–28%；长 prefill 没有一致的大幅加速。当前版本在启动时捕获全部配置桶，请求期间只回放，不再承担图捕获成本。下方性能表是上一版分段图热回放测量，启动捕获的验证单列在文末。

## 原版参考

检查的是远程 `/sjtu/yaosikai/sglang` 提交 `15b256bdb`。该版本的 `PrefillCudaGraphRunner` 支持 `breakable`、`full` 和 `tc_piecewise`，CUDA 默认选择 `breakable`。其构造函数在 `prefill_cuda_graph_runner.py:606` 调用 `self.capture()`，`_capture_one_stream` 按 token 桶从大到小捕获，完成后才提供服务。Breakable 不依赖 torch.compile，在 attention 边界分段；prefill 按 aggregate token 数分桶，动态 attention 与输出尾部可以保留 eager。

主要参考文件为：

- `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`
- `python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py`
- `python/sglang/srt/model_executor/cuda_graph_config.py`

原版当前 `tmp.sh` 显式设置 decode、prefill backend 均为 disabled。代码支持 prefill graph，与该脚本实际启用它是两回事。

## 实现与边界

Rust 不依赖 Python 的装饰器或 torch.compile。Dense Qwen3 层拆为输入归一化、QKV/位置处理、attention、输出投影/MLP 等共享方法，eager 和 graph 使用同一组算子。

一个 token 桶捕获 `num_layers + 1` 个图。首段执行 embedding、第一层输入 RMSNorm、QKV 投影、QK RMSNorm 和 RoPE。每层 attention 返回后，下一段执行本层输出投影、残差 RMSNorm、MLP，并接续下一层的输入 RMSNorm/QKV/QK RMSNorm/RoPE。最后一段执行最后一层输出计算及最终 RMSNorm。Qwen3-0.6B 的 28 层对应每桶 29 个图。

每次 prefill 只在图外准备一次真实 attention 元数据，所有层共享。图内计算按桶大小 padding，但只有真实 token 的 Q/K/V 进入 attention 和 KV cache，因此不需要创建虚拟请求或为 padding 分配 KV 页。改变请求数、query 边界、缓存前缀和 paged/ragged 模式不需要重新捕获同一 token 桶。

Attention 输出复制到下一段的固定缓冲区，残差缓冲区也独立持有。原地 RMSNorm 不会覆盖上一段仍需读取的残差。Logits gather 和 LM head 按真实请求的 `logits_indices` 在图外执行，避免捕获请求数变化的输出形状。返回 logits 不复用下次回放的 segment 输出存储。

`NativeCudaGraph::capture` 统一处理 side-stream 预热、同步、捕获和错误退出。每个分段图持有固定输入、输出和 native graph；释放整桶前等待 GPU 完成使用其输出，再释放 private allocation pools。模型重新加载权重或重新绑定 KV cache 前清理图，避免图引用旧地址。

接口和代码主要位于：

- `src/models/qwen3/prefill_graph.rs`：桶选择、启动捕获、分段回放、GPU 回归测试。
- `src/models/qwen3/model.rs`：共享层算子和 prefill graph 分流。
- `src/engine/graph.rs`：通用原生分段捕获封装，既有 decode graph 保留。
- `src/engine/model_runner.rs`：`configure_prefill_graph`、`capture_prefill_graphs`、配置和清理生命周期。
- `src/engine/engine.rs`、`src/server/cli.rs`、`src/main.rs`：参数、解析和启动日志。

## 配置与缓存

`--prefill-cuda-graph-max-tokens N` 设置整批未缓存 token 总数的最大捕获桶，默认 2048；设为 0 禁用。它独立于 `--cuda-graph-bs`，后者只控制 decode。原有 Rust `tmp.sh` 无需修改，CUDA BF16/FP16 运行时默认启用分段 prefill graph。

图在权重及 KV cache 绑定后、HTTP 监听前捕获。日志 `capturing prefill segmented CUDA Graphs at startup` 给出全部桶，逐桶 `prefill segmented CUDA Graph capture complete` 给出桶大小和段数，最后 `prefill CUDA Graph startup capture complete` 列出成功及失败的桶。CPU、float32、缺少 native bridge、`SGLANG_PROFILE_STEPS=1` 或超过 token 上限时使用 eager。返回错误的捕获会标记对应桶失败，后续该桶使用 eager，重新配置会清理失败标记。

桶按以下规则向上取整，最后受 N 限制：

| 实际未缓存 tokens | 桶间隔 |
|---:|---|
| 1–128 | 2 的幂 |
| 129–512 | 64 |
| 513–1024 | 128 |
| 大于 1024 | 256 |

例如 265 tokens 使用桶 320，516 使用桶 640。仅使用 2 的幂会使这两个请求接近翻倍计算，首轮实测出现了明显回退，因此最终采用更细的分桶。

所有成功捕获的桶保持常驻；请求中不执行捕获、LRU 淘汰或重新捕获。未成功捕获的桶和超上限请求回退 eager。重新绑定模型或清理后，需要显式重新初始化图。默认上限 2048 有 22 个 token 桶，每桶 29 个分段图；启动时间和常驻图内存随配置桶增加，调低上限可减少捕获规模。

上一版使用按需捕获和 LRU，与原版启动阶段捕获的生命周期不一致；本次已取消这两项设计。

## 正确性验证

- 本地 `cargo test --lib`：102 项通过。
- 远程 `cargo test --release --lib -- --test-threads=1`：109 项通过，1 项已有采样微基准默认忽略；最终 release 服务构建通过。
- 新增 GPU 对照使用 BF16 和 FP16、3 层 Qwen3、attention width 大于 hidden size，与 eager 比较 logits 和整个 KV cache；logits 最大绝对误差小于 0.005，KV 小于 0.03125。
- 场景覆盖多请求不等长、零/非零混合前缀、跨页、变化的 position/token、桶切换、padding 不写 KV、超上限回退、禁用/清理、启动预捕获、清理后重新初始化、缺失桶回退且请求期间不新增图。
- 完整 Qwen3 API：12 项通过，包括流式/非流式、并发请求、max_completion_tokens 和异构 joint 采样。
- 60 个性能请求的 text 和 usage，与改动前二进制及最终二进制的关闭模式均完全一致。
- `--prefill-cuda-graph-max-tokens 0` 的服务没有 prefill capture 日志，decode 仍成功捕获 `[1, 2, 4]`。

GPU 验证均使用 GPU 1，通过 `CUDA_VISIBLE_DEVICES=1` 映射为进程 CUDA 0，没有使用 GPU 0 上的用户工作负载。

## 分段图热回放性能对照

下表记录改为启动捕获之前的热回放性能；attention 外的分段计算与当前实现相同，当前生命周期改为启动捕获。环境为 A100-SXM4-80GB、GPU 1、Qwen3-0.6B BF16、release、FlashInfer、max-running-req=4、max-seq-len=40960、decode graph 上限 4。两列使用同一版本代码，关闭模式将 prefill graph 上限设为 0。

文本请求生成 1 token，temperature=0、ignore_eos=true，包括 HTTP、prefill、LM head 和采样的完整延迟。每组 5 次，舍弃第一次，取余下 4 次中位数。Fresh 请求在开头改变标记以避免 16-token radix page 命中；prefix 请求预先填充共享前缀，再改变末尾。下表 prefix 的 token 数是完整 prompt 长度，图处理的是尚未缓存的 token。

| Fresh prompt tokens | 关闭 prefill graph | 开启 | 延迟降低 |
|---:|---:|---:|---:|
| 54 | 6.98 ms | 5.24 ms | 24.9% |
| 265 | 8.80 ms | 8.92 ms | -1.4% |
| 516 | 12.48 ms | 11.85 ms | 5.1% |
| 1016 | 15.18 ms | 14.48 ms | 4.6% |
| 2016 | 26.46 ms | 25.45 ms | 3.8% |
| 4016 | 49.72 ms | 49.91 ms | -0.4% |

4016-token fresh 请求超过默认上限，两种配置均使用 eager，其微小差异属于测量波动。

| Prefix prompt tokens | 关闭 prefill graph | 开启 | 延迟降低 |
|---:|---:|---:|---:|
| 61 | 6.27 ms | 4.60 ms | 26.6% |
| 273 | 7.53 ms | 5.39 ms | 28.4% |
| 525 | 8.06 ms | 5.86 ms | 27.3% |
| 1025 | 8.80 ms | 6.71 ms | 23.8% |
| 2025 | 10.17 ms | 9.24 ms | 9.1% |
| 4025 | 12.84 ms | 11.90 ms | 7.3% |

旧按需捕获版本在新桶首次请求上的完整延迟，54/265/516/1016/2016 tokens 分别约为 88.71/41.00/64.04/113.53/94.16 ms。这些单样本包含捕获、预热和请求成本，不能当作纯捕获耗时。这些额外图捕获成本现在转移到服务启动阶段，不再在请求阶段发生。

这是串行、单轮每组 5 次请求的测量，有调度和时钟波动；没有测量高并发吞吐或其他模型。较短 prefill 和缓存前缀场景受益更明显，长 prefill 仍由矩阵计算主导。不要把表中 prefill 收益等同于完整 1024-token 生成请求的收益。

原始结果、日志和测量脚本位于 Git 忽略目录 `benchmark/results/prefill-graph-2026-10-07/`。`codex-pcg-after.json` 为上一版细分桶热回放开启结果，`codex-pcg-disabled.json` 为同实现关闭结果，`codex-pcg-before.json` 为重构前结果。初轮重构前 prefix 的一组测量有较大波动，因此最终表使用明确开关的对照。

## 启动捕获验证

最终 release 服务在 GPU 1 上从 06:27:03.067 UTC 启动，06:27:09.006 UTC 开始 HTTP 监听，整体约 5.94 秒。Prefill 捕获从 06:27:07.769 到 06:27:08.975，约 1.21 秒；全部 22 个桶、每桶 29 段均成功，`failed={}`。这些时间包括该次实际运行条件，不代表其他模型或设备的固定启动耗时。

完成 60 个测量请求和 12 项 API 测试后，逐桶 capture complete 日志仍为 22 条，没有请求触发捕获。60 个请求的 text 和 usage 与 eager 对照一致。

本次启动捕获版同协议测得 fresh prompt 54/265/516/1016/2016/4016 tokens 的中位延迟依次为 5.08/9.01/12.82/16.93/25.58/50.29 ms；prefix prompt 61/273/525/1025/2025/4025 tokens 依次为 4.75/5.38/5.88/6.89/9.32/13.38 ms。首个 HTTP 请求仍有初始化开销，54-token 的单次延迟为 37.55 ms，但日志确认它没有执行图捕获，不能将所有首请求开销归因于 CUDA Graph。

本次服务启动捕获后 GPU 1 总占用约 35.6 GiB（36411 MiB）（包含模型、KV cache 和图等），说明全部桶常驻需要更多显存；该数值不是图本身的独立内存测量。原始启动日志和此次测量保存为 `codex-pcg-startup.log`、`codex-pcg-startup.json`，测试和 API 日志也存于上述结果目录。

## 当前范围

已实现 dense Qwen3 的分段 prefill graph，不是 full prefill graph。FlashInfer paged/ragged attention 与规划继续 eager，decode 图和 joint top-k/top-p 采样保持现有实现。没有依赖 torch.compile，没有增加 MoE、张量并行或 prefill/decode 异步重叠。

本地与远程源码、文档已同步，远程 release 二进制可用原 `tmp.sh` 启动；启动脚本未修改。临时测试服务已停止。
