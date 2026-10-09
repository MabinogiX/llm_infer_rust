# Qwen3 prefill 与非贪心采样优化

日期：2026-10-07。项目：`sglang-test:/sjtu/yaosikai/llm_infer_rust`。

本轮在前一轮 decode 优化基础上，接入 FlashInfer ragged/paged prefill 和 GPU 概率采样。后续根据用户要求，采样已切换为原版 SGLang 当前采用的 joint top-k/top-p；下文原始性能表记录的是切换前的顺序采样版本，prefill 部分未变。此前的 release 二进制保存在远程 `/tmp/codex-prefill-sampling-before`，用于相同 GPU、相同配置的请求对照。

## 实现

### Prefill

无缓存前缀的 batch 调用 `PrefillPlan` 与 `BatchPrefillWithRaggedKVCacheDispatched`，直接读取当前层 packed QKV 的实际 stride。有任何请求带前缀时，整批调用 `BatchPrefillWithPagedKVCacheDispatched`，直接读取各层 KV cache 和请求页表。两者都使用 causal mask；paged 路径按 KV 长度与 query 长度之差对齐 query 的绝对位置，覆盖混合前缀与跨页请求。

规划在每次 model forward 准备一次，所有层共享。Prefill 和 decode 分别持有持久 plan，工作区根据官方调度器实际选出的 tile 数、split-KV 与 merge 大小扩容，后续容量足够时复用。前一批仍持有 plan 时不允许覆盖。

这消除了 FlashInfer prefill 路径上的逐层 KV gather、padding、GQA repeat 和显式注意力 mask，也消除了逐层 prefix 元数据的 GPU→CPU 读取。准备阶段仍有元数据读取和同步，paged 页索引仍通过 GPU `masked_select` 整理。本次没有接入 CUDA Graph。

涉及 `src/models/attention/backend.rs`、`flashinfer_bridge.cpp`、`flashinfer_plan_run.cu` 和 `build.rs`。

### 非贪心采样

CUDA 且构建包含 FlashInfer 时，temperature 缩放后的完整 FP32 softmax 概率直接交给 `TopKTopPSamplingFromProb`，在原始分布上同时计算 top-k/top-p 截断，取交集采样。不在 top-k 与 top-p 之间重新归一化，与原版 SGLang 的 `filter_apply_order="joint"` 一致。

不限制 top-k 时，使用 `TopPSamplingFromProb` 或 `SamplingFromProb`。这样同时省去 libtorch topk、top-k 阈值 mask，以及原先 top-p 的全词表排序、cumsum、scatter、第二次 softmax和 multinomial。同参数 batch 直接采样，避免 row-index 上传、logits gather 和输出 scatter；异构参数批次仍分组，保持输出行顺序。贪心仍用 argmax，`top_p=0` 直接选最高 logit。CPU 和没有 FlashInfer 的构建用 libtorch 在未过滤分布上计算两个阈值，采用相同 joint 语义。边界并列概率遵循 FlashInfer 的阈值语义。

例如完整概率 `[0.4, 0.3, 0.2, 0.1]`、`top_k=2, top_p=0.5`：joint 保留前两个 token，重新归一化后的概率为 `[4/7, 3/7]`。旧顺序在 top-k 后先重新归一化，随后 top-p 只留下第一个 token。回归测试已在旧实现上失败，并在 CPU joint fallback 上通过。

随机数从 libtorch CUDA generator 在锁内获取 seed 和互不重叠的 Philox offset。新实现可通过 `tch::Cuda::manual_seed` 重复，但随机算法变化后不保证与旧 multinomial 逐 token 相同。内核无效分布通过 GPU 标记为 -1，在整个 batch 的 token IDs 回传后报告错误，不为每组新增 CPU 同步。

真实词表微基准发现 A100 上向量化采样 kernel 的寄存器资源超限。采样独立编译并设置 `-maxrregcount=64`，不影响 attention 内核；151,936 词表大小已纳入必跑 CUDA 测试。

涉及 `src/engine/sampling.rs`、`sampling_bridge.cpp`、`sampling_flashinfer.cu` 和 `build.rs`。

## 验证方法

- Attention：BF16/FP16、非连续 QKV、不同请求长度、零/非零混合前缀、跨页和长前缀，与现有 SDPA 对照，最大绝对误差限制 0.03125。
- Sampling：与 joint 阈值过滤计算的理论概率对照，每组 20,000 个样本，频率误差不超过 0.02；检查被截断 token 不出现、top-k 并列值、异构参数行顺序与 seed 复现；另测 Qwen3 实际词表大小。
- HTTP：已有完整 API 测试，加上 12 个混合温度/top-k/top-p 的并发请求。服务配置 `max-running-req=2`，覆盖非贪心、贪心、top-k=1 和 top-p=0。
- 请求延迟：GPU 1，A100-SXM4-80GB，Qwen3-0.6B，FlashInfer，关闭 CUDA Graph，最多两个 running requests，但计时请求串行发送。每组 5 次，舍弃第一次，取剩余 4 次的中位数。无前缀请求在 prompt 开头更换标记，防止命中 radix cache；缓存请求共享长前缀、变更末尾。1-token 请求衡量的是 HTTP 全部延迟，以 prefill 为主，并非单独 kernel 时间。

原始请求结果和测量脚本保存在本地 Git 忽略目录 `benchmark/results/prefill-sampling-2026-10-07/`。GPU 采样微基准可通过 `cargo test --release --lib benchmark_sampling_cuda -- --ignored --test-threads=1 --nocapture` 复测；包含 global CUDA RNG 的测试应串行运行。

## 原始实测结果（改为 joint 之前）

无前缀、生成 1 token（包括 HTTP、完整 prefill 与采样）：

| Prompt tokens | 优化前 | 优化后 | 延迟降低 |
|---|---:|---:|---:|
| 44 | 9.41 ms | 5.56 ms | 40.9% |
| 525 | 14.09 ms | 12.02 ms | 14.7% |
| 2062 | 32.55 ms | 26.41 ms | 18.9% |
| 8206 | 121.85 ms | 110.71 ms | 9.1% |

共享缓存前缀、变更末尾、生成 1 token：下表按测量脚本中重复单词数量标识前缀规模，实际 cache match 是按 page 对齐的 token 数。

| 前缀规模 | 优化前 | 优化后 | 延迟降低 |
|---|---:|---:|---:|
| 约 512 tokens | 21.50 ms | 6.43 ms | 70.1% |
| 约 2048 tokens | 24.91 ms | 9.72 ms | 61.0% |
| 约 8192 tokens | 44.03 ms | 18.33 ms | 58.4% |

Temperature=0.8、短 prompt、生成 128 tokens 的完整请求：

| Top-k | Top-p | 优化前 | 优化后 | 延迟降低 |
|---|---:|---:|---:|---:|
| 不限制 | 1.0 | 597.66 ms | 529.73 ms | 11.4% |
| 50 | 0.9 | 588.61 ms | 545.57 ms | 7.3% |
| 不限制 | 0.9 | 567.53 ms | 531.61 ms | 6.3% |

采样微基准在 GPU 0（同为 A100-SXM4-80GB）运行，FP32 logits、151,936 词表、10 次预热后平均 100 次，每次包含 token IDs 回传。旧路径在同一个测试中显式执行原有过滤与 multinomial；这里隔离的是采样，不包括 model forward，也不包括旧 `sample_batch` 的分组拷贝。

| Batch | Top-k | Top-p | 旧采样 | 新采样 | 加速比 |
|---:|---|---:|---:|---:|---:|
| 1 | 不限制 | 1.0 | 0.1938 ms | 0.1519 ms | 1.28× |
| 1 | 不限制 | 0.9 | 0.4761 ms | 0.1669 ms | 2.85× |
| 1 | 50 | 0.9 | 0.5341 ms | 0.3008 ms | 1.78× |
| 8 | 不限制 | 1.0 | 0.1786 ms | 0.1471 ms | 1.21× |
| 8 | 不限制 | 0.9 | 0.8933 ms | 0.1757 ms | 5.08× |
| 8 | 50 | 0.9 | 0.9963 ms | 0.3590 ms | 2.78× |

切换 joint 之前的验证：本地 `cargo test --lib` 99 项通过；远程 `cargo test --release --lib -- --test-threads=1` 104 项通过、1 项微基准默认忽略；显式执行该微基准通过；release 服务构建通过；优化前后完整 HTTP 测试均为 12 项通过。35 个贪心对照请求的输出完全一致，全部 50 个对照请求的 usage 完全一致。非贪心请求未做逐 token 一致性要求。

这些是单轮、每组 5 次请求的测量，有 GPU/CPU 调度与时钟波动；没有将三组非贪心请求的耗时差当作相同采样算法的输出轨迹。结果表明缓存前缀 prefill 和 nucleus 采样本身有明显收益，完整生成请求仍主要由 model decode 占据。没有测量高并发吞吐、不同模型或其他 GPU，不能把采样内核加速比当作整个服务加速比。

本地与远程源码均已更新。临时测试服务已停止，远程 release 二进制可由原 `tmp.sh` 启动。启动脚本未修改。


## Joint 更新验证

代码已按原版 SGLang 当前的 joint 模式更新，CUDA 不再先用 libtorch top-k 过滤；CPU fallback 同步改为 joint。`joint_top_k_top_p_uses_original_probability_mass` 在旧实现上以第二个 token 的频率为 0 失败，修改后 CPU 和 CUDA 均符合理论频率 3/7。其他采样分布测试也已改用 joint 参考阈值。

本地 100 项测试通过；远程 release/CUDA 105 项通过、1 项微基准默认忽略；显式微基准通过；release 服务构建通过；完整 API 测试 12 项通过，包含异构非贪心并发请求。临时服务已停止。

以下是本轮同一个 GPU 微基准内的对比。旧采样列仍是原始 libtorch 顺序过滤，不是上一版已经优化的 FlashInfer 顺序过滤。两列算法语义不同，仅用于记录成本，不能视为逐 token 等价实现的性能比较。

| Batch | Top-k | Top-p | 原始 libtorch 顺序采样 | FlashInfer joint |
|---:|---|---:|---:|---:|
| 1 | 不限制 | 1.0 | 0.1926 ms | 0.1468 ms |
| 1 | 不限制 | 0.9 | 0.4752 ms | 0.1717 ms |
| 1 | 50 | 0.9 | 0.6301 ms | 0.3961 ms |
| 8 | 不限制 | 1.0 | 0.1795 ms | 0.1463 ms |
| 8 | 不限制 | 0.9 | 0.8938 ms | 0.1754 ms |
| 8 | 50 | 0.9 | 0.9899 ms | 0.4951 ms |

Joint 直接在完整概率分布上拒绝采样。随机 logits、top-k=50、top-p=0.9 用例中，本轮比上一轮已经优化的顺序 FlashInfer 路径耗时更高；不承诺语义切换在所有分布上更快。本次没有重新测量完整请求延迟，上面的 128-token 请求表仍为 joint 切换前结果。原始微基准输出为 `benchmark/results/prefill-sampling-2026-10-07/sampling-joint-microbench.log`。
