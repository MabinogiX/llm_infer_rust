# Qwen3 FlashInfer decode CUDA Graph

日期：2026-10-07。项目：`sglang-test:/sjtu/yaosikai/llm_infer_rust`。

本文记录 decode graph 接入时的实现与测量。后续已增加 prefill 分段图，参见 [Prefill CUDA Graph](cuda-graph-prefill.md)。

FlashInfer decode 已接入 CUDA Graph。模型 forward 在启动时捕获，运行时更新输入和分页规划后回放；prefill、采样和调度仍在图外执行。Qwen3-0.6B 在 A100 上的 1024-token 请求中位延迟从 4.04 秒降至 2.98 秒，约降低 26%。

## 前置条件与处理

原有 GraphRunner 已具备固定输入缓冲区、按 batch 大小捕获和 padding 的基础，但 FlashInfer 后端明确禁用了 graph，还存在三个阻碍：

1. Decode 准备阶段通过 GPU `masked_select` 生成变长页索引，并读取 indptr 到 CPU，再调用规划器。不能在 stream capture 中执行这些动态操作和同步。
2. Eager decode 只有一个可复用 plan。不同大小的图需要独立持有地址固定的页索引、indptr、last-page-length 和 workspace；捕获后不能更换这些地址。
3. Padding 行共用一个 KV 写入位置，多个线程写同一位置存在冲突。

现在每个 captured batch 持有独立 backend state。捕获前完成第一次规划；捕获内的 attention prepare 只获取对应 plan，所有层复用它。回放前，在图外依据更新后的 block table 和 cache lengths 重做规划，将结果写入固定缓冲区。

Native FlashInfer 使用 `DecodePlan(enable_cuda_graph=true)`，按固定 GPU 几何保留 split-KV 容量，并传入 `block_valid_mask`。短序列与长序列的有效任务数量可以变化，捕获的 launch 结构与 workspace 地址保持固定。已捕获的 workspace 若需超出预留容量，返回错误，避免释放图仍引用的内存。

GraphRunner 为每个可能的 padding 行预留独立 KV 页。缩小 batch 时重置 padding 行的 input ID、position、长度、页表和写入位置。FlashInfer graph 不需要 `req_to_token`，因此不在回放时复制完整 token 映射。

采样本来就位于 model forward 之外，joint top-k/top-p 无需改动。模型内的 KV 写入、RoPE、RMSNorm、投影等算子具备捕获基础。`SGLANG_PROFILE_STEPS=1` 会在模型阶段计时中同步设备，因此自动禁用 graph；正常性能测试设为 `0`。

图持有 backend state 的强引用，backend 注册表只持有弱引用。清理图时先释放 native graph，再释放 plan 和缓冲区，最后归还 padding 页；模型卸载前先清理图。重新捕获不会残留旧 plan。

## 使用

原 `tmp.sh` 无需修改：它设置 `SGLANG_PROFILE_STEPS=0`、FlashInfer 和 `--max-running-req 1`，会默认捕获 batch 1。

- `--cuda-graph-bs N`：最大捕获 batch，不超过 `max-running-req`。默认使用 `max-running-req`。
- `--cuda-graph-bs 0`：禁用，使用 eager decode。
- 捕获大小为不超过上限的 1、2、4 等既有序列，并补上上限本身。实际 batch 使用最小可容纳的图；例如 batch 3 使用图 4 并 padding。
- 超出已捕获大小的 batch 使用 eager。单个大小捕获失败时记录 warning；没有可用图时使用 eager。
- 启动日志 `CUDA Graph capture complete` 列出成功捕获的大小。CPU 或缺少 native CUDA Graph bridge 的构建使用 eager。

本轮完整模型服务成功捕获 `[1, 2, 4]`；与原启动参数相同的单请求配置另行验证默认 batch 1 捕获。

## 验证

- 修改前显式指定 `--cuda-graph-bs 4`，FlashInfer 服务没有 capture-complete 日志，断言失败；修改后断言通过，捕获 `[1, 2, 4]`，没有 capture-failed warning。
- 本地 `cargo test --lib`：100 项通过。
- 远程 `cargo test --release --lib -- --test-threads=1`：106 项通过，1 项已有采样微基准默认忽略；release 服务构建通过。采样测试使用全局 CUDA RNG，应串行运行。
- 新增真实 FlashInfer graph 测试：batch 1/2/3/4 切换、batch 5 回退、不同请求长度、1/16/17/128/255/1023 的长度变化、页表换序、实际 KV 写入、padding、返回输出不会被后续回放覆盖、释放 padding 页、清理后重新捕获，以及显式禁用。与 eager FlashInfer 最大绝对误差不超过 0.03125。
- 完整 Qwen3 API 测试：12 项通过，包括流式与非流式、max_completion_tokens、并发请求行压缩和异构 joint 采样。
- 12 个串行贪心基准请求的完整 message 和 usage 均与修改前一致。

GPU 测试均在 GPU 1 执行，通过 `CUDA_VISIBLE_DEVICES=1` 映射到进程的 CUDA 0；没有使用正在运行用户工作负载的 GPU 0。

## 性能

同一台 A100-SXM4-80GB、GPU 1、Qwen3-0.6B BF16、release、FlashInfer、max-running-req=4、max-seq-len=40960、graph 上限 4。修改前后都明确指定 graph 参数，修改前 FlashInfer 后端不支持 graph，因此实际走 eager。短 chat prompt 为 18 tokens，temperature=0、ignore_eos=true；请求串行发送，每组 4 次，舍弃第一次，取余下 3 次中位数。重复 prompt 命中相同的 16-token radix 前缀。时间包含 HTTP、prefill、全部 decode 与采样。

| 生成 tokens | 修改前 eager | CUDA Graph | 延迟降低 |
|---:|---:|---:|---:|
| 32 | 129.04 ms | 91.83 ms | 28.8% |
| 128 | 534.43 ms | 353.66 ms | 33.8% |
| 1024 | 4.045 s | 2.982 s | 26.3% |

这是单轮每组 4 次请求的比较，包含调度与时钟波动；修改前的一个 1024-token 样本为 4.635 秒，其余保留样本约 4.02–4.04 秒。没有测量高并发吞吐，不能将此表推广至其他模型或 GPU。原始输出、capture 日志、API 和单元测试记录保存在 Git 忽略目录 `benchmark/results/cuda-graph-2026-10-07/`，测量脚本为其中的 `measure.py`。

## 后续成本

Graph 捕获的前置条件已满足，但 graph 不会消除整个服务的开销。当前回放前仍通过 `masked_select` 整理页索引、读取 indptr 到 CPU，并运行 FlashInfer 规划器；调度侧输入上传、采样和 token IDs 回传也在图外。这些是后续端到端优化的候选项，需要单独测量，不能据此断言已达到原版 SGLang 的请求延迟。

Prefill 继续使用既有 FlashInfer paged/ragged eager 路径。本轮没有接入 prefill graph、异步重叠采样与调度、张量并行或其他模型。CUDA Graph 回归测试和完整模型测量使用 BF16；FP16 的 attention 已有参考测试，但本轮没有单独测量 FP16 graph。

源码与文档同步到本地和远程。临时测试服务已停止，远程 release 二进制可用原 `tmp.sh` 启动；脚本未改动。
