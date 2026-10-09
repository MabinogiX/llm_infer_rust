# Qwen3-0.6B：Rust 与 SGLang 剩余 decode 差距

日期：2026-10-08。分析远程 `sglang-test:/sjtu/yaosikai/llm_infer_rust`（HEAD `694df85`）和 `/sjtu/yaosikai/sglang`（HEAD `15b256bdb`）。本次没有修改模型、依赖、启动脚本或编译二进制；只使用 GPU 1 启动临时诊断服务。

## 用户 benchmark 的实际含义

两份日志均为 100 个请求、随机 ISL/OSL=64/32、temperature=0、seed=42，服务最大运行请求数均为 1。客户端并发增加到 8、32 主要增加排队，没有扩大模型 decode batch。

| 客户端并发 | Rust 输出 tok/s | SGLang 输出 tok/s | Rust TTFT ms | SGLang TTFT ms | Rust TPOT ms | SGLang TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 363.58 | 410.31 | 8.12 | 21.62 | 2.56 | 1.81 |
| 8 | 369.59 | 354.68 | 587.88 | 639.66 | 2.57 | 1.82 |
| 32 | 368.06 | 361.60 | 2271.14 | 2337.54 | 2.57 | 1.81 |

单并发输出吞吐低约 11.4%，但 decode TPOT 高约 41.4%；Rust 更低的 TTFT 抵消了一部分 decode 差距。后两组 Rust 吞吐略高，不能概括为所有场景 Rust 都更慢，也不能据此判断批量模型吞吐。

两份日志的 reported output 均为 3200 tokens，retokenized output 分别为 3100/3200。重分词与原始生成 token 数不必一致，这不是错误的充分证据，但对照时需保留两种统计口径，并另行检查输出或 token IDs。日志中的 peak concurrent requests 也不能当作实际模型 batch size。

## 配置与版本核对

- GPU：A100-SXM4-80GB；诊断进程 `CUDA_VISIBLE_DEVICES=1`。
- 模型：同一目录 Qwen3-0.6B，BF16，28 层，hidden=1024，16 Q heads、8 KV heads、head_dim=128，MLP=3072，vocab=151936。
- 两边当前默认都启用 decode graph，原版默认 prefill backend 为 breakable。
- 两边 attention 都为 FlashInfer；Rust page_size=16，原版 page_size=1。
- 两边 torch 都为 `2.13.0+cu130`，FlashInfer 分别为 **0.6.14 / 0.6.18**。Rust 从自己环境中的 FlashInfer 头文件编译内核，版本差异会进入实际二进制。
- 原版 BF16 GEMM backend 配置为 auto。本次 trace 中两边实际使用相同的 cuBLAS GEMV/GEMM kernel 家族。
- 原版默认 `disable_overlap_schedule=False`；Rust 当前调度、forward、采样和发布串行执行。

## 模型 graph 的 GPU 证据

Nsight Systems 使用 CUDA node tracing，仅采集服务就绪后的短窗口。固定 `/v1/completions` prompt 为 `The sky is blue. ` 重复 16 次，两边都是 81 个输入 tokens，生成 128 tokens，temperature=0、ignore_eos=true、stream=true，串行请求。重复 prompt 可以命中前缀；此测量主要隔离 decode，不能替代用户随机 chat benchmark。

按 kernel 的 correlationId 聚合 graph launch，仅选择包含 FlashInfer decode 且超过 200 个 kernel 的组。Rust 有 2153 个完整 395-kernel 图，原版有 3002 个完整 340-kernel 图；窗口结束时各有一个不完整组。下面为各类 kernel 累计时长的中位数，单位微秒；分组中位数之和不必严格等于总时长中位数。

| 每个 decode graph | Rust | SGLang |
|---|---:|---:|
| GEMM/GEMV（含 LM head） | 1150.91 | 1146.64 |
| RMSNorm、残差 RMSNorm、QK Norm | 335.68 | 196.83 |
| K/V 写入 | 173.66 | 55.55 |
| RoPE | 57.47 | 53.22 |
| SiLU × up | 65.92 | 53.06 |
| FlashInfer decode + merge | 533.76 | 374.83 |
| 其他（含索引转换、embedding） | 61.28 | 8.22 |
| 所有 kernel 的累计时长 | 2379.26 | 1888.19 |
| graph 首 kernel 开始到末 kernel 完成 | 2401.18 | 1907.14 |

Node tracing 会扰动运行，尤其是大量小 kernel。这个表用于识别候选瓶颈，不能把差值直接视为上线后可兑现的加速，也不能将 CPU CUDA API 时间与 GPU 时间相加。最终收益需同环境、无 profiler 的 A/B 验证。

### 1. KV 写入是最明确的结构性浪费

Rust `src/models/attention/base.rs:71` 每层执行 `write_loc.to_kind(Int64)`，再用两个 `index_copy_` 分别写 K/V。因此每个 decode step 有 28 次转换、56 次通用 scatter。索引内容并不随层变化，转换也被每层重复捕获进 graph。

原版 `python/sglang/srt/mem_cache/memory_pool.py:148` 的 `_set_kv_buffer_impl` 调用 `store_cache`，trace 确认 Qwen3 CUDA 路径使用单个 `sglang::store_kvcache`，同时写 K/V。不是推断它用了 RoPE+KV 三者融合；本次原版 trace 的 RoPE 与 KV store 是两个独立 kernel。

优先实现一个支持现有 cache 布局、QKV stride 和 int32/int64 location 的向量化 KV store，将每层 3 个 kernel 减为 1 个。至少先把相同 write_loc 的类型转换移到 batch/metadata 准备阶段。两层次的改动都需覆盖 prefill/decode、跨页、非连续 QKV、padding 和写入位置边界。

### 2. 已融合残差 RMSNorm，但内核效率仍有差距

Rust `src/models/qwen3/fused_ops.cu:83` 的 add_norm 为标量循环，每个 token 一个 256-thread block，并通过 shared memory 和两次 block 同步做归约。原版此次使用 FlashInfer/CuTe 的 FusedAddRMSNorm kernel，1024 hidden 的一个 token 对应 128-thread block。trace 中单次残差 RMSNorm 约为 Rust 4.6–4.8 μs、原版 2.4–2.6 μs；每 step 有 56 次，微小差异会累积。

Rust QK Norm 也用每 head 一个 256-thread block，head_dim=128 时一半线程没有元素处理；原版 `sglang::fused_qknorm_warp` 将多个 head 打包，trace 的 grid 为 6 而 Rust 为 24。应优先参考原版的 warp/head 分工、向量读写和 BF16 rounding 语义，而不是再次增加“残差融合”接口。

### 3. Attention 存在差距，但尚未隔离原因

两边都是 FlashInfer decode 和 merge；本次 Rust 累计 attention 时间更长。已确认 page_size 和 FlashInfer 版本不同，graph 内 decode grid/plan 参数也有差异，不能把这部分全归因于 Rust 桥接代码。

下一步需先统一 FlashInfer 头文件/运行库版本，在独立试验环境中重编译 Rust，再用相同 page size、序列长度、split-KV 和 graph 配置测量。若差距仍在，再检查 work estimation、graph padding、split-KV 及 merge。当前没有证据建议重新实现 attention。

### 4. GEMM 与 joint sampling 不是当前首要目标

原版 `UnquantizedLinearMethod.apply` 常规 CUDA 路径使用 `F.linear`；Rust `linear` 使用 `x.matmul(weight.transpose(0,1))`。trace 证实本次 GEMV/GEMM 家族和总时间近似相同。LM head 确实很大，但两边都有同一成本；暂不建议先重写所有线性层。

用户 benchmark temperature=0，两边使用 argmax，没有调用 joint top-k/top-p。继续优化非贪心采样不能解决这里的 decode 差距。

## 图外成本

Rust `src/models/attention/flashinfer_bridge.cpp:75` 每 step 通过 arange、cumsum、masked_select 等 GPU 操作生成页索引，`indptr.to(CPU)` 同步读回后重新运行 DecodePlan。`masked_select` 的动态输出长度也可能引入同步；GPU trace 确认它执行了选择、计数与索引写入 kernel。原版使用专门的 KV index kernel，并利用 CPU `seq_lens_cpu` 构造 `global_override_indptr_cpu` 传给 fast_decode_plan，避免为已知长度同步读回 GPU 结果。

Rust 调度侧还构造多个小 CPU tensor 并上传，graph 回放前复制 metadata，回放后复制整行 logits；greedy 采样用 `Vec::try_from(argmax)` 立即回传 token，随后 CPU 才构造下一步。这些都是 graph 外开销。原版采样仍是 argmax，优势候选在 pinned buffers、非阻塞传输和调度重叠，而非另一个采样算法。

应从 CPU 已知页数直接构造 planner indptr，并用一个专用 kernel 填固定地址的 GPU 页索引；让调度直接写稳定 graph 输入，减少小 tensor 分配和多次拷贝。同步替换需要确保 plan workspace 不会被上一轮仍在执行的图读取，不能直接删除当前同步屏障。

## 调度重叠的单变量验证

另外启动不带 Nsight 的正常服务，沿用同一固定 81-token completions prompt、max_tokens=128、temperature=0、ignore_eos=true。每组 12 个串行请求，舍弃前 2 个，取后 10 个样本中位数。两边每请求均收到 128 个非空 token 文本 chunk。原版只改变 `--disable-overlap-schedule`，仍启用 decode CUDA Graph。

| 无 profiler 的服务 | ITL ms（各请求中位数再取中位数） | TTFT ms | 完整请求 ms |
|---|---:|---:|---:|
| Rust 当前默认 | 2.805 | 7.790 | 358.687 |
| SGLang 默认 overlap | 1.902 | 32.608 | 272.097 |
| SGLang 关闭 overlap | 2.889 | 30.741 | 397.290 |

原版关闭 overlap 后 ITL 增加约 0.987 ms，约 51.9%，接近 Rust 的水平。这是默认调度重叠对该场景有显著贡献的直接 A/B 证据。不能把它与 kernel 表中的差值相加，也不能说 Rust 实现 overlap 必然兑现同样的 0.987 ms；两边图外 planner、拷贝、CPU 代码不同。实验为单轮、小样本，正式收益还需重复交替 A/B 及更长输出、更多模型 batch 验证。

本次没有新测真实 batch 8/32 吞吐；需要将双方服务的 max-running-requests 与 decode graph 上限一起放大，再测客户端并发。否则仍是在串行服务前排队。

## 建议实施顺序

1. **模型层先优化 KV store 和 Norm 内核**。边界清晰且 trace 有直接证据；K/V 合并写入、消除每层 location 转换、warp/向量化 RMSNorm 和 QK Norm。先分别改动并测量，避免多个改动混在一起。
2. **并行规划下一阶段调度重叠和稳定输入缓冲区**。端到端证据表明它是大项。采样结果保留在 GPU，CPU 准备下一批与上一批计算/回传交错；需要明确 event、buffer 双缓冲、请求结束/取消和 KV cache 发布的生命周期。这属于执行引擎改造，不能仅在模型内增加一个 kernel 达成。
3. **先统一 FlashInfer 版本再诊断 attention**，并控制 page size、split-KV 参数。升级/重编译应使用隔离环境，保留当前可回退二进制，当前分析没有修改原有依赖。
4. **收紧 graph 外 metadata 路径**：CPU indptr、固定 GPU buffers、专用页索引 kernel、异步回传；与第 2 项的正确性约束一起设计。
5. 小 batch GEMM 和非贪心 sampling 暂不作为本 benchmark 的首要改动。

GPU trace 的约 0.49 ms 模型 graph 差距与关闭 overlap 的约 0.99 ms ITL 变化，是两个不同测量口径：前者包含 node tracing 扰动、页大小/版本差异，后者为正常 HTTP 服务 A/B。它们共同证明模型算子和执行管线均有优化空间，但不能据此预测各项收益的简单总和。

## 复现与证据

轻量复测脚本为 `codex-decode-profile.py`，支持 `rust plain`、`sglang plain`、`sglang plain no-overlap`，对指定服务在 GPU 1/端口 8002 上启动、等待就绪、发送 12 个请求、保存请求时间并终止临时进程组。未带 plain 时通过 Nsight 采集延迟窗口；Nsight 结束会截断末个请求，分析已排除不完整 graph。

原始用户日志、逐请求测量 JSON、graph breakdown 和 kernel/API 汇总已保存至 Git 忽略目录 `benchmark/results/decode-gap-2026-10-08/`。原始完整 trace 保留远程 `/tmp/codex-decode-rust.nsys-rep`、`/tmp/codex-decode-sglang.nsys-rep`；SQLite 为 `/tmp/codex-decode-rust-final.sqlite`、`/tmp/codex-decode-sglang.sqlite`。

Nsight 导入所需 `libdw` 仅解包至远程 `/tmp/codex-nsys-libs/`，没有安装系统包。临时服务使用 GPU 1，测试完成后全部停止；原有两个 `tmp.sh`、`benchmark.sh` 和模型实现未改动。
