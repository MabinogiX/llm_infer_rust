# FlashInfer 对齐、归一化及 KV 写入优化

日期：2026-10-08。远程项目：`sglang-test:/sjtu/yaosikai/llm_infer_rust`。基于 Rust HEAD `694df85`；原版 SGLang HEAD `15b256bdb`。原有 `tmp.sh`、`benchmark.sh` 未修改。

## 实现

### FlashInfer 0.6.18

远程 Rust 的 FlashInfer 从 0.6.14 对齐到原版的 0.6.18。由于网络下载慢，使用原版已安装包重新打包的本地 wheel，由 uv 安装；原版环境未改动。逐文件校验两边 official include 目录的 241 个文件，内容完全相同。

版本现由根目录 `pyproject.toml` 的 Linux `cuda` extra 与 `uv.lock` 统一管理；`build.rs` 监视 FlashInfer include 目录，wheel 升级会触发 adapter 重编译。0.6.18 新的 `uniform_q_len` 参数使用 -1，表示异构 query 长度。Prefill workspace 改用官方 `PrefillPlanWorkspaceSize` sizing pass，再 reserve 和调用 `PrefillPlan`，不继续复制旧版本的容量估算公式。Decode、paged/ragged prefill、joint top-k/top-p 均已在新头文件下重编译和验证。

Torch 仍为 `2.13.0+cu130`；native adapter 不需要在推理路径导入 FlashInfer Python API。`uv sync --locked --extra cuda --group dev` 同时安装其完整 Python 依赖。

### 归一化

`src/models/qwen3/fused_ops.cu` 新增两个快路径：

- hidden=1024 的 RMSNorm 和 residual RMSNorm：128-thread block，每线程一个 16-byte BF16/FP16 向量；残差先舍入到 activation dtype，再在寄存器里保留到归约完成，减少标量加载和重复读写。
- head_dim=128 的 QK Norm：一个 warp 处理一个 head，四个 head 共用一个 128-thread block；向量读取四个元素、warp 归约，不使用原先每 head 一个 256-thread block 的 shared-memory 归约。

输入、权重、行 stride 均满足对齐要求才选择快路径；其余维度及未对齐输入保留通用内核。返回 Tensor 地址、原地残差更新和 packed QKV 的 V 部分保持现有接口语义。

数学与 BF16/FP16 舍入规则保持现有设计，但并行归约的加法顺序有变化，不承诺逐 bit 相同。数值容差和整模型验证见下文。

### 融合 KV store

新增 `src/models/attention/kv_store.rs`、`kv_store_bridge.cpp` 和 `kv_store.cu`，由独立 `has_cuda_kv_store` build cfg 启用。

CUDA BF16/FP16 直接由单个 kernel 将 K/V 同时写到 flattened page slots，支持 int32 和 int64 location、不同 K/V token/head stride 和非连续 packed QKV；不再每层将 location 转为 int64，也不再分别执行两个 `index_copy_`。满足对齐条件时使用 16-byte 向量拷贝，否则逐元素复制。复制为 dtype 位模式复制，没有浮点转换。

负位置（padding）及超过 capacity 的位置在 kernel 内不写缓存；正常位置仍由 scheduler 检查。CPU 的非法正位置继续返回错误；CUDA 的越界 guard 不引入同步读取。CPU、float32 和未编入 native bridge 的路径保留原有实现。

## 正确性

- 本地 `cargo test --lib`：102 项通过。
- 远程 GPU 1 `cargo test --release --lib -- --test-threads=1`：111 项通过，1 项已有微基准默认忽略；release 构建成功。
- 新增 BF16/FP16、aligned/unaligned 行的归一化参考对照，验证残差舍入和 padding 不被覆盖；QK Norm 覆盖 packed stride、非整 block 的 head 数及 V 不变。
- 新增 KV store 测试覆盖 BF16/FP16、int32/int64、跨页、负 padding、capacity guard、未写入区域、非连续 K/V、标量 fallback 和 decode 单槽覆盖，结果与 scatter 参考逐 bit 一致。
- 已有 decode graph 和 prefill segmented graph 的动态 bucket、批次 padding、不同长度及 logits/整个 KV cache 对照通过。
- 完整 Qwen3 API：12 项通过，包含 HF golden prompts、流式/非流式、并发、max_completion_tokens、tools、reasoning 和 joint sampling。

额外发送 8 组 prompt、每组首次与命中前缀各一次，共 16 个 greedy 请求，输出长度为 32。旧二进制和仅对齐 FlashInfer 的版本 text/usage 为 16/16 完全一致；最终版本 usage 为 16/16 一致、text 为 12/16 一致。文本差异集中在两组重复语句 prompt，归一化改动后出现；KV 拷贝对照逐 bit 一致，归约顺序变化是合理的数值解释，但本次没有测首个不同 token 的 logit margin，不宣称已证明都是 argmax 近似并列。旧版本自身在其中一组首次/缓存前缀请求也产生不同文本。

因此，这次结果是数值容差和 API 正确性验证通过，不能称为旧版本逐 token 完全一致。

## 无 profiler 的端到端测量

同一 A100-SXM4-80GB GPU 1、BF16 Qwen3-0.6B、release、max-running-req=1、max-seq-len=40960、FlashInfer、默认 prefill/decode graph。固定 `/v1/completions` prompt 为 `The sky is blue. ` 重复 16 次，两边都是 81 输入 tokens、128 输出 tokens，temperature=0、ignore_eos=true、stream=true。每轮 12 次，丢弃前两次，对后十次取中位数；ITL 是各请求的 token 间隔中位数再取中位数。

| 版本 | ITL ms | 完整请求 ms |
|---|---:|---:|
| 原二进制 | 2.826 | 361.962 |
| 仅对齐 FlashInfer | 2.656 | 340.954 |
| 最终版本，第一轮 | 2.353 | 301.887 |
| 原二进制，交替复测 | 2.825 | 361.334 |
| 最终版本，交替复测 | 2.355 | 301.534 |
| 最终版本，显式 page size 1 | 2.372 | 304.800 |

最终版本 ITL 降低约 16.7%，完整请求延迟降低约 16.6%。相同固定输出长度下，约对应 20% 的串行请求吞吐提升；该固定 workload 的收益不推广为所有场景；原脚本随机 smoke 复测结果见下一节。

仅升级 FlashInfer 的该轮 ITL 降低约 6%；归一化与 KV 的端到端收益按组合测量，没有分别隔离二者的 HTTP 收益。单项 kernel 时间使用 profiler 检查，不能与正常 HTTP 延迟相加。

保留默认 page size 16：本次 page size 1 小样本未更快，且会扩大当前图外页表整理和复制规模。版本对齐不意味着必须改动该调度参数；`--page-size 1` 仍可显式选择。

## 原 smoke benchmark 复测

使用原 `benchmark/sglang_bench/smoke.sh`，每轮 100 请求、随机 64 输入/32 输出 tokens、seed=42，客户端并发 1/8/32；服务 max-running-req=1，仍是串行调度。每轮 100 请求全部成功、usage 输出 3200 tokens、retokenized 输出 3100 tokens，与此前 Rust 日志一致。

临时复测将 BENCH_MODEL 设为本地模型路径，并设置 HF_HUB_OFFLINE=1，避免 benchmark 查询模型名 default 时联网重试；生产配置和原脚本未修改。输出吞吐按 usage tokens 计算。

| 客户端并发 | 原 Rust 日志吞吐 tok/s | 本次吞吐 tok/s | 增幅 | 本次 TTFT ms | 本次 TPOT ms |
|---|---:|---:|---:|---:|---:|
| 1 | 363.58 | 414.70 | 14.1% | 7.44 | 2.24 |
| 8 | 369.59 | 424.09 | 14.7% | 512.85 | 2.23 |
| 32 | 368.06 | 425.99 | 15.7% | 1962.31 | 2.22 |

对比为此前用户日志与本次运行，并非同轮 A/B。此前原版 SGLang 的并发 1 吞吐为 410.31 tok/s、TPOT 为 1.81 ms；本次 Rust 吞吐接近这一记录，但 TPOT 仍较大，不能宣称稳定超过原版。原版没有在本次重新启动或复测。固定 prompt 的交替二进制对照更适合确认改动本身的收益。

原始日志保存为归档目录中的 `codex-kernel-smoke.log`，远程原 benchmark.log/benchmark_sglang.log 未覆盖。

## GPU trace 验证

对最终版本再次采集相同 81/128 workload 的 Nsight node trace，排除采集窗口末尾的不完整图。共有 2567 个完整 decode graph，每图 **339 个 kernel**，此前为 395 个：每层三个索引转换/K scatter/V scatter 替换为一个 KV store，28 层共少 56 个 kernel。

| 每个 decode graph 的累计 GPU kernel 时间 μs（中位数） | 改动前 | 最终版本 |
|---|---:|---:|
| Norm（含 QK） | 335.68 | 194.53 |
| KV store（不含之前的 dtype 转换） | 173.66 | 64.29 |
| FlashInfer decode + merge | 533.76 | 372.26 |
| GEMM/GEMV | 1150.91 | 1153.60 |
| 所有 kernel | 2379.26 | 1913.56 |
| graph GPU 时间跨度 | 2401.18 | 1932.80 |

GEMM 近似不变，三个目标环节都有实际改善。最终模型 graph 的跨度接近之前原版 SGLang 的 1907.14 μs；这不是严格同轮对照，不据此宣称完全等效。Profiler 会扰动小 kernel 时间，实际端到端收益使用上面的无 profiler A/B。当前 Rust 调度仍是串行，图外 metadata 和采样回传尚未实现 overlap。

## 文件与回退

源码和文档已同步远程，远程 `target/release/sglang-rust` 已更新，可继续使用原 `tmp.sh`。原版 SGLang 环境没有改动。

远程 `/tmp/codex-kernel-upgrade-backup/` 保存原二进制 `sglang-rust-before`、仅对齐版本 `sglang-rust-fi018`、旧 FlashInfer package/dist-info，以及从原版重新打包的 0.6.18 wheel。旧二进制包含编译好的原生内核，可以直接运行用于回退/对照，无须先还原 Python FlashInfer 包。

原始 JSON、测试日志、trace breakdown 与复测脚本保存于 Git 忽略目录 `benchmark/results/kernel-optimizations-2026-10-08/`；完整 trace 留在远程 `/tmp/codex-kernel-after-trace.nsys-rep`、SQLite 为 `/tmp/codex-kernel-after-trace.sqlite`。所有临时服务使用 GPU 1，验证结束后停止。

## 后续：复用上游内核（2026-10-08）

用户已选择：对齐原版残差数值语义、采用合法保留槽和 GPU 越界断言、移植设备内核并保留 ATen 薄适配层。

- 普通 RMSNorm 和融合残差 RMSNorm：直接调用 FlashInfer 0.6.18 `flashinfer::norm::{RMSNorm,FusedAddRMSNorm}`，传递 ATen current stream 和独立 input/residual 行 stride。满足向量读取对齐要求的 BF16/FP16 走官方实现；无 FlashInfer 或不满足对齐条件的布局走通用 fallback。
- 融合残差语义：归一化使用未舍入的 FP32 残差和，写回 residual 才舍入到 BF16/FP16；CPU 和 CUDA fallback 同步采用此规则。此前自定义 1024 向量化残差内核已删除。两种 FP16/BF16 参考对照更新，并增加能区分旧舍入行为的 fixture。原版已安装 `sgl_kernel.fused_add_rmsnorm` 对该 fixture 返回 `[1.0, 1.0]`、residual `[1.0, 1.0078125]`，与新规则一致。
- QK Norm：移植原版 `fused_qknorm_warp/cta`，支持 head_dim=64/128/256/512/1024；正常 packed QKV 在 head/dim 内连续，token stride 可不同。仍使用原版 warp/CTA norm helper 和 persistent launch 的 block 上限；PDL 在 A100 上关闭。原自定义 128-width warp 内核已删除。其他 head stride 或未对齐输入保留通用 fallback。
- KV store：移植原版 `store_kvcache` 与 warp copy helpers，常用 row bytes 128/256/512/1024/2048/4096/8192 原生实例化，num_split 按原版 2048/1024 对齐选择 4/2/1。K/V source token stride 独立，head/dim 内连续；其他 row width、head stride 或未对齐布局使用原有通用拷贝 fallback。全部 CUDA 路径采用同一非法索引断言和 reserved_skip_index 约定。

`third_party/sglang/` 保存设备内核、所需辅助头文件、原 Apache-2.0 LICENSE 和来源说明。六个辅助文件为原版逐字复制；utils.cuh 仅移除 TVM/DLPack host 内容并提供其基础 include，qknorm/kvcache 只裁去 TensorView host wrapper，设备算法保持原样。构建采用 C++20、上游需要的 relaxed constexpr、目标 SGL_CUDA_ARCH 和启用断言的编译参数。不依赖原版目录或 TVM 运行时。

### 保留槽与接口

更正此前说明：旧 decode graph 已使用私有合法 padding 页；-1 是旧通用写入接口和单元测试支持的行为。新 KV 池的 backing tensor 物理 page 数为 `layout.num_pages + 1`，page 0 初始化为零且不参与分配，真实 page IDs 为 1..=layout.num_pages，可用页数保持既有定义；内存预算扣除保留页。禁止将 page 0 归还空闲池。

Engine 给 model 配置 reserved_write_slot=0；decode graph 的 block table、req_to_token、write_loc padding 全部为 0，不再从池中另外分配 batch-size 个 padding 页。slot 0 写入直接跳过，因此各 padded row 不会互相改写。原版同样允许通过 reserved_skip_index=-1 禁用保留槽；低层直接绑定测试缓存的默认行为对应这一选项。无论是否禁用保留槽，负索引和超过 capacity 的索引都属于错误。

这些槽位与 backing shape 改动已由用户明确选择；Rust 模型 forward 参数和 ATen 指针桥接保持原有体系，仅补充保留槽配置。验证覆盖全容量分配/释放、page 0 初始化与排除、graph padding、packed stride、BF16/FP16、int32/int64 和未写区域。非法 CUDA 索引在独立子进程测试，避免 device assert 污染其余 GPU 测试的上下文。

本节迁移改变了残差数值语义，前文性能和 12/16 文本一致性统计属于迁移前版本，不能直接作为此版本的数据。最新验证结果如下。

### 迁移后的验证与测量

本地 `cargo test --lib`：105 passed。远程 GPU 1 `cargo test --release --lib -- --test-threads=1`：115 passed、1 ignored、0 failed；release 构建成功。完整 Qwen3 API 12 项通过，含 HF golden、流式/非流式、并发、token 限制、工具、reasoning 和非贪心采样。16 个额外 greedy 请求的 usage 全部与最早旧版一致，text 为 12/16 一致，不要求与旧残差语义逐 token 相同。

同前文固定 81 输入/128 输出 workload、每轮 12 请求丢前 2 次，交换运行顺序复测：

| 版本与顺序 | ITL ms | 完整请求 ms |
|---|---:|---:|
| 迁移前优化版，第一轮先运行 | 2.366 | 303.706 |
| 上游内核版，第一轮后运行 | 2.391 | 307.257 |
| 上游内核版，复测先运行 | 2.363 | 302.979 |
| 迁移前优化版，复测后运行 | 2.342 | 300.302 |

这两组配对测量中，上游内核版 ITL 增加约 0.9%–1.1%，完整延迟增加约 0.9%–1.2%。这是一次小样本测量，说明此次迁移并未进一步提速；性能接近上一个自定义优化版本，主要收益是复用官方实现、对齐语义并降低自研内核维护量。不将其宣称为严格性能等价，也没有以此前 smoke 数据作为这次迁移的最新结果。

最新日志、JSON 和移植辅助脚本归档于 `benchmark/results/upstream-kernels-2026-10-08/`。迁移前优化二进制保存为远程 `/tmp/codex-kernel-upgrade-backup/sglang-rust-before-upstream-norm`。远程 release 已更新，tmp.sh 和 benchmark.sh 未修改；临时测试服务结束后停止。

## 环境统一迁移到 uv

根目录 `pyproject.toml` 声明 Torch `2.13.0` 与 Linux CUDA 可选依赖 `flashinfer-python[cu13]==0.6.18`；`uv.lock` 统一锁定传递依赖，原独立 `requirements-flashinfer.txt` 已删除。原锁文件中已有包的版本未变。FlashInfer 由官方 wheel 安装，包含 Python API 所需依赖，替代此前临时重新打包的 wheel。

```bash
# Linux CUDA，构建与测试均使用项目 .venv
uv sync --locked --extra cuda --group dev
uv run --locked --extra cuda --group dev python -m unittest discover -s tests -p test_api_e2e.py -v

# macOS / CPU
uv sync --locked --group dev
```

CUDA 环境后续同步或运行时应继续指定 `--extra cuda`。启动脚本使用同步后的 `.venv`，缺少环境或依赖时提示对应 uv 命令。自定义环境可使用 `UV_PROJECT_ENVIRONMENT=/path/to/venv uv sync ...`，再设置 `VENV_DIR=/path/to/venv` 启动。

远程下载受限，本次从 `uv.lock` 下载 21 个官方 Linux wheel，并在本地与远程逐个校验 SHA256。先使用 `uv pip install` 从临时 wheelhouse 安装锁定版本，再执行 `uv sync --locked --extra cuda --group dev --offline` 做最终精确同步；`uv pip check` 通过，FlashInfer Python 导入与 CUDA 可用性检查通过。官方 wheel 的 241 个 include 文件与之前使用的头文件完全一致。

迁移后验证：本地 `cargo test --lib` 105 项通过；远程 GPU 1 `cargo test --release --lib -- --test-threads=1` 115 项通过、1 项已有微基准忽略；release 构建成功，完整 API 回归 12 项通过。远程 release 二进制已更新，原启动脚本可继续使用。
