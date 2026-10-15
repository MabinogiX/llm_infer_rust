# 共享 CUDA graph pool：组合验收

日期：2026-10-10；远端 `sglang-test`，GPU 1，A100-SXM4-80GB。前序实现与显存记录见[共享 pool 评估](shared-cuda-graph-pool-assessment-2026-10-10.md)。

## 方法与可比范围

复现脚本：[benchmark-shared-graph-pool.py](../scripts/benchmark-shared-graph-pool.py)。五种配置：阶段二 Rust graph、阶段三 Rust graph（新增 capture 峰值日志）、当前 Rust eager、原版 SGLang 默认 overlap、原版关闭 overlap。每种配置独立启动，前一服务完全停止后再启动下一服务；三轮轮换执行顺序。此处阶段二/三对比隔离非 owning 中间量收益；graph/eager 对比包含前序 graph 与元数据优化，不只代表 pool 的作用。

同一 Qwen3-0.6B 目录、BF16、FlashInfer、GPU、page_size=16、max-running-requests=4、context=40960；Rust memory-ratio=0.4、decode bs=4、prefill limit=2048。原版 mem-fraction-static=0.4，额外限制 max-total-tokens=163904，与 Rust 的 `4 × (40960/16 + 1)` 个 usable pages 对齐，避免相同内存比例产生不同 KV 容量。原版禁用 chunked prefill，使用 full decode / breakable prefill，显式设置与 Rust 相同的 1/2/4 decode 桶及 22 个 prefill token 桶。

原版远端 HEAD 为 `15b256bdb04be6cfdbc56c359af27ffd5abc18fe`，与早期源码研究的本地 HEAD 不同，使用其现有环境而未升级依赖。双方 torch 均为 2.13.0+cu130；本轮运行时 FlashInfer 均为 0.6.18（以本轮环境检查和 metadata 为准）。CPU 实现、attention 版本及调度不同，因此跨实现结果是现有部署参考，不能把全部差异归因于 Rust 语言或共享 pool。测试不进行跨实现输出逐 token 等价声明。

八种 workload：64/512/2048-token 冷 prefill，4096-token 超限 eager 回退（各输出 1 token）；64/128、C=1 的生成；512/64、C=4 与 1024/32、C=4 的生成；1024-token 共享前缀、变更末尾的缓存请求（输出 32）。C 表示客户端并发，不保证每个模型 forward 恰好形成相同大小的 batch。提示以唯一 hash 开头，冷请求避免命中既有 radix prefix；输入使用同一模型 tokenizer 预先核对。全部 temperature=0、ignore_eos=true，流式最终 usage 验证实际 ISL/OSL。长前缀的额外尾部长度随 tokenizer 分词略变，原始输入长度逐请求保存。

每轮每 case 先预热 max(2,C) 次，随后 C=1 计时 12 个请求、C=4 计时 32 个请求；每种配置共 408 个计时请求，15 次启动共 2040 个计时请求。另有每次启动 21 个预热/seed 请求，初始组合验收共 2355 个请求。记录所有计时请求，不挑选最快样本。表中的 TTFT/TPOT 为每轮请求中位数的三轮中位数；吞吐为每轮总 reported output tokens / workload wall time 的三轮中位数，包含客户端与 HTTP 开销。

TTFT 从客户端发送开始到第一个非空文本 chunk；TPOT=(完整 SSE 请求耗时−TTFT)/(reported completion tokens−1)，不是独立 GPU kernel 时间。特殊/空文本 token 与网络 chunk 合并可能影响观察；保存 content_chunks，但不把 chunk 数当作 token 数。冷 prefill 的 1-token 延迟包含 HTTP、prefill、采样和输出。

## 显存口径

Rust 在整个 decode+prefill catalog 捕获之前 reset allocator peak；原版通过临时 `sitecustomize` import hook 包装 `ModelRunner.init_cuda_graphs`，在同一入口 reset/read torch allocator peak。hook 仅存在于本次子进程环境，不修改原版源码。统计 absolute allocated/reserved peak、捕获前基线与差值；这些是 device allocator 计数，不含 FlashInfer 直接 cudaMalloc、CUDA context、graph driver 等非 allocator 存储。

同时每 20 ms 采样 NVML v1，保存完整启动/请求时间序列。该驱动的 v1 `used` 包含约 869 MiB driver-reserved framebuffer，而 `nvidia-smi memory.used` 不含这部分。原始 JSON 保留未扣减的 NVML 值，summary 另外扣除 CLI 报告的 driver-reserved MiB，得到约 1 MiB 精度的驻留近似；不与历史 nvidia-smi 整数值混用。采样 startup peak 仅是启动峰值的下界，不等于 allocator 的精确 capture peak。ready 在 health 就绪时读取，steady 在所有 case 完成后读取，不是最大请求峰值。

## 复现

```sh
cd /llm_infer_rust
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/benchmark-shared-graph-pool.py \
  --output-dir benchmark/results/shared-pool-stage4-2026-10-10 \
  --rounds 3 --modes sglang rust-stage2 rust eager sglang-serial --resume
```

`--resume` 复用已经完整写出的单次启动结果，脚本不会复用半成品结果。旧阶段二二进制为 `/tmp/shared-pool-stage2-bin`，阶段三备份为 `/tmp/shared-pool-stage3-bin`。保存 metadata、workload、逐启动 JSON/server log、NVML 时间序列、汇总及同模式 output hashes；CPU/CUDA 日志位于 `logs/shared-pool-stage4-*-20261010.log`。原始结果位于 Git 忽略的 `benchmark/results/shared-pool-stage4-2026-10-10/`。

## 组合验收结果

以下驻留值为上述 NVML 校正近似，取三轮中位数；包含权重、相同容量的 KV cache 和运行时存储。阶段二和阶段三使用同一 CPU 默认配置，隔离非 owning 输出迁移的影响。

| 配置 | health 就绪 MiB | 请求结束 MiB | 启动秒 |
| --- | ---: | ---: | ---: |
| Rust 阶段二 | 23575 | 23603 | 6.23 |
| Rust 阶段三 | 20821 | 20849 | 6.13 |
| Rust eager | 20543 | 20591 | 5.47 |
| SGLang 默认 overlap | 20638 | 20770 | 38.74 |
| SGLang 关闭 overlap | 20636 | 20770 | 40.83 |

阶段三保留阶段二的输出：408/408 个计时请求内容 SHA-256 完全一致；steady 显存继续保持减少 2754 MiB。生成吞吐基本持平：64/128、C=1 为 419.8→420.0 token/s，512/64、C=4 为 1203.0→1201.1 token/s，1024/32、C=4 为 961.0→957.7 token/s。非 owning 迁移的实测收益是显存，未显示生成吞吐提速。启动秒包含权重加载、Python 导入、依赖初始化等，不能据此解释语言或 graph 本身的速度。

| catalog 统计 | Rust 阶段三 | SGLang 默认 overlap |
| --- | ---: | ---: |
| capture peak allocated bytes | 20393494016 | 20685191680 |
| capture peak reserved bytes | 21143486464 | 20986200064 |
| peak allocated 相对捕获前 MiB | 66.36 | 87.93 |
| peak reserved 相对捕获前 MiB | 126 | 194 |
| 捕获结束 allocated 增量 MiB | 19.12 | 49.32 |
| 捕获结束 reserved 增量 MiB | 126 | 132 |
| capture 耗时 ms（三轮中位数） | 801 | 5354 |

Rust peak/read 日志覆盖一次完整 decode+prefill catalog，而非单个 segment。三轮上述 byte 计数一致；阶段二旧二进制没有 peak 日志，不补造其历史峰值。absolute peak 包含捕获前已分配的权重/KV，增量也不是 pool-only。不同实现的捕获前 allocator cache、attention workspace、graph 初始化和版本路径不同，不能将两列差值全部解释为 pool 策略。SGLang 关闭 overlap 的捕获峰值相同，capture 中位数 5054 ms。

## 发现并修复 CPU 线程引起的延迟尖峰

初始组合基准发现 Rust graph 和 eager 均偶发约 50–60 ms 的短请求 TTFT。以固定 workload 中 24 个 512-token、1-token 输出、串行冷请求构造回归探针，并在同一服务重放作为 cached 对照。旧二进制默认 LibTorch intra-op 线程为 64，cold 最大 TTFT 52.88 ms，cached 最大 65.56 ms；关闭 tokenizer parallelism 后 cold 最大仍有 54.54 ms，未消除症状。只设置 `OMP_NUM_THREADS=1`、保持同一二进制时，cold 最大降至 8.17 ms，cached 最大 5.11 ms，24 个样本均低于探针的 30 ms 阈值。

原版 `model_runner.py` 的 GPU 初始化执行 `torch.set_num_threads(1)`。因此 `Engine::load_for_serving` 在 CUDA serving 且未显式配置 `OMP_NUM_THREADS`/`MKL_NUM_THREADS` 时设置 LibTorch intra-op=1，并记录实际线程数。CPU serving 保持既有配置，显式线程环境变量保持优先；不修改 inter-op 线程。单变量实验证实 CPU 线程配置对症状有因果影响，没有进一步证明具体的 OS 调度或 quota 机制。

最终二进制不设置 OMP 环境变量重跑原探针，cold 最大 7.99 ms、cached 最大 4.68 ms，全部通过。探针是隔离机器上的性能回归工具，30 ms 阈值不作为跨硬件 CI 标准：

```sh
.venv/bin/python scripts/probe-streaming-latency.py \
  --workload benchmark/results/shared-pool-stage4-2026-10-10/workload.json \
  --output benchmark/results/shared-pool-stage4-2026-10-10/latency-fixed-cold512.json
```

调用前须启动空闲服务在 18115；cold 必须 fresh server，重放同一 fixture 是 cached 请求。

## 最终三轮 A/B

另用相同输入、三轮轮换顺序独立启动旧 graph（二进制含峰值日志、CPU 默认 64）、新 graph（CPU 默认 1）、新 eager，共 1224 个计时请求，加预热/seed 共 1413 个请求。旧新 graph 的 408/408 内容 SHA-256 完全一致，usage 均符合目标长度。结果保存在 `benchmark/results/shared-pool-stage4-final-2026-10-10/`。

以下每格为 **TTFT ms / output token/s**；冷请求输出 1 token，因此其 token/s 也等于请求/s。生成场景使用完整请求 wall time 计算吞吐，不能把短请求改善外推到长生成。

| case | 旧 graph | 新 graph | 新 eager |
| --- | ---: | ---: | ---: |
| cold64 | 3.69 / 175.0 | 3.29 / 291.6 | 4.08 / 232.9 |
| cold512 | 8.44 / 60.7 | 7.59 / 129.4 | 9.10 / 109.0 |
| cold2048 | 21.66 / 45.6 | 20.82 / 47.6 | 20.81 / 46.4 |
| fallback4096 | 44.94 / 22.1 | 43.56 / 22.8 | 43.63 / 22.8 |
| generate64c1 | 6.72 / 416.9 | 5.91 / 422.3 | 7.67 / 304.8 |
| generate512c4 | 23.13 / 1192.4 | 21.90 / 1206.4 | 24.20 / 915.4 |
| generate1024c4 | 36.81 / 956.2 | 35.77 / 965.3 | 37.60 / 786.3 |
| prefix1024 | 8.97 / 385.3 | 7.92 / 391.5 | 9.62 / 287.2 |

512-token cold 的 36 个计时样本合并后，P95 TTFT 为 53.44→7.76 ms，最大 58.80→7.94 ms；旧图 6 个样本超过 30 ms，新图 0 个。P95 用排序样本的线性插值，仅 36 个样本，不能视为生产尾延迟 SLA。64-token cold 最大 44.59→3.59 ms。长生成吞吐仅有约 1% 的变化；本轮修复的主要收益是短请求长尾及其串行吞吐。

最终 graph health/steady 仍为 20821/20849 MiB，eager 为 20543/20591 MiB，峰值计数与旧 graph 完全相同。最终 capture 三轮为 803/798/823 ms，中位数 803 ms；没有捕获提速结论。

复现最终 A/B（必须保留旧二进制）：

```sh
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/benchmark-shared-graph-pool.py \
  --output-dir benchmark/results/shared-pool-stage4-final-2026-10-10 \
  --rounds 3 --modes rust-before-threads rust eager --resume
```

旧 graph SHA-256：`8772626c3be7f240d3dc97777266e451a76c782aa58e8177fb0a08157074ae37`，备份 `/tmp/shared-pool-stage4-before-latency-bin`。最终服务 SHA-256：`f0544bbd0189a27f06ef1875c5608fa1c0a19633f2db91b7b947a27f7bb4e37a`。metadata 的 Git HEAD 仅描述基线提交，本轮变更仍在工作区，须结合二进制 hash 和同步源码复现。

## 后续方向与验收

共享 pool 四阶段已完成。现有数据支持保留串行共享 pool 与非 owning 输出契约；下一项值得独立设计和验证的是 Engine CPU/GPU overlap。原版默认/关闭 overlap 的生成 TPOT（64/128、C=1）为 1.883/3.140 ms，512/64、C=4 为 2.483/3.819 ms，1024/32、C=4 为 2.617/3.971 ms；当前 Rust 对应 2.339/3.010/3.106 ms。原版自身的开关实验显示 overlap 的价值，但不能承诺移植后获得相同收益。

overlap 必须保持共享 pool 在同一执行 stream 上串行 replay，并避免未消费的非 owning activation 被覆盖；前一轮 logits/采样结果、CPU 元数据与 GPU event 的所有权和完成边界需要单独设计。scheduler 选择 prefill 或 decode 的次序本身不是共享 pool 冲突；真实并行执行或跨轮保留非 owning 值才会改变当前安全前提。

最终 CPU 124 项、CUDA 18 项、HTTP 12 项通过，固定 cold/cached 性能探针通过。源码、文档、脚本与原始基准本地/远端同步；临时服务退出后 GPU 0/1 各 4 MiB。allocator peak reset 仅在初始化 catalog 使用，device-wide 统计要求验收时没有其他 concurrent capture。该日志不提供运行时每个请求的独立峰值。


## 2026-10-11：review 修复

基准脚本已移除对未选择的阶段二/线程优化前 binary 的强制依赖。完整 `--resume` 直接复用原 metadata/workload 与逐轮结果，不再要求 binary、GPU 或 tokenizer 可用；未完整数据仍检查所需二进制 hash。此前记录的原始 hash 与性能数值保持原口径，本次接口安全契约和回归测试修改没有重新测量吞吐。

非 owning 接口改为显式 unsafe trait，并补充同一 runner 的 decode→prefill→prefill→decode KV/logits/保留输出回归；接口约束和测试方法见[review 修复记录](execution-cuda-graph-design-2026-10-10.md)。
