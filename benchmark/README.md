# 推理服务 benchmark

这套脚本面向已经启动的 `sglang-rust` OpenAI Chat Completions 服务。三层用途：

| 脚本 | 用途 |
| --- | --- |
| `pr.sh` | OpenAI SDK 接口测试、GSM8K 前 50 题、SGLang 100 请求 × 1/8/32 并发 |
| `sglang_bench/smoke.sh` | 单独运行日常服务性能 smoke |
| `guidellm/release.sh` | 5 个 ISL/OSL × 4 个并发档位的正式性能矩阵 |
| `lm_eval/gsm8k.sh` | 单独运行 GSM8K；`GSM8K_LIMIT=''` 运行全量 |

## 准备

先用适合目标负载的服务配置启动。项目的 `scripts/run-server.sh` 默认限制为 `--max-running-req 4 --max-seq-len 512`；性能测试可设置 `SGLANG_BUILD_PROFILE=release`。Qwen3-0.6B 的 `max_position_embeddings` 为 40960，服务会拒绝更大的 `--max-seq-len`；因此它无法运行包含 64K 输入的完整 release 矩阵。要跑完整矩阵，需要模型配置支持至少约 65808 tokens，并确认可用内存足够。

```bash
SGLANG_BUILD_PROFILE=release ./scripts/run-server.sh --max-running-req 32 --max-seq-len 40960
```

工具建议安装在**独立** Python 环境中；当前仓库的 Python 环境主要用于 PyTorch/libtorch。请按各项目安装文档安装 `sglang`、`guidellm`、`lm_eval[api]`，以及本仓库开发依赖中的 `openai`。`SGLANG_BENCH_PYTHON` 可以指定安装了 SGLang 的 Python；`PROTOCOL_PYTHON` 可以指定装有 `openai` 的 Python（默认优先使用本仓库 `.venv/bin/python`）；其他两个工具由 `PATH` 查找。SGLang 的服务端无需使用 Python 版 SGLang。

```bash
export BENCH_BASE_URL=http://127.0.0.1:8000
export BENCH_MODEL=default
export BENCH_TOKENIZER=/path/to/Qwen3-0.6B
export BENCH_MAX_SEQ_LEN=40960 # 与服务实际 --max-seq-len 一致
```

服务目前没有 `GET /v1/models`，必须显式配置 `BENCH_MODEL`（缺省为服务默认的 `default`）。`BENCH_TOKENIZER` 缺省读取相邻 `mini-sglang/Qwen/Qwen3-0.6B`。三个工具都走 `/v1/chat/completions`；`BENCH_BASE_URL` 只填服务根 URL，不加 `/v1`。

## 启动

```bash
./benchmark/pr.sh
./benchmark/sglang_bench/smoke.sh
./benchmark/guidellm/release.sh
GSM8K_LIMIT='' ./benchmark/lm_eval/gsm8k.sh
```

默认结果写到 `benchmark/results/<时间戳>/`，可用 `BENCH_RESULTS_DIR` 指定路径。`BENCH_DRY_RUN=1` 只打印外部命令。`pr.sh` 的三个阶段依次执行，任一失败即停止；各脚本也可单独运行。

可调参数：`SMOKE_ISL`、`SMOKE_OSL`（默认 64/32）、`SMOKE_REQUESTS`（默认每档 100）、`GSM8K_LIMIT`（默认 50）、`GSM8K_CONCURRENCY`（默认 1）、`RELEASE_REQUESTS`（默认每格 1000）、`RELEASE_TIMEOUT`（默认 3600 秒）。正式矩阵在开始前检查全部 ISL+OSL+16 是否不超过 `BENCH_MAX_SEQ_LEN`；这只是保守的客户端检查，模型本身和 KV cache 容量仍需满足要求。

## 解读结果

GuideLLM JSON/终端统计用于查看 TTFT、TPOT、请求延迟分位数、请求与 token 吞吐及错误数；SGLang 的 JSONL 用于快速回归。固定并发表示**客户端维持的请求流数**，不是服务同时执行的请求数；`max-running-req` 较低时，排队时间会计入端到端延迟。P99 需要足够样本，因此正式矩阵每格默认 1000 请求；100 请求 smoke 仅判断是否有明显退化。

固定模型、tokenizer、硬件、服务参数、缓存状态、工具版本和负载数据，才能横向比较。合成文本适合控制长度和压测，但不代表真实业务提示；后续可增补真实数据集与固定输入种子。GSM8K `--limit 50` 只用于发现明显正确性回归，不能视作正式准确率。当前 chat API 不支持 logprobs，lm-eval 的 loglikelihood 类任务不能直接在这个接口上评估；GSM8K 是生成式任务。Qwen3 的 thinking 模式、最大输出长度及评分提示格式会影响成绩，正式比较时需固定这些设置并检查样例输出。

参考：[SGLang bench_serving](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/base/benchmarks/autoregressive_model_benchmark.mdx)、[GuideLLM benchmark](https://github.com/vllm-project/guidellm/blob/main/docs/getting-started/benchmark.md)、[GuideLLM metrics](https://github.com/vllm-project/guidellm/blob/main/docs/guides/metrics.md)、[lm-evaluation-harness interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)。
