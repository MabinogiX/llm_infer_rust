# mini-sglang Rust 服务

Rust 版本的 mini-sglang 推理服务。目前支持本地 dense Qwen3 模型、CPU/CUDA float32 eager 执行，以及 OpenAI 风格的文本与聊天生成接口。

后续 dense、MoE 及不同状态缓存模型的接入计划见 [多模型架构优化与分阶段实施方案](docs/multi-model-architecture-roadmap-2026-10-08.md)。

## 运行

Python 构建与测试环境统一由根目录的 `pyproject.toml` 和 `uv.lock` 管理，默认环境为 `.venv`。先同步依赖：

```bash
# macOS / CPU 环境
uv sync --locked --group dev

# Linux CUDA 环境，包含 FlashInfer 及其完整 Python 依赖
uv sync --locked --extra cuda --group dev
```

CUDA 环境后续执行 `uv sync` 或 `uv run` 时应保留 `--extra cuda`，否则 uv 的精确同步会移除该可选依赖。更新依赖时修改 `pyproject.toml`，执行 `uv lock`，再按对应环境执行上述同步命令。使用自定义 `VENV_DIR` 时，可通过 `UV_PROJECT_ENVIRONMENT=/path/to/venv uv sync ...` 同步同一份锁文件。

本机可以用脚本编译并启动服务，默认构建 debug 版本、读取相邻 `mini-sglang/Qwen/Qwen3-0.6B` 模型：

```bash
./scripts/run-server.sh
```

设置 `SGLANG_BUILD_PROFILE=release` 可构建并运行 `target/release/sglang-rust`；`debug` 对应 `target/debug/sglang-rust`。例如在自己的 `tmp.sh` 中先导出环境变量，再调用同一个脚本：

```bash
export SGLANG_BUILD_PROFILE=release
./scripts/run-server.sh --model-path /path/to/Qwen3-0.6B
```

性能测试时不要设置 `SGLANG_PROFILE_STEPS=1`，它会在阶段边界同步 CUDA。

指定其他模型或端口时，直接传入服务参数；脚本默认设置 `max-running-req=4` 和 `max-seq-len=512`，也可在命令行覆盖：

```bash
./scripts/run-server.sh --model-path /path/to/Qwen3-0.6B --port 8001 --max-seq-len 1024
```

脚本使用项目的 `.venv` 查找 PyTorch/libtorch；需要使用其他虚拟环境时设置 `VENV_DIR`。也可以手动配置下文的 libtorch 环境后运行：

```bash
cargo run -- --model-path /path/to/Qwen3-0.6B --port 8000
```

`main.rs` 解析模型路径和服务参数，调用 `server::serve`。默认监听 `127.0.0.1:8000`，可通过 `--host` 和 `--port` 修改。服务启动时加载模型与 tokenizer；加载失败会退出并显示原因。`--device auto`（默认）在 libtorch 检测到 CUDA 时使用 `cuda:0`，否则使用 CPU；`--device cuda` 要求 CUDA 可用，`--device cpu` 强制使用 CPU。启动日志会打印实际选择的设备。当前仅支持 `tp-size 1`。

Linux CUDA 环境中，模型权重和 KV cache 绑定后会捕获 decode CUDA Graph；图捕获失败的批次回退到 eager。`--cuda-graph-bs N` 设置最大捕获批量（默认使用 `--max-running-req`），设为 `0` 可禁用。构建图桥接层需要与 PyTorch 对应的 CUDA Toolkit；缺少时服务仍使用 eager decode。CPU 不启用图捕获。设置 `SGLANG_PROFILE_STEPS=1` 时，逐步计时会同步设备，因此自动禁用图捕获；性能测试应设为 `0`。

FlashInfer decode 图的实现、前置条件与测量结果见 [CUDA Graph 验证记录](docs/cuda-graph-decode.md)。

Qwen3 的 CUDA BF16/FP16 prefill 在启动时捕获全部配置的分段图，按整批未缓存 token 总数分桶。`--prefill-cuda-graph-max-tokens N` 设置最大桶（默认 2048），设为 `0` 独立禁用 prefill graph；`--cuda-graph-bs 0` 只禁用 decode graph。图从大桶到小桶捕获，完成后服务才开始监听；请求期间不捕获、不淘汰图，超限或缺失的桶使用 eager。启动耗时和显存需求随捕获桶增加，可调低 token 上限。CPU、float32 和逐步同步 profiling 使用 eager prefill。实现和验证见 [Prefill 分段图验证记录](docs/cuda-graph-prefill.md)。

### FlashInfer 分页 decode 与 prefill

参考 SGLang 环境的 FlashInfer 版本固定为 `0.6.18`，由 `pyproject.toml` 的 Linux `cuda` extra 声明，`uv.lock` 锁定全部传递依赖；使用 `uv sync --locked --extra cuda --group dev` 安装。升级头文件后 Cargo 会重新编译原生 adapter。CUDA KV 写入使用单个向量化 kernel 同时写 K/V，支持 int32/int64 位置及 packed QKV stride；普通 RMSNorm 在满足对齐条件时直接复用 FlashInfer 官方内核并传入行 stride；残差 RMSNorm 同样直接复用 FlashInfer；QK Norm 和 KV store 的正常布局移植原版 SGLang 设备内核，保留 ATen 薄适配层，特殊布局使用通用 fallback。上游源码、许可证与移植范围见 `third_party/sglang/README.md`。

在 Linux CUDA 环境中，可以将 `--attention-backend flashinfer` 用于 BF16/FP16、head_dim=128 的模型。构建时需要在 `VENV_DIR` 指向的环境中安装带 CUDA 头文件的 `flashinfer-python` 和与 `tch` 一致的 PyTorch/libtorch。构建脚本从 FlashInfer 包中编译 CUDA 内核；服务运行时不加载 Python。默认编译目标为 A100（sm_80），其他架构可设置 `FLASHINFER_CUDA_ARCH`。FlashInfer 后端的 decode 直接使用现有 KV cache 的分页视图、页表和请求实际长度，prefill 无缓存前缀时使用 FlashInfer ragged KV，有前缀时使用 paged KV，避免 SDPA 的 padding、逐层缓存 gather 和显式注意力 mask。decode 支持完整 CUDA Graph 捕获与回放；prefill 支持在 attention 边界分段的 CUDA Graph，attention 与分页规划仍以 eager 执行。每个捕获批量独立持有地址固定的 FlashInfer plan、页索引与 workspace，回放前在图外更新分页规划。batch padding 行使用永久保留的合法 KV 第 0 页。

```bash
VENV_DIR=/path/to/cuda-venv ./scripts/run-server.sh \
  --model-path /path/to/Qwen3-0.6B \
  --attention-backend flashinfer \
  --max-running-req 4 --max-seq-len 40960
```

这里调用 FlashInfer 官方的 `DecodePlan` 与分页 decode dispatch；每轮 decode 更新分页元数据和规划，供所有模型层复用。工作区由 attention backend 持有，跨 decode step 复用，仅在容量不足时扩容、backend 销毁时释放；前一批仍持有规划时不允许覆盖。FlashInfer 的规划器会按请求长度和 GPU 并行度决定是否拆分 KV。若构建环境缺少 FlashInfer 头文件，服务会在模型初始化时报错，不会静默回退到原先的 SDPA decode。

prefill 调用官方 `PrefillPlan` 与 ragged/paged causal dispatch，规划和工作区同样由 backend 持有、各层共享，按实际调度需求扩容。Q/K/V 使用实际 stride，支持 packed QKV 的非连续视图。CPU 继续使用 eager。`pt` 和 `fa` 的 attention 算法不变；分段图仅捕获模型中的 attention 外计算，本轮 GPU 验证使用 FlashInfer。

CUDA 非贪心采样在构建包含 FlashInfer 时，使用与原版 SGLang 相同的 joint top-k/top-p：在 temperature 缩放后的完整 softmax 分布上同时确定 top-k 与 top-p 截断，再从交集采样，不在两种过滤之间重新归一化。直接调用 FlashInfer `TopKTopPSamplingFromProb`，省去 libtorch topk、阈值 mask、全词表排序、cumsum 和 multinomial；同参数批次直接采样，避免分组拷贝。不限制 top-k 时走 top-p 或普通概率采样内核。并列概率按 FlashInfer 阈值语义处理。随机数来自 libtorch CUDA generator，可用 `tch::Cuda::manual_seed` 复现新实现，但不会与旧采样算法逐 token 一致。CPU 和未编入 FlashInfer 的构建以 libtorch 实现相同 joint 过滤语义。

如果服务器使用 CUDA 兼容驱动，启动前先设置 `export LD_LIBRARY_PATH="/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"`，再运行上述脚本。

诊断请求延迟时，在启动命令前设置 `SGLANG_PROFILE_STEPS=1`。日志会按请求 UID 和输出 token 序号打印 `request step timing`，包含批次准备、模型前向、采样和 KV 状态更新的微秒耗时；`model forward timing` 汇总规划、各 decoder 层、RoPE、归一化、MLP 和 attention backend 等耗时。非流式请求还会打印入口及输出处理耗时。诊断模式会在阶段边界同步 CUDA，可能降低吞吐；正常运行时不设置该变量。

Linux GPU 部署时，`VENV_DIR` 指向的环境需要安装与 `tch` 兼容的 CUDA 版 PyTorch；CPU 版 PyTorch 即使机器有 GPU，也会让 `auto` 选择 CPU。`./scripts/run-server.sh --device cuda` 可用于明确检查 CUDA 是否可用。模型权重加载后，服务通过该环境的 PyTorch 查询剩余显存，再按 `--memory-ratio` 分配 GPU KV cache。

模型专有代码集中在 `src/models/<模型名>/`：`config.rs` 解析和校验模型强类型配置，`model.rs` 实现模型结构及持有配置的 factory，`template.rs` 处理输入模板，`output.rs` 解析输出，`definition.rs` 提供配置解析器、factory 和默认模板／parser 的注册定义。启动时，`src/server/components/builder.rs` 读取 Hugging Face `config.json` 的 `model_type` 和 `architectures`，通过 `src/models/registry.rs` 选择模型，并统一初始化 tokenizer、Engine、Scheduler 和输出解析器。不支持的模型或执行组合会在加载权重前报错。新增模型时实现配置、运行时描述及能力校验并登记，无须在模型目录中复制运行时装配；请求处理过程不切换模型。

共享计算位于 `src/layers/`：bias-free linear、embedding／logits 行选择、`PackedQkv`、`DenseSwiGlu`、RMSNorm／residual RMSNorm／QK norm 和 `HalfSplitRope`，包括对应 CUDA 内核。接口目前只表达 Qwen3 已验证的直接 norm scale、FP32 累积和全维 half-split RoPE 语义，CUDA Q/K 变换会原地更新 projection 视图。模型保留权重命名和绑定、QK norm 的选用、残差组合、attention 调用及 graph 分段。第二个真实模型的接口验证仍待完成。

```bash
curl http://127.0.0.1:8000/health

curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"prompt":"Hello","max_tokens":16}'

curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":16,"stream":true}'
```

`server/manager.rs` 在专用线程中运行 Scheduler，通过通道收发请求和 token；`server/api.rs` 提供普通 JSON 和 SSE 响应。流式连接中断时会取消对应请求。

## 端到端接口测试

完整的 PR smoke 与 release 性能测试入口见 [benchmark/README.md](benchmark/README.md)。

先启动默认 Qwen3 模型服务，再在另一个终端安装测试依赖并运行：

```bash
./scripts/run-server.sh --port 8000
```

```bash
# Linux CUDA 环境；macOS / CPU 环境去掉 --extra cuda
uv sync --locked --extra cuda --group dev
SGLANG_E2E_BASE_URL=http://127.0.0.1:8000/v1 \
  uv run --locked --extra cuda --group dev python -m unittest discover -s tests -p test_api_e2e.py -v
```

聊天接口支持 `max_completion_tokens` 和 `max_tokens`；两者同时提供时优先使用非空的 `max_completion_tokens`。新字段缺省或为 `null` 时使用 `max_tokens`，两者均未指定时默认生成上限为 1024。输出长度包含思考内容对应的 tokens，流式与非流式请求使用相同上限。

可通过 `SGLANG_E2E_MODEL` 指定请求中的模型名。测试需要 Qwen3 tokenizer；它覆盖 `max_tokens`、`max_completion_tokens` 的优先级、`stream`、用量统计、`enable_thinking`、错误请求、思考内容和工具调用，并复用 `tests/fixtures/qwen3_chat_golden.json` 对照聊天模板。

## 日志

服务使用 `tracing` / `tracing-subscriber`，同时输出到终端和按 UTC 日期滚动的 `logs/llm-infer-rust.YYYY-MM-DD.log`。终端输出保留 ANSI 样式，日志文件仅保存纯文本。每行包含 UTC 时间、`service=llm-infer-rust`、`level`、消息及结构化字段，例如：

```text
2026-09-28T06:46:40.039232Z service=llm-infer-rust level=INFO starting model server model_path=/path/to/model log_dir=logs
```

可通过启动参数设置目录和等级：

```bash
./scripts/run-server.sh --log-dir ./logs --log-level debug
```

日志等级优先级为 `--log-level`、`RUST_LOG`、默认 `info`。例如 `RUST_LOG=sglang_rust=debug,info ./scripts/run-server.sh` 可按模块过滤。文件由后台线程写入；队列满时对产生日志的线程施加反压，而非丢弃日志。正常退出时会刷新缓冲；旧日期文件不会自动删除。

## 已迁移的 KV Cache 基础模块

`src/engine/kvcache/` 已包含 Rust 版本的 `BaseCacheHandle`、`CacheManager` trait、`KVCachePool` 与 `KVCacheAllocator`：

- 页表、空闲页栈、内存页数计算由 Rust 管理；
- `KVCachePool::new` 使用 `tch::Tensor::f_empty` 创建 `(2, layers, pages, page_size, kv_heads, head_dim)` Tensor；
- `get_all_kv_cache` 返回 `tch::Tensor` K/V 切片，供之后迁移的 attention 层使用；
- `RadixCacheManager` 已迁移，支持页对齐前缀匹配、共享前缀引用计数、插入回滚、请求释放和页粒度驱逐；
- Naive cache manager 已迁移；CUDA 显存通过启动环境中的 PyTorch 查询。

本机使用 `tch-rs` 构建时，需将 `LIBTORCH_USE_PYTORCH=1` 指向含 libtorch 的 Python 环境。`tch 0.26` 的官方目标版本是 PyTorch/libtorch 2.13，本项目当前环境为 2.13.0。macOS 运行时还需设置 `DYLD_LIBRARY_PATH`，让动态链接器找到 libtorch：

```bash
VIRTUAL_ENV=/Users/dp/code/sglang-rust/.venv \
PATH=/Users/dp/code/sglang-rust/.venv/bin:$PATH \
LIBTORCH_USE_PYTORCH=1 \
DYLD_LIBRARY_PATH=/Users/dp/code/sglang-rust/.venv/lib/python3.12/site-packages/torch/lib \
cargo test --lib
```

## Engine

`src/engine/engine.rs` 提供了 `mini-sglang` `Engine` 的 Rust 生命周期骨架：它会校验本地 Hugging Face 模型目录；当 `--max-seq-len` 超过模型配置的 `max_position_embeddings` 时，启动会报错退出，不再静默截断。Qwen3-0.6B 的配置上限为 40960。通过校验后，已迁移的 `KVCacheAllocator` 才会创建 libtorch KV Cache。

`ModelRunner` 通过 `Engine::attach_model_runner` 绑定 Rust `ModelExecutor`。`Engine::forward(&batch)` 在 libtorch `no_grad` 环境中执行 prefill，传入 `logits_indices`；decode 优先回放已捕获的 CUDA Graph，没有可用图时走 eager 路径。模型构建、权重加载、scheduler 的 `BatchContext` 和 CUDA Graph 已接入；分布式张量并行仍未迁移。未绑定 runner 的 `Engine::forward` 或指定 `tp_size > 1` 会返回明确错误。

`Engine::sample(&logits, &params)` 已接入 Rust `Sampler`。`logits` 为 `(num_reqs, vocab_size)`，`params` 必须有同样数量的 `SamplingParams`；它支持 greedy、temperature、top-k 与 top-p，并将相同采样参数的请求合并为一次 libtorch 调用。

## BatchContext 与权重加载

`BatchContext::prepare_prefill` 会把 scheduler 请求的未缓存 token 拼接为 `Batch`，并生成 `write_loc`、`cu_seqlens_q`、`prefix_lens`、`block_table`、`req_to_token` 与 `logits_indices`。这部分不依赖模型结构，已使用 libtorch Tensor 实现。

## Scheduler 接口

`scheduler::Scheduler` 提供请求提交、取消、空闲判断和 `step` 接口，以及请求状态、输出 token 和终止原因类型。启动模块优先读取 `generation_config.json` 的 `eos_token_id`，字段缺失时读取 `config.json`，将解析后的 EOS 集合显式传给 Scheduler；Scheduler 不再自行读取配置。显式 `null`／`[]` 表示无 EOS，缺失、非整数、负数或越界 ID 会启动失败，不从 tokenizer 配置或 token 0 回退。`step` 先运行 `PrefillManager`，再由 `DecodeManager` 对所有仍在运行的请求执行一次 decode（包括本轮新进入运行状态的请求）。Decode 使用可复用的 libtorch 输入和页表缓冲区；超长或无法容纳的请求返回 `Abort`，前向/采样失败返回 `Error`，具体错误可由 `last_step_error()` 查看。Prefill 失败回滚 radix 插入，decode 失败按已写入的 KV 前缀清理请求。

`load_hf_safetensors` 支持读取 `model.safetensors` 或 `model.safetensors.index.json` 所列的 shards。`Engine::build_model(&factory)` 通过模型层提供的 `ModelFactory` 创建 Rust 模型，随后 `Engine::load_model_weights()` 把实际读取到的具名 Tensor 交给 `ModelExecutor::load_weights` 绑定；未实现绑定的模型会明确报错。

## Qwen3（dense）

`models::Qwen3ForCausalLM` 已实现 dense Qwen3 的 RMSNorm、SwiGLU、QK-RMSNorm、RoPE、GQA 与因果注意力，并按 Hugging Face 标准权重键加载。`Qwen3Factory` 持有 `Qwen3Config`；Engine 在构造模型前检查其层数、KV 几何、词表和上下文与 `RuntimeModelConfig` 一致。通常通过公开的统一启动入口加载模型与服务所需对象：

```rust
use sglang_rust::{
    engine::ServerArgs,
    server::{ServeArgs, build_components},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = ServeArgs::new(ServerArgs::new("/path/to/qwen3"));
    let components = build_components(&args)?;
    let _engine = components.scheduler.engine();
    Ok(())
}
```

缓存声明由 `ModelFactory::cache_spec` 提供：普通 full-history MHA／GQA／MQA 模型可按层声明 KV heads 和 head dimension，运行时把同几何层合并为缓存组，并通过 `ModelExecutor::bind_state_cache` 绑定稳定的逐层视图。各组共用页表，显存预算计入所有组及保留页；支持异构缓存布局不代表已经实现新的模型或 backend。线性 attention／SSM、MLA 与滑窗独立淘汰不在此次实现范围。详见 [分层状态缓存设计](docs/layered-state-cache-design-2026-10-09.md)。

当前 dense Qwen3 支持 eager/分段图 prefill、带缓存前缀的 prefill 和 paged-KV decode；模型接入 `Engine` 时会自动绑定 `KVCachePool` 的逐层 K/V 切片。Qwen3-MoE 与张量并行尚未迁移，调用时会返回明确错误。

Decode 的 KV 映射按请求 ID、已有长度和页号快照增量更新；稳定请求只上传新增 token 的映射，换行、页号变化或序列回退时更新相应区间。CUDA BF16/FP16 的 Qwen3 decoder 通过独立 residual 流使用原位 fused add-RMSNorm，按原版语义使用未舍入的 FP32 残差和计算方差与归一化，写回的 residual 仍舍入到激活类型。KV 池额外预留并清零 page 0，真实请求的 page IDs 从 1 开始；decode graph padding 读取该页并跳过 slot 0 写入。非法 KV 写入索引触发 GPU 断言（CPU 返回错误），不再接受 -1 padding。

Attention 通过 `ServerArgs::attention_backend` 选择后端，默认 `"fa"` 使用 LibTorch 的 scaled dot product attention。Prefill 会把不同长度的请求填充成一个 batch，每层调用一次 SDPA，再还原输出顺序；无缓存前缀时使用 causal mask，让符合条件的 CUDA BF16/FP16 输入可由 LibTorch 选择 fused FlashAttention kernel。带缓存前缀时使用显式掩码，LibTorch 可能选择其他 SDPA kernel。`"pt"` 保留原有 eager 实现。`--dtype auto` 读取模型 `config.json` 中的 `torch_dtype`（或 `dtype`），CPU 推理回退到 float32。

## TokenizerWorker

`src/tokenizer.rs` 使用 Hugging Face 的原生 Rust `tokenizers` crate 加载模型目录中的 `tokenizer.json`，不需要 Python 或 `transformers` 运行时：

Qwen3 聊天模板支持请求中的 `tools` 完整 JSON 定义、消息的 `tool_calls`（OpenAI 嵌套结构或直接结构）、`reasoning_content`、可为空的 `content`、连续 `tool` 响应，以及 `enable_thinking` / `chat_template_kwargs.enable_thinking`。其他消息扩展字段会保留并传给通用 Jinja 模板。

聊天响应会把模型生成的 `<think>...</think>` 解析为 `reasoning_content`，把完整且合法的 `<tool_call>...</tool_call>` JSON 解析为 `tool_calls`。纯工具调用的 `content` 为 `null`；正常停止且包含工具调用时，`finish_reason` 为 `tool_calls`。SSE 将思考、正文和完整工具调用分别放在 `delta` 中，工具调用的 `function.arguments` 是 JSON 字符串。未闭合或无效的工具调用标签保留为正文。`/v1/completions` 继续返回原始生成文本。

`src/server/output/` 定义通用的 `ChatOutputParser` 接口；Qwen3 实现位于 `src/models/qwen3/output.rs`，保存单个请求的解析状态。服务启动时在 `build_components` 中选择一次解析器构造函数；每个请求调用这个固定函数创建独立状态，因此并发流不会混用尚未闭合的标签。新增模型时，在其模型目录中实现解析器并接入组件组装，API 和 SSE 路径无需改动。

```rust
use sglang_rust::tokenizer::{ChatMessage, TokenizerWorker};

let worker = TokenizerWorker::new("/path/to/model", false)?;
let ids = worker.encode("hello")?;
let text = worker.decode(&ids, true)?;
let prompt = worker.apply_chat_template(
    &[ChatMessage::new("user", "hello")],
    true,
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

模型目录中的 `tokenizer_config.json` 若包含字符串形式的 `chat_template`，通用模板会在初始化时编译为 Jinja 模板，并提供 `messages`、`tools`、`add_generation_prompt` 及 `chat_template_kwargs`。Qwen3 组件在启动时显式选择与 Hugging Face 输出对照验证的 Rust 渲染器；独立调用 `TokenizerWorker::new` 时保留一次性的模板自动识别。没有模板时，通用路径以空行拼接非空消息内容。`trust_remote_code=true` 不受支持；通用路径遇到非字符串模板配置会明确报错。原生 Rust 加载器不会执行 Python 远程代码。
