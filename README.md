# mini-sglang Rust 服务

Rust 版本的 mini-sglang 推理服务。目前支持本地 dense Qwen3 模型、CPU/CUDA float32 eager 执行，以及 OpenAI 风格的文本与聊天生成接口。

## 运行

本机可以用脚本编译 debug 版本并启动服务，默认读取相邻 `mini-sglang/Qwen/Qwen3-0.6B` 模型：

```bash
./scripts/run-debug.sh
```

指定其他模型或端口时，直接传入服务参数；脚本默认设置 `max-running-req=4` 和 `max-seq-len=512`，也可在命令行覆盖：

```bash
./scripts/run-debug.sh --model-path /path/to/Qwen3-0.6B --port 8001 --max-seq-len 1024
```

脚本使用项目的 `.venv` 查找 PyTorch/libtorch；需要使用其他虚拟环境时设置 `VENV_DIR`。也可以手动配置下文的 libtorch 环境后运行：

```bash
cargo run -- --model-path /path/to/Qwen3-0.6B --port 8000
```

`main.rs` 解析模型路径和服务参数，调用 `server::serve`。默认监听 `127.0.0.1:8000`，可通过 `--host` 和 `--port` 修改。服务启动时加载模型与 tokenizer；加载失败会退出并显示原因。`--device auto`（默认）在 libtorch 检测到 CUDA 时使用 `cuda:0`，否则使用 CPU；`--device cuda` 要求 CUDA 可用，`--device cpu` 强制使用 CPU。启动日志会打印实际选择的设备。当前仅支持 float32、`pt` attention 和 `tp-size 1`。

Linux CUDA 环境中，模型权重和 KV cache 绑定后会捕获 decode CUDA Graph；图捕获失败的批次回退到 eager。`--cuda-graph-bs N` 设置最大捕获批量（默认使用 `--max-running-req`），设为 `0` 可禁用。构建图桥接层需要与 PyTorch 对应的 CUDA Toolkit；缺少时服务仍使用 eager decode。CPU 不启用图捕获。

Linux GPU 部署时，`VENV_DIR` 指向的环境需要安装与 `tch` 兼容的 CUDA 版 PyTorch；CPU 版 PyTorch 即使机器有 GPU，也会让 `auto` 选择 CPU。`./scripts/run-debug.sh --device cuda` 可用于明确检查 CUDA 是否可用。模型权重加载后，服务通过该环境的 PyTorch 查询剩余显存，再按 `--memory-ratio` 分配 GPU KV cache。

模型专有代码集中在 `src/models/<模型名>/`：`model.rs` 实现模型结构，`template.rs` 处理输入模板，`output.rs` 解析输出，`components.rs` 组装 tokenizer、engine、scheduler 和输出解析器。启动时，`src/server/components/builder.rs` 读取 Hugging Face `config.json` 的 `model_type` 和 `architectures`，通过 `src/models/registry.rs` 选择一次模型；不支持的模型会在加载权重前报错。新增模型时只需在 `src/models/` 下增加目录并登记，无须修改服务端和 tokenizer 的模型分支；请求处理过程不切换模型。

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
./scripts/run-debug.sh --port 8000
```

```bash
uv sync --locked --group dev
SGLANG_E2E_BASE_URL=http://127.0.0.1:8000/v1 \
  uv run --locked python -m unittest discover -s tests -p test_api_e2e.py -v
```

可通过 `SGLANG_E2E_MODEL` 指定请求中的模型名。测试需要 Qwen3 tokenizer；它覆盖 `max_tokens`、`stream`、用量统计、`enable_thinking`、错误请求、思考内容和工具调用，并复用 `tests/fixtures/qwen3_chat_golden.json` 对照聊天模板。

## 日志

服务使用 `tracing` / `tracing-subscriber`，同时输出到终端和按 UTC 日期滚动的 `logs/llm-infer-rust.YYYY-MM-DD.log`。终端输出保留 ANSI 样式，日志文件仅保存纯文本。每行包含 UTC 时间、`service=llm-infer-rust`、`level`、消息及结构化字段，例如：

```text
2026-09-28T06:46:40.039232Z service=llm-infer-rust level=INFO starting model server model_path=/path/to/model log_dir=logs
```

可通过启动参数设置目录和等级：

```bash
./scripts/run-debug.sh --log-dir ./logs --log-level debug
```

日志等级优先级为 `--log-level`、`RUST_LOG`、默认 `info`。例如 `RUST_LOG=sglang_rust=debug,info ./scripts/run-debug.sh` 可按模块过滤。文件由后台线程写入；队列满时对产生日志的线程施加反压，而非丢弃日志。正常退出时会刷新缓冲；旧日期文件不会自动删除。

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

`scheduler::Scheduler` 提供请求提交、取消、空闲判断和 `step` 接口，以及请求状态、输出 token 和终止原因类型。它从模型配置按 generation、tokenizer、model 的顺序加载 EOS ID。`step` 先运行 `PrefillManager`，再由 `DecodeManager` 对所有仍在运行的请求执行一次 decode（包括本轮新进入运行状态的请求）。Decode 使用可复用的 libtorch 输入和页表缓冲区；超长或无法容纳的请求返回 `Abort`，前向/采样失败返回 `Error`，具体错误可由 `last_step_error()` 查看。Prefill 失败回滚 radix 插入，decode 失败按已写入的 KV 前缀清理请求。

`load_hf_safetensors` 支持读取 `model.safetensors` 或 `model.safetensors.index.json` 所列的 shards。`Engine::build_model(&factory)` 通过模型层提供的 `ModelFactory` 创建 Rust 模型，随后 `Engine::load_model_weights()` 把实际读取到的具名 Tensor 交给 `ModelExecutor::load_weights` 绑定；未实现绑定的模型会明确报错。

## Qwen3（dense）

`models::Qwen3Factory` 已实现 dense Qwen3 的 RMSNorm、SwiGLU、QK-RMSNorm、RoPE、GQA 与因果注意力，并按 Hugging Face 标准权重键加载：

```rust
use sglang_rust::{
    engine::{Engine, ModelArgs, ServerArgs},
    models::Qwen3Factory,
};

let model_path = "/path/to/qwen3";
let model_args = ModelArgs::from_pretrained(model_path)?;
let mut engine = Engine::new(ServerArgs::new(model_path), model_args, 0)?;
engine.build_model(&Qwen3Factory)?;
engine.load_model_weights()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

当前 dense Qwen3 支持 eager prefill、带缓存前缀的 prefill 和 paged-KV decode；模型接入 `Engine` 时会自动绑定 `KVCachePool` 的逐层 K/V 切片。Qwen3-MoE 与张量并行尚未迁移，调用时会返回明确错误。

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
