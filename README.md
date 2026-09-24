# 最小 Axum API 服务器

一个使用 Axum 与 libtorch Rust bindings 的 Rust HTTP 服务示例。

## 运行

```bash
cargo run
```

服务默认监听 `http://127.0.0.1:3000`。

在另一个终端请求接口：

```bash
curl http://127.0.0.1:3000/
# Hello from Axum!

curl http://127.0.0.1:3000/health
# ok
```

`controllers::router` 负责注册路由；`hello` 和 `health` 是异步处理函数，返回的字符串会自动成为 HTTP 响应体。

## 校验与 API 文档示例

启动服务后访问 `http://127.0.0.1:3000/swagger-ui/`，可以查看和调用自动生成的 Swagger UI；原始 OpenAPI 文档位于 `http://127.0.0.1:3000/api-docs/openapi.json`。

`POST /users` 使用 `serde` 反序列化 JSON、`validator` 声明字段规则，并用 `axum-valid` 在进入 controller 前自动执行校验：

```bash
curl -X POST http://127.0.0.1:3000/users \
  -H 'content-type: application/json' \
  -d '{"name":"alice","email":"alice@example.com"}'
```

`name` 必须有 3 至 30 个字符，`email` 必须是合法邮箱。响应为 `201 Created`；非法 JSON 或不符合规则的字段会得到 `400 Bad Request`（字段规则错误会以 JSON 返回），且 controller / service 不会执行。

## 目录结构

```text
src/
├── main.rs                         # 启动 HTTP 服务
├── controllers/
│   ├── mod.rs                      # 路由注册
│   └── health_controller.rs         # HTTP 请求处理
└── services/
    ├── mod.rs
    └── health_service.rs            # 业务逻辑
```

controller 只处理 HTTP 层的输入和输出，并调用 service；service 保持与 Axum 解耦，放置业务规则。当前服务没有数据库或外部依赖，因此不额外引入 repository / data-access 层。

## 已迁移的 KV Cache 基础模块

`src/engine/kvcache/` 已包含 Rust 版本的 `BaseCacheHandle`、`CacheManager` trait、`KVCachePool` 与 `KVCacheAllocator`：

- 页表、空闲页栈、内存页数计算由 Rust 管理；
- `KVCachePool::new` 使用 `tch::Tensor::f_empty` 创建 `(2, layers, pages, page_size, kv_heads, head_dim)` Tensor；
- `get_all_kv_cache` 返回 `tch::Tensor` K/V 切片，供之后迁移的 attention 层使用；
- `RadixCacheManager` 已迁移，支持页对齐前缀匹配、共享前缀引用计数、插入回滚、请求释放和页粒度驱逐；
- Naive cache manager、Rust attention layer 绑定，以及加速器空闲显存查询尚未迁移；调用相应入口会返回 `未实现` 错误。

本机使用 `tch-rs` 构建时，需将 `LIBTORCH_USE_PYTORCH=1` 指向含 libtorch 的 Python 环境。`tch 0.26` 的官方目标版本是 PyTorch/libtorch 2.13，本项目当前环境为 2.13.0。macOS 运行时还需设置 `DYLD_LIBRARY_PATH`，让动态链接器找到 libtorch：

```bash
VIRTUAL_ENV=/Users/dp/code/sglang-rust/.venv \
PATH=/Users/dp/code/sglang-rust/.venv/bin:$PATH \
LIBTORCH_USE_PYTORCH=1 \
DYLD_LIBRARY_PATH=/Users/dp/code/sglang-rust/.venv/lib/python3.12/site-packages/torch/lib \
cargo test --lib
```

## Engine

`src/engine/engine.rs` 提供了 `mini-sglang` `Engine` 的 Rust 生命周期骨架：它会校验本地 Hugging Face 模型目录、将 `max_seq_len` 收敛到模型的上下文窗口，并通过已迁移的 `KVCacheAllocator` 创建和释放 libtorch KV Cache。

`ModelRunner` 已迁移为 eager 执行器：通过 `Engine::attach_model_runner` 绑定 Rust `ModelExecutor` 后，`Engine::forward(&batch)` 会在 libtorch `no_grad` 环境中执行 prefill 或 decode。prefill 会传递 `logits_indices`，decode 则始终走 eager 路径；GraphRunner 按当前迁移范围不创建。模型构建、权重加载、scheduler 的 `BatchContext`、CUDA Graph 和分布式张量并行仍未迁移；未绑定 runner 的 `Engine::forward` 或指定 `tp_size > 1` 会返回明确错误。

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

Attention 通过 `ServerArgs::attention_backend` 选择后端，当前默认且唯一可执行的值是 `"pt"`。`"fa"` / `"flashattention"` 已保留为同一抽象的占位后端，调用时会返回未实现错误，便于后续接入 FlashAttention binding 而无需改动 Qwen3 层。

## TokenizerWorker

`src/tokenizer.rs` 使用 Hugging Face 的原生 Rust `tokenizers` crate 加载模型目录中的 `tokenizer.json`，不需要 Python 或 `transformers` 运行时：

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

模型目录中的 `tokenizer_config.json` 若包含字符串形式的 `chat_template`，会以 Jinja 语法渲染，并提供 `messages` 与 `add_generation_prompt`。没有模板时，保持 mini-sglang 原始实现的回退行为：以空行拼接非空消息内容。`trust_remote_code=true` 和非字符串模板配置会明确返回 `未实现` 错误；原生 Rust 加载器不会执行 Python 远程代码。
