# 测试分层与运行

| 位置 | 责任 | 前置条件 |
| --- | --- | --- |
| `src/**` 中的 `#[cfg(test)]` | 模块契约、错误路径、数值回归 | Rust + 当前项目的 PyTorch/libtorch 环境 |
| `tests/layered_state_cache.rs` | 通过公开 factory/runtime 接口验证异构 KV 缓存 | 同上；CUDA 版本单独启用 |
| `tests/e2e/test_protocol.py` | OpenAI 协议、长度限制、usage、流式等价性 | 已启动的服务 + `openai` 开发依赖 |
| `tests/e2e/test_qwen3.py` | Qwen3 模板、thinking、工具调用 | Qwen3-0.6B 模型与对应 tokenizer |
| `tests/e2e/test_concurrency.py` | 并发请求、混合采样、不同输出长度 | 已启动的服务；批处理回归需 `max-running-req >= 2` |
| `tests/fixtures/qwen3_chat_golden.json` | Rust 模板与 Python HTTP 测试共用的模板基准 | 保留上游模板对应语义，不能按错误的实际输出覆盖 expected |
| `benchmark/` | 正确性 smoke 与服务性能评估 | 见 `benchmark/README.md` |

单元测试保留在所属模块旁，跨模块的公开接口验证放在 `tests/`；共享算子的测试归 `src/layers/`，模型测试验证模型组合和权重绑定。CPU 形状测试仍用于接口检查，但缓存正确性须比较非零权重下的数值结果。不要用示例程序的运行成功代替断言。

## Rust

```bash
./scripts/run-rust-tests.sh cpu

# sglang-test 使用与 tmp.sh 相同的 release、虚拟环境和兼容驱动路径
PATH="$HOME/.cargo/bin:$PATH" \
LD_LIBRARY_PATH="/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
SGLANG_BUILD_PROFILE=release ./scripts/run-rust-tests.sh cuda
```

`VENV_DIR` 可选择 Python 环境，`SGLANG_BUILD_PROFILE` 支持 `debug`（默认）和 `release`。脚本不安装依赖、不启动服务，也不需要真实模型权重。CPU 模式隐藏 GPU，运行单元、集成和文档测试。CUDA 模式先检查 GPU 与 native CUDA/FlashInfer 测试是否编入，再单线程运行显式忽略的 GPU 正确性用例；缺失必要能力会失败，不能产生静默跳过后的通过结果。两种模式均须执行，CUDA 模式不重复运行 CPU 用例。

直接运行 `cargo test` 时，GPU 测试显示为 ignored；其编译条件仍由 `build.rs` 决定。显式执行 GPU 用例后，GPU 不可用会触发断言。CUDA RNG 和 graph capture 具有进程级状态，因此 GPU 用例必须使用 `--test-threads=1`。非法 KV 索引测试在子进程内触发设备断言，避免污染其他用例的 CUDA context。

`engine::sampling::tests::benchmark_sampling_cuda` 是手工微基准，继续单独忽略，CUDA 正确性入口会排除它。它对照历史顺序 top-k/top-p 实现，两个采样路径的分布语义不同，耗时比较不能证明数值等价。需要测量时，在上述 CUDA 环境中执行：

```bash
cargo test --locked --release --lib benchmark_sampling_cuda -- --ignored --nocapture --test-threads=1
```

## Python HTTP 测试

安装项目 `dev` 依赖并启动服务后，显式设置地址运行：

```bash
# 在 sglang-test 的仓库内，另一个终端使用 bash tmp.sh 启动 8001 服务
SGLANG_E2E_BASE_URL=http://127.0.0.1:8001/v1 \
  .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v

# 只运行协议或并发测试；对应模块名也可精确到方法
SGLANG_E2E_BASE_URL=http://127.0.0.1:8001/v1 \
  .venv/bin/python -m unittest discover -s tests -p test_protocol.py -v
```

不设置 `SGLANG_E2E_BASE_URL` 时，三个测试类会明确 skip，不会意外请求本机端口。已设置地址时，连接失败会报错。`SGLANG_E2E_MODEL` 默认 `default`。全部测试只做请求，不负责启停服务。

`tmp.sh` 的 `--max-running-req 1` 适合验证排队和请求复用；批处理还应使用独立端口运行 `max-running-req >= 2` 的服务，再跑同一套测试。客户端同时提交请求不保证服务器一定形成特定大小的 batch；精确的行压缩、页映射和 graph padding 由 Rust 用例验证。测试提示较长，建议 `max-seq-len >= 2048`，远端现有 `40960` 配置满足要求。

Qwen3 thinking/工具调用测试依赖 Qwen3-0.6B 的贪心生成行为，换模型或权重后需重新确认；确定性的标签切分和无效工具调用解析由 Rust parser 测试保障。HTTP golden 测试对照聊天接口与原始 completion 的 prompt tokens；模板字符串本身由 Rust golden 测试逐例精确比较。
