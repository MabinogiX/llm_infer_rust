# 多模型 LLM 架构优化与分阶段实施方案

日期：2026-10-08。本文记录架构分析与后续计划，不代表相关能力已实现。

分析基于本地 `sglang-rust` HEAD `c21fde4` 与原版 `sglang` HEAD `ac656ee79a` 的当前源码。远程 Rust 项目为 `sglang-test:/sjtu/yaosikai/llm_infer_rust`。本文中的原版路径指 `sglang/python/sglang/srt/`，主要参考 Python SRT 的模块职责与接口设计。

## 1. 目标与待确认事项

用户计划继续使用本项目接入不同的语言模型，需要同时覆盖 dense、MoE，以及目标模型实际采用的不同 attention／状态缓存模块。当前只考虑 LLM 语言推理入口。

用户提到的目标为“Qwen 27B”和“GLM5.3 Flash”。两者的完整模型标识、版本、checkpoint 目录和配置尚未确认，不据简称推断其具体模型结构、量化格式或硬件需求。“自带多模块”究竟指多模态模块还是内部不同计算模块，也待确认。

在阶段 0 获取实际配置后，再确定必须实现的功能。如果模型包含多模态模块，先明确文本推理依赖哪些模块和权重，以及 checkpoint 能否按文本路径加载；不能直接假定视觉等权重均可省略。

第一阶段保持每个 Runtime 加载一个模型。支持多个模型架构、按启动参数切换 checkpoint，与同进程同时承载多个模型是不同的工作范围。未来如需同进程多模型，由前端把 served model name 映射到独立 Runtime，各自管理权重、状态缓存、graph 和显存预算。

## 2. 当前架构与原版参考

Rust 已具备 HTTP、FrontendManager、Scheduler、Engine、ModelRunner、ModelExecutor、attention backend 和缓存管理模块。模型计算不直接处理 HTTP 请求，启动时也已有模型注册机制，可以在此基础上演进。

原版 Python SRT 的主要链路为：

```text
HTTP / OpenAI 协议
  → TokenizerManager
  → Scheduler
  → ModelRunner
  → Model + AttentionBackend
  → DetokenizerManager
  → 前端响应
```

原版使用进程和 IPC 分离部分职责；Rust 当前使用线程和通道。参考重点是模块职责、状态所有权和执行约定。现阶段继续使用线程与通道，是否引入进程隔离取决于后续故障恢复和部署需求。

| 模块 | Rust 当前情况 | 原版可参考的设计 | 优化方向 |
|---|---|---|---|
| 模型注册 | 注册项构造 Scheduler、Tokenizer 和 OutputParser | Registry 选择模型实现，运行时负责加载和执行 | 注册提供模型定义，启动流程集中组装 |
| 模型配置 | 通用 ModelArgs 包含 Qwen 风格计算字段与缓存尺寸 | 保留 HF 配置并提取运行时信息 | 每模型强类型配置 + 公共运行时描述 |
| 基础算子 | Qwen3 已使用共享 layers；第二模型验证待完成 | layers 共享算子，模型组合这些算子 | 提取共享 layers，保留模型数值差异 |
| 状态缓存 | 已支持逐层声明与按几何分组的 full-history 分页 K/V | 不同 attention 与状态缓存实现 | 模型声明分层缓存需求，运行时管理资源 |
| 模型执行 | 已实现带模式约束的 ForwardBatch／ForwardOutput | ForwardBatch 与结构化执行输出 | 统一批次输入和输出语义 |
| CUDA graph | decode／分段 prefill 管理由 Runner 持有，模型提供计算段 | runner 与 backend 分工，明确图内／图外操作 | graph 生命周期通用化，模型提供计算段 |
| 聊天格式 | Qwen3 使用固定自定义模板与输出 parser | 模板、reasoning parser、tool parser 分别配置 | 独立 ChatProfile |
| 权重加载 | 全量收集 checkpoint，随后模型绑定与 packing | 迭代读取权重，模型负责名称映射 | 逐 shard／tensor 消费并最终校验 |
| 输出处理 | 每步重新解码全部累计 tokens | 维护增量解码位置，并支持批量处理 | 增量解码与统一输出事件 |
| 调度 | 同步运行 prefill 后再 decode | token budget、chunked prefill、overlap | 先整理批次与状态约定，再逐步优化 |

## 3. 需要调整的接口

### 3.1 模型注册与配置

当前 `models/registry.rs` 中注册项的 builder 返回 `ServeComponents`，模型模块依赖 `server::ServeArgs` 并组装运行时。新增模型容易重复 tokenizer 初始化、权重加载、缓存分配和 graph capture 流程。

建议注册项提供模型定义：配置解析器、模型 factory、公共能力描述，以及可选的默认 ChatProfile。通用启动模块统一完成以下顺序：

1. 读取模型标识与配置，选择模型实现。
2. 校验模型、dtype、backend、量化和并行设置的组合。
3. 加载 tokenizer／聊天配置并构造模型、绑定权重。
4. 根据模型状态需求和剩余显存分配缓存。
5. 初始化 backend 与执行资源，执行支持的 graph capture。
6. 创建 Scheduler 并发布就绪状态。

每个模型拥有自己的强类型配置，运行时只获取公共信息，例如词表、有效上下文上限、状态缓存需求和支持能力。不要把未来所有架构字段都塞进 `ModelArgs`，也不要在每步 forward 解析 JSON。

当前配置未表达 RoPE scaling、projection bias、滑窗和逐层不同的状态需求。配置中出现尚未支持的计算特性时，应启动失败并说明原因，避免忽略字段后输出错误结果。公共默认值与模型专属默认值分别定义。

EOS、特殊 token 和生成默认设置由统一模型／tokenizer 初始化流程读取并传给 Scheduler。当前 Scheduler 自行读取配置，缺失或无法得到 EOS 时回退到 token 0；后续应区分明确的“无 EOS”与无效配置，不能把 token 0 当作通用模型约定。

### 3.2 共享 layers 与模型结构

提取 RMSNorm／residual RMSNorm、RoPE、embedding、线性投影、packed QKV、packed gate-up、dense MLP 和 logits 处理中的可复用实现。

模型实现负责组合算子并选择正确语义，包括是否存在 QK norm、bias、RoPE 变体、激活和输出处理。共享实现仍需表达数值差异；例如不能因为同名 RMSNorm 就假定参数和计算语义完全相同。

Dense FFN 与 MoE FFN 可以在 decoder 组合处替换。MoE 的 routing、专家权重布局和计算策略应由专门模块承载，避免把 MoE 逻辑散落到通用 Scheduler。

第二个真实模型的完整名称与 checkpoint 尚未确认。经用户确认，先完成 Qwen3 共享层提取与回归，第二模型验证保留为待办；在其接入前不把现有接口视为覆盖所有模型的稳定公共 API。避免通过几十个开关构造覆盖所有架构的通用 Transformer，也避免给每个标量算子增加热路径动态分发。

### 3.3 分层状态缓存

已于 2026-10-09 实现通用 full-history 分页 KV 的分层声明与分组存储：`ModelFactory::cache_spec` 声明每层 KV 几何，`KVCachePool` 按相同几何合并层，`ModelExecutor::bind_state_cache` 接收逐层稳定视图。各组共用逻辑页表与生命周期，支持层间不同的 KV heads／head dimension；Qwen3 已迁移到新接口。详见 [分层状态缓存设计与实现](layered-state-cache-design-2026-10-09.md)。

本次支持范围为普通 MHA／GQA／MQA 的 full-history 分页 K/V；FFN 为 dense 或 MoE 不改变其缓存约定，但没有新增 MoE 模型计算支持。线性 attention／SSM、MLA、滑窗独立淘汰和量化 KV 不纳入本次实现，不以普通 KV 接口冒充这些状态。
目标是由模型声明每层／每组需要的状态，由 Runtime 负责显存预算、分配与释放，模型通过稳定句柄或逐层视图访问状态。具体类型由目标模型配置决定；现有分页 K/V 实现应作为一种受支持的缓存实现保留。

必须明确以下生命周期约定：

- 请求创建、扩容、结束、取消及故障清理。
- 前缀命中时可以复用哪些状态；哪些状态需要复制、重算或禁止共享。
- 分支／fork 的状态隔离，避免修改共享可变状态。
- chunked prefill 的中间状态发布与回退。
- graph padding 对保留槽位和其他状态的处理，不能污染真实请求或共享状态。
- state／weight／backend 地址发生变化时，相关 graph 的失效与释放顺序。

不同状态未必适合统一成分页 KV 或复用同一种 radix cache。以真实目标模型验证状态接口，尚不支持的前缀缓存组合应明确禁用。

### 3.4 执行与 CUDA graph

已实现模型以统一 `ForwardBatch` 接收输入，返回 `ForwardOutput`（当前只有 logits），详见 [执行与 CUDA graph 设计](execution-cuda-graph-design-2026-10-10.md)。保留 Scheduler 批次与模型执行批次的职责区分，在执行入口完成参数准备和校验。对 prefill／decode 使用有明确约束的输入，减少大量 Option 字段产生的非法组合和重复 phase 信息。

ForwardOutput 初期以生成所需 logits 为核心，后续按真实需求增加 hidden states、logprobs 等输出。embedding、多模态输入和 speculative decoding 暂不为占位而加入接口。

已落实的 graph 管理职责如下：

| 负责方 | 职责 |
|---|---|
| Runner／执行管理 | 桶选择、固定缓冲区、捕获、回放、失效与释放 |
| 模型 | 定义计算结构，提供经过验证的可捕获计算段 |
| Attention／状态 backend | 准备计划与元数据，声明图内和图外操作及 padding 约定 |

已将 prefill 的桶目录、固定缓冲区、捕获、回放与失效释放移到通用 `SegmentedGraphRunner`。Qwen3 通过 `PrefillGraphProgram` 提供计算段和图外 attention；通用管理器不访问 decoder layers。另一个三段算术测试程序验证接口不依赖 Transformer 层结构，第二个实际模型的验证仍待后续接入。

graph 已使用按模式划分的 `GraphCapabilities` 显式声明，默认不启用；声明根据实际模型、backend、dtype 和 KV 绑定状态产生，并携带请求数／token／上下文与 padding 约束。未验证的组合回退 eager，启动日志说明能力拒绝原因，动态形状回退记录 debug 原因。

原版 attention backend 将元数据处理区分为图外准备与图内静态形状操作，可以参考这一契约。保留已实现的启动时捕获策略和 graph 生命周期约定。

当前 FlashInfer adapter 限制 `head_dim=128`。其他 head dimension 需要扩展原生实例化与测试，不能只改 Rust 配置检查。遇到内核接口或布局不兼容时，先整理差异和修改选项，再由用户判断。

### 3.5 ChatProfile 与输出事件

聊天模板、思考格式、工具格式和特殊 token 不应仅由模型计算架构决定。同一 architecture 的不同 checkpoint 可能采用不同聊天协议。

独立 ChatProfile 提供这些格式及 parser 配置，支持模型目录配置与明确的用户 override。当前 Qwen3 自定义 renderer 保留为已验证实现；通用模板兼容性经 golden 测试确认后逐步启用，避免直接替换导致工具和思考格式变化。

ChatOutputParser 应输出强类型的内容、思考和工具调用事件，由协议模块生成 OpenAI JSON／SSE。普通响应和流式响应共用解析逻辑，并保持跨 token 标记、UTF-8 边界和结束刷新的语义一致。

### 3.6 权重加载

当前 checkpoint 全量读入 CPU 内存；模型构造时预分配零权重，packed 参数绑定时还有临时 Tensor。大模型启动时需重点控制内存峰值、零初始化和复制成本。

建议拆为：

- checkpoint reader：发现 shard、读取索引，逐步提供带名称的 tensor。
- 模型权重映射：处理命名、packing、共享 embedding、量化或专家布局。
- 加载完成校验：检查缺失、重复、shape、dtype，以及允许忽略的权重。

可以先以逐 shard 加载降低峰值，再演进到逐 tensor／mmap 或受控预取。加载失败时应清理未完成资源；完成校验前不能对外发布就绪或捕获模型 graph。MoE 和量化格式的布局支持由实际 checkpoint 决定。

## 4. 服务接口与运行时优化

### 4.1 请求与错误

当前 HTTP 请求的 `model` 用于回显，没有参与模型选择或校验。先增加 served model name／alias 的校验，响应使用已解析的模型标识；未来再扩展到多 Runtime 路由。

在进入 Scheduler 前统一做参数和能力校验。当前非法 top-p 和生成长度等会被静默修正，应明确区分默认值、合法规范化和错误请求；行为变化需同步接口测试和兼容性说明。

建立强类型内部请求／输出／错误事件。保留取消、过载、无效输入、执行失败等原因，再映射到 HTTP 和 SSE。当前 API 的 `detail` 错误格式与 Scheduler 输出中的 Error 标记不足以表达完整错误信息。

### 4.2 就绪、容量与 CPU 工作

- 区分进程存活与模型运行时就绪，反映 worker 退出、初始化失败和执行故障；当前 `/health` 固定成功。
- 限制等待队列和输出缓冲，定义过载与慢客户端策略。不能简单用阻塞发送让一个慢客户端卡住所有 GPU 请求。
- 为模板、tokenization 和输出处理提供受控的 CPU 执行资源。当前同步 CPU 操作直接运行在 async handler 中，后续用长 prompt 与并发场景测量是否需要 offload 或 batching。
- 改进增量 detokenization。当前每 token 重新解码完整累计序列，总工作量随输出长度显著增加；需保留特殊 token 和不完整 UTF-8 的正确性。
- 统一可配置的排队、首 token、token 间隔或总请求时限。当前超时主要围绕每次等待 token，应明确其含义与取消清理规则。

### 4.3 调度与性能

当前同步 step 先运行选中的 prefill，再运行 decode。后续增加独立的 prefill token budget 和 chunked prefill，减少长 prompt 对已有 decode 请求的干扰。中间 chunk 的模型执行、状态发布和是否产生 logits／采样需要明确约定。

CPU 调度与 GPU 执行 overlap 放在状态和批次所有权明确之后实施，需要完成事件、缓冲区生命周期和结果发布顺序约定。参考原版设计，但收益使用本项目的交替 A/B 测量验证。

## 5. 分阶段实施与验收

各阶段可独立评审；实施时更新本节状态并记录实际模型、测试和测量结果。每阶段继续保留已验证的 Qwen3 路径。

| 阶段 | 状态 | 主要产物 | 验收条件 |
|---|---|---|---|
| 0：目标模型确认 | 待实施 | 完整模型名／目录、配置和权重结构；功能与资源矩阵 | 明确 attention、状态、dense／MoE、量化、上下文、文本路径和硬件需求；列出支持／缺失／待验证项 |
| 1：注册与配置 | 已实现；本地 CPU 与远端 Qwen3／CUDA 验证通过 | 模型定义注册、每模型配置、公共能力与运行时描述；统一启动流程 | Qwen3 保持可运行；未知 architecture 和不支持配置在加载前报错；模型模块不再组装 Scheduler 和服务对象 |
| 2：共享算子与加载 | 部分实施：Qwen3 共享层提取完成；第二模型与逐步加载待办 | 共享 layers、逐步加载与权重映射接口 | Qwen3 数值与 API 回归通过；测试覆盖 packed／tied 权重和加载错误；记录启动时间与 CPU/GPU 内存峰值 |
| 3：执行与状态缓存 | 分层 full-history 分页 KV 与执行输入输出已实现；其他状态待实施 | ForwardBatch／ForwardOutput、分层状态需求与生命周期 | 现有分页 KV 路径通过；按目标模型验证所需状态的取消、前缀复用和中间 chunk 行为；无资源泄漏或共享状态污染 |
| 4：graph 接口整理 | 已实现通用管理、能力和执行接口；第二实际模型验证待办 | 通用 graph 管理、模型计算段和 backend 元数据契约 | Qwen3 eager／decode graph／prefill graph 对照通过；不同桶、padding、前缀命中、失效与回退正常；未验证组合不启用 |
| 5：目标模型接入 | 待实施 | 实际 dense／MoE／其他所需模块的模型实现与注册 | 对齐可信参考的模板、logits／数值容差、生成和用量语义；多批次与不同长度测试通过；记录支持限制与性能 |
| 6：协议与运行时完善 | 待实施 | ChatProfile、统一事件、模型名／参数校验、readiness、容量控制和增量解码 | 普通／流式响应一致；思考与工具格式、UTF-8、断连、超时、过载和故障测试通过 |
| 7：调度与性能 | 待实施 | token budget、chunked prefill、CPU offload／batching、overlap | 长 prompt 与 decode 混合场景正确；确认无状态／缓存发布竞态；按模型和 workload 报告可重复 A/B 结果 |

阶段 2～4 可以穿插接入第二个真实模型，用它验证接口；具体模型由阶段 0 的配置和可用权重决定。阶段 6 中模型名／参数校验和 readiness 可独立提前实施。不要等全部抽象完成后才验证第二个 Adapter。

纯接口迁移先保持既有数值和服务行为。需要更改参数校验、模板、数值语义、槽位或布局时，分别记录原因和兼容性影响。优先复用 FlashInfer 或原版已验证内核；接口、布局不兼容的选择先告知用户。性能检查在有实际执行变化或退化疑点时进行，不把架构重构本身当作提速证据。

### 5.1 阶段 1 实施记录（2026-10-09）

本次确认注册与配置分离的设计合理，并完成以下调整：

- `models/registry.rs` 注册配置解析器、模型 factory 和现有模板／输出 parser 的默认选择；不再接收 `ServeArgs` 或返回 `ServeComponents`。Qwen3 的定义移到 `models/qwen3/definition.rs`，模型模块不再创建 Engine、Scheduler 或 tokenizer。
- `Qwen3Config` 保存计算字段并在启动时强类型解析、校验。通用 `ModelArgs` 移除，`RuntimeModelConfig` 只保存当前分页 KV 几何、词表、上下文上限及已解析的 checkpoint dtype。`ModelFactory` 持有模型专属配置，通过 `validate_runtime` 校验实际执行组合，再通过 `create` 构造模型。
- `server/components/builder.rs` 统一选择定义、解析配置、校验执行组合和 EOS、初始化 tokenizer、加载 Engine 并创建 Scheduler。权重加载、剩余显存缓存分配和 graph capture 的原有顺序仍由 Engine 执行。模型配置不再由 Engine 或 Scheduler 重读。
- 未知 architecture、非法维度／dtype／backend／并行配置，以及未支持的 RoPE scaling、量化、projection bias、滑窗、非 SiLU 激活和逐层 attention 类型，在 tokenizer／权重加载之前失败。停用滑窗时保留其窗口元数据，普通 HF 元数据允许存在；这不是对任意未知计算扩展的兼容承诺。
- EOS 策略已由用户确认：优先取 `generation_config.json` 的 `eos_token_id`，该字段缺失时使用 `config.json`；显式 `null`／`[]` 表示无 EOS。缺失、非整数、负数或超出词表的 ID 启动报错。移除 tokenizer 配置中的 ID 回退和 token 0 的通用 EOS 回退。Scheduler 接收已解析的 EOS 集合，无 EOS 模型仍按长度结束并支持取消／错误清理。
- 未显式声明 graph 支持的 `ModelExecutor` 默认禁用 decode graph；Qwen3 保留现有 backend 声明和 prefill capture 路径。

本阶段采用两个范围收敛：当前公共状态描述只表达已实现的统一分页 KV，不提前设计分层状态类型；能力通过实际的模型／dtype／device／backend 组合校验表达，完整 graph 模式／形状能力描述留在阶段 4。ChatProfile 的独立 override 和通用模板迁移仍属于阶段 6，当前保留 Qwen3 renderer 和 parser。模型／生成配置以外的请求默认采样参数保持现状。

兼容性变化：不完整的模型维度配置、此前被静默忽略的不支持特性、无效 EOS 不再被接受。缺省的上下文、RoPE theta、norm epsilon 和 tied embedding 继续保留本项目原有默认值；显式非法类型不再静默取默认值。Rust 构造接口相应迁移到 `Qwen3Config`、已配置的 factory、`RuntimeModelConfig` 及显式 EOS 参数。

本地 macOS／CPU 验证：`cargo test --all-targets --no-fail-fast`，109 项测试通过。覆盖配置拒绝、EOS 来源优先级与严格校验、统一启动加载微型 dense Qwen3 safetensors checkpoint、prefill／decode 生成、无 EOS 时 token 0 不误停及 abort；原有 Qwen3 前向、权重绑定、模板／parser、调度和缓存测试通过。

按用户指定的工作方式，在本地开发并同步至 `sglang-test:/sjtu/yaosikai/llm_infer_rust`，以该远端代码和测试结果为验收依据。两端基于同一提交 `da6c5c1`，同步文件通过 SHA-256 校验，远端 `tmp.sh`／benchmark 脚本保留。

远端 A100 GPU 1 验证：`cargo test --release --all-targets -- --test-threads=1`，119 项通过、1 项已有采样微基准默认忽略；release 服务二进制已更新。首次 CUDA 回归暴露测试执行器未声明 graph 支持，已在本地补齐其 backend 能力转发并同步，完整回归通过。真实 `/sjtu/yaosikai/Qwen3-0.6B`、BF16、FlashInfer、max-running-req=4、max-seq-len=40960，eager（decode/prefill graph 均关闭）和 graph（decode 上限 4、prefill token 上限 2048）各 12 项 HTTP 回归通过；8 组含不同长度和重复前缀的贪心请求，文本、usage 和 finish_reason 完全一致。已有 CUDA kernel／graph 数值回归随 Rust 测试通过，服务日志确认 decode 与 segmented prefill graph 捕获。临时服务均已退出。

远端记录保存在项目 `logs/registration-config-tests-20261009.log`、`logs/registration-config-build-20261009.log`、`logs/registration-config-http-summary-20261009.log` 和 `logs/registration-config-eager-graph-comparison-20261009.json`，两种模式各自的服务／API 日志也位于 `logs/`。本次未做性能测量，重构不作为推理提速证据。

### 5.2 阶段 1 审查修复（2026-10-09）

经多 agent 独立审查及交叉验证，用户授权修复 R1–R4，详情见 [审查报告及修复记录](model-registration-config-review-2026-10-09.md)。factory 的运行时校验现在对照公共描述与模型强配置，Engine 的加载／构造入口提前拒绝不一致组合；缓存配置映射共用一个私有函数；Qwen3 固定默认值由 serde 与程序构造共用；README 的装配职责、EOS 说明和公开构造示例已更新。未增加新的配置层或已校验加载入口，缓存仍在权重加载后分配。

修复在本地开发并同步远端。本地 111 项 Rust 测试通过，README Rust 示例编译检查通过；远端 GPU 1 release 全目标测试 121 项通过、1 项已有微基准忽略，release 二进制更新。真实 Qwen3 eager／CUDA graph 各 12 项 HTTP 回归通过，8 组贪心 text／usage／finish_reason 对照一致，临时服务退出。远端修复验收日志使用 `logs/registration-r1-r4-*-20261009.*`，与此前阶段 1 的日志分开保留。

### 5.3 共享 layers 与 Qwen3 结构实施记录（2026-10-09）

共享层提取的设计合理，但接口应按已知计算语义命名。此次将 bias-free linear、embedding／logits 行选择、packed QKV、dense SwiGLU、RMSNorm／residual RMSNorm／QK norm 与全维未缩放 half-split RoPE 移至 `src/layers/`，CUDA 桥接与内核也移入该目录。接口限定 crate 内可见，保留原算法、dtype 舍入及 CUDA 原地更新约定，不引入逐算子动态分发、MoE 占位或通用 Transformer 的特性开关。

`PackedQkv` 统一持有 projection 几何和权重，返回原 packed allocation 的 Q/K/V 视图，支持 query width 与 hidden size 不同。`DenseSwiGlu` 封装 gate-up packing、激活和 down projection。Qwen3 decoder 不再重复保存这些参数和几何字段；模型仍负责 checkpoint 命名与绑定、tied embedding、QK norm 选用、残差顺序、attention 与 graph 分段。prefill 和 decode 共用 logits 行选择／投影逻辑。权重 reader 的逐 shard 加载留在后续工作。

新增 packed QKV 对照独立 projection、packed SwiGLU 对照 gate/up/down、embedding／tied weight 与 live logits 行重排测试；原有 norm、RoPE、CUDA 和 graph 数值测试随提取保留。用户已确认先完成 Qwen3 提取和回归，第二模型验证待办；尚未完成阶段 2 的全部加载与内存验收，本次不作为推理提速证据。

本地 macOS／CPU 全目标 Rust 测试 114 项通过。最终源码同步至 `sglang-test:/sjtu/yaosikai/llm_infer_rust` 并以 SHA-256 校验；远端 A100 GPU 1 release 全目标测试 124 项通过、1 项已有采样微基准忽略，release 二进制更新。真实 Qwen3-0.6B／BF16／FlashInfer 下，eager 和 decode＋segmented prefill graph 各 12 项 HTTP 测试通过；每种模式的 8 组不同长度与重复前缀请求，其 text／usage／finish_reason 均与提取前基线完全一致，两种模式也互相一致。临时服务退出后 GPU 显存恢复。远端验收记录为 `logs/shared-layers-tests-20261009.log`、`logs/shared-layers-build-20261009.log`、`logs/shared-layers-http-summary-20261009.log`、`logs/shared-layers-eager-graph-comparison-20261009.json` 及各模式服务／API 日志。

### 5.4 分层分页 KV 实施记录（2026-10-09）

已完成逐层 `ModelCacheSpec`、同几何缓存组、公共页布局与模型逐层视图绑定。模型计算不再接收两个全局同尺寸 K/V Tensor；生产加载使用 factory 的状态声明，运行时预算按所有层的实际几何累加，并在预算不足以容纳保留页和一页真实 KV 时拒绝启动。现有 radix／naive ownership 统一涵盖所有组，取消、前缀分支与 graph padding 的已有约定保持有效。

范围按用户要求收敛到通用 full-history Transformer KV，没有增加少数架构的特殊分支。新增异构 A／B／A 缓存组的生命周期测试，以及通过 Engine 加载和绑定接口的异构 attention 数值对照；此测试 factory 不作为生产模型注册。Qwen3 的 backend／内核限制继续校验。详细接口、生命周期、支持范围和后续工作见 [设计文档](layered-state-cache-design-2026-10-09.md)。

本地 macOS／CPU 全目标测试 121 项通过，`cargo fmt --all -- --check` 和 `git diff --check` 通过；CUDA 专用分支在本地无 GPU 时不执行实际计算。最终代码同步至 `sglang-test:/sjtu/yaosikai/llm_infer_rust`，两端基于 `0904a14`，本次修改及新增文件通过 SHA-256 校验；远端 `tmp.sh`、benchmark 脚本和已有日志保留。

远端 A100 GPU 1 使用完整 CUDA／FlashInfer 构建，release 全目标测试 131 项通过、1 项已有采样微基准忽略，release 服务二进制已更新。真实 Qwen3-0.6B／BF16／FlashInfer 下，eager 和 decode＋segmented prefill graph 各 12 项 HTTP 回归通过；8 组不同长度及重复前缀请求的 text／usage／finish_reason 在两种模式间、与既有重构前基线均完全一致。服务日志确认 graph 捕获，临时服务已退出，GPU 1 显存恢复到测试前的 7 MiB。

远端验收记录为 `logs/layered-cache-tests-20261009.log`、`logs/layered-cache-build-20261009.log`、`logs/layered-cache-http-summary-20261009.log`、`logs/layered-cache-eager-graph-comparison-20261009.json` 及两种模式的服务／API 日志。本次没有性能测量，不作为推理提速证据。

### 5.5 执行与 CUDA graph 实施记录（2026-10-10）

完成 `ForwardBatch`／`ForwardMode`／`ForwardOutput` 模型接口，以及 graph 前的输入／元数据形状校验。Scheduler 批次仍独立持有数据，在 Runner 入口借用转换；Engine 在对接 Scheduler 时提取 logits。decode／prefill 的模式约束统一，backend 的既有 phase 字段由入口校验保持一致。

完成通用 `SegmentedGraphRunner` 和模型 `PrefillGraphProgram`／`PrefillGraphReplay` 接口；Qwen3 不再持有 graph、桶目录和失败桶。Graph 能力默认禁用，按模式、实际运行组合、形状与 padding 声明；backend 的 decode 计划更新命名为图外 `prepare_replay`。权重／缓存／reserved-slot 变化前统一清理 graph，设备同步后释放 native 资源。有效批次与静态表宽不匹配时回退 eager，非法批次在复制前报错。

随后统一两种 graph 回放接口：decode 执行器明确命名为 `DecodeGraphRunner`，与 `SegmentedGraphRunner` 均接收 `ForwardBatch`、返回 `Option<ForwardOutput>`；Scheduler 批次转换和输入校验由 `ModelRunner` 入口负责，图执行器只做各自的形状准入和回放编排。

接口统一后重新验收：本地 125 项、远端 GPU 1 release 全目标 136 项 Rust 测试通过，1 项已有采样微基准忽略；eager／graph 各 12 项真实 Qwen3 HTTP 回归及 8 组输出基线对照通过。release 二进制已更新，临时服务退出后 GPU 1 显存为 4 MiB。此次追加回归记录使用 `logs/graph-interface-*-20261010.*`，与首轮执行接口重构日志分别保存。

新增不同段输入形状的算术测试程序，验证通用接口及失败／缺失／超限桶、padding、保留输出和地址重绑定；没有新增生产模型。Qwen3 的分段 prefill 数值回归扩展至 BF16／FP16 与 `pt`／`fa`／FlashInfer，保留动态长度、前缀命中、clear／recapture 和既有 decode 数值测试。设计与接口约定见 [执行与 CUDA graph 设计](execution-cuda-graph-design-2026-10-10.md)。

本地 macOS／CPU 全目标测试 125 项通过，`cargo fmt --all -- --check` 和 `git diff --check` 通过；本地无 CUDA，实际 GPU 数值验证在远端执行。本次变更及新增文件共 16 个同步至 `sglang-test:/sjtu/yaosikai/llm_infer_rust` 并通过 SHA-256 校验。两端 Git HEAD 不同，开发前已逐文件核对源码一致；远端已有脚本和日志保留。

远端 A100 GPU 1 release 全目标测试 136 项通过、1 项已有采样微基准忽略。BF16／FP16 × `pt`／`fa`／FlashInfer 六种分段 prefill 对照通过；真实 Qwen3-0.6B／BF16／FlashInfer 的 eager 与 decode＋segmented prefill graph 各 12 项 HTTP 回归通过。8 组不同长度和重复前缀请求的 text／usage／finish_reason 在两种模式间、与已有重构前基线均完全一致。服务日志确认两种 graph 捕获，release 二进制已更新，临时服务退出后 GPU 1 显存恢复至 7 MiB。

远端验收记录为 `logs/execution-graph-tests-20261010.log`、`logs/execution-graph-build-20261010.log`、`logs/execution-graph-http-summary-20261010.log`、`logs/execution-graph-eager-graph-comparison-20261010.json` 及各模式服务／API 日志。本次没有性能测量；第二个实际模型的执行与 graph 接入验证仍待后续完成。

## 6. 本阶段范围之外

当前未决定实现同进程多模型驻留、多模态请求、跨进程 worker、Tensor／Expert Parallel、speculative decoding 或所有量化格式。若目标模型的体积或结构要求其中某项，应在阶段 0 明确列为必要前置条件，再安排对应阶段，不能因本文未展开就视为已支持。

## 7. 源码参考

Rust：

- [模型注册](../src/models/registry.rs)与[启动组装](../src/server/components/builder.rs)。
- [Qwen3 定义](../src/models/qwen3/definition.rs)、[强类型配置](../src/models/qwen3/config.rs)、[模型与权重绑定](../src/models/qwen3/model.rs)、[共享 layers](../src/layers/mod.rs)、[prefill graph](../src/models/qwen3/prefill_graph.rs)。
- [Engine／RuntimeModelConfig](../src/engine/engine.rs)、[ModelExecutor／ModelRunner](../src/engine/model_runner.rs)、[checkpoint reader](../src/engine/model_loader.rs)、[KVCachePool](../src/engine/kvcache/pool.rs)。
- [AttentionSpec／backend](../src/models/attention/backend.rs)、[HTTP](../src/server/api.rs)、[请求 schema](../src/server/schemas.rs)、[FrontendManager／detokenizer](../src/server/manager.rs)、[输出 parser](../src/server/output/parser.rs)。
- [Scheduler／EOS](../src/scheduler/scheduler.rs)、[prefill 调度](../src/scheduler/prefill.rs)、[tokenizer／模板](../src/tokenizer.rs)。

原版 SGLang（相邻 checkout）：

- [HTTP 启动链路](../../sglang/python/sglang/srt/entrypoints/http_server.py)、[OpenAI serving 公共转换流程](../../sglang/python/sglang/srt/entrypoints/openai/serving_base.py)。
- [ModelRegistry](../../sglang/python/sglang/srt/models/registry.py)、[ModelConfig](../../sglang/python/sglang/srt/configs/model_config.py)、[checkpoint loader](../../sglang/python/sglang/srt/model_loader/loader.py)。
- [Qwen3 共享层与加载映射](../../sglang/python/sglang/srt/models/qwen3.py)、[RadixAttention](../../sglang/python/sglang/srt/layers/radix_attention.py)、[AttentionBackend 图内／图外契约](../../sglang/python/sglang/srt/layers/attention/base_attn_backend.py)。
- [ForwardBatch](../../sglang/python/sglang/srt/model_executor/forward_batch_info.py)、[logits 输出](../../sglang/python/sglang/srt/layers/logits_processor.py)。
- [聊天格式与 parser 配置](../../sglang/python/sglang/srt/entrypoints/openai/serving_chat.py)、[增量 detokenization](../../sglang/python/sglang/srt/managers/detokenizer_manager.py)、[调度与 overlap](../../sglang/python/sglang/srt/managers/scheduler.py)。

相关项目记录：

- [Prefill 分段 graph](cuda-graph-prefill.md)。
- [FlashInfer、归一化、KV 内核与 uv 环境](kernel-optimizations-2026-10-08.md)。
- [Decode 性能差异与 overlap 测量](benchmark-decode-gap-2026-10-08.md)。
- [KV cache forward 后发布方案](kv-cache-publish-after-forward-design.md)。实施状态和缓存接口迁移前应重新核对该方案与源码。
