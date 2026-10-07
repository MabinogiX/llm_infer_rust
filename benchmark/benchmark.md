# Benchmark优化记录

## 2026.10.07
发现请求中的max_completion_tokens没有被正确处理，导致速度缓慢。修复之后单个请求约0.2s.

## 2026.10.07
使用release构建，单个请求达到了19s。优化下述算子之后，单个请求降低到了5s，GPU利用率约56%。
|环节|Rust 实现|SGLang 实现|
| --- | --- | --- |
|QKV|三次矩阵乘法|QKVParallelLinear 一次投影，再切分 Q/K/V|
|MLP gate/up|两次矩阵乘法，分别做 SiLU 和乘法|合并投影，并使用 SiluAndMul 算子|
|RMSNorm、QK norm|多个 libtorch 张量操作|CUDA 融合 RMSNorm、残差加法与归一化，并在条件满足时使用原地融合 QK norm|
|RoPE|查表后用乘、加、拼接旋转 Q/K|使用缓存和 CUDA 原地 RoPE 算子|

## 2026.09.30
从LibTorch SDPA切换到了FlashInfer，导致无法使用graph，单个请求速度下降至23s左右，gpu利用率约40%；

## 2026.09.30
发现昨天的测试有误，启动参数变成了--max-running-req 4 --max-seq-len 512，因而单个请求变快。优化prefill kv cache不再跟cpu通讯，单个请求依然3s左右。使用--max-running-req 4 --max-seq-len 40960单个请求长达86s，出现显著性能回退。cpu和gpu利用率接近100%

## 2026.09.29
数据类型改成BF16（原来是Float32）,使用LibTorch SDPA，显存占用大幅度下降，单个请求速度约3s，gpu利用率约90%

## 2026.09.29
增加graph runner之后，gpu利用率达到了99%，但显存占用大幅提升，单个请求速度降低至45s

## 2026.09.29
首次测试使用sglang benchmark，单个请求长达30s，A100 gpu利用率33%，未完成测试。同样的模型和配置sglang在15s内完成了100个请求（无并发），gpu利用率80+%。