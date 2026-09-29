# Benchmark优化记录

## 2026.09.29
首次测试使用sglang benchmark，单个请求长达30s，A100 gpu利用率33%，未完成测试。同样的模型和配置sglang在15s内完成了100个请求（无并发），gpu利用率80+%。