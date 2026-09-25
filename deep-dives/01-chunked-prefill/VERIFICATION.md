# 01 · Chunked Prefill — VERIFICATION

## 验证基线

- 验证日期：2026-09-23
- SGLang：main @ `5c154c21`
- vLLM：main @ `c4cfd70`（v0 部分对照 tag v0.10.2）

## 关键 claim 追溯表（摘要，完整版见 NOTES.md 与 index.html 内联引用）

| Claim | 代码位置 |
|-------|----------|
| SGLang chunk 切分围绕 `full_untruncated_fill_ids` / `prefix_indices` / `extend_range` 三个字段 | sglang@5c154c21 `python/sglang/srt/managers/schedule_batch.py:977` |
| `PrefillAdder` 双预算：`rem_input_tokens`（整批）+ `rem_chunk_tokens`（单 chunk） | `schedule_policy.py:1383` / `:1466` |
| 同一时刻只有一个 chunked request（单例 `Scheduler.chunked_req`） | `scheduler.py:1321`，`assert` 见 `:4063`，"at most one request" 注释见 `batch_result_processor.py:418` |
| 续算优先于 waiting queue | `scheduler.py:3921-3929` |
| main 已无 `IS_CHUNKED_PREFILL`；纯 chunk batch = EXTEND，混批 = MIXED | `forward_batch_info.py:177` |
| SGLang 默认 prefill-first | `scheduler.py:3799` / `:3769-3779` |

## 未验证/存疑清单

见 NOTES.md 文末"未验证/存疑"清单——讲的时候别当事实说。

## 勘误记录

| 日期 | 内容 | 处理 |
|------|------|------|
|      |      |      |
