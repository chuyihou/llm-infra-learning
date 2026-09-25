# 选题 Roadmap

> 只排 deep dive。`notes/` 是学习现场，不排期，学到什么写什么。

## 排序原则

1. **先写已验证的**：已经动手验证过的主题，出货快、质量稳，先用来立口碑
2. **再写现学现卖的**：PD 分离、多模态这些 roadmap 剩余项，学完就写，学习和输出一次完成
3. **训练/RL 等 A5**：GRPO 那期等 CS336 A5 做完再写，那是含金量最高的一期，不赶时间
4. **求职优先**：这个系列是为求职服务的，任何一期都不能挤占面试准备的主线；时间只是建议，随时可调

工作量：S < 1 周，M = 2–3 周，L = 1 个月以上（含现学）

---

## Phase 1 · 推理引擎（强项区，快速立口碑）

| 期号 | 主题 | 你的 edge | 工作量 | 建议时间 |
|------|------|-----------|--------|----------|
| 01 | Chunked Prefill | ✅ 已完成发布 | — | 2026-09 |
| 02 | Continuous Batching 与调度器全景 | scheduler、continuous batching 已掌握；vLLM vs SGLang 调度策略代码级对照 | M | 2026-10 |
| 03 | PagedAttention 与 KV Cache 管理 | radix/paged KV cache 已掌握；prefix caching、page size tradeoff | M | 2026-10/11 |
| 04 | FlashAttention 深潜 | **绝活**：Triton 手写过 forward，正确性验到 1 fp16 ULP；差异化最强的一期 | M | 2026-11 |
| 05 | Attention 变体：MQA / GQA / MLA | 已掌握；可独立成轻量篇，或并入 04 | S | 2026-11 |
| 06 | Speculative Decoding 家族 | 投机解码家族已掌握；Eagle / Medusa / 标准 spec decode 对照 | M | 2026-11/12 |
| 07 | 推理量化：INT4 / FP8 / MXFP4 | 格式与误差、W4A16 kernel、打包/解包、layout 敏感性全部动手写过 | M | 2026-12 |
| 08 | MoE 推理：Expert Parallelism 与负载均衡 | MoE 已掌握；EP、EPLB、all-to-all 开销 | M | 2027-01 |
| 09 | PD Disaggregation | roadmap 剩余重点，需现学；Mooncake-style 架构 | L | 2027-01+ |

## Phase 2 · 训练与 RL（与 A5 联动）

| 期号 | 主题 | 你的 edge | 工作量 | 建议时间 |
|------|------|-----------|--------|----------|
| 10 | ZeRO / FSDP 深潜 | 概念级已掌握，动手进行中 | M | 等动手完成 |
| 11 | 并行策略全景：TP / PP / CP / EP / SP | 集合通信概念级；顺带补 NCCL deep dive（面试题库缺口） | L | 2027-Q1 |
| 12 | GRPO 实战 | **王牌期**：CS336 A5 全程手写实现；SFT → reward → RL 完整 pipeline | L | A5 做完后 |
| 13 | RL 框架对照：verl / slime / OpenRLHF | 有 LLM_RL_Frameworks_Notes.md 全套笔记 | M | 2027-Q1 |

## Phase 3 · Agent 与多模态

| 期号 | 主题 | 你的 edge | 工作量 | 建议时间 |
|------|------|-----------|--------|----------|
| 14 | Agent Harness：ReAct loop 与 context 管理 | harness 循环、code-protocol vs text-protocol 已掌握；CMU agent 课 Phase 2 联动 | M | 2027-Q1 |
| 15 | 多模态推理 | roadmap 剩余重点，需现学 | L | 2027-Q2 |

## 备选池（轻量 notes，不占 deep dive 编号）

- CUDA Graph 在推理引擎里的用法
- Sampling 实现细节：top-k / top-p / min-p 的 kernel 视角
- vLLM vs SGLang 全面对照（架构、调度、取舍）
- 生产 serving 可观测性：metrics、profiling、debug 手段

备选池走轻量路线：markdown 直发，不配交互页，适合填补 deep dive 之间的空档。
