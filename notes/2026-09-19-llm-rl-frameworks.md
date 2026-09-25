# LLM 强化学习框架学习笔记：slime、verl、Relax、Miles 与 OpenRLHF

**最后更新：** 2026-09-16  
**适合读者：** 了解 SFT、PPO/GRPO 基础、分布式训练与 LLM serving 的工程师  
**完成后能够：** 理解大模型 RL post-training 的完整数据流，分析同步/异步与 colocated/disaggregated 架构，并根据训练后端、rollout 引擎、agentic/multimodal 需求选择框架。

**名称说明：** 本文的 Relax 指小红书 AI Infra 开源的 [redai-studio/Relax](https://github.com/redai-studio/Relax)，不是 Google DeepMind 的 RLax 算法组件库。

---

## 0. 一分钟图景

大模型 RL 框架不是一个“PPO optimizer”。它要把多个不同工作负载组成长期稳定的闭环：

```text
Prompts / Environments
        ↓
Rollout Actor / Inference Engines
        ↓ trajectories: tokens, logprobs, masks, metadata
Reward / Verifier / Judge / Environment
        ↓ rewards
Advantage Estimation
        ↓ training samples
Actor Training + optional Critic Training
        ↓ new weights
Weight Synchronization
        └────────────────────→ Rollout Engines
```

框架间的主要差异不只是支持 PPO 还是 GRPO，而是：

- Training backend：FSDP/FSDP2、Megatron-LM、DeepSpeed 等。
- Rollout backend：vLLM、SGLang、HF Transformers 等。
- Orchestration：Ray actor、Ray Serve、single controller、data buffer/queue。
- Placement：模型角色是否 colocate、分池、跨集群。
- Execution：同步、异步、允许多少 weight/data staleness。
- Correctness：token/logprob、模型版本、MoE routing 与 mask 是否一致。
- Agentic：多轮工具、环境、sandbox、长尾 rollout 是否是一等公民。
- Operations：fault tolerance、checkpoint、observability、elastic scaling。

五个框架的第一印象：

| 框架 | 核心取向 |
|---|---|
| slime | 深度绑定 Megatron + SGLang，数据生成灵活、路径直接 |
| verl | HybridFlow/Hybrid Controller，后端与 placement 组合灵活、生态广 |
| Relax | Ray Serve 服务化、完全异步、omni-modal 与弹性 rollout |
| Miles | 基于 slime 演进，强调企业运维、超大模型、低精度、TITO 与故障恢复 |
| OpenRLHF | Ray + vLLM + DeepSpeed/相关后端，易用、完整、agent execution 统一 |

---

## 1. RL Post-training 的角色

### 1.1 Actor / Policy

正在训练的模型 `πθ`。它既要：

- 在 rollout 阶段生成 token；
- 在 training 阶段计算当前策略 log probability 和梯度。

同一个 Actor 的 serving layout 与 training layout 往往不同，因此需要 reshard 或独立副本。

### 1.2 Rollout Engine

面向高吞吐生成：continuous batching、paged KV、prefix cache、TP/EP、agent multi-turn。它产生的训练轨迹至少需要：

```text
prompt tokens
response/action tokens
behavior logprobs
loss masks
model/weight version
sampling metadata
environment observations
reward/verifier outputs
```

只保存文本再 retokenize 可能改变 token boundary，导致训练样本与真实行为策略不一致。

### 1.3 Reference Model

常用于 KL penalty：限制新 policy 偏离 reference policy。Reference 只 forward，不更新；但它仍消耗权重内存和计算。

### 1.4 Reward / Verifier / GenRM

- Reward model：学习得到的偏好分数。
- Rule-based verifier：数学答案、代码测试、格式、环境成功。
- Generative reward model / judge：另一个模型评分。
- Environment reward：多轮 agent interaction 的状态变化。

Reward 不只是 scalar function，还可能涉及 sandbox、搜索、浏览器、图像、音频和长时间外部任务。

### 1.5 Critic 与 Advantage

PPO 常使用 value/critic；GRPO、RLOO、REINFORCE++ 等采用不同 baseline 或 advantage estimator。框架需要支持：

- sequence-level 与 token-level reward；
- GAE/return/baseline；
- group normalization；
- KL、clip、importance ratio；
- padding、packing 和 response mask。

### 1.6 Weight Sync

训练权重必须更新到 rollout engines。方式包括：

- 同 GPU time-share，切换训练/推理状态；
- GPU-to-GPU collective/broadcast；
- CUDA IPC / NCCL / RDMA；
- checkpoint/shared filesystem；
- full weights、delta weights 或 LoRA adapters。

Weight sync 的时间、峰值内存和一致性是系统核心，而不是外围功能。

---

## 2. 算法只占系统的一部分

### 2.1 PPO 简化目标

```text
r_t(θ) = πθ(a_t|s_t) / πold(a_t|s_t)

L_clip = E[min(r_t A_t,
               clip(r_t, 1-ε, 1+ε) A_t)]
```

需要 behavior logprobs、当前 policy logprobs、advantages，通常还需要 critic。工程成本较高，但控制手段成熟。

### 2.2 GRPO / Group-based 方法

同一 prompt 采样多个 responses，用组内 rewards 构造相对 advantage，避免单独 critic。好处是系统角色减少；代价是每 prompt 多样本 rollout、组内方差和采样成本。

### 2.3 RLOO / REINFORCE++ 等

通过 leave-one-out baseline、batch/group baseline、KL 与 normalization 等设计，在不使用 critic 或减少 critic 依赖的情况下训练。

### 2.4 框架比较时必须固定算法

不同框架默认：

- reward normalization；
- KL 形式；
- clipping；
- token/sequence averaging；
- group sampling；
- loss mask；
- reference logprob 计算；

可能不同。直接比较最终 reward 曲线，可能是在比较算法实现而不是系统吞吐。

---

## 3. 同步、异步与资源放置

### 3.1 Colocated Synchronous

Actor training 与 rollout 共享 GPU，阶段交替：

```text
rollout → reward/advantage → train → weight update → rollout
```

**优点**

- GPU 需求少；权重不需要跨独立集群长期复制。
- 严格 on-policy，数据 staleness 低。
- 容易理解和调试。

**缺点**

- Training 与 inference 最佳并行布局不同。
- 阶段切换和 reshard 有开销。
- 一个阶段运行时另一阶段资源可能闲置。

### 3.2 Disaggregated Synchronous

Rollout 与 training 使用不同 GPU，但每轮仍等待完整 batch 和 weight sync。

- 可独立选择 vLLM/SGLang 与 FSDP/Megatron layout。
- 资源利用更灵活。
- 仍存在 stage barrier 和长尾 rollout bubble。

### 3.3 Fully Asynchronous

Rollout、reward、advantage、training 持续流水运行：

```text
Rollout version k ── samples ──→ Trainer version k+n
        ↑                          │
        └──── async weight sync ───┘
```

**收益**

- 隐藏 rollout、reward、train 与 weight sync latency。
- 适合 agentic 长尾和独立弹性扩容。

**代价**

- 数据可能来自旧 policy，产生 off-policy bias。
- 必须记录 weight version、behavior logprobs 和 staleness。
- Backpressure、queue、故障恢复和 checkpoint 一致性复杂。

### 3.4 Hybrid

只拆开最昂贵或最不匹配的角色。例如 rollout 独立，reference/actor-forward/advantage 与 trainer colocate。目标是在 throughput、GPU 成本和 on-policy correctness 间折中。

---

## 4. 性能与正确性模型

### 4.1 同步周期

```text
T_iteration
  ≈ T_rollout
  + T_reward
  + T_advantage
  + T_train
  + T_weight_sync
  + bubbles
```

Agentic RL 中，rollout 常因工具、环境和 response length 产生重尾。等待最慢 sample 会形成 straggler bubble。

### 4.2 异步稳态

流水线吞吐近似由最慢 stage 决定：

```text
throughput ≤ min(rollout_rate,
                 reward_rate,
                 train_consume_rate,
                 weight_sync capacity)
```

Queue 只隐藏短期波动。长期生产率不匹配会导致 backlog、样本过旧或内存爆炸。

### 4.3 On-policy 与 Staleness

设 trajectory 由 policy version `v` 生成，trainer 当前为 `v+k`：

```text
staleness = current_version - rollout_version
```

异步框架需要限制最大 staleness、丢弃过旧样本，或使用 importance correction。提高吞吐不能以不可控 policy lag 为代价。

### 4.4 五个常见 Silent Bug

1. **Token mismatch：**生成后 detokenize/retokenize，tokens 改变。
2. **Logprob mismatch：**rollout 与 trainer kernel、precision、temperature 或 masking 不一致。
3. **Weight mismatch：**样本和 reference/current actor 版本关联错误。
4. **MoE routing mismatch：**rollout 与 train forward 路由不同。
5. **Mask mismatch：**environment observation 或 prompt token 被错误计入 policy loss。

训练“不报错”不代表 RL loop 正确。每个 framework 的 correctness tooling 比单个 benchmark 更重要。

---

## 5. slime

[slime](https://github.com/THUDM/slime)由 THUDM 开源，定位是面向 RL scaling 的 LLM post-training framework。

### 5.1 核心架构

```text
Megatron training
       ↕ weight sync
SGLang rollout + router
       ↕
Data Buffer
       ↕
custom generation / reward / verifier / environment
```

- Training 深度使用 Megatron-LM。
- Rollout 深度使用 SGLang。
- Ray 管理 placement 与 worker。
- Data Buffer 连接 prompt、生成、reward 与训练。
- Custom generation interface 承载 math、code、search、tools、sandbox、多 agent。

### 5.2 设计取向

slime 不追求抽象所有 training/serving backend，而是保留 Megatron 与 SGLang 的原生参数和能力。好处是上游新特性容易直接使用，代价是 backend 选择更 opinionated。

### 5.3 适合

- 已经使用 Megatron + SGLang。
- 大模型/MoE，重视 SGLang routing、cache、PD 和 weight sync。
- Rollout/data generation 逻辑高度自定义。
- 希望核心数据流相对直接、便于单独 replay rollout/train。

### 5.4 注意

- 对其他 trainer/rollout backend 的可替换性不是核心目标。
- Megatron checkpoint、模型适配和大规模并行有学习成本。
- 异步模式仍需自己确定 staleness 与算法接受范围。

官方仓库称 slime 被用于多个 GLM 系列 release 的完整 post-training loop；这是项目方的公开生产使用声明。

---

## 6. verl

[verl](https://github.com/verl-project/verl)由 ByteDance Seed 发起，是 [HybridFlow](https://arxiv.org/abs/2409.19256) 的开源实现，强调 hybrid-controller programming model 和灵活资源映射。

### 6.1 核心思想

- 把 RL dataflow 与 worker/backend 解耦。
- Controller 表达 PPO、GRPO、DAPO 等流程。
- Worker group 封装 actor、rollout、reference、critic/reward。
- 支持不同角色 colocate 或分布在不同 GPU sets。
- 通过 Hybrid Engine/resharding 在 generation 与 training layout 间切换。

### 6.2 Backend 生态

官方仓库当前列出：

- Training：FSDP、FSDP2、Megatron-LM 等。
- Rollout：vLLM、SGLang、HF Transformers。
- 模型：Hugging Face 生态以及大型 dense/MoE。
- 算法/recipes：PPO、GRPO、DAPO、GSPO、RLOO、REINFORCE++ 等。
- Multi-turn、tool calling、multimodal 和多种硬件路径。

### 6.3 适合

- 需要在 FSDP/FSDP2/Megatron 与 vLLM/SGLang 间组合。
- 希望快速实现新的 RL dataflow 或复现实验 recipe。
- 团队重视社区规模、模型覆盖和 extensibility。
- Placement 会随集群规模变化。

### 6.4 注意

- 灵活性带来配置面和版本兼容复杂度。
- Recipe、主库、rollout engine 与训练 backend 的 commit 需要固定。
- 比较不同模式前，必须确认 algorithm config 和 token/logprob semantics 一致。

官方资料公开列出 DAPO、Seed-Thinking 等基于 verl 的训练案例；具体性能数字应按其模型与集群环境解释。

---

## 7. Relax

[Relax](https://github.com/redai-studio/Relax)全称 Reinforcement Engine Leveraging Agentic X-modality，由小红书 AI Infra 团队开源，重点是 asynchronous、service-oriented、omni-modal RL。

### 7.1 六层服务化架构

官方文档将系统拆成：

1. Entrypoints；
2. Orchestration；
3. Actor/Rollout/Critic/ActorFwd/Advantages/GenRM components；
4. Engine 与 reward/router/filter；
5. Megatron-LM + SGLang backends；
6. Ray actor groups、TransferQueue 与 DCS distributed layer。

每个核心角色可以作为独立 Ray Serve deployment，适合独立健康检查、弹性和恢复。

### 7.2 三种执行模式

- **Colocate Sync：**Actor/Rollout time-share GPU，严格 on-policy。
- **Fully Async：**Rollout、Actor、ActorFwd、Reference、Advantages 分布在独立 GPU clusters，通过 TransferQueue 流式交换。
- **Hybrid：**Actor 与 Rollout 分离，但 ref/actor_fwd/advantages 在 Actor 侧进程内执行。

DCS（Distributed Checkpoint Service）使用 NCCL/GLOO 路径同步权重，并尝试与训练 overlap。

### 7.3 适合

- Text、vision、audio 的统一 RL post-training。
- 多轮 agent/environment，rollout 时间重尾。
- 需要独立扩缩 rollout engines 或跨集群 federation。
- 团队接受 service-oriented deployment 与明确 staleness budget。

### 7.4 注意

- Fully async 的吞吐收益必须与 policy staleness/样本丢弃一起报告。
- 更多独立服务意味着更多 queue、registry、health 和 checkpoint 状态。
- 项目在 2026 年开源，成熟度应按目标模型、backend 和版本实际验证。

---

## 8. Miles

[Miles](https://github.com/radixark/miles)由 RadixArk 开源，fork 自 slime 并持续与 slime 协同演进，定位是 enterprise-grade large-scale model post-training。

### 8.1 核心栈

- SGLang：rollout 与 agentic inference。
- Megatron-LM：主要大规模 training backend。
- FSDP2：保留 Hugging Face 实现的可选训练路径。
- Fully async pipeline 与可配置 on/off-policy schedule。
- P2P RDMA 等大规模 weight update 路径。

### 8.2 Correctness 与 Ops 特性

- **TITO（Token-In-Token-Out）：**避免文本级 detokenize/retokenize mismatch。
- **R3（Rollout Routing Replay）：**记录并重放 MoE expert routing，减少 rollout/train forward mismatch。
- **Fault tolerance：**官方文档描述 SGLang engine 的 in-place recovery。
- Low-precision RL、LoRA/multi-LoRA、observability 与硬件适配。

### 8.3 适合

- 超大 dense/MoE 模型，已经选择 SGLang/Megatron。
- 企业环境重视 fault tolerance、硬件覆盖、监控和版本支持。
- Token correctness、MoE routing 和快速 weight sync 是核心风险。
- 希望采用 slime 思路但需要更多产品化运维能力。

### 8.4 注意

- Miles v0.1 于 2026-08 发布，功能和兼容矩阵变化快。
- “Day-0 model support”和性能数据来自项目方，应在自身环境复现。
- 与 slime 共享大量思想和代码；选型应比较真正需要的 Miles 增量，而不是只看功能列表。

---

## 9. OpenRLHF

[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)是较早开源的完整 RLHF 框架之一，当前强调 Ray + vLLM 分布式架构与统一 agent execution。

### 9.1 核心架构

- Ray 调度 Actor、Reward、Reference、Critic 和 rollout engines。
- vLLM 负责高吞吐 generation。
- DeepSpeed 及项目当前列出的其他训练 backend 负责参数训练与内存优化。
- Hugging Face Transformers 提供模型接口。
- 支持 colocated/hybrid 与 async 模式。

### 9.2 Agent-based Execution

官方仓库强调 token-in-token-out 的 AgentExecutor：

- Single-turn 与 multi-turn 共用抽象。
- Environment observation 用 mask 与 model action 分离。
- RL algorithm 与 agent execution mode 解耦。
- Custom reward/environment 可插拔。

### 9.3 适合

- 希望快速跑通完整 PPO/GRPO/RLOO/REINFORCE 类流程。
- 已熟悉 Ray、vLLM、Hugging Face/DeepSpeed。
- 中小规模研究、课程、复现，以及逐步扩展到 multi-turn agent。
- 重视文档、脚本和较低上手门槛。

### 9.4 注意

- 当前主分支功能变化快，脚本与依赖版本必须固定。
- 大规模 Megatron/SGLang 专用优化不是其唯一中心。
- 项目 README 中的性能和“production-ready”等表述属于项目方声明，应以目标集群验证。

---

## 10. 横向比较

| 维度 | slime | verl | Relax | Miles | OpenRLHF |
|---|---|---|---|---|---|
| 核心取向 | SGLang-native、直接数据流 | HybridFlow、后端/placement 灵活 | 服务化、fully async、omni-modal | slime 系企业化与超大模型 | 易用完整的 Ray+vLLM RLHF |
| 主要 Trainer | Megatron | FSDP/FSDP2/Megatron 等 | Megatron | Megatron，另有 FSDP2 | DeepSpeed/项目当前后端 |
| 主要 Rollout | SGLang | vLLM/SGLang/HF | SGLang | SGLang | vLLM |
| Orchestration | Ray + Data Buffer | Hybrid Controller + Ray workers | Ray Serve + TransferQueue | slime-derived + async ops | Ray actors/controllers |
| Backend 自由度 | 较低但深 | 高 | 中，明确 Megatron/SGLang | 中，深度 SGLang/Megatron | 中，偏 vLLM/HF |
| Fully Async | 支持相关模式 | 支持/持续演进 | 核心能力 | 核心能力 | 支持 async agent RL |
| Agentic | 强，自定义 generation | 强，生态/recipes 广 | 强，服务化多轮 | 强，enterprise connectors | 强，统一 AgentExecutor |
| Multimodal | 支持并持续扩展 | VLM/omni/VLA 生态 | 核心 omni-modal | LLM/VLM/diffusion 扩展 | VLM/multi-turn 支持 |
| 突出正确性 | 显式 dataflow/replay/debug | 可复现 recipes/backend consistency | Version/staleness/服务状态 | TITO、R3、fault tolerance | TITO agent execution |
| 主要代价 | Backend opinionated | 配置与组合复杂 | 分布式服务状态复杂 | 新项目、功能面宽 | 极大规模专用路径需验证 |

这张表是截至 2026-09 官方资料的快照，不是永久结论。

---

## 11. 选型建议

### 选择 slime，当

- Megatron + SGLang 已经是确定栈。
- 想直接使用 SGLang 原生 serving 能力。
- 自定义 rollout/reward/environment 比多 backend 抽象更重要。

### 选择 verl，当

- 需要在 FSDP2/Megatron 与 vLLM/SGLang 间灵活组合。
- 算法 recipe、社区生态和 placement 实验是重点。
- 团队愿意管理更大的配置与版本矩阵。

### 选择 Relax，当

- Fully async、弹性 rollout 和跨集群是核心。
- Text/Vision/Audio omni-modal RL 是一等需求。
- 希望把每个 RL role 当独立服务管理。

### 选择 Miles，当

- 已选择 slime/SGLang/Megatron 路线，但需要更强 enterprise operations。
- 超大 MoE、快速 weight sync、TITO/R3、低精度与故障恢复是重点。

### 选择 OpenRLHF，当

- 优先快速搭建 Ray + vLLM + Hugging Face/DeepSpeed 工作流。
- 需要完整算法脚本和统一 single/multi-turn agent interface。
- 规模和模型适配可以由现有 backend 覆盖。

### 不要只凭框架名字选

先固定：

```text
model + algorithm + reward
prompt/response distribution
rollout engine version
trainer backend version
GPU/network topology
on-policy/staleness requirement
```

然后用同一 workload 做端到端 benchmark。

---

## 12. 性能分析

### 12.1 Stage Metrics

- Rollout generated tokens/s、active requests、长尾完成时间。
- Reward/verifier tasks/s、sandbox queue、timeout。
- Actor/Critic train tokens/s、MFU、optimizer step time。
- Actor forward/reference logprob tokens/s。
- Weight sync seconds、bytes、阻塞时间。
- Data queue depth、age、policy version/staleness。
- End-to-end samples/s、tokens/s、GPU-hours/step。

### 12.2 Bubble 分析

每张 GPU 标记状态：

```text
ROLL_OUT
WAIT_DATA
RESHARD
SYNC_WEIGHT
TRAIN_FWD_BWD
OPTIMIZER
REWARD
IDLE
```

只看平均 GPU utilization 无法说明 idle 是等待数据、权重、最慢 agent 还是通信。

### 12.3 资源配比

```text
rollout_capacity ≥ prompt_rate × samples_per_prompt × output_tokens
train_capacity ≥ accepted_sample_tokens
reward_capacity ≥ generated_trajectories
weight_sync_capacity ≥ update_frequency × model_bytes
```

异步情况下应让各 stage 在目标 staleness 内平衡，而不是让 queue 无限吸收不匹配。

### 12.4 成本指标

- GPU-hours per 1M rollout tokens。
- GPU-hours per optimizer step。
- Reward/environment cost per accepted trajectory。
- Dropped/stale/invalid sample ratio。
- Weight sync 和 reshard 占总时间比例。
- Cost per quality improvement，而不只是 samples/s。

---

## 13. Correctness Checklist

**Trajectory**

- Rollout tokens 与 train input token IDs 完全一致。
- Loss mask 只覆盖 policy actions。
- 多轮 observation、tool result 和 special tokens 边界正确。

**Policy Version**

- 每个 trajectory 带 rollout weight version。
- Staleness policy 有上限、告警和丢弃规则。
- Resume 后版本号与 queue/checkpoint 一致。

**Logprob**

- Sampling temperature/top-p 的 behavior distribution 明确。
- Rollout/train 的 vocabulary、tokenizer、chat template 一致。
- Precision/kernel mismatch 有量化误差门槛。

**Reward**

- Reward 不读取答案泄漏字段。
- Verifier timeout、exception、invalid output 有稳定语义。
- Group reward normalization 不跨错误边界。

**Distributed**

- MoE routing、sequence packing、padding 和 global batch 可复现。
- Weight sync 是原子版本切换，不混合新旧参数。
- Failed rollout/retry 不被重复训练。

---

## 14. 最小公平 Benchmark

### 14.1 固定条件

- 同一 checkpoint、tokenizer、dtype。
- 同一 prompts、samples/prompt、max length 与 sampling seed。
- 同一 algorithm equations、KL、clip、advantage normalization。
- 同一 reward/verifier。
- 同一 GPU 数、型号、网络和 fault policy。

### 14.2 三组实验

**单阶段能力**

- Trainer tokens/s。
- Rollout tokens/s。
- Weight sync GB/s 与 peak memory。

**同步闭环**

- Time/iteration、bubble breakdown。
- Strict on-policy reward/learning curve。

**异步闭环**

- Throughput vs max staleness。
- Sample age distribution。
- 同等 wall-clock/GPU-hours 的质量曲线。

### 14.3 Agentic Stress Test

- 50% 短 rollout、45% 中等、5% 极长或 timeout。
- Tool/environment failure 和重试。
- Rollout engine crash 与恢复。
- Reward service 降速。
- Trainer checkpoint/restart。

观察 backpressure、queue growth、sample loss、duplicate training 和最终一致性。

---

## 15. 常见故障模式

| 现象 | 可能根因 | 优先检查 |
|---|---|---|
| Reward 上升但评估下降 | Reward hacking/leakage | verifier input、held-out eval |
| KL 突然爆炸 | 旧样本、logprob/version mismatch | staleness、behavior logprob |
| GPU utilization 高但训练慢 | Rollout/forward 重复或无效样本 | accepted ratio、bubble states |
| Rollout queue 无限增长 | Rollout 比 trainer 快 | queue age、stage rates、backpressure |
| Trainer 等数据 | Rollout/agent 环境长尾 | completion distribution、partial rollout |
| Async 吞吐高但学不动 | Policy lag 太大 | version gap、importance ratio |
| MoE 训练不稳定 | Rollout/train routing mismatch | expert route replay/consistency |
| Resume 后曲线变化 | Queue/RNG/version 未恢复 | checkpoint coverage |
| 多轮样本 loss 错 | Observation/action mask 错位 | token-level trajectory audit |
| Weight sync OOM | 同时存在多份权重/buffer | peak memory timeline、sync mode |

---

## 16. 最新发展与行业采用（截至 2026-09）

### 16.1 技术趋势

**从单轮 RLVR 走向 agentic long-horizon RL。** Tool、browser、code sandbox、GUI 和多 agent 让 rollout 成为外部服务编排问题，长尾与 failure recovery 比纯生成更重要。

**从 bulk synchronous 走向 async/hybrid。** 框架普遍增加独立 rollout pools、streaming queue、partial rollout 和 configurable staleness，目标是减少训练/推理气泡。

**Correctness 成为一等系统能力。** TITO、zero-mismatch rollout、MoE routing replay、versioned trajectories、replay/debug pipeline 被公开框架强调，因为错误通常不会 crash，只会悄悄改变学习目标。

**多模态扩展。** VLM、omni-modal、VLA 和 diffusion RL 进入同一 post-training stack，environment payload 不再只有文本 tokens。

**更大模型与低精度。** Megatron/EP、P2P weight sync、FP8/FP4、LoRA 和 multi-LoRA 被用于降低超大模型 RL 的显存、通信和成本。

**Observability 与 fault tolerance 产品化。** 框架开始提供 trace viewer、online RL state、health manager、in-place engine recovery 和跨 stage metrics。

### 16.2 公开项目状态

| 框架 | 发起/维护方 | 公开时间线与采用证据 |
|---|---|---|
| slime | THUDM / 社区 | 2025 开源；仓库声明用于多个 GLM 系列完整 post-training loop |
| verl | ByteDance Seed / verl community | HybridFlow/EuroSys；公开 DAPO、Seed-Thinking 等案例与广泛社区集成 |
| Relax | 小红书 AI Infra / RedAI Studio | 2026-04 开源；官方提供 text/VLM/omni-modal 与 fully async recipes |
| Miles | RadixArk / SGLang 生态 | 2026-08 v0.1；基于 slime，强调 enterprise ops 与超大模型路径 |
| OpenRLHF | OpenRLHF community | 2023 开源；Ray+vLLM 路线，公开课程、研究和社区案例 |

这些是官方仓库/项目方公开信息。生产声明和性能数据不应被当作跨环境的独立验证。

---

## 17. 自测题

1. 为什么 LLM RL framework 不只是 optimizer？
2. Rollout trajectory 至少应保存哪些字段？
3. Colocated sync 与 fully async 的核心权衡是什么？
4. 为什么异步 queue 不能解决长期 stage rate 不匹配？
5. TITO 解决什么 correctness 问题？
6. verl 与 slime 的设计取向有何差异？
7. Relax 的 service-oriented 架构适合什么 workload？
8. Miles 的 R3 为什么对 MoE 重要？
9. 为什么不能直接比较两个框架默认 GRPO 的 reward curve？
10. 公平 benchmark 必须固定哪些条件？

### 参考答案

1. 它还需生成、reward、advantage、训练、weight sync、调度、容错和观测。
2. Tokens、behavior logprobs、mask、reward、sampling metadata、weight version 与环境状态。
3. 同步更 on-policy、简单但有 bubbles；异步吞吐高但有 staleness 和复杂状态。
4. 若生产率长期大于消费率，backlog 和样本年龄仍会无限增长。
5. 避免文本 detokenize/retokenize 后 token 与 logprob/loss 对不上。
6. verl 强调 backend/placement/dataflow 灵活；slime 深度选择 Megatron+SGLang 并暴露原生能力。
7. Omni-modal、长尾 agentic、独立弹性 rollout 和跨集群角色分离。
8. Rollout 与 trainer expert routing 不同会使训练 logprob/gradient 对不上行为轨迹。
9. KL、normalization、mask、sampling 与 objective 细节可能不同。
10. 模型、tokenizer、算法公式、reward、数据、采样、硬件、backend 版本和 staleness policy。

---

## 18. 主要资料

- [slime GitHub](https://github.com/THUDM/slime)：Megatron + SGLang、Data Buffer 与 agentic rollout。
- [slime Documentation](https://thudm.github.io/slime/)：配置、weight sync、debug 与 fault tolerance。
- [verl GitHub](https://github.com/verl-project/verl)：HybridFlow、后端矩阵和 recipes。
- [HybridFlow Paper](https://arxiv.org/abs/2409.19256)：verl 的 programming model 与资源映射。
- [verl Documentation](https://verl.readthedocs.io/en/latest/)：worker、placement、算法与性能调优。
- [Relax GitHub](https://github.com/redai-studio/Relax)：Ray Serve 六层架构、TransferQueue、DCS 与 omni-modal RL。
- [Relax Paper](https://arxiv.org/abs/2604.11554)：fully async omni-modal post-training 系统。
- [Miles GitHub](https://github.com/radixark/miles)：企业级 slime-derived framework、TITO/R3/weight sync。
- [Miles Documentation](https://miles.radixark.com/docs)：training backends、agentic rollout、low precision 与 fault tolerance。
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF)：Ray + vLLM、agent execution 与 RL algorithms。
- [OpenRLHF Documentation](https://openrlhf.readthedocs.io/)：安装、训练脚本与扩展接口。

**验证原则：** 这些框架变化非常快。记录 repo commit、container、trainer/rollout engine 版本和 recipe SHA，所有性能与稳定性结论都用目标模型和集群复现。
