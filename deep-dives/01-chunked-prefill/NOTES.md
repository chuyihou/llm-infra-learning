# Chunked Prefill 讲解文档（SGLang 为主，vLLM 对照）

- 验证基准：2026-09-23 对着最新代码确认 —— sglang main @ `5c154c21`，vllm main @ `c4cfd70`（v0 部分对照 tag v0.10.2）
- 听众：infra 同事，重点是 SGLang 的实现和怎么调，vLLM 作为对照
- 文末有"未验证/存疑"清单，讲的时候别当事实说

## 0. 为什么需要 chunked prefill（一句话开场）

- prefill 是 compute-bound（attention O(n²)），decode 是 memory-bound；长 prompt 一次算完的问题：
  - 整个 step 被一个 prefill 独占，所有 decode 请求原地等待 → TPOT 抖、ITL 长尾
  - attention 中间结果显存峰值 ~ chunk_size × seq_len，容易 OOM
- chunked prefill = 把长 prompt 切成固定大小的 chunk，逐个 step 算，每个 chunk 和 decode 请求混批（思想来自 Sarathi，arXiv 2308.16369）
- 效果：长 prompt 不再独占 GPU 一个 step，decode 每个 step 都能往前走 → TPOT 平滑
- 代价：同一个 prompt 要多个 step 才算完 → TTFT 上升；每个 chunk 的 attention 都要重扫一遍 prefix KV

## 1. SGLang 实现深挖

### 1.1 请求状态记账：`Req`（`python/sglang/srt/managers/schedule_batch.py:977`）

三个核心字段，chunk 切分永远围绕它们：

- `full_untruncated_fill_ids`：完整 prompt，切分永远参照它
- `prefix_indices`：已算出 KV、写进 paged KV pool 的 token 对应的 pool slot —— KV 进展的账本
- `extend_range = Range(start, end)`：本轮 forward 要算的 chunk，用 `set_extend_range`（schedule_batch.py:1530）设置

### 1.2 切分决策：`PrefillAdder`（`python/sglang/srt/managers/schedule_policy.py`）

- `_select_prefill_admission`（1383）：剩余 token = `len(full) - len(prefix_indices)`，超过 chunk 上限 `rem_chunk_tokens`（= `chunked_prefill_size`）就截断 `extend_len`，置 `is_chunked=True, max_new_tokens=0`
- `_commit_prefill_admission`（1466）：执行 `set_extend_range`；若 chunked 则 `adder.new_chunked_req = req`
- 注意这里是**两个预算**：`rem_input_tokens = max_prefill_tokens`（整批 prefill 总预算），`rem_chunk_tokens = chunked_prefill_size`（单个 chunk 上限）；mixed 时 decode token 也占预算

### 1.3 续算：同一时刻只有一个 chunked request

- `Scheduler.chunked_req` 是单例（scheduler.py:1321 初始化；4063 有 `assert self.chunked_req is None`）；`batch_result_processor.py:418` 注释原文："There is only at most one request being currently chunked."
- 下一轮续算在 `Scheduler._get_new_batch_prefill_raw`（scheduler.py:3921-3929）：**续算优先于 waiting queue** —— 先 `chunked_req.init_next_round_input()`，再 `PrefillAdder.add_chunked_req`（schedule_policy.py:1060）按本轮预算 `fit_chunk` 切下一块；算完返回 None（scheduler 清掉 `chunked_req`），没算完继续挂着

### 1.4 chunk 之间 KV 怎么连续

- KV 写 paged KV pool，`prefix_indices` 记录已写入的 slot
- `ScheduleBatch.prepare_for_extend`（schedule_batch.py:2691）每轮重建 forward 输入：`input_ids = get_fill_ids()[len(prefix_indices):]`（只取本 chunk 做 query），`seq_lens = extend_range.end`（prefix KV 从 pool 按 prefix_indices 读）

### 1.5 调度循环（`scheduler.py`）

- 主循环 `event_loop_normal`（1913）/ `event_loop_overlap`（1948），每轮调 `get_next_batch_to_run`（3637）
- 循环内顺序：
  - `process_pending_chunked_abort`（3534，chunked abort 延迟到调度步边界）
  - 上轮 prefill batch `merge_batch` 进 running（`chunked_req` 被排除，3656）
  - `stash_chunked_request`（3530）把上 chunk 的新 KV 插 radix tree
  - **prefill-first**：`get_new_batch_prefill`（3799）→ `_get_new_batch_prefill_raw`（3824），返回 None 才 `update_running_batch` 跑 decode（3769-3779）
- SGLang 默认 prefill-first；想让 decode 不被 prefill 饿死要显式开混批（见 1.7）

### 1.6 ForwardMode：纯 chunked prefill batch 还是 EXTEND

- 当前 main **已没有 `IS_CHUNKED_PREFILL`**（`forward_batch_info.py:177`）
- 纯 chunk prefill batch = `EXTEND`；混批 = `MIXED`（注释原文 "Contains both EXTEND and DECODE when doing chunked prefill"）
- `ScheduleBatch.mix_with_running`（schedule_batch.py:3101）产生 MIXED：把 running decode batch 里每个 req 看成 1-token extend 拼进同一 forward
- `is_mixed_chunk`（scheduler.py:1323）= `chunked_prefill_size is not None and enable_mixed_chunk`；混批时 `new_batch.mix_with_running(running_batch)`（4114-4138）
- `init_chunked_prefill`（1301）：`chunked_prefill_size <= 0`（含 CLI 的 -1）归一为 None 即禁用；multimodal + Transformers backend 强制禁用

### 1.7 调参旋钮（字段在 `python/sglang/srt/arg_groups/fields/schedule.py`）

- `--chunked-prefill-size`（schedule.py:59）：单 chunk 上限；CLI 默认 None → 按 GPU 显存解析（`arg_groups/memory_hook.py:100-176`）：<35GB（A10/4090/5090）=4096；<60GB（A100 40G/L40）=4096；<90GB（H100/A100 80G）=8192；<160GB（H20/H200）=8192；更大（B200/MI300）=16384；未知=4096；-1 禁用；DP attention 会再减半（server_args.py:304 注释）
- `--max-prefill-tokens`（schedule.py:73）：默认 **16384**；整批 prefill token 总预算，实际 bound = max(16384, context_len)
- `--max-running-requests`（schedule.py:37）：默认 None → profiling 解析（`model_executor/pool_configurator.py:1403`）：`int(full_token/context_len*512)` clamp 到 [2048, 4096] 再 `min(..., full_token//2)`；调大：并发升但 decode 易 OOM/频繁 retract；调小：省 KV 但并发受限
- `--max-total-tokens`（schedule.py:42）：默认 None → 按 mem-fraction 自动算；官方注明主要用于开发调试
- `--schedule-policy`（schedule.py:88）：默认 **fcfs**；选项 lpm/random/fcfs/dfs-weight/lof/priority/routing-key/hrrn/shortest-prefill-first；官方 tuning doc：共享 prefix 多用 lpm（调度开销大），否则 fcfs 开销最低
- `--enable-mixed-chunk`（schedule.py:209）：默认 **False**；打开后 prefill chunk 和 decode 混批（MIXED），decode 不被 prefill 饿死
- `--cuda-graph-max-bs-decode` / `--cuda-graph-max-bs-prefill`（`fields/exec_.py:507/510`）：默认 None → 按显存解析（如 H100 tp<4 时 decode max_bs=256，tp>=4 时 512）；调大：大 batch 也进 cuda graph；代价：多占显存，官方建议配合降 mem-fraction-static
- `--prefill-decode-interval`（schedule.py:63）：跑完一个 prefill batch 后先跑 N 轮 decode；decode-heavy 可用
- `--enable-dynamic-chunking`（schedule.py:66）：默认 False；PP 下按 latency 模型动态调 chunk 大小
- `--mem-fraction-static`：默认约 0.9；OOM 时降到 0.7~0.8，prefill/decode 都缓解但降并发/峰值吞吐（官方 tuning doc）

### 1.8 调参实战

- prefill-heavy（长 prompt 多）：`chunked-prefill-size` 往显存上限调（如 H100 上 8192），减少 chunk 数 → 长 prompt TTFT 降；但单 forward 变大、decode 侧 TPOT 抖 → 考虑开 `--enable-mixed-chunk`
- decode-heavy（短 prompt 高并发）：`chunked-prefill-size` 保持默认或调小；重点调 `max-running-requests` 和 `cuda-graph-max-bs-decode`
- chunk 太大症状：decode 请求 TPOT 尖刺（默认 prefill-first，decode 被大 chunk 挡住）；prefill attention 中间显存 OOM（~chunk_size × seq_len）；官方 doc：prefill OOM → 先降到 4096/2048
- chunk 太小症状：长 prompt 切成很多 chunk → 每个 chunk 一次 forward（launch 开销 + 每 chunk attention 遍历一遍 prefix KV）→ 长 prompt TTFT 变差
- 观察是否生效：`--log-requests` 看单请求 TTFT/ITL；prometheus（`--enable-metrics`，`observability/metrics_collector.py`）：`sglang:num_queue_reqs` / `num_running_reqs` / `gen_throughput` / `token_usage` / `cache_hit_rate`；scheduler 日志的 prefill batch stats（`metrics_reporter.py:364-396`：num_prefill_requests / sum_prefill_tokens / sum_prefill_kv_tokens）

### 1.9 与 RadixAttention / CUDA graph 的交互

- Radix：每 chunk 跑完 `stash_chunked_request` → `maybe_cache_unfinished_req`（`mem_cache/common.py:156`）→ `cache_unfinished_req(req, chunked=True)`：**未完成的 chunk 前缀也被插进 radix tree**；下轮 `init_next_round_input`（schedule_batch.py:1556）重新 `match_prefix` —— prefix cache 以 chunk 粒度渐进可用，其他请求可命中"还在 prefill 中"的前缀
- CUDA graph 是 per-phase backend（`model_executor/cuda_graph_config.py:32-52`：decode/prefill × full/breakable/tc_piecewise/disabled）：prefill 在 CUDA 默认 **BREAKABLE**（`default_prefill_backend`，cuda_graph_config.py:122），full prefill 按模型 opt-in；`full_prefill_max_req` 默认 = `chunked_prefill_size // 512`；不支持的形状回退 eager
- 逃生开关 `--disable-cuda-graph` 仍在官方 doc 里

## 2. vLLM 对应部分（v1，main @c4cfd70）

### 2.1 先说版本事实

- 当前 main **只有 v1**：`vllm/core/scheduler.py`（v0 的 `_schedule` / `_schedule_chunked_prefill` / `_schedule_prefills`）已从 main 删除；v0 结构只存在于历史 tag（如 v0.10.2）
- 当前实现位置：`vllm/v1/core/sched/scheduler.py`（3276 行），`class Scheduler`，主入口 `schedule()` 在 553 行

### 2.2 统一调度：没有 prefill/decode 两阶段

- `schedule()` 开头注释（555-565）原话：没有 "decoding phase" 也没有 "prefill phase"；每个 request 只有两个计数器，scheduler 每步给 request 分配 token 让前者追上后者：
  - `num_computed_tokens`：已计算过的 token 数（`vllm/v1/request.py:201`）
  - `num_tokens_with_spec`：`len(all_token_ids) + len(spec_token_ids)`（`vllm/v1/request.py:299`），即 prompt + 已产出 output + spec draft
- 这个统一公式天然覆盖 chunked prefill、prefix caching、spec decode

### 2.3 两个 budget + 调度顺序

- `token_budget = max_num_scheduled_tokens`（scheduler 意图），`input_budget = max_num_batched_tokens`（forward 物理上限）（571-574）；每 schedule 一个 request 就扣减（808-809，1304-1305）
- 顺序：**RUNNING 先，WAITING 后**；RUNNING 里 decodes 和未完成的 prefill chunk 混在一起按 FCFS 走
- decode 优先是隐含的：官方文档（`docs/configuration/optimization.md:53`）原话 "scheduling policy prioritizes decode requests. It batches all pending decode requests before scheduling any prefill operations." —— 机制上因为 RUNNING（含 decode）先消费 budget，prefill 只吃剩下的

### 2.4 切分算式

- RUNNING 中 request（657-675）：
  - `num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens`
  - `long_prefill_token_threshold` clamp（662-663）：设为 N 后单个 prefill chunk 最多吃 N 个 token，防长 prefill 饿死其他人；batch 里只有一个 request 时自动豁免（604-609 注释原话：only request 没人可饿死，就让它用满 budget）
  - 再 `min(剩余 budget)`，`min(max_model_len clamp)`
  - decode 自然得到 1（`num_tokens_with_spec - num_computed_tokens = 1`）；未完成的 prefill 得到 min(剩余 prompt, 剩余 budget) → 这就是 chunk
- WAITING 新 prefill（1065-1116）：`request_token_budget = min(token_budget, input_budget - draft_slots)`；`num_new_tokens = min(剩余 prompt, request_token_budget)`；`enable_chunked_prefill=False` 且 prompt 装不下 → break，整队卡住（head-of-line blocking）

### 2.5 乐观推进与修正

- `_update_after_schedule`（1571）：schedule 之后**立刻乐观推进** `num_computed_tokens`（1584），注释说这样下个 step 可以马上再 schedule 这个 prefill
- `is_prefill_chunk = num_computed_tokens < num_tokens + num_output_placeholders`（1589-1591），标记"没做完的 prefill chunk"，供 DP prefill balancing 等使用（640 行）
- spec token 被 reject → `update_from_output`（1906）回调修正

### 2.6 preemption = recompute，没有 swap

- `_preempt_request`（1526）：`_free_request_blocks` + `num_computed_tokens = 0`（1548），request 被 prepend 回 waiting 队列重算；v1 scheduler 全文**零处出现 "swap"** —— v0 的 CPU swap 机制被拿掉了
- 牺牲者选择：FCFS 下是 `running[-1]`（最后到达的）

### 2.7 调参旋钮（`vllm/config/scheduler.py` + `vllm/engine/arg_utils.py`）

- `max_num_batched_tokens`：每 step 物理 token 上限；字段默认 2048，但**实际按硬件+场景走** `get_batch_defaults`（arg_utils.py:2756）：B200/B300（≥160GB）→16384；H100/H200（≥70GB）→ LLM 类 16384 / API server 8192；其他 → LLM 类 8192 / API server 2048；A100 被显式排除在 H100 档之外（注释引 PR #17885：A100 上设太大反而掉 throughput）；"默认 2048"只对最小配置成立 —— 讲的时候要强调这是硬件相关的
- `max_num_scheduled_tokens`：默认 None → 等于 max_num_batched_tokens；比 batched 小的唯一场景是 speculative decoding（draft token 在 runner 里凭空多出来，要预留 headroom）；CLI `--max-num-scheduled-tokens`
- `max_num_seqs`：字段默认 128，实际大卡 1024 / 其他 256；另有新参数 `max_num_active_seqs`（默认 None）：只收紧 RUNNING 准入、不缩小 runner/CUDA graph 容量
- `enable_chunked_prefill`：v1 默认 True；手动关掉打 warning（"may cause the engine to crash or produce incorrect outputs"）；`verify_max_model_len` 强制 `max_num_batched_tokens >= max_model_len`，否则启动直接 ValueError
- `long_prefill_token_threshold`：默认 0（关闭）
- `scheduler_reserve_full_isl`：默认 True；admit 新 request 时按**完整 prompt 长度**检查 KV 是否放得下，而不是只看第一个 chunk —— 修 chunked prefill 下过度准入 → KV 打满 → 反复 preempt 的 thrash（对应 issue #37307 场景）
- `watermark`：默认 0.0；预留 KV block 比例做 headroom，减少 KV 吃紧时反复 evict
- `prefill_schedule_interval`：默认 1；DP 部署下每 N 个 step 才准入新 prefill，对齐各 rank forward 时间
- `max_num_queued_reqs` / `max_num_queued_tokens`：API server 层 503 拒绝阀，TTFT QoS 用（`max_num_queued_tokens ≈ target_TTFT × prefill_throughput`）
- `performance_mode="throughput"`：未手动设时，max_num_batched_tokens 和 max_num_seqs 都 ×2
- `scheduler_delay_factor`：v1 已删除（v0 参数）
- 注意：`max_num_batched_tokens` / `max_num_seqs` 进 `compute_hash` —— 改这两个值触发 torch.compile 重编译，线上热调有编译代价

### 2.8 调参实战（官方 `docs/configuration/optimization.md:53-80`）

- 调小（如 2048）：ITL 更好，prefill 对 decode 干扰小 → decode-heavy / 在线聊天
- 调大：TTFT 更好，一 batch 塞更多 prefill → prefill-heavy；throughput 最优 `max_num_batched_tokens > 8192`，尤其小模型+大卡
- 设太大：单 step prefill chunk 过大 → decode 被 piggyback 拖慢，ITL/TPOT 长尾变差；A100 上实测反而掉 throughput；极端大逼近 `max_num_seqs × max_model_len` 触发 warning
- 设太小：长 prompt 切很多 chunk → TTFT 上升，prefill 总吞吐降，但 ITL 最平滑 —— 在线低延迟场景就是想要这个 tradeoff
- prefill-heavy（长文档问答/batch job）：往大调（8192→16384 看卡），配 `long_prefill_token_threshold` 防单个超长 prompt 饿死其他人；考虑 `max_num_queued_tokens` 做 TTFT 保护
- decode-heavy（chat/多轮）：保持小（2048 档），decode 优先天然友好；KV 压力大调小 max_num_seqs 或 watermark>0 减 preemption
- KV 打满反复 preempt（吞吐突然崩）：先看过度准入 —— `scheduler_reserve_full_isl=True` 是默认已开的防护；再看 watermark 和 max_num_seqs

### 2.9 v0 vs v1（一页，听众问历史时用）

- v0（tag v0.10.2）：三路分发 —— `_schedule`（1467）按 `chunked_prefill_enabled` 选 `_schedule_chunked_prefill`（1345）或 `_schedule_default`（1217）；后者调 `_schedule_prefills`（1027，`token_chunk_size`）/ decode 路径；`SequenceData.num_computed_tokens`（`sequence.py:347`）
- v0 默认 prefill 优先、prefill/decode 不混批；开 chunked 后才变 decode 优先 + 混批；preemption 有 swap（`blocks_to_swap_in/out`）也有 recompute
- v1：单一 `schedule()`，decode 优先 + 混批是设计默认值；只有 recompute，无 swap；新增 `scheduler_reserve_full_isl` + `watermark` 反 thrash

## 3. SGLang vs vLLM 对照（讲解收尾用）

- 同源：都是 Sarathi 式（2308.16369），两边官方文档都引用；核心思想一致：decode 优先、prefill chunk piggyback、token budget 限每步计算量
- 旋钮数量：vLLM 一个 `max_num_batched_tokens` 既做 chunk 上限又做整批预算（另有 `max_num_scheduled_tokens` 管 spec headroom）；SGLang 拆成 `chunked_prefill_size`（单 chunk）+ `max_prefill_tokens`（整批）两个预算
- 默认策略：vLLM v1 天然混批、decode 优先；SGLang 默认 prefill-first，混批要显式开 `--enable-mixed-chunk`
- 并发 chunk：SGLang 同一时刻只跟踪一个未完成的 chunked request（续算优先于 waiting queue）；vLLM 队列里多个请求可各自 chunked，都在 RUNNING 里按 FCFS 吃 budget
- 记账风格：SGLang 显式三字段（full prompt / prefix_indices / extend_range）+ radix tree 渐进缓存未完成 chunk 前缀；vLLM 统一成 `num_computed_tokens` 追 `num_tokens_with_spec` 一个算式，prefill chunk / decode / spec decode / prefix-cache 续算全走同一条路
- 代价：vLLM scheduler.py 3000+ 行，各种 clamp（mamba 对齐、encoder budget、spec pad）挤在一个循环里；SGLang 切分逻辑分散在 PrefillAdder + Scheduler，但 chunked_req 单例让"谁在续算"一目了然
- 硬件默认值：两边 chunk 上限都按 GPU 显存解析，不是固定值；vLLM 还分 LLM 类 vs API server 场景，A100 被特殊对待（PR #17885）

## 4. Scheduler 代码阅读路线（今天看代码用）

SGLang（`python/sglang/srt/managers/`）：

- `scheduler.py`：`event_loop_normal`（1913）→ `get_next_batch_to_run`（3637）→ `_get_new_batch_prefill_raw`（3824，续算 3921-3929）/ `get_new_batch_prefill`（3799）
- `schedule_policy.py`：PrefillAdder 构造（在 scheduler.py 3902-3918）、`_select_prefill_admission`（1383）、`_commit_prefill_admission`（1466）、`add_chunked_req`（1060）
- `schedule_batch.py`：`Req`（977）、`set_extend_range`（1530）、`init_next_round_input`（1556）、`prepare_for_extend`（2691）、`mix_with_running`（3101）
- `forward_batch_info.py`：ForwardMode（177），EXTEND vs MIXED
- 调参字段：`arg_groups/fields/schedule.py`；显存解析：`arg_groups/memory_hook.py:100-176`

vLLM（`vllm/v1/`）：

- `v1/core/sched/scheduler.py`：`schedule()`（553，注释 555-565）→ RUNNING 切分（657-675）→ WAITING 切分（1065-1116）→ `_update_after_schedule`（1571）→ `_preempt_request`（1526）
- `v1/request.py`：`num_computed_tokens`（201）、`num_tokens_with_spec`（296-299）
- `config/scheduler.py`：SchedulerConfig 默认值；`engine/arg_utils.py`：`get_batch_defaults`（2756）、throughput ×2（2985）
- 官方调参：`docs/configuration/optimization.md:53-80`；SGLang 官方 tuning：docs.sglang.ai 的 hyperparameter_tuning

## 5. 可能被问的刁钻问题

- chunk 太小为什么反而伤 TTFT？每个 chunk 一次 forward launch 开销 + attention 每 chunk 重扫一遍 prefix KV；chunk 数 = ceil(prompt_len / chunk_size)
- chunk 太大为什么伤 TPOT？SGLang 默认 prefill-first 下 decode 请求要等整个大 chunk 的 forward 跑完；vLLM 里大 chunk 占满 budget，decode 被 piggyback 拖慢 → ITL 尖刺
- 为什么 SGLang 只允许一个 chunked request？简化续算记账：续算优先、不插队 waiting queue；代价是多个长 prompt 串行 prefill —— vLLM 允许多个并发 chunk
- 关掉 chunked prefill 会怎样？vLLM：prompt 超 budget 整队 head-of-line blocking，且官方 warning 可能 crash；SGLang：`--chunked-prefill-size -1`，长 prompt 一次算完，显存峰值和 TPOT 抖都回来
- `max_num_batched_tokens` 改了为什么服务抖一下？进 `compute_hash`，触发 torch.compile 重编译（vLLM）
- prefix cache 和 chunked prefill 打架吗？SGLang 不打：未完成 chunk 的前缀也进 radix tree，以 chunk 粒度渐进可用
- DP attention 为什么 chunk 要减半？（server_args.py:304 注释；展开讲 DP 下每 rank 的 token 分布）
- A100 为什么 vLLM 默认不给 16384？PR #17885：A100 上设太大反而掉 throughput —— 显存带宽/计算比和 H100 不同，大 chunk 的 attention 中间结果吃带宽

## 未验证 / 存疑（讲的时候别当事实说）

- chunk 大小对 TTFT/TPOT 的定量关系是基于机制的推导，官方无定量公式
- SGLang `--enable-torch-compile` 与 chunked prefill 的确切限制没找到代码 guard
- SGLang `enable_mixed_chunk` 与 overlap/spec decode 组合的边界行为没深挖
- vLLM 硬件默认值里 TPU/CPU 档只读了代码未实测
