# Chunked Prefill: Turn Long Prompts into *Schedulable* Work

Chunked Prefill is not a new attention mechanism, nor does it split the model context. Its job is simpler: instead of forcing a long prefill to finish in one shot, it returns control to the scheduler at safe boundaries.

> *The key idea: Chunked Prefill does not reduce the total work. It controls how frequently the system regains an opportunity to schedule something else.*

**What this article explains**: Why long prompts interfere with online decode; why chunking preserves model semantics; which workloads benefit and which pay overhead; and how SGLang represents the relevant state, budgets, and parameters.

**Intended audience**: Infrastructure engineers familiar with Transformers and KV caches who care about TTFT, ITL, throughput, GPU memory, and scheduler implementation.

**Code baselines**: SGLang c19dc43 · vLLM 4edb551; verified on 2026-09-22.

---

*This post is adapted from the [full interactive deep dive](https://chuyihou.github.io/llm-infra-learning/deep-dives/01-chunked-prefill/) — with live scheduler diagrams and a chunk-size lab. Code references below were verified against the pinned commits.*

## 01 / WHY — Why Chunked Prefill Exists

The problem is not simply that long prompts are slow. Prefill and decode stress the GPU in different ways, while an online scheduler must protect both time to first token for new requests and streaming latency for active requests.

### 1. Prefill and Decode stress different hardware bottlenecks

**Prefill** processes many prompt tokens at once. In a linear layer, the input can be viewed as `[L, H]`, where `L` is the number of Prompt tokens in this round,`H` is the hidden size. Multiplying it by the weight matrix produces a large GEMM. As `L` grows, more tokens reuse the same weights and GEMM arithmetic intensity rises. Meanwhile, self-attention compute grows approximately quadratically with sequence length: `L²` . Past the hardware roofline crossover, prefill usually shifts from underutilizing compute units toward being ** compute-bound**.

**Decode** generates roughly one token per sequence per iteration. At small batch sizes, linear layers resemble GEMV or very thin GEMMs: little math is performed even though large model weights must be read, while attention also reads the entire previous KV cache. As context length `S` grows, each new token reads more KV. Decode therefore tends to hit HBM bandwidth first and remain ** memory / I/O-bound**.<sup>[[1]](#ref-sarathi)</sup>

*This is a mechanism diagram, not a benchmark. The actual ridge point and workload paths depend on model architecture, batch shape, precision, kernels, peak GPU compute, and HBM bandwidth.*

**Approximate KV bytes read per generated token** 2 × N<sub>layers</sub> × S × N<sub>kv_heads</sub> × d<sub>head</sub> × bytes<sub>element</sub>The factor of 2 accounts for K and V. This approximation ignores cache hierarchy, paging, and kernel fusion; it only shows why traffic grows linearly with context length S.

“More tokens” means different things in the two phases

In prefill, more tokens means more queries computed in parallel. In decode, a longer context mainly increases the KV history read by each query.

Longer contexts by themselves do not make Decode automatically compute-bound

It usually makes Decode more bandwidth limited. Common variables that make Decode tend to be compute-bound are larger batches, more parallel tokens, or kernels that significantly increase data reuse.

### 2. TTFT versus ITL: competing goals on one GPU

**TTFT (Time to First Token)** measures the delay from request arrival to its first output token; ** ITL (Inter-Token Latency)** measures the gap between adjacent streamed tokens. Running a new request's full Prefill earlier improves its TTFT, but the same long forward pass delays the next Decode step for active requests and can create ITL spikes.

The real concern is not only average ITL, but its **variance and tail**. Users notice sudden pauses in a stream even when average tokens/s barely changes. Yet always prioritizing Decode leaves newly arrived long prompts queued and worsens TTFT. The scheduler must balance both.

*Chunked Prefill targets this non-preemptible window. It prevents the scheduler from having to choose between one entire prefill and all pending decode work at once.*

**Background: TP is communication-bound; PP is bubble-bound ADDITIONAL READINGSupplementary background on multi-GPU deployment and pipeline bubbles**

When a model does not fit on one GPU, Tensor Parallelism (TP) and Pipeline Parallelism (PP) distribute it across multiple GPUs. Their bottlenecks differ, which determines where Chunked Prefill can help.

#### Tensor Parallelism

TP shards each layer’s matrices across GPUs. Nearly every layer requires an All-Reduce, Reduce-Scatter, or All-Gather, so performance is sensitive to inter-GPU bandwidth and collective latency. Larger TP degrees make communication more likely to dominate the critical path.

#### Pipeline Parallelism

PP assigns different layers to different stages and transfers activations only at stage boundaries. A stage must still wait for its upstream microbatch; pipeline fill, drain, and stage-time imbalance create bubbles in which a GPU is ready but has no work.

**Expand: how pipeline bubbles form**

STAGE

t0

t1

t2

t3

t4

t5

GPU 0

M0

M1

M2

BUBBLE

BUBBLE

BUBBLE

GPU 1

WAIT

M0

M1

M2

BUBBLE

BUBBLE

GPU 2

WAIT

WAIT

M0

M1

M2

BUBBLE

GPU 3

WAIT

WAIT

WAIT

M0

M1

M2

The green diagonal shows a microbatch advancing through the stages; gray cells are fill and drain bubbles. Too few microbatches—or large differences in their duration—increase idle time.`Idealized utilization ≈ m / (m + p − 1) m=3, p=4 → 50%`

More, more-uniform microbatches can reduce bubbles, at the cost of additional scheduling and communication overhead. Later we will see how Chunked Prefill turns a very long prefill into more regular work units. It reduces only part of the bubble; it cannot eliminate fill/drain, communication, or an imbalanced stage partition.

## 02 / CORE MECHANISM — How Chunked Prefill Works

The mechanism has three layers: split prefill into resumable chunks; combine chunks with decode work in mixed batches; then shape chunks around long-context cost and the PP stage cadence.

### 1. Split a full prefill into smaller chunks

Let the prompt length be `L` and the per-request chunk cap be `C`. Without chunking, the Scheduler submits `[0,L)` once. With chunking, it submits `[0,C)`, `[C,2C)`, and so on until the prompt is complete. The request therefore needs roughly `ceil(L/C)` Prefill iterations.

This does not turn one message into unrelated prompts. After the first chunk, its K/V is stored in the KV cache. The next chunk computes Q/K/V only for new tokens, while each new query can still attend to the entire legal prefix. Intermediate chunks are not sampled; normal Decode starts only after all prompt tokens have been processed.

*C=4: Process up to 4 new tokens per round; each boundary allows the Scheduler to re-evaluate Decode, waiting queue, budget and KV.*

### Why chunking preserves model semantics

Consider the second chunk, `T4…T7`. It generates queries for only four new tokens, but each query's Key/Value visibility still follows its original global position. Q7 can read K0…K7, not merely K4…K7. Three invariants preserve equivalence with a one-shot Prefill.

**KV is appended continuously Open example The second round does not recalculate T0–T3; it reads the historical KV and appends the new KV from T4–T7. KV 0–3→KV 4–7**

Assume `chunk_size=4`. Physical KV pages can be dispersed, but the logical token order cannot be changed.

ITERATION 1

Write KV0Write KV1Write KV2Write KV3

ITERATION 2

Read KV0Read KV1Read KV2Read KV3Write KV4Write KV5Write KV6Write KV7

**Check:** after the second iteration, the logical cache covers `KV[0:8)`. It was built across two iterations, but still represents one sequence.

**Positions do not reset Open example Chunks are just execution boundaries. T4 for the second round is still at global position 4, rather than starting over at position 0. 0 1 2 3|4 5 6 7**

RoPE or other position encoding must see the original sequence position, otherwise the second chunk will be mistaken for a new prompt.

Correct

T0·p0T1·p1T2·p2T3·p3T4·p4T5·p5T6·p6T7·p7

Incorrect reset

T4·p0T5·p1T6·p2T7·p3

**Check:** the starting position equals the number of prompt tokens already processed; here, `prefix_len=4`.

**The causal mask does not stop at chunk boundaries Open example The second round of Query can not only read this chunk, but also read all legal history on the left side of the boundary. K0…K3+K4…Ki**

For a query at global position `i`, the rule remains: it may read keys satisfying `j ≤ i`. The chunk boundary introduces no new attention mask.

Q4 can read

K0K1K2K3K4K5K6K7

Q7 can read

K0K1K2K3K4K5K6K7

**Check:** Q7 can see K0–K7, not just K4–K7; KVs belonging to other sequences remain invisible.

**Correctness conclusion:** as long as KV state, global positions, and causal visibility match a one-shot Prefill, chunking does not change the model's dependency graph. Different kernels or floating-point reduction orders may still produce small, non-bitwise-identical numerical differences.

### 2. Mixed batching: put decode and prefill chunks in one forward pass

Chunking alone only creates more scheduling boundaries. The scheduler can insert a decode-only iteration between two chunks—interleaving—or go further and place decode tokens from running requests together with a waiting request’s prefill chunk in **the same batch and the same model forward pass**. That is mixed batching.

A common decode-first policy reserves one Decode token for each running request, then fills the remaining token budget with one or more Prefill chunks. Prefill continues to advance without forcing streaming requests to wait behind a full-prompt forward pass.

![TensorRT-LLM Chunked Prefill before and after comparison. The complete Prefill in the first half causes other requests to wait, while the Prefill in the second half is cut into multiple chunks so that Prefill and Decode for different requests can be advanced in an interleaved manner.](https://developer-blogs.nvidia.com/wp-content/uploads/2024/11/Chunked-Prefill-Process.png)

*It shows that chunking shortens the interval during which the scheduler cannot intervene, making prefill and decode work from different requests easier to interleave. The timeline alone does not tell us which tokens share one forward pass.[Source: NVIDIA Technical Blog](https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/)*

*The upper half uses chunks but still runs prefill and decode in separate iterations. The lower half is mixed batching: latency-sensitive decode tokens are placed first, then the remaining token budget is filled with a prefill chunk.*

one forward = {Decode tokens} ∪ {Prefill chunk tokens}

This is the precise meaning of mixed batching. A common decode-priority policy places running decode work first, then fills the remaining token budget with prefill chunks. A production scheduler must also check KV capacity, priority, CUDA Graph coverage, and feature compatibility.

#### Why mixing can improve GPU efficiency

The two workloads place complementary pressure on the hardware. Mixed batching can reduce queuing and may also raise the effective arithmetic intensity of the combined forward pass.

Prefill chunk

compute-heavy

Many queries run in parallel, weights are reused across tokens, and larger GEMMs utilize Tensor Cores more effectively.

Decode tokens

memory-heavy

Each sequence contributes only one query per decode iteration. At small batch sizes, compute utilization is low while the cost of reading model weights and a long KV history dominates.

Mixed forward

better balance

Combining latency-sensitive decode tokens with a compute-dense prefill chunk lets shared linear layers run on a larger token batch and amortize weight reads. Within a suitable range, adding decode tokens can have modest incremental cost.<sup>[[1]](#ref-sarathi)</sup>

**This does not literally mean “compute and I/O run at the same time.”** The framework unifies token metadata and can merge linear layers into larger GEMMs, while Prefill attention and Decode attention may still use different kernels. The gain depends on kernel support, batch shape, KV access, and graph capture.

| execution mode | Contents of one forward pass | What it addresses |
| --- | --- | --- |
| Full Prefill | Entire Prompt Prefill. | The path is simple, but the rescheduling window is the longest. |
| Chunked, non-mixed | Only Prefill chunk, or only Decode. | Decode can be inserted at chunk boundaries, but will still wait during each Prefill forward. |
| Chunked + mixed | Decode token coexists with Prefill chunk. | Use the remaining token budget to continue advancing Prefill while protecting Decode cadence. |

> **Do not conflate chunking with mixed batching** Chunking determines how soon control returns to the Scheduler. Mixed batching determines whether Prefill and Decode execute together in the next iteration. Enabling chunking does not automatically enable mixed batching.

### 3. Multimodal: how image and video tokens enter Chunked Prefill

A multimodal request does not feed JPEG pixels or an entire video directly into the LLM. Images are typically resized or tiled, videos are frame-sampled, and a vision encoder produces visual features. The processor then expands a prompt placeholder into a span of visual embeddings. **Ordinary Chunked Prefill partitions the expanded LLM-side token/embedding sequence—not raw pixels or video frames, and not automatically the vision encoder.**

There are therefore at least three granularities: media preprocessing, vision-encoder item/cache management, and the LLM Prefill token budget. Image resolution and tile count—or sampled video frames multiplied by visual tokens per frame—can turn a short-looking text prompt into a long Prefill. Video is especially likely to create a burst of visual tokens.

***Tuning order:** decompose TTFT into media fetch/decode → preprocessing → vision encoder → queueing → LLM Prefill → first-token release. If visual-token volume is the problem, first adjust resolution, tiling, or frame sampling within quality constraints, then tune the LLM chunk size. Otherwise encoder latency can be mistaken for ineffective Chunked Prefill.*

**Deep optimization: turn fixed-token chunks into near-constant-time microbatches for long contexts and PP ADDITIONAL READINGSGLang DynamicChunkSizer, latency model, and PP cadence**

The long-context problem is that **equal-size chunks become progressively slower later in the prompt**. Let cached history length be `H` and the next chunk contain `x` new tokens. Those queries attend over roughly `H+x` keys and values, so the dominant incremental attention work grows approximately as `x(H+x)`. With fixed `x`, per-iteration latency rises as `H` grows.

This hurts two goals at once. The non-reschedulable window grows again, reintroducing Decode ITL jitter. Under PP, consecutive microbatches become uneven, so stages wait around the slower late-prompt chunks. The target should therefore shift from “always process 4K tokens” to “keep each chunk near a target duration.”

SGLang's empirical latency model

f(l) = a·l² + b·l + c

ΔT(H, x) = f(H+x) − f(H) = a·x² + (2aH+b)·x

Choose x such that ΔT(H, x) ≈ T_target

`f(l)` is not a theoretical FLOPs equation; it is fitted to measured Prefill latency for the current model, dtype, kernels, and GPU. As `H` grows, the positive root `x` that satisfies the same `T_target` naturally becomes smaller.

*These steps map directly to the [DynamicChunkSizer source](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler_components/dynamic_chunk_sizer.py). The result combines an empirical model with safety constraints; it is more than a single `min(remaining_tokens, cap)`.*

#### How this optimizes long contexts

First, query-side activations, temporary workspace, and admission work are bounded by `x` rather than expanding with the entire prompt at once. Second, the predictor reduces `x` as history grows, bringing late-chunk latency back toward the target and preventing the non-preemptible window from growing again. Third, page/tile alignment avoids fragmentation and unfavorable kernel shapes.

This does not compress the final KV cache: every layer still stores K/V for the complete prompt, so capacity grows roughly linearly with total context. It controls **new work and temporary memory per iteration**. Because smaller late chunks add iteration, launch, and metadata overhead, the implementation uses smoothing, lower bounds, and alignment instead of shrinking without limit.

#### How this reduces PP bubbles

PP needs multiple microbatches occupying different stages concurrently. In an idealized forward-only pipeline with `p` stages and `m` equal-duration microbatches, utilization is approximately `m/(m+p−1)`. Chunking increases `m`; dynamic sizing also keeps successive microbatch durations closer, reducing the **extra imbalance bubble caused by slower late-context chunks**.

*The long-tail microbatches in the upper row will lengthen the beats of subsequent stages together; the lower row uses different token numbers in exchange for closer execution times, making it easier for multiple in-flight microbatches to overlap stably.*

> **Important implementation detail: this is not closed-loop control of the slowest stage** In the pinned commit, PP0 profiles synthetic Prefill inputs, broadcasts the samples to all ranks, and each rank fits the same model. This improves PP cadence through consistent, near-constant-time chunks; it does not measure every stage independently and feed the bottleneck stage back into an online controller.<sup>[[4]](#ref-dynamic)</sup>

It targets

Iteration-latency drift from long history, per-iteration activation/workspace peaks, and extra bubbles from uneven PP microbatch duration.

It cannot eliminate

PP's inherent fill/drain bubbles, uneven layer partitioning, cross-stage communication, final KV capacity, or inefficiencies in the kernel/backend itself.

It may add

Forward-launch, scheduling, and metadata overhead. If the fitted model differs from the online mixed workload, actual latency may miss the target.

Online verification

At the same time, observe the chunk latency distribution, each PP stage time, pipeline idle, p99 ITL, long prompt TTFT, tokens/s and KV/activation headroom.

## 03 / IMPACT & LIMITS — What Problems Does Chunked Prefill Solve, and Which Metrics Can It Improve?

Its most direct effect is to bound how long and how many resources one prefill occupies in an iteration. That can improve decode tail latency, goodput, and PP utilization, but it does not reduce the model’s total required computation or final KV-cache capacity.

Evaluate three categories separately: the per-iteration boundaries chunking controls directly; system metrics that may improve with appropriate scheduling and mixed batching; and model work that no partitioning scheme can remove.

| Problem or metric | Typical effect when enabled | Why it changes | Conditions and limits |
| --- | --- | --- | --- |
| Problems addressed and metrics that may improve |  |  |  |
| Decode p95 / p99 ITL, TPOT | Usually lower | The long Prefill is split into shorter non-preemptible windows, and Decode gets the next scheduling opportunity faster. | Chunks must be small enough; the Scheduler must actually prioritize or mix in Decode at boundaries. |
| ITL Fluctuation and Generation Stall | Usually lower | A single long prompt no longer monopolizes one long forward pass, limiting the duration of interference. | If the chunk is still large, or multiple prefills continuously occupy the budget, spikes will still occur. |
| Peak execution time of a single round | Lower and more predictable | Each round processes at most one prefill chunk of limited size, making the Scheduler cadence more predictable. | For long contexts, sizing should account for history length; the latency of a fixed-token chunk still grows over time. |
| Activation / Workspace Peak | Usually lower | Query-side activations and temporary workspace are bounded by the current chunk rather than the entire prompt at once. | The specific peak value depends on Attention backend, CUDA Graph, parallel mode and batch composition. |
| Throughput and GPU utilization | May improve | Mixed Batching allows memory-heavy Decode and compute-heavy Prefill chunks to share forward and reduce idle token budget. | Simple slicing does not guarantee throughput improvement; chunks that are too small will increase launch, metadata, and scheduling overhead. |
| SLO Goodput | Often more useful than raw throughput | Lower ITL tails can allow more requests to satisfy TTFT and ITL SLOs at the same provisioned capacity. | Test against the real distributions of input length, output length, arrival rate, and concurrency. |
| PP Imbalance Bubble | May decrease | More, latency-shaped microbatches make a stable pipeline cadence easier to maintain. | Requires multiple microbatches in flight; it cannot eliminate fill/drain, communication, or an imbalanced layer partition. |
| Long Prompt TTFT | May improve or regress | Better queuing and batching may shorten the wait; but more iteration, launch and decode queue jumping may also delay the first token. | TTFT is not a guaranteed win; measure both p50 and p99. |
| What chunking does not change or eliminate |  |  |  |
| Theoretical total FLOPs / Attention dependence | Same asymptotic order | Subsequent chunks still need to read the complete legal prefix; the same dependency graph is just executed in multiple rounds. | Kernel shape, padding, and fusion can change wall-clock time, but do not remove the model’s logical work. |
| Final prompt KV-cache capacity | Essentially unchanged | The K/V of each layer of the complete prompt still needs to be saved eventually, and the capacity still grows with the context length. | Chunking reduces per-iteration temporary peaks. KV compression, quantization, and offload address a different problem. |
| Model weights and output dependencies | Unchanged | When KV, global position and causal mask are correct, Chunked Prefill retains the same logical dependencies as full Prefill. | Different kernels and floating-point reduction orders may cause small numerical differences that are not bitwise-identical. |
| Fundamental TP/PP communication | Does not disappear | TP collectives and PP activation transfers remain. Chunking mainly changes their granularity and timing. | Insufficient network bandwidth or uneven stage partition still requires independent optimization. |

How to decide whether to enable it

Do not compare only average tokens/s. At minimum, examine p95/p99 ITL, the TTFT distribution, SLO goodput, peak iteration latency, activation/workspace headroom, and PP stage idle time. Chunked Prefill redistributes when work happens and bounds per-iteration peaks; it does not reduce the total model workload.

### Takeaway: optimize an operating point, not the fastest isolated prefill

Full Prefill tends to optimize **finishing one prompt as quickly as possible in isolation**. Chunked Prefill changes the objective to finding an acceptable operating point across TTFT, Decode ITL, online throughput, and peak memory. The rule is not “smaller is always better,” but ** choose the largest chunk that still meets p99 ITL and GPU-memory constraints**, avoiding unnecessary iteration and launch overhead.

## 04 / DESIGN & IMPLEMENTATION — How Four Policies Jointly Shape Chunked Prefill

Chunk size answers only “how much may run in this iteration.” A production scheduler must also decide which requests enter, how Prefill and Decode share a batch, and when unfinished work resumes.

The four policies below are engineering responsibilities, not necessarily four classes with matching names. They make a joint decision within each scheduling iteration: one policy’s output becomes another policy’s budget or constraint.

POLICY 01

### Chunk SizingWhich token range does this Prefill specifically advance this round?

#### Inputs

Remaining tokens, history/prefix length, static cap, target iteration latency, page/tile granularity, current batch-token budget, and context limit.

#### Decision

Choose `extend_range=[start,end)`; determine whether it is an intermediate or final chunk, and how many KV pages to reserve.

#### Invariants

`0<x≤remaining`; positions remain continuous; alignment never crosses the context limit; charge the budget for actual new tokens, never the original prompt length again.

#### Metrics to watch

Chunk latency distribution, number of chunks per request, p99 ITL, long prompt TTFT, activation peak, kernel shape and launch overhead.

Typical failures

**Too large:** the non-preemptible window, ITL spikes, PP imbalance, and per-iteration memory peak all return. ** Too small:** iteration, launch, metadata, and scheduling costs consume the benefit.

SGLang mapping

`chunked_prefill_size` provides the base cap. The PP dynamic path predicts the next chunk from `history_len`, then applies `page_size / 64` alignment, lower bounds, and capacity clipping.

POLICY 02

### New-work AdmissionIn addition to the existing Decode and continuation, what new prompts can be started?

#### Inputs

Waiting-queue order, priority/arrival time, prefix-cache hits, input tokens still to compute, current and future KV budget, request slots, and compatibility constraints such as LoRA or grammar.

#### Decision

Produce the current `can_run_list`, or stop admitting work when token, KV, slot, or tile budgets are exhausted; optionally choose lower-priority work to preempt.

#### Invariants

KV allocation and token budget are feasible at the same time; the real extend length is recalculated after cache match; temporary locks, slots or staged metadata are rolled back when the request is rejected.

#### Metrics to watch

queue age, admission reason, prefix hit length, KV usage, actual Prefill batch size, number of skipped and preempted requests.

Typical failures

Pure FCFS can miss valuable cache hits; favoring only short requests or cache hits can starve long prompts; checking only current input budget while ignoring future decode KV can over-admit work and trigger retraction later.

SGLang mapping

`SchedulePolicy.calc_priority()` sorts first; `PrefillAdder` maintains remaining input/chunk-token budgets, KV memory, request slots, and backend tile budget in one place.

POLICY 03

### Batch CompositionIs this round Prefill-only, Decode-only, or Mixed Forward?

#### Inputs

Running Decode number, batch token budget, selected Prefill chunks, mixed switch, CUDA Graph shape, and feature matrix such as logprob, input embeds, beam, spec decode, etc.

#### Decision

Choose the forward mode. A common decode-first policy places one decode token for every running request, then fills the remaining capacity with prefill tokens.

#### Invariants

Each Decode row still has only one new token; Prefill rows maintain their own prefix / extend length; after merging, input, position, KV location and output split must correspond one-to-one.

#### Metrics to watch

Mixed forward proportion, number of Decode/Prefill tokens in each round, CUDA Graph hit, graph fallback reasons, kernel time, ITL and tokens/s.

Typical failures

If mixed batching is enabled without kernel or metadata support, the result may be a shape mismatch, incorrect sampling, or graph fallback. Too much decode reserve stalls prefill; too little recreates ITL spikes.

SGLang mapping

`PrefillAdder` uses `num_mixed_decode_tokens` to pre-charge the input and chunk budgets. Once compatibility checks pass, `mix_with_running()` converts each Decode row to a one-token extend and sets the forward mode to `MIXED`.

POLICY 04

### Continuation & PreemptionWhen should a partially processed prompt that already owns KV resume?

#### Inputs

Partial request's prefix/extend progress, occupied KV, defer times and age, waiting queue priority, free pages, abort/pause status, PP microbatch status.

#### Decision

Resume first, defer, age, preempt another request, or retract and recompute under memory pressure. Normal sampling/decode begins only after the final prompt chunk.

#### Invariants

continuation cannot lose or duplicate KV; intermediate chunks are not sampled in advance; abort, pause, weight update, and PP must safely release or retain resources across microbatch paths.

#### Metrics to watch

continuation wait time, number of consecutive defers, chunks/requests, retract and recompute tokens, KV residency, partial request age.

Typical failures

If new requests continually jump ahead, a partial prefill can hold KV for a long time without reaching its first token. Overprotecting continuations hurts fairness for new work. Incorrect cleanup can also cause KV leaks or double frees.

SGLang mapping

`chunked_req` stores the continuation explicitly. Before admitting new work, the Scheduler calls `add_chunked_req()` and updates `inflight_middle_chunks` and `contains_last_prefill_chunk`.

Tune the four policies together

Sizing bounds per-iteration cost; admission chooses who receives resources; composition decides whether prefill and decode share a forward pass; continuation prevents partial requests from starving. A chunk-size sweep alone cannot explain problems caused by queue priority, KV admission, or mixed-feature compatibility.

### The four policies form a feedback loop, not four independent switches

This is not a fifth policy. It explains how the four policies interact. They do not form a strictly one-way pipeline: **Batch Composition must first declare Decode reserve**, so admission can know how much prefill budget remains. After execution, ** Continuation brings the partial request back to the next round**. The correct model is feed-forward decision making plus cross-iteration feedback.

| Order | Primary policy | Consumes | Hands off |
| --- | --- | --- | --- |
| 0. Reserve mandatory work | 03 Composition + 04 Continuation | `running_batch` plus an existing `chunked_req`, fairness/age | Decode reserve and continuations that must be tried first |
| 1. Size each request | 01 Chunk Sizing | remaining tokens, history length, static/dynamic cap, alignment granularity | candidate `extend_range` with KV page demand |
| 2. Decide admission | 02 Admission | Candidate demand, remaining token/slot/KV budget, priority and prefix hit | `can_run_list`; or a skip, stop, or preempt decision |
| 3. Materialize the batch | 03 Batch Composition | Accepted Prefill chunks, running Decode, feature compatibility | Prefill-only/Mixed `ScheduleBatch` with complete metadata |
| 4. Commit next state | 04 Continuation | forward results, completion interval, sample/abort status | `chunked_req`, `running_batch` or complete/free |

> **The three critical couplings** Sizing determines the demand seen by admission. Composition’s decode reserve reduces admission’s available budget. Continuation influences both current priority and the next iteration’s input. Break any link and the system risks budget mis-accounting, KV leaks, premature sampling, or starvation of partial requests.

## 05 / SGLANG — How SGLang Implements Chunked Prefill in Its Scheduler

SGLang represents the scheduler state explicitly: new Prefill work lives in `waiting_queue`, active Decode work in `running_batch`, and an unfinished Prefill continuation in `chunked_req`.

*Interactive demo — try it live on the [full interactive version](https://chuyihou.github.io/llm-infra-learning/deep-dives/01-chunked-prefill/).*

### Code map: jump from the diagram to source

All links are pinned to commit `c19dc43`. On a first pass, follow ①→②→③→④→⑤ through the main scheduling path; then use the rightmost column to inspect budget accounting, mixed batching, execution, and result write-back.

| Reading order | Diagram location | Source entry point | What to inspect |
| --- | --- | --- | --- |
| ① Scheduler heartbeat | Iteration entry point | [`run_event_loop()`](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler.py#L1862) [`event_loop_normal()`](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler.py#L1913) | The outer loop of request entry, batch selection, run batch, and processing of results. |
| ② Batch selection | Request state → Scheduler | [`get_next_batch_to_run()`](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler.py#L3622) | When to continue running Decode and when to try to generate a new Prefill batch. |
| ③ Prefill path | Priority / size / continuation | [`get_new_batch_prefill()`](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler.py#L3790) [`_get_new_batch_prefill_raw()`](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler.py#L3817) | Run `calc_priority`, choose dynamic size, construct the adder, try `chunked_req` first, then scan the waiting queue. |
| ④ Budget & admission | PrefillAdder / KV gate | [`PrefillAdder`](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/schedule_policy.py#L620) [`add_chunked_req()`](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/schedule_policy.py#L1022) · [`add_one_req()`](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/schedule_policy.py#L1232) | How `rem_input_tokens`, `rem_chunk_tokens`, memory, request slots, and tile budgets are charged together. |
| ⑤ Batch materialization | ScheduleBatch / Mixed | [`mix_with_running()`](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/schedule_batch.py#L3096) | How to convert Decode row into 1-token extend, and how to merge seq length, KV location and forward metadata. |
| ⑥ PP dynamic size | DynamicChunkSizer | [`DynamicChunkSizer`](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler_components/dynamic_chunk_sizer.py#L38) [`predict()`](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler_components/dynamic_chunk_sizer.py#L115) | Startup profiling, quadratic latency model, history-aware chunk prediction, and safety clipping. |
| ⑦ Execute | ModelRunner / GPU | [`run_batch()`](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler.py#L4311) | How do the batches produced by Scheduler enter the forward, sampling and overlap paths. |
| ⑧ Feedback | Result → next state | [`process_batch_result()`](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler.py#L4738) [`update_running_batch()`](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler.py#L4167) | Sampling results, completing requests, running batch compression, and how the next round of status is formed. |

This design makes “is there an unfinished chunk to resume?” a direct Scheduler input. Each Prefill scheduling iteration chooses the current chunk size, creates `PrefillAdder`, tries to advance the existing `chunked_req`, and then scans the waiting queue according to the scheduling policy. It materializes a `ScheduleBatch` and each request's `extend_range`, allocates KV, and—when configured—mixes the Prefill batch with running Decode work.

`waiting_queue`

Requests not yet admitted to the current Prefill iteration. Schedule policy, priority, and prefix-cache state determine scan order.

new work

`running_batch`

Usually contains active Decode requests. With mixed chunking enabled, their Decode tokens consume part of the current iteration's budget first.

decode work

`chunked_req`

A request whose prompt is only partially processed and whose partial KV remains resident. SGLang tries to resume it before scanning new waiting work.

continuation

`PrefillAdder`

Centralizes remaining input-token, chunk-token, KV-memory, and related budgets to decide whether a request fits the current Prefill batch.

admission gate

scheduler.py · The critical path of a round of Prefill scheduling (simplified excerpt)[upstream @ c19dc43](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler.py#L3790)

```
# 1. Static chunk cap; PP may override it dynamically from history length
chunked_prefill_size = self.chunked_prefill_size
if self.chunked_req is not None and self.dynamic_chunk_sizer:
    dynamic_size = self.dynamic_chunk_sizer.predict(history_len)
    if dynamic_size is not None:
        chunked_prefill_size = dynamic_size

# 2. PrefillAdder accounts for batch, chunk, mixed-Decode, and KV-memory constraints together
adder = PrefillAdder(
    ..., self.max_prefill_tokens, chunked_prefill_size,
    running_bs if self.is_mixed_chunk else 0, ...)

# 3. Resume the continuation before admitting new waiting work
if self.chunked_req is not None:
    self.chunked_req.init_next_round_input()
    self.chunked_req = adder.add_chunked_req(self.chunked_req)

for req in self.waiting_queue:
    req.init_next_round_input(self.tree_cache)
    res = adder.add_one_req(req, has_chunked_req=...)
```

### The key parameters operate at different levels

| Parameter | What it controls | What it does not mean |
| --- | --- | --- |
| `--chunked-prefill-size` | Maximum tokens in one prompt chunk; `-1` disables chunking. | It is not the total token budget for the entire Prefill batch. |
| `--max-prefill-tokens` | Total input-token budget for the Prefill batch. | It is shared; each request does not receive this full amount. |
| `--enable-mixed-chunk` | Allows Prefill and Decode to share a batch. | It does not enable chunking itself; this option is disabled by default in the checked commit. |
| `--enable-dynamic-chunking` | Under PP, selects chunk size dynamically from a profiled latency model. | It is not automatically better for non-PP deployments. |

`PrefillAdder` matters because it closes all budgets in one place. With mixed chunking enabled, running Decode tokens are deducted from the available input and chunk budgets, while the KV allocator reserves space for those mixed tokens. Changing only the chunk size without updating every related accounting path can produce an implementation that appears to work, but fails under pressure.

### Why token count alone is insufficient for DynamicChunkSizer

A 4K-token chunk with zero history is not equivalent to a 4K-token chunk after a 64K-token prefix: Attention must read a very different amount of history, so forward latency changes. SGLang's PP dynamic-chunking path profiles Prefill latency, fits an approximate quadratic model of sequence length versus runtime, and predicts the next chunk size from the current history length. The target is closer to a fixed-duration microbatch than a mechanically fixed token count.<sup>[[4]](#ref-dynamic)</sup>

> **Interaction with speculative decoding** Speculative decoding primarily optimizes Decode. With mixed chunking enabled, the selected speculative algorithm must support that execution path. Validate draft/verify metadata, lookahead tokens, CUDA Graph behavior, and the final Prefill chunk—not merely whether the server starts.

## 06 / VLLM — vLLM: Let the Per-Iteration Token Budget Create Chunks

vLLM V1 does not maintain a separate queue of fixed-size chunks. Instead, each request advances its `num_computed_tokens` toward the required token count, clipped by the iteration's global token budget. Any Prefill work that does not fit naturally remains for the next iteration.

### The same mechanism, represented differently

At the start of each iteration, the Scheduler establishes `token_budget` and `input_budget`. It scans `running` requests first, then `waiting` requests. For each request, it computes the number of tokens still required and clips that amount to the remaining budget. If the KV allocator can provide new blocks, the request joins the iteration; otherwise, the Scheduler may preempt according to policy or stop admitting more work.

This is why vLLM does not need an exact equivalent of SGLang's dedicated `chunked_req` slot. A request may already have some `num_computed_tokens` but still be short of the prompt boundary; it simply continues as in-progress work in later iterations. The V1 documentation summarizes the default policy as: schedule pending Decode first, then fill the remaining `max_num_batched_tokens` budget with Prefill. Prefill work that does not fit is chunked automatically.<sup>[[8]](#ref-vllm-tuning)</sup>

*The crucial distinction: vLLM's `max_num_batched_tokens` is the token budget for **the entire iteration**, not a fixed chunk size assigned to each request.*

vllm/v1/core/sched/scheduler.py · How chunks are generated from the remaining budget (simplified excerpt)[upstream @ 4edb551](https://github.com/vllm-project/vllm/blob/4edb55169f69c76bb87ab9f154585917cb01bb17/vllm/v1/core/sched/scheduler.py#L573-L665)

```
# Shared budget for each round; there may be both Decode and in-progress Prefill in RUNNING
token_budget = self.max_num_scheduled_tokens
input_budget = self.scheduler_config.max_num_batched_tokens

for request in self.running:
    num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens

    if 0 < long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = long_prefill_token_threshold

    # Key: The maximum amount of budget left in this round will be pushed forward; the remaining part will be left to the next round.
    num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)

    if new_blocks is not None:
        num_scheduled_tokens[request.request_id] = num_new_tokens
        token_budget -= num_new_tokens
        input_budget -= num_new_tokens + draft_slots

# WAITING requests are also subject to request_token_budget
if not enable_chunked_prefill and num_new_tokens > request_token_budget:
    break
num_new_tokens = min(num_new_tokens, request_token_budget)
```

### How to read the main knobs

| Parameter | What it actually controls | Typical direction |
| --- | --- | --- |
| `--enable-chunked-prefill` | Allows Prefill to be advanced in multiple rounds when the single-round budget cannot be filled; V1 is enabled by default when available. | When closed, the full Prefill must fit into the batch token budget. |
| `--max-num-batched-tokens` | The total number of tokens schedulable per iteration—the most important operating-point knob. | Smaller values usually favor ITL; larger values usually favor TTFT and throughput. The official documentation suggests testing values above 8192 for small models on large GPUs when optimizing throughput. |
| `--max-num-seqs` | Number of sequences per round and runner / CUDA Graph capacity bounds. | It constrains the concurrent shape and is not equivalent to the token budget. |
| `--long-prefill-token-threshold` | When there are competing requests, there is a limit on how many tokens a single long Prefill can advance in this round; this cap is not imposed when running alone. | Used to reduce the risk of a long Prefill eating up the entire round budget; the default value is 0, which means no additional restrictions.<sup>[[10]](#ref-vllm-config)</sup> |

> **A unifying abstraction for reading the vLLM source**`num_computed_tokens` tracks how far computation has progressed toward the request's required token count. That difference can represent Chunked Prefill while also accommodating prefix caching, speculative decoding, and resumed requests. Chunking emerges naturally whenever the current iteration's budget cannot close the gap in one pass.<sup>[[9]](#ref-vllm-scheduler)</sup>

## 07 / PERFORMANCE EVIDENCE — Existing Performance Evidence: Benefits and Costs

This section retains the original paper figures instead of quoting only headline multipliers. Before transferring a result to your SGLang deployment, verify the model, GPU, parallelism strategy, dataset, and SLO.

### Read the Sarathi-Serve numbers in context

Sarathi-Serve combines Chunked Prefill with stall-free batching and compares serving capacity under tail-latency constraints. The paper reports 2.6× capacity over the then-current vLLM for Mistral-7B on one A100; up to 3.7× for Yi-34B on two A100s with TP=2; and up to 5.6× end-to-end capacity for Falcon-180B on eight A100s with PP=2, TP=4, and commodity Ethernet. See the original [Figure 10](https://arxiv.org/html/2403.02310v3#S5.F10), [Figure 11](https://arxiv.org/html/2403.02310v3#S5.F11), and [Figure 13](https://arxiv.org/html/2403.02310v3#S5.F13).<sup>[[1]](#ref-sarathi)</sup>

These are not isolated “Chunked Prefill speedups,” and they should not be used to predict current SGLang performance directly. They include the combined effects of scheduling, hybrid batching, parallelism, and a particular SLO; the baseline is also the serving stack available when the paper was evaluated.

*Summary chart redrawn from values reported in the paper; it is not an original paper figure. Sources: [Sarathi-Serve Figure 10](https://arxiv.org/html/2403.02310v3#S5.F10) and [Figure 11](https://arxiv.org/html/2403.02310v3#S5.F11). The horizontal axis is capped at 6×.*

![Sarathi-Serve Original Figure 12: Latency-throughput tradeoff of vLLM and Sarathi-Serve on Mistral-7B and Yi-34B.](https://chuyihou.github.io/llm-infra-learning/deep-dives/01-chunked-prefill/assets/sarathi-fig12-latency-throughput.svg)

*Original image and caption:[Sarathi-Serve v3 · Figure 12](https://arxiv.org/html/2403.02310v3#S5.F12). This page saves the original SVG provided by arXiv HTML, without redrawing the data.*

![Sarathi-Serve Figure 13a: Comparison of decode-only TBT for Falcon-180B TP and PP configurations.](https://chuyihou.github.io/llm-infra-learning/deep-dives/01-chunked-prefill/assets/sarathi-fig13a-pp-tp-latency.svg)

![Sarathi-Serve Figure 13b: Falcon-180B serving capacity under strict and relaxed SLOs.](https://chuyihou.github.io/llm-infra-learning/deep-dives/01-chunked-prefill/assets/sarathi-fig13b-pp-tp-capacity.svg)

***FIGURE 13(a) · DECODE-ONLY TBT** Cross-node collectives in TP-8 enter the critical path; TP-4 + PP-2 achieves lower median TBT.*

*Original image and complete analysis:[Sarathi-Serve v3 · Figure 13](https://arxiv.org/html/2403.02310v3#S5.F13). This page uses original SVG.*

![Sarathi-Serve Original Figure 14: Prefill runtime overhead of Yi-34B under different prompt lengths and chunk sizes.](https://chuyihou.github.io/llm-infra-learning/deep-dives/01-chunked-prefill/assets/sarathi-fig14-chunking-overhead.svg)

*Original image and caption:[Sarathi-Serve v3 · Figure 14](https://arxiv.org/html/2403.02310v3#S5.F14). Only trends can be migrated here, and 512 / 2048 cannot be regarded as a universal answer to SGLang.*

Smaller chunks incur more overhead

In the Yi-34B, TP=2 Prefill ablation, chunk=512 adds up to about 25% to total Prefill runtime, while chunk=2048 adds almost none. The exact values are experiment-specific, but the direction is important.

Chunking and hybrid batching must be coordinated

The ablation shows that chunking alone can increase TTFT through inefficient slicing, while hybrid batching alone can still suffer TBT stalls from full-length Prefill. Combining the two improves both dimensions.

> What the public evaluation gives is not "set the parameter to 2048", but an experimental hypothesis: there is an interval that is small enough to protect tail latency, but large enough not to be swallowed up by slicing overhead.

### SGLang's current default is just a hardware heuristic

In the checked SGLang commit, when the user does not set a value explicitly, the memory hook chooses a starting point from visible GPU memory: 2K below 20 GiB, 4K from 20–60 GiB, 8K from 60–160 GiB, and 16K at 160 GiB or above. The code describes this as a heuristic based on the observation that more memory often accompanies a stronger GPU; it also helps reserve headroom for activations and CUDA Graphs. It is not a workload-aware optimum.<sup>[[5]](#ref-memory)</sup>

SGLang's tuning guide offers symptom-driven guidance: if OOM occurs during Prefill, reduce `--chunked-prefill-size` to 4096 or 2048. This saves memory but slows Prefill for long prompts.<sup>[[6]](#ref-tuning)</sup>

### Mapping paper trends to SGLang parameters

The table preserves a “symptom → preferred knob → related checks” mapping without prescribing a fixed tuning sequence. The paper's token budget does not map one-to-one to an SGLang parameter; evaluate `--chunked-prefill-size`, `--max-prefill-tokens`, and `--enable-mixed-chunk` separately when transferring its findings.

| Observed symptom | Try first | Check at the same time |
| --- | --- | --- |
| p99 ITL has obvious spikes | Reduce chunk; enable/check mixed; increase Decode reserve. | Whether the spikes are related to long prefill arrival times and whether graph fallback occurs. |
| Long-prompt TTFT regresses | Increase the chunk or total Prefill budget; reduce unnecessary deferral. | Scheduler/launch overhead, continuation starvation, and cache-hit bucketing. |
| Prefill OOM | Reduce chunk to 4K/2K; reduce `mem_fraction_static`. | activation headroom, CUDA Graph buffers, and the actual parsed configuration. |
| Queue is deep but GPU utilization is low | Check admission, `max_prefill_tokens`, KV capacity, and the CPU scheduler. | Do not simply keep increasing the chunk; Prefill granularity may not be the bottleneck. |
| PP stage latency varies widely | Try `--enable-dynamic-chunking`. | Per-stage latency, in-flight microbatches, communication, and uneven layer partitioning. |
| Incorrect results after enabling mixed batching | Disable mixed batching, then re-enable features one at a time. | Speculative decoding, logprob, beam search, LoRA, multimodal paths, and CUDA Graphs. |

> **Selection rule** Once the configuration is correct, avoids OOM, and meets the Decode p99 ITL target, choose the largest chunk that still satisfies those constraints. Then tune the total Prefill budget, mixed batching, and continuation policy. This reduces chunking rounds without allowing long Prefill work to monopolize an iteration.

### References

Version-sensitive content was verified on 2026-09-22; recheck your installed version before deployment.

1. [Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference](https://arxiv.org/abs/2403.02310) — Defines chunked prefills and stall-free batching, with ablations and serving-capacity evaluation.

1. [SGLang Scheduler](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler.py#L3790) — Main path for Prefill scheduling, continuation, and batch construction.

1. [SGLang PrefillAdder](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/schedule_policy.py#L620) — Input, chunk, and KV-memory budget accounting.

1. [SGLang DynamicChunkSizer](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler_components/dynamic_chunk_sizer.py) — PP Prefill latency profiling and dynamic chunk prediction.

1. [SGLang memory hook](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/arg_groups/memory_hook.py) — GPU-memory default buckets and activation/CUDA Graph headroom heuristic.

1. [SGLang Hyperparameter Tuning](https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html) — Guidance for throughput, KV usage, OOM, and chunk-size tuning.

1. [SGLang Bench Serving Guide](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/docs/docs/developer_guide/bench_serving.mdx) — Benchmark methods for TTFT, ITL, E2E latency, throughput, and request rate.

1. [vLLM Chunked Prefill Tuning](https://github.com/vllm-project/vllm/blob/4edb55169f69c76bb87ab9f154585917cb01bb17/docs/configuration/optimization.md) — V1 defaults and the `max_num_batched_tokens` TTFT/ITL/throughput trade-off.

1. [vLLM V1 Scheduler](https://github.com/vllm-project/vllm/blob/4edb55169f69c76bb87ab9f154585917cb01bb17/vllm/v1/core/sched/scheduler.py) — RUNNING/WAITING scheduling, token-budget clipping, KV-slot allocation, and preemption.

1. [vLLM SchedulerConfig](https://github.com/vllm-project/vllm/blob/4edb55169f69c76bb87ab9f154585917cb01bb17/vllm/config/scheduler.py) — Configuration semantics for `max_num_batched_tokens`, Chunked Prefill, and the long-Prefill threshold.

1. [SGLang Multimodal Chunk Guard](https://github.com/sgl-project/sglang/blob/c19dc43cc732a3a7e9b487ba26f4af866cbe620b/python/sglang/srt/managers/scheduler.py#L1304-L1318) — Multimodal Chunked Prefill compatibility protection for the Transformers backend; the request-preparation path in the same file expands placeholders and computes M-RoPE.

1. [vLLM Multimodal Encoder Scheduling](https://github.com/vllm-project/vllm/blob/4edb55169f69c76bb87ab9f154585917cb01bb17/vllm/v1/core/sched/scheduler.py#L1760-L1822) — Encoder compute/cache budgets, multimodal-item boundary clipping, and `disable_chunked_mm_input` behavior.
