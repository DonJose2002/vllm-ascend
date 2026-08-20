# vLLM-Ascend 投机解码（draft_model）Eager 模式完整流程

> 适用版本：vllm-ascend 0.22.1rc1 + vllm 0.22.1
> 配置示例（`test_SD.sh`）：
> - 目标模型 Qwen3-8B（hidden=4096），草稿模型 Qwen3-0.6B（hidden=1024）
> - `--speculative-config '{"method": "draft_model", "num_speculative_tokens": 5}'`
> - `-tp 1`，`cudagraph_mode=NONE`（eager）
>
> 本文所有 `文件:行号` 引用均基于上述版本。`vllm_ascend/...` 指本仓库，`vllm-0.22.1/vllm/...` 指上游 vllm 源码。

---

## 0. 一句话总览

每个调度 step：

```
Scheduler(把上轮草稿token附到请求) → 目标模型前向(验证) → 拒绝采样(accept/reject)
   → 草稿模型前向(生成下一批草稿) → 草稿token回传Scheduler → 下一step
```

目标模型一次前向处理 `1 + num_speculative_tokens` 个 token；草稿模型则自回归跑 `num_speculative_tokens` 次（第 1 次吃多个 seed token，后面每次吃 1 个）。

---

## 1. 关键组件与类继承关系

### 1.1 草稿器（Drafter）

`vllm_ascend/spec_decode/draft_proposer.py:8`

```python
class AscendDraftModelProposer(DraftModelProposer, AscendSpecDecodeBaseProposer):
```

- `DraftModelProposer`（上游 `vllm-0.22.1/vllm/v1/spec_decode/draft_model.py:17`，继承 `SpecDecodeBaseProposer`）
- `AscendSpecDecodeBaseProposer`（本仓库 `vllm_ascend/spec_decode/llm_base_proposer.py:133`，继承上游 `SpecDecodeBaseProposer`）

**MRO**：`AscendDraftModelProposer → DraftModelProposer(上游) → SpecDecodeBaseProposer(上游) → AscendSpecDecodeBaseProposer(本仓库) → ...`

要点：
1. 上游 `DraftModelProposer` **不重写** `propose`，只重写模型加载/配置相关方法，且 `pass_hidden_states_to_model=False`（`draft_model.py:27`）——即草稿模型吃 `input_ids`，不吃目标模型 hidden states。
2. **在 vllm-ascend 上，`draft_model` 路径并不走上游的 `propose()`**。本仓库的 `model_runner_v1.py` 直接调用 `self.drafter._propose(...)`（`vllm_ascend/worker/model_runner_v1.py:1847`），而 `_propose` 是 `AscendSpecDecodeBaseProposer` 自己实现的（`vllm_ascend/spec_decode/llm_base_proposer.py:645`），用 ACLGraph/合并多步 attention 重写了整个提议逻辑。
3. 草稿器实例化入口：`vllm_ascend/spec_decode/__init__.py:46` `get_spec_decode_method("draft_model", ...)`。

### 1.2 目标侧 Model Runner

`vllm_ascend/worker/model_runner_v1.py` 的 `NPUModelRunner`。spec decode 相关方法：

| 方法 | 行号 | 作用 |
|------|------|------|
| `execute_model` | 1900 | 目标模型前向 + 计算采样元数据，存到 `execute_model_state` 后返回 |
| `sample_tokens` | 2326 | 拒绝采样 + 调用草稿器生成草稿 |
| `_calc_spec_decode_metadata` | 1538 | 构造 `SpecDecodeMetadata`（logits 索引等） |
| `propose_draft_token_ids` | 1638 | 草稿提议总入口（含 padded 分支） |
| `_sample` | 2520 | 调用 rejection sampler 做验证 |

### 1.3 Scheduler（上游）

`vllm-0.22.1/vllm/v1/core/sched/scheduler.py`：负责把草稿 token 附到请求、接收回采样结果、回滚 KV。

---

## 2. 端到端时序（单个 decode step）

```
┌─────────────── Scheduler 进程 ───────────────┐
│ 1. request.spec_token_ids = 上轮草稿提议         │  scheduler.py:510
│ 2. scheduled_spec_decode_tokens[req] = 草稿token  │  scheduler.py:513
│ 3. num_scheduled_tokens 含 1(bonus)+k(草稿)       │
└──────────────────┬───────────────────────────┘
                   │ SchedulerOutput（带 spec tokens）
                   ▼
┌─────────────── Worker: execute_model ────────┐
│ 4. _update_states: 拼出 input_ids(含草稿)、positions │  model_runner_v1.py:1981
│ 5. _calc_spec_decode_metadata → SpecDecodeMetadata   │  model_runner_v1.py:1307 / def:1538
│ 6. 目标模型前向 → hidden_states                       │
│ 7. compute_logits(sample_hidden_states[logits_indices])│  model_runner_v1.py:2280
│ 8. 存 execute_model_state，return None               │  model_runner_v1.py:2303
└──────────────────┬───────────────────────────┘
                   ▼
┌─────────────── Worker: sample_tokens ────────┐
│ 9. _sample → rejection_sampler(验证/接受/拒绝)        │  model_runner_v1.py:2378/2540
│    → sampler_output.sampled_token_ids  (接受位为token,拒绝位为-1) │
│10. propose_draft_token_ids(sampled_token_ids)        │  model_runner_v1.py:2437
│    ├─ prepare_next_token_ids_padded (算 bonus/接受数) │  model_runner_v1.py:1759
│    ├─ prepare_inputs_padded  (保留拒绝token作padding) │  model_runner_v1.py:1829
│    └─ drafter._propose(...)                           │  model_runner_v1.py:1847
│         └─ _run_merged_draft: 草稿模型跑 k 次         │  llm_base_proposer.py:989
│11. _copy_draft_token_ids_to_cpu                      │  model_runner_v1.py:2404
└──────────────────┬───────────────────────────┘
                   │ ModelRunnerOutput（带新草稿token）
                   ▼
┌─────────────── Scheduler: update ────────────┐
│12. update_draft_token_ids → request.spec_token_ids=新草稿│
│13. request.num_computed_tokens -= num_rejected (KV回滚) │  scheduler.py:1381
└───────────────────────────────────────────────┘
```

下面逐步展开。

---

## 3. Scheduler 侧：草稿 token 如何进入目标模型输入

每个 request 维护 `request.spec_token_ids`（上一轮草稿器的输出，长度 = `num_speculative_tokens`）。

`scheduler.py:502-517`：

```python
if request.spec_token_ids:
    num_scheduled_spec_tokens = num_new_tokens + request.num_computed_tokens \
                                - request.num_tokens - request.num_output_placeholders
    if num_scheduled_spec_tokens > 0:
        spec_token_ids = request.spec_token_ids[:num_scheduled_spec_tokens]
        scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
    request.spec_token_ids = []   # 本step用完清空，下step由 update_draft_token_ids 重新填
```

这些 spec token 会被拼到 request 的 token 序列尾部。于是目标模型这一步要处理的 token 数 = `1（上轮最后接受/bonus）+ k（待验证的草稿）`。对 `num_speculative_tokens=5`，即 6 个 token —— 这正是 profiling 里目标模型 embedding 算子输入 `6,1` 的来源。

---

## 4. 目标模型前向 + SpecDecodeMetadata

### 4.1 元数据构造 `_calc_spec_decode_metadata`（`model_runner_v1.py:1538`）

给定每个请求的 `num_draft_tokens` 和累计调度 token 数，构造 `SpecDecodeMetadata`（`vllm-0.22.1/vllm/v1/spec_decode/metadata.py:10`），关键字段：

- `logits_indices`：目标模型需要对哪些 token 位置算 logits（= 草稿token对应位置 + bonus 位置）
- `bonus_logits_indices`：每个请求的 bonus 位置（用于生成最后一个新 token）
- `target_logits_indices`：草稿 token 在 logits 中的索引（用于和草稿做对比）
- `draft_token_ids`：待验证的草稿 token（从上轮 `input_ids` 取出）
- `num_draft_tokens` / `cu_num_draft_tokens`：每请求草稿数及前缀和

示例（取自源码注释 `model_runner_v1.py:1544-1552`）：5 个请求，草稿数 `[3,0,2,0,1]`，则 `logits_indices` 长度 = `sum(num_draft+1)`，覆盖所有需要采样的位置。

### 4.2 目标模型前向

- `_update_states`（`model_runner_v1.py:1981`）把 spec token 拼入 `input_ids`，构造 positions、attn metadata。
- 目标模型 `forward(input_ids, positions)` → `hidden_states`。
- `sample_hidden_states = hidden_states[logits_indices]`（`model_runner_v1.py:2279`）→ `compute_logits` → `logits`。

> **embedding 算子调用链（目标模型）**
> `Qwen3Model.forward` → `self.embed_tokens(input_ids)` → 实例是 `AscendVocabParallelEmbedding`
> → `forward()`（`vllm_ascend/ops/vocab_parallel_embedding.py:163`）
> → `_forward_origin()`（同文件:186，`-tp 1` 时 `masked_input = input_`）
> → `self.quant_method.embedding(self, masked_input.long())`（:200）
> → 上游 `UnquantizedEmbeddingMethod.embedding` → `F.embedding` → NPU 算子 `aclnnEmbedding_GatherV2AiCore_GatherV2`
>
> 权重 `[151936, 4096]`，输入 `[6,1]` → 输出 `[6,4096]`。替换注册在 `vllm_ascend/utils.py:697`。

### 4.3 `execute_model` 返回 None（为 async 拆分）

`model_runner_v1.py:2303`：把 `(scheduler_output, logits, spec_decode_metadata, hidden_states, ...)` 存进 `self.execute_model_state` 后 `return None`。真正的采样在 `sample_tokens()` 里完成。eager 模式下二者仍先后执行，但拆分是为了兼容 async scheduling。

---

## 5. 拒绝采样：验证/接受/拒绝

`sample_tokens()`（`model_runner_v1.py:2326`）→ `_sample()`（:2520）：

```python
sampler_output = self.rejection_sampler(
    spec_decode_metadata, None, logits, sampling_metadata,
)
```

rejection sampler（上游 `vllm-0.22.1/vllm/v1/sample/rejection_sampler.py`）逻辑：
- 对每个草稿位置，比较目标 logits 采样结果与草稿 token：相同→接受，不同→拒绝，且该位置之后全部拒绝。
- 最后 bonus 位置总是产生 1 个新 token（无论接受多少）。
- 输出 `sampler_output.sampled_token_ids`：形状 `(num_reqs, num_spec_tokens+1)`，**接受位填 token id，拒绝位填 `-1`**。

例如目标输入 `[a,b,c,d,e,f]`（a=bonus，b..f=5 个草稿），若 e、f 被拒：该请求采样输出可能是 `[b,c,d,g,-1,-1]`（b,c,d 接受，g 为 d 之后的新 bonus，e,f 位 -1）。接受数 = 非负个数 - 1（减去 bonus）。

---

## 6. 草稿提议：`propose_draft_token_ids`（核心）

`model_runner_v1.py:1638`。对 `draft_model`/eagle 走 `:1735` 分支。

### 6.1 选 padded 还是 non-padded

`:1739`：

```python
if self.vllm_config.speculative_config.disable_padded_drafter_batch:
    next_token_ids = self.drafter.prepare_next_token_ids_cpu(...)   # CPU list，有阻塞同步
else:
    next_token_ids, valid_sampled_tokens_count = self.drafter.prepare_next_token_ids_padded(...)  # 默认
```

**默认 `disable_padded_drafter_batch=False`，走 padded 路径。** 这与 `sample_tokens` 里的 `use_padded_batch`（`:2424`）一致。

### 6.2 `prepare_next_token_ids_padded`（`llm_base_proposer.py:1663`）

输入 `sampled_token_ids`（GPU tensor，拒绝位 -1）。纯 GPU 计算：

- `valid_mask = (token_ids != -1) & (token_ids < vocab_size)`
- `valid_sampled_tokens_count = valid_mask.sum(dim=1)` —— 每请求接受数（含 bonus）
- `next_token_ids`：每请求最右一个有效 token（= bonus token），无有效则用 `request.get_token_id()` 兜底

输出 `next_token_ids`（每请求的 bonus/seed）和 `valid_sampled_tokens_count`。

### 6.3 `prepare_inputs_padded`（`llm_base_proposer.py:1844`）—— "7 token 之谜"的根源

docstring（`:1850`）原文：

> "...does not consider the rejected tokens. Instead, **all tokens are included as inputs to the speculator, with the rejected tokens used as padding and filtered out later by `token_indices_to_sample`.** No blocking CPU operations should be introduced in this function."

它**不剔除**被拒 token，而是：
- 用 triton kernel `prepare_inputs_padded_kernel`（`vllm_ascend/ops/triton/spec_decode/utils.py:22`）在 GPU 上算出 `token_indices_to_sample`（每请求要采样的有效位置）和 `num_rejected_tokens_gpu`。
- `token_indices` 保持全部位置（含被拒的 padding 位）。
- 注释（`:1904`）："`prepare_inputs_padded` does not change `seq_lens` (rejected tokens are kept as padding and filtered out later)."

回到 `model_runner_v1.py:1840`：

```python
target_token_ids = self.input_ids.gpu[token_indices]
```

`token_indices` 含 padding 位 → `target_token_ids` 就是草稿模型第 1 次前向的输入，长度 = 接受位数 + bonus + 拒绝padding位。

**对应到你的例子**：目标 `[a,b,c,d,e,f]`，e、f 被拒 → padding；接受 a,b,c,d + bonus g + 2 个 padding = `[a,b,c,d,g,0,0]` 共 **7** 个。这就是 embedding 算子在草稿模型上看到 `7,1` 的原因。

### 6.4 为何保留 padding 而不剔除（设计权衡）

对比 non-padded 版本 `prepare_inputs`（`llm_base_proposer.py:1720`）：它**会**剔除拒绝 token（重排 `slot_mapping`、`query_start_loc`、`seq_lens`），但实现里用了 `np.cumsum` / `np.repeat` / `.item()` / `torch.from_numpy().to(device)`（`:1770-1804`），这些是 **CPU 侧阻塞同步** 操作。

padded 方案把全部准备留在 GPU（triton kernel），代价是**多算几个 padding token 的 FLOPs**，换来：
- batch shape 稳定、零 CPU↔NPU 同步（对 async scheduler / kernel 友好）；
- padding 位的 KV 通过 `slot_mapping=-1`（`PADDING_SLOT_ID`，`vllm-0.22.1/vllm/v1/spec_decode/utils.py`）不写入有效槽；其 logits 被 `token_indices_to_sample` 在采样时跳过；`num_rejected_tokens_gpu` 用来 `seq_lens -= num_rejected`（上游 `propose` 里 `:562-566`）让有效 token 不 attend 到 padding。

即：**padding token 被真实地过了一遍 embedding/FFN/attention（所以你看到所有算子维度都是 7），但其结果在采样阶段被丢弃、KV 不污染有效序列**——"算了但扔掉"。这与你项目 AGENTS.md 中"热路径避免 `.item()`/CPU-NPU 同步"的原则一致。

> 想关掉这点冗余算力（代价是引入 CPU 同步）：在 speculative_config 里设 `disable_padded_drafter_batch=True`，会改走 `prepare_inputs`。

---

## 7. 草稿器 `_propose` → `_run_merged_draft`：草稿模型跑 k 次

### 7.1 `_propose`（`llm_base_proposer.py:645`）

- 构造 `multi_steps_attn_metadata`：为 `num_speculative_tokens` 个草稿步各建一份 attention metadata（`:839` 起，`:918-936` 循环建后续步）。
- 把第 1 步要用的 `target_token_ids`（含 padding）拷进 `self.input_ids`，positions、slot_mapping 就位。
- `run_draft = partial(self._runnable, **model_inputs)`（`:968`）。eager 模式下 `_runnable = _run_merged_draft`；graph 模式下是 `ACLGraphWrapper`。
- `draft_token_ids = run_draft()`（`:974`）。

### 7.2 `_run_merged_draft`（`llm_base_proposer.py:989`）

对 `draft_model`（`pass_hidden_states_to_model=False`）：

**第 1 次前向（seed，多个 token）**

```python
model_input_ids = self.input_ids[:num_input_tokens]   # = target_token_ids，含 padding（7个）
ret_hidden_states = self.model(input_ids=model_input_ids, positions=..., inputs_embeds=None)  # :1022
sample_hidden_states = last_hidden_states[token_indices_to_sample]   # 只取有效位
logits = self.model.compute_logits(sample_hidden_states)             # :1089
draft_token_ids = logits.argmax(dim=-1)                              # 第 1 个草稿 token
```

> **embedding 算子调用链（草稿模型）**：与目标模型完全同链，但权重 `[151936,1024]`，输入 `[7,1]` → 输出 `[7,1024]`。第 1 次前向调用 1 次 `aclnnEmbedding_GatherV2`。

**后续 `num_speculative_tokens-1` 次前向（每次 1 token）**（`:1132-1249` 循环）

```python
for draft_index in range(self.num_speculative_tokens - 1):
    input_ids = draft_token_ids_tensor[draft_index]   # 上一步输出的 1 个 token
    self.input_ids[:batch_size] = input_ids
    model_input_ids = self.input_ids[:input_batch_size]
    ret_hidden_states = self.model(input_ids=model_input_ids, positions=..., ...)  # 1 token 前向
    sample_hidden_states = last_hidden_states[token_indices_to_sample]
    draft_token_ids = self.model.compute_logits(sample_hidden_states).argmax(dim=-1)
```

每次输入 `1` 个 token → embedding 算子输入 `1,1` → 输出 `1,1024`。

**次数对账**：`num_speculative_tokens=5` → 草稿模型共前向 5 次 = 1 次（7 token seed）+ 4 次（各 1 token）。与你 profiling 观察（目标 1 次 6-token，草稿 1 次 7-token + 4 次 1-token）完全吻合。

最终返回 `draft_token_ids`，形状 `(batch_size, num_speculative_tokens)`。

### 7.3 回传 Scheduler

- `_copy_draft_token_ids_to_cpu`（`model_runner_v1.py:1869`，调用点 `:2404`）把草稿 token 拷到 CPU。
- 经 `ModelRunnerOutput` 回到 Scheduler。
- Scheduler `update_draft_token_ids` 把新草稿写入 `request.spec_token_ids`，供**下一** step 使用（见第 3 节）。

---

## 8. KV cache 回滚

目标模型与草稿模型各自维护 KV cache。

**目标侧**：`scheduler.py:1371-1385`

```python
num_draft_tokens = len(scheduled_spec_token_ids)
num_accepted = len(generated_token_ids) - 1
num_rejected = num_draft_tokens - num_accepted
if request.num_computed_tokens > 0:
    request.num_computed_tokens -= num_rejected   # ← 把被拒位置"退回"，下step重算
```

即被拒的 token 在下一步会被重新当作"未计算"，从而被重新前向（这正是投机解码能纠正错误草稿的方式）。

**草稿侧**：通过 `prepare_inputs_padded` 的 `token_indices` / `slot_mapping`（padding 位 `-1`）和 `num_rejected_tokens_gpu` 调整 `seq_lens`，保证草稿模型只把有效 token 写入/读到 KV。

---

## 9. 一个完整轮次的 token 流转（结合你的例子）

设 `num_speculative_tokens=5`，已接受前缀末位为 `a`。

| 阶段 | token 流 |
|------|---------|
| 上轮草稿提议 | 草稿自回归生成 `b,c,d,e,f`（5 个） |
| Scheduler 拼接 | 目标输入 = `[a, b,c,d,e,f]`（6 个：1 bonus + 5 草稿） |
| 目标前向 + 验证 | 接受 `b,c,d`；拒绝 `e,f`；bonus 生成 `g` |
| 草稿 seed 前向（padded） | 输入 `[a,b,c,d,g,0,0]`（7：接受4+bonus1+padding2）→ 产出草稿 `h` |
| 草稿第 2~5 前向 | `[h]→i`，`[i]→j`，`[j]→k`，`[k]→l` |
| 草稿回传 | 新草稿 `[h,i,j,k,l]` → `request.spec_token_ids` |
| 下一轮目标输入 | `[g, h,i,j,k,l]`（6：1 bonus + 5 草稿） |

> `g` 是上轮 bonus（d 之后的新 token），作为下轮 seed；`h..l` 是本轮草稿提议。

---

## 10. 文件索引

### vllm-ascend（本仓库）

| 文件 | 关键位置 | 内容 |
|------|---------|------|
| `worker/model_runner_v1.py` | `execute_model:1900` | 目标前向主循环 |
| | `sample_tokens:2326` | 采样 + 触发草稿 |
| | `propose_draft_token_ids:1638` | 草稿提议总入口（padded/non-padded 分支） |
| | `_calc_spec_decode_metadata:1538` | 构造验证元数据 |
| | `_sample:2520` | rejection sampler |
| | `1819-1832` | padded/non-padded 选择 |
| `spec_decode/__init__.py` | `46` | `draft_model` → `AscendDraftModelProposer` |
| `spec_decode/draft_proposer.py` | `8` | `AscendDraftModelProposer` 类 |
| `spec_decode/llm_base_proposer.py` | `_propose:645` | 草稿提议（vllm-ascend 重写） |
| | `_run_merged_draft:989` | 草稿多步前向循环 |
| | `prepare_next_token_ids_padded:1663` | 算 bonus/接受数（GPU） |
| | `prepare_inputs_padded:1844` | 保留拒绝 token 作 padding |
| | `prepare_inputs:1720` | non-padded 版（剔除拒绝，有 CPU 同步） |
| `ops/vocab_parallel_embedding.py` | `163/186/200` | embedding 前向（aclnn 算子落点） |
| `ops/triton/spec_decode/utils.py` | `22` | `prepare_inputs_padded_kernel` |
| `utils.py` | `697` | `VocabParallelEmbedding→AscendVocabParallelEmbedding` 注册 |

### 上游 vllm 0.22.1

| 文件 | 关键位置 | 内容 |
|------|---------|------|
| `v1/spec_decode/draft_model.py` | `17` | `DraftModelProposer`（仅模型加载，`pass_hidden_states_to_model=False`） |
| `v1/spec_decode/llm_base_proposer.py` | `propose:427` | 上游提议（eager draft_model 实际被本仓库 `_propose` 取代） |
| | `prepare_inputs_padded:965` / `prepare_inputs:1027` | 上游 padded/non-padded（被本仓库同名方法覆盖） |
| `v1/spec_decode/metadata.py` | `10` | `SpecDecodeMetadata` |
| `v1/core/sched/scheduler.py` | `502-517` | 把草稿 token 附到请求 |
| | `1368-1392` | 接受/拒绝统计 + `num_computed_tokens -= num_rejected` |
| `v1/sample/rejection_sampler.py` | — | 拒绝采样实现 |
| `model_executor/models/qwen3.py` | `forward` | `self.embed_tokens(input_ids)`（目标/草稿均经此） |
| `model_executor/layers/vocab_parallel_embedding.py` | `UnquantizedEmbeddingMethod.embedding` | `F.embedding` → aclnn 算子 |

---

## 11. 调试Tips：打印实际输入 token

在 `vllm_ascend/ops/vocab_parallel_embedding.py:186` 的 `_forward_origin` 里加：

```python
def _forward_origin(self, input_):
    print(f"[embed_tokens] dim={self.embedding_dim} tokens={input_.tolist()}")
    ...
```

- `embedding_dim==4096` → 目标模型；`==1024` → 草稿模型。
- 目标会打印 6 个 id；草稿第 1 次打印 7 个（含 padding 的 0），后 4 次各打印 1 个。
- `.tolist()` 会触发一次 NPU→CPU 同步，仅用于调试，勿留生产热路径。
