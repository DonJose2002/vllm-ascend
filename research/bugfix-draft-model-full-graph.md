# Debug 笔记:bug3 — draft_model + FULL 图:drafter token 数失配导致输出形状错误

> 日期:2026-08-17 | 基线:vllm-ascend v0.22.1rc1 + vllm 0.22.1 | 状态:**方案 C(降级)已提交待服务器验证;方案 A(正解)待做**
> 前序:bug1 update_stream MRO(`bugfix-draft-model-update-stream.md`)、bug2 模式跨模型污染(`bugfix-qknorm-pattern-cross-model.md`)。

## 现象

`cudagraph_capture_sizes=[6,12]` + FULL + draft_model:运行期首步崩溃
```
RuntimeError: The expanded size of the tensor (1) must match the existing size (2)
at non-singleton dimension 0.  Target sizes: [1, 5].  Tensor sizes: [2, 5]
```
位置:`model_runner_v1.py _copy_draft_token_ids_to_cpu`(`draft_token_ids_cpu[:1].copy_([2,5])`)。

MindStudio Insight profiling 另证实:`test_SD.sh` 原配置(cudagraph_mode=NONE)下 drafter 本来就没走图——eager 一直是对的,图模式是未开垦之地。

## 用户分析核验(全部成立)

1. `target_token_ids` 长度 6 = **合理**:scheduler dump `num_scheduled_tokens=6`(1 采样 + 5 spec tokens)= R×(K+1),由 scheduler 按标准 spec-decode verify batch 设定
2. `set_inputs_first_pass` 填充 +1:**上游设计行为**。`llm_base_proposer.py:95-101`:非 parallel_drafting → `extra_slots_per_request=1`;draft_model `pass_hidden_states_to_model=False` → `net_num_new_slots=1-0=1` → `total_num_output_tokens = 6 + 1×1 = 7`(drafter merged forward 需要额外输入槽产出第一个 draft token 的 logits)
3. `[6]` 时 7>6=max_size → dispatcher 直接返回 NONE(cudagraph_dispatcher.py:282)→ eager 跑,不崩
4. `[6,12]` 时 pad 到 12 → `_create_padded_batch_descriptor` FULL-uniform 分支 `num_reqs = 12//(K+1) = 2`(assert 12%6==0 通过)→ **虚构 phantom 请求**
5. 图按 batch_size=2 烘焙输出 [2,5];运行时真实 num_reqs=1 → copy_ 到 [1,5] buffer → RuntimeError
6. 去掉 ACLGraphWrapper → eager 用真实 batch_size=1 → 输出 [1,5] 正确

## 根因(结构性)

**drafter 每步 token 数 = R×(K+2)**(K+1 verify + 1 extra slot),**永远不是 (K+1) 的倍数**;而 dispatcher 的 FULL-uniform 分支硬性假设 num_tokens = R'×(K+1)(整除断言 + num_reqs 推导)。draft_model 的 drafter 在这条 dispatch 路径上没有正确入口——非特例,是结构失配。

对照:EAGLE/MTP/DFlash 的 `net_num_new_slots=0`(`pass_hidden_states=True` 或 parallel drafting),token 数保持 (K+1) 倍数 → 与 dispatcher 假设兼容 → 它们的 FULL 支持(PR #11473 MTP、#8589 DFlash)不中招。**唯独 draft_model 是 1。**

## 上游对照(vllm 0.22.1)

- 上游 drafter **有独立 dispatcher 实例**,注释明说 "PIECEWISE-only dispatching in eagle"(llm_base_proposer.py:139-143);`initialize_cudagraph_keys`:"Only supports PIECEWISE cudagraphs"(380-395)
- PIECEWISE dispatch 用 relaxed key(`num_reqs=None, uniform=False`,cudagraph_dispatcher.py:325-327):padding 只 round up num_tokens,**无整除断言、不虚构请求** → R×(K+2) 无害
- 上游 draft 循环是 K-1 次独立 model 调用,**没有** "merged draft + FULL 图" 路径——那是 vllm-ascend 特有设计(`_run_merged_draft` + `ACLGraphWrapper(FULL)`)

## vllm-ascend issue 检索(2026-08-17)

无 draft_model+FULL 相关报告。现有 FULL 适配:MTP(#11473)、DFlash(#8589)——均为 net_new_slots=0 的方法。**这条路径从未有人跑通**(bug1 的 MRO 遮蔽一直在门口挡着)。

## 修复方案

### 方案 C(已实现,本次提交):drafter 对 draft_model 禁用图模式

`llm_base_proposer.py` `use_cuda_graph` 追加 `and not speculative_config.uses_draft_model()`:
- ACLGraphWrapper 不创建 → drafter eager(数值/行为 = 上游 drafter 的 PIECEWISE-only 语义,甚至更保守)
- target 模型的 FULL 图不受影响
- `_propose`/`dummy_run`/`_run_merged_draft` 的所有 `use_cuda_graph` 门槛点已逐一核验:全链路一致走 eager 路径
- test_server.sh / API **零变化**(用户配置的 capture_sizes 仍约束 target 图)

### 方案 A(待做):drafter 专属 FULL 图支持

- drafter 捕获尺寸表:从用户配置 `[R×(K+1)]` 派生 **`[R×(K+2)]`**(如 [6,12] → [7,14])
- drafter dispatch 绕开 FULL-uniform 断言:pad 到 ≥num_tokens 的最小 R×(K+2)
- `dummy_run` 的 `batch_size = num_tokens // (K+1)` 需显式按 R 计算(R≥K+1 时 7//6 式的整除巧合失效)
- 基础设施现成:`set_draft_graph_params` 已按 draft 单独建表(acl_graph.py:368),wrapper 已有 use_eagle per-method 分支先例
- 价值:把 vllm-ascend 的 merged-draft 设计对 draft_model 补完,可向上游提 PR

## 验证清单(服务器,方案 C)

```bash
git pull   # research/main
# 配置: cudagraph_mode FULL + cudagraph_capture_sizes=[6,12] + draft_model
# 期望:
#   1. 不再出现 expanded size (1)/(2) 错误
#   2. 正常出 token(端到端跑通,顺带验证 bug1/2/3 修复链)
#   3. MindStudio/profiler 或日志:target 走 FULL 图;drafter eager(无 "Wrapping draft model with ACLGraphWrapper")
```

## 遗留

- [ ] 服务器验证方案 C
- [ ] **方案 A 前置调研(用户要求):查上游 vllm 后续版本(release notes / issues / PRs)有没有给 draft_model 加 FULL 图模式**——若有,对齐其设计;若无,自行实现并考虑提 PR
- [ ] 方案 A 实现与验证
