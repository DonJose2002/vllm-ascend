# Debug 笔记:bug3 — draft_model + FULL 图:drafter token 数失配导致输出形状错误

> 日期:2026-08-17 | 基线:vllm-ascend v0.22.1rc1 + vllm 0.22.1
> 状态:**✅ 已解决(2026-08-17 深夜服务器验证通过)**——方案 C(默认)+ 方案 A(`VLLM_ASCEND_DRAFT_MODEL_FULL_GRAPH=1`,接受长度恢复、双向波动经复检为正常)
> 完整历程(含 bug1/bug2 前序)见姊妹篇:`draft-model-full-graph-journey.md`(自然语言复盘)

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

### 方案 C(✅ 已实现并经服务器验证 2026-08-17):drafter 对 draft_model 禁用图模式(默认行为)

`llm_base_proposer.py` `use_cuda_graph` 追加 `and not speculative_config.uses_draft_model()`:
- ACLGraphWrapper 不创建 → drafter eager(数值/行为 = 上游 drafter 的 PIECEWISE-only 语义,甚至更保守)
- target 模型的 FULL 图不受影响
- `_propose`/`dummy_run`/`_run_merged_draft` 的所有 `use_cuda_graph` 门槛点已逐一核验:全链路一致走 eager 路径
- test_server.sh / API **零变化**(用户配置的 capture_sizes 仍约束 target 图)

### 方案 A(已实现,实验性,门控 `VLLM_ASCEND_DRAFT_MODEL_FULL_GRAPH=1`,默认关=方案 C 行为)

**上游调研结论(2026-08-17,实现前完成)**:
- vllm 上游 drafter 仍 PIECEWISE-only;**#45258**(RFC,open,2026-06)实测 drafter 占 decode step 15-18%,外部原型 11 次迭代失败后撤回 RFC,地雷图:FULL keys+wrapper 可行、需预捕获 warmup、BatchDescriptor 全程透传、**KV-store 隔离(draft 垃圾写入 target KV page 0 的危险失效模式)**、未捕获 shape 优雅降级
- **#34880**(Eagle drafter FULL 尝试,**closed 未合并**):首步与 target 共享 uniform_decode、后续步骤独立 keys
- **#47460**(merged):draft_model PIECEWISE keys 初始化缺失已修(仍非 FULL)
- 结论:无上游 FULL 先例可对齐,方案 A 为新 territory

**实现(4 文件)**:
1. `envs.py`:`VLLM_ASCEND_DRAFT_MODEL_FULL_GRAPH`(默认 0)
2. `model_runner_v1.py _check_and_update_cudagraph_mode`:flag 开时派生 `drafter_sizes = {s + s//(K+1)}`(如 [6,12]→[7,14]),`set_draft_graph_params(drafter_sizes)`,存 `self._draft_model_graph_sizes`
3. `model_runner_v1.py _dummy_run`:捕获循环(target 尺寸)内,drafter.dummy_run 调用翻译为 R×(K+2) tokens / R reqs / `BatchDescriptor(R*(K+2), R, uniform=True)`——drafter 图在 target 每个捕获尺寸对应翻译尺寸处惰性捕获
4. `llm_base_proposer.py`:
   - `__init__`:`use_draft_model_full_graph` flag;`use_cuda_graph` 对 draft_model 仅在 flag 开时为 True
   - `dummy_run`:draft_model 时 query_start_loc 填 K+2 步长(不用 runner 的 K+1 步长 buffer)、`max_query_len=K+2`、batch_size 用显式 num_reqs(规避 R≥K+1 的整除陷阱)
   - `_propose`:draft_model 专属 dispatch——真实 num_tokens=R×(K+2) pad 到捕获表内最小尺寸;无匹配(如超 max)→ eager 优雅降级;FULL 时 K+2 式 padding(qsl 步长 K+2、seq_lens/block_table 零填充,phantom 追加在真实请求后,FIA 跳过 len-0 请求);复用 eagle 的 `_adjust_tensor`/group buffers/`attn_update_stack_num_spec_norm` 机制

**安全性设计**:phantom 输出行 [R',K] 由 `_copy_draft_token_ids_to_cpu` 按 shape[0] 拷贝(eagle 同款容忍);slot_mapping 尾部 -1;未捕获 shape 回 eager(对齐 RFC #45258 建议)。

**CPU 已验证**:尺寸派生、pad 选择、qsl 步长、eager 降级、捕获/运行时尺寸精确互配。

### 事故 A-1(2026-08-17 服务器验证发现):接受率崩塌(>3 → 1.3)

**现象**:方案 A 部署成功、正常出 token,但平均接受长度从 >3 掉到 1.3(≈每步只接受 1 个)——第 1 个之后的 draft token 全错。

**根因**:FULL padding 分支的 `seq_lens` 抄了 eagle 分支的 `runner.seq_lens`,但语义不同:
- eagle:`needs_extra_input_slots=False`,不经 extend,runner.seq_lens(verify 后)本就覆盖 step0 全部 KV → 直接用正确
- draft_model:`set_inputs_first_pass` 走 extra-slots 分支,末尾 `extend_all_queries_by_N` **把 seq_lens 每行 +1**(覆盖 step0 写入的 extra seed KV);用未 +1 的 runner.seq_lens 覆盖 → **所有 q 的 causal attend 边界左移 1** → 全部 draft hidden 错位 → 草稿质量崩
- **dflash 分支(同样走 extend)的先例正是用 `cad.seq_lens`——当时抄错了对象**

**修复**(`33205ca2d`):seq_lens/seq_lens_cpu 改用 extend 后的 `common_attn_metadata` 值(CPU mirror 缺失时 fallback `optimistic + N`),对齐 dflash 与已验证的 eager 语义。qsl/max_query_len 的 K+2 步长经核对与 `extend_all_queries_by_N` 输出一致,无需改。

**教训**:与 eager 路径做逐字段 diff(set_inputs_first_pass 的返回值是"语义基准"),跨方法抄分支时要检查该方法是否走了不同的 set_inputs 分支。

## 验证清单(服务器,方案 A)

```bash
git pull && export VLLM_ASCEND_DRAFT_MODEL_FULL_GRAPH=1
# 配置: cudagraph_mode FULL + cudagraph_capture_sizes=[6,12] + draft_model
# 期望:
#   1. 启动日志见 "draft_model drafter FULL graph enabled: target sizes [6,12] -> drafter sizes [7,14]"
#   2. 部署完成、正常出 token;MindStudio/profiler: drafter 前向出现图 replay("Replaying aclgraph")
#   3. 输出质量正常(R>1 时尤其注意 phantom 行不污染——与方案 C 对照生成结果)
#   4. **平均接受长度恢复 >3**(事故 A-1 修复后的关键指标;若仍 ~1.3 见下"事故 A-1")
#   5. 崩溃/异常 → 贴日志,回退 export ...=0 即恢复方案 C
```

### 事故 A-1 后续:接受长度在 3 附近波动(✅ 2026-08-17 深夜复检通过,结案)

现象:修复后接受长度回到 ~3 但围绕原值双向波动。用户逐项复检后确认无系统性偏差——双向波动符合图编译(static kernel tiling/累加顺序差异)在 BF16 + greedy argmax 下的数值噪声特征,属良性,结案。定量工具保留(`research/bench_sd.py`,双口径接受长度 + ITL/TTFT + per-position 接受率),后续回归验证可复用。

## 遗留

- [x] 方案 C 服务器验证(2026-08-17 用户确认通过)
- [x] 方案 A 前置调研(上游无 FULL 先例;#45258 RFC 地雷图 + #34880 未合并设计为参考)
- [x] 方案 A 实现与服务器验证(含事故 A-1 修复;2026-08-17 深夜用户确认通过)
- [x] **pull request**(2026-08-18):三个 PR 分支已就绪(main 基线):`pr/bugfix-draft-model-mro`、`pr/bugfix-pattern-shape-scoping`、`pr/draft-model-full-graph`(plan C + plan A,env 已转 `--additional-config '{"draft_model_full_graph": true}'`,env 变量保留为弃用回退);文案在会话记录 / `/tmp/opencode/pr-descriptions.md`
- [x] **v0.23.0 验证**(2026-08-18,见上节:5a/5a'/5b/5c/5d 全通过,方案 A 接受长度零损失 + ITL 5.7×)
- [ ] A/C 的 ITL 收益定量对比已出数(v0.23.0 单请求 5.7×);多 batch 场景可后续补充

## v0.23.0 服务器验证(2026-08-18 部署)

D5 PR 提交前在官方 v0.23.0 基线复现 + 验证(经 Docker `quay.io/ascend/vllm-ascend:v0.23.0`,CANN 9.1.0):
- `myfork/server/v0.23.0-baseline`(= tag v0.23.0):三 bug 均在——**但触发顺序反转**:bug2 先挡路(split ValueError),需先修 bug2 才暴露 bug1(0.22.1rc1 上 bug1 先崩)。证据 `logs/v0.23.0-baseline-bug2-split-crash.txt` + `logs/v0.23.0-baseline-pr2-bug1-update-stream.txt`(tag+PR2 后编译越过 split 点,图捕获阶段崩 update_stream——一石二鸟:PR2 生效 + bug1 复现)
- `myfork/server/v0.23.0-fixes`(baseline + PR1/PR2 cherry-pick + PR3 适配移植,4 commits):PR3 移植差异——v0.23.0 的 dummy_run 用 `copy_snapshot_to_gpu`、`_propose` FULL 分支带 `_pad_query_start_loc_for_fia`(与 main 同构),K+2 分支作为独立 if 插在共享分支前

### 验证结果(✅ 全部通过,2026-08-18)

| 步骤 | 结果 | 证据 |
|---|---|---|
| 5b 方案 C(默认) | 捕获 2/2 + startup complete;无 "Wrapping draft model" 行(drafter eager,符合方案 C 语义) | `logs/v0.23.0-fixes-planC-serve-ok.txt` |
| 5c 方案 A | **"Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL"** + **"target sizes [6, 12] -> drafter sizes [7, 14]"** + 捕获 2/2 + startup complete | `logs/v0.23.0-fixes-planA-drafter-full-graph-ok.txt` |
| 5d bench(30 req × 2670 draft steps) | **接受长度完全一致**(2.876 vs 2.876,per-position 5 个位置全部相同,greedy 确定性下零漂移);**ITL mean 281.8 → 49.7 ms(5.7×)**,p50/p90 同幅,TTFT 持平 | `logs/v0.23.0-fixes-planA-vs-planC-bench.txt` |

注:bench "burst est" 口径显示 1.000/1.000 属单请求负载(max_num_seqs=1)下突发估计退化的已知伪影,主口径(counters)2.876 两模式一致且与 0.22.1rc1 基线(~3)吻合。

**结论**:方案 A 在 v0.23.0 上正确性零损失(接受长度逐位一致 = A-1 修复在新基线同样有效),单请求 ITL 收益 5.7×。结果回填三个 PR 描述("validated on 0.22.1rc1 & 0.23.0")后提交。
