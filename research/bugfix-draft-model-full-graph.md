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

### 事故 A-1 后续:接受长度在 3 附近波动(2026-08-17 观测,待定量)

现象:修复后接受长度回到 ~3 但围绕原值双向波动。判据与定位方法:
- **双向波动 + 均值差小** → 判定为图编译(static kernel tiling/累加顺序)的 BF16 数值噪声,经 greedy argmax 放大,属良性
- **均值显著偏低(>0.1)或单边** → 系统性问题,用 per-position 接受率定位:
  - pos 0 就低 → step0 元数据(extend 相关)
  - pos 0 正常、pos 1+ 逐位衰减过快 → chain 步 replay 元数据(attn_update_stack_num_spec_norm)
- 定量工具:`research/bench_sd.py`(独立脚本,不改代码主体),从 /metrics 取
  `vllm:spec_decode_num_drafts/accepted/per_pos` counters,输出每请求接受长度、
  per-position 接受率、ITL/TTFT 分位;A/C 两模式各跑一次后 compare

```bash
# 服务器,两模式各起一次服务后:
python3 research/bench_sd.py bench  --base-url http://127.0.0.1:8007 \
    --model /nfs-share/hf_weights/Qwen3-8B --tag A --out bench_A.json
python3 research/bench_sd.py compare bench_C.json bench_A.json
# 判定:mean accept len 差 <0.1 → 噪声结案;ITL 应见 A 显著优于 C(方案 A 的目的)
```

## 遗留

- [x] 方案 C 服务器验证(2026-08-17 用户确认通过)
- [x] 方案 A 前置调研(上游无 FULL 先例;#45258 RFC 地雷图 + #34880 未合并设计为参考)
- [ ] 方案 A 服务器验证(清单见上)
- [ ] (远期)若方案 A 稳定,考虑向上游提 PR(可引用 #45258 的 KV 隔离担忧的解法)
