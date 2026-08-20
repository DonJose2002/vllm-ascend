# 上游 vllm 0.22.1 vs vllm-ascend 0.22.1rc1:调度器与投机解码对照

> Phase 0 任务:Week1 "对照上游 vllm 的 scheduler / spec_decode 与 vllm-ascend 的差异" 的产出。
> 版本锚点:上游快照 `/home/hp/projects/vllm/vllm-ascend-0.22.1/vllm-0.22.1/`(下称 `vllm/`),
> 工作仓库 `repos/vllm-ascend`(research/main,与官方 tag v0.22.1rc1 一致,下称 `vllm_ascend/`)。
> 另附 5080 环境的 vllm 0.26.0(`~/vllm-env/.../site-packages/vllm/`)作演进参考。
> 本文所有 `文件:行号` 均基于上述版本。已有的三篇 spec decode / attention 笔记不重复展开,只给链接。

---

## 0. 一句话结论

**默认配置下,vllm-ascend 的调度器就是上游 `Scheduler`,一行没改。** 差异全部藏在四个"可选拓扑调度器"和一层平台 patch 里,由 `NPUPlatform.check_and_update_config()`(`vllm_ascend/platform.py:665-704`)按配置动态换入;而 spec decode 则相反——**调度协议(记账方式)两侧同构,执行协议(drafter 怎么跑)被 vllm-ascend 整体重写**。一句话:调度器是"换着用",spec decode 是"重写了执行层"。

---

## 1. 母本:上游 0.22.1 的调度器长什么样

要看懂 vllm-ascend 改了什么,得先记住母本的样子。上游 `vllm/v1/core/sched/scheduler.py`(2337 行)的核心思想写在 schedule() 的注释里(这个注释被 vllm-ascend 的每个 fork 原样抄走):

> **没有 "prefill 阶段" 和 "decode 阶段" 之分。** 每个 request 只有 `num_computed_tokens` 和 `num_tokens_with_spec`(= prompt + output + spec tokens)。每一步,调度器只是尽量给各请求分配 token,让 `num_computed_tokens` 追上 `num_tokens_with_spec`。

围绕这个思想,schedule()(`scheduler.py:329`)做三件事:

1. **RUNNING 优先**:遍历 running 队列,给每个请求算 `num_new_tokens`(受 `long_prefill_token_threshold` 和 token budget 双重限制),`kv_cache_manager.allocate_slots()` 分不到块就按策略抢占(PRIORITY 抢最低优先级,FCFS 抢队尾,`scheduler.py:194-226`);
2. **WAITING 准入**:没有抢占发生时,从 waiting 队列准入新请求(查 prefix cache / KV connector,决定本次 prefill 多少 token);
3. **产出 SchedulerOutput**:`num_scheduled_tokens`、`scheduled_spec_decode_tokens`、新块表等,交给 worker 执行。

**AsyncScheduler**(`v1/core/sched/async_scheduler.py:12`)是调度与执行 overlap 的变体:调度器提前一步发排,不知道本步会采样出什么 token,所以用 `num_output_placeholders` 记账(`async_scheduler.py:18-35`:每步给每请求预扣 `1 + cur_num_spec_tokens` 个占位,输出回来再 `_update_request_with_output` 里扣减),spec token 用占位符列表先顶着,实际值由 worker 侧回填。这套"占位记账"贯穿 0.22.1 的 scheduler,是理解 spec decode 调度协议的钥匙。

**关键钩子:`scheduler_cls`**(`vllm/config/scheduler.py:127`,解析在 `:168-188`)。上游官方的调度器插件化入口:`scheduler_config.scheduler_cls` 填类名字符串,`EngineCore` 用 `get_scheduler_cls()`(`vllm/v1/engine/core.py:135`)动态加载。默认 None → `async_scheduling` 开则 AsyncScheduler,否则 Scheduler。**vllm-ascend 的三个新调度器走这个正门,一个走猴子补丁的偏门(见下)。**

spec decode 与调度的接口只有两处,两侧完全一致(此处不重复,详见 `spec_decode_eager_flow.md` §3/§8):

- 附着:`scheduler.py:500-514` 把上轮草稿 `request.spec_token_ids` 切进本步输入,用完清空;
- 回滚:`scheduler.py:1374-1381` 按接受数回退 `num_computed_tokens`,被拒 token 下一步重算。

---

## 2. vllm-ascend 的四个自有调度器

先给接线总表(`vllm_ascend/platform.py:665-704`,`check_and_update_config` 里按序检查):

| 调度器 | 开关(additional_config) | 拓扑约束 | 接线方式 |
|---|---|---|---|
| `BalanceScheduler` | `enable_balance_scheduling` | 仅 PD-mixed(kv_both) | **猴子补丁**:模块属性替换(`patch_balance_schedule.py:702-703`) |
| `RecomputeScheduler` / `AsyncRecomputeScheduler` | `recompute_scheduler_enable` | 仅 PD-disagg(producer/consumer) | `scheduler_cls` 正门(`platform.py:688-689`) |
| `SchedulerDynamicBatch` | `SLO_limits_for_dynamic_batch != -1` | 无(仅 910B3 验证过) | `scheduler_cls` 正门(`platform.py:693-697`) |
| `ProfilingChunkScheduler` | `profiling_chunk_config.enabled` | 无 | `scheduler_cls` 正门(`platform.py:701-703`)+ patch 链 |

四个调度器有一个**共同的宿命**:把上游 `schedule()` 整个抄下来,在循环里插自己的逻辑。`scheduler_profiling_chunk.py` 的 docstring 自己承认:"Compatible with vLLM v0.15.x scheduler. When the upstream schedule() method is refactored, this override should be updated accordingly."——上游每重构一次 schedule(),这边四个文件都要跟着重抄。这是插件化架构对"调度器这种核心热路径"还没开够钩子的代价(contrast:`scheduler_cls` 开了类级钩子,但方法级复用还是只能靠抄)。

### 2.1 BalanceScheduler:DP rank 间负载均衡(`patch/platform/patch_balance_schedule.py`)

**解决什么**:数据并行(DP>1)时,各 rank 的 EngineCore 独立调度,请求数会漂移——某个 rank 先跑满 `max_num_seqs`,其他 rank 空转等它,尾部延迟被最慢 rank 拖死。

**怎么做的**:三个动作。

1. `BalanceDPEngineCoreProc.run_busy_loop()`(`:601`)在每步末尾的集合通信窗口里加一次 `balance_gather`——`dist.all_gather` 各 rank 的 running 请求数(CPU tensor,`:47-56`),成本被摊进本来就要做的 all-reduce 里;
2. `schedule()`(整抄上游)在 **WAITING 准入循环的头部**插一行判断(`:282-284`):`max(t.item() for t in self.balance_queue) == self.max_num_running_reqs` 就 break——**任何一个 rank 跑满了,所有 rank 都停止准入**,新请求等下一轮一起进,保证 wave 级对齐;
3. 模块加载时直接替换 `EngineCoreProc.run_engine_core` 和 `vllm.v1.core.sched.scheduler.Scheduler`(`:702-703`)。注意这是**类替换**而非 `scheduler_cls` 配置——所以它只能在 patch 模块里干,而且替换发生在 `pre_register_and_update()`(全局 patch 阶段),比配置解析更早。

平台侧还锁死拓扑(`platform.py:665-673`):PD-disagg 模式下开它直接 ValueError。相关上游 PR #29721 还在推进中(patch 索引里写了 "Remove this patch when vLLM merge the PR")。

### 2.2 RecomputeScheduler:PD-disagg 下的"重算式抢占"(`core/recompute_scheduler.py`,1049 行)

**这是四个里最"换语义"的一个。** 上游的抢占是本地行为:挤出 KV 块、请求回 WAITING、重新 prefill。但 PD-disagg 的 kv_consumer 端,请求的 KV 是从 prefill 节点拉来的——本地没有原始计算过程,抢占后"重放"的代价和语义都变了。

vllm-ascend 的解法(名字由此而来):kv_consumer 端抢占的请求**不回队,直接终局**——`schedule()` 把被抢占请求收进 `recomputed_reqs`(`:280-286`),`update_from_output()`(`:798-806`)给客户端发一个 `finish_reason=STOP, stop_reason="recomputed"` 的输出,由 serving 层/客户端**重新提交**,走全新 prefill。本质是把"抢占-恢复"退化成"放弃-重来",换来 KV 块管理的简单性。

两个顺带的特色:

- **MTP KV consumer 的 FULL 图占位**(`:149-152`):consumer 拉远端 KV 时还没有草稿 token,`spec_token_ids` 填 `PLACEHOLDER_TOKEN_ID * num_spec_tokens`,让形状先对齐,ACL FULL 图的模式匹配才不会失败(否则降级 eager)。这和我们在 draft_model+FULL 系列里处理的"形状稳定优先、内容后到"是同一哲学;
- **EPLB 路由记账**(`update_from_output` 里 `enable_return_routed_experts` 分支):每步把 worker 回传的 routed experts 存进 `routed_experts_mgr`——**Phase 3 MoE 热点专家分析的现成数据入口**,到时直接回来看这段。

### 2.3 SchedulerDynamicBatch:查表实现 SLO 感知 token budget(`core/scheduler_dynamic_batch.py`)

**解决什么**:chunked prefill 的 `max_num_batched_tokens` 是静态的——prefill chunk 太大,同 batch 的 decode 请求 ITL 就被打爆;太小,吞吐又亏。想要"chunk 大小随当前 decode 负载自适应,并守住延迟 SLO"。

**怎么做的**:核心是 `BudgetRefiner`(`:35`)。启动时读一张离线 profiling 表 `profile_table.csv`(列:ctx_len, d_num, chunk_size, cost),按 `(上下文长度档, decode 请求数档)` 建索引;每步 `refine_budget()`(`:107`)统计 running 里 decode 请求的平均 token 数和个数,对齐到表里最近的档位,查出"cost ≤ SLO 限"前提下的最大 chunk size,替换默认 token budget(`:154-156`)。另外把 running 队列重排成 **decode 严格在前**(`:160-163`)——decode-first 的 chunked prefill,和上游 FCFS(先到先算)不同。

注意它**强制打开 chunked_prefill**(`platform.py:696`),且注释自述"目前只在 910B3 上验证过"。

### 2.4 ProfilingChunkScheduler:在线拟合的动态分块(`core/scheduler_profiling_chunk.py` + `patch/platform/patch_profiling_chunk.py`)

DynamicBatch 用离线表,这个用**在线模型**:启动后经 `collective_rpc` 让每个 worker 跑几档 chunk size 的 prefill profiling,拟合二次延迟模型(`ProfilingChunkManager` / `profiling_chunk_predictor.py`);调度时按每个 waiting 请求的 `num_computed_tokens` 预测最优 chunk。运行期每步回传 `execution_time_ms`(由 NPU model runner 动态挂在 model output 上)在线精修;拟合收敛后自动关掉计时同步,避免 pipeline stall(`patch_profiling_chunk.py:58-72`)。

工程上最绕的是**补丁跨进程存活**:`EngineCore.__init__` 的补丁在 spawn 出的子进程里会丢,所以又包了一层 `EngineCoreProc.run_engine_core`——子进程 unpickle wrapper 时触发 import 本模块,把补丁在子进程里重打一遍(`patch_profiling_chunk.py:216-229`)。读懂这个模式对以后写任何"进程级 monkey patch"都有参考价值。

---

## 3. 平台 patch 层:对调度语义的小修正

`vllm_ascend/patch/__init__.py` 是全插件 patch 的**自述索引**(每个 patch 带 Why/How/上游 PR/Future Plan),值得一字不落读一遍。与调度/KV 相关的归纳:

| Patch | 动机(一句话) |
|---|---|
| `patch_scheduler.py` | 重写 `_mamba_block_aligned_split`:去掉上游 assert,Mamba 状态块对齐切分(eagle 时多留一块防 prune 导致 cache miss),支持 external KV connector |
| `patch_kv_cache_utils.py` | 上游 #40860 禁止"混合 KV 组 + 多 block size + CP",NPU 对 MLA/SWA-MLA 分层实现了 CP,改回 lcm(block_sizes)×dcp×pcp 而非报错 |
| `patch_kv_cache_interface.py` | `MLAAttentionSpec` 子类化,让 DSA(DeepSeek 稀疏注意力)模型复用 MLA 描述 + Sparse C8 支持 |
| `patch_kv_cache_coordinator.py` | PD-disagg + hybrid Mamba:D 侧只收到 FullAttention 块,hit 长度取 FA 组而非全组 min,避免前缀复用被清零 |
| `patch_mamba_manager.py` | 混合 Mamba 前缀缓存查找支持 PCP/DCP(上游 #40996 只支持 DCP) |
| `patch_speculative_config.py` | **MTP 模型族大表**(详见 §4.3) |
| `patch_pp_mtp.py` | PP>1 时 drafter 只在最后一个 stage 本地加载,绕过上游对 PP size 的均匀划分校验 |
| `patch_camem_allocator.py` | sleep mode 的 allocator 可用性检查:上游只认 CuMem,NPU 用自家 CaMem |

规律:**每一个 patch 的 Future Plan 都是"上游合了 XXX 就删"**——这层 patch 的本质是"上游还没有平台钩子的地方先打洞"。对我们要给上游贡献代码的启示:优先推"加钩子"型 PR(比如我们的 #14510 就是门控 + 优雅降级,而非硬改),而不是"打洞"型。

---

## 4. spec decode 对照

### 4.1 方法矩阵

上游 0.22.1 `vllm/v1/spec_decode/` 支持的 method(`vllm/config/speculative.py:534-582` 解析):`ngram`、`ngram_gpu`、`suffix`、`draft_model`、`medusa`、`eagle`/`eagle3`、`mtp`、`dflash`、`extract_hidden_states`、`custom_class`(用户自定义类路径)、以及模型专属的 `gemma4` / `step3p5`。

vllm-ascend `vllm_ascend/spec_decode/__init__.py:33-51` 的 `get_spec_decode_method` 支持:`ngram`、`ngram_gpu`、`suffix`、`medusa`、`eagle`/`eagle3`/`mtp`、`dflash`、`draft_model`、`extract_hidden_states`。

差异:

- **`custom_class` 不支持**:全仓库 0 处引用,`NPUModelRunner` 建 drafter 走自己的工厂(`model_runner_v1.py:616`),传 custom_class 会直接 `ValueError: Unknown speculative decoding method`;
- **`gemma4`/`step3p5` 不支持**:上游按 hf_config.model_type 特判加载专属 proposer,NPU 侧没接;
- **`ngram_gpu` 是空壳**:`ngram_proposer_npu.py` 的 `propose()` 是 `pass`——注册了方法名但没实现(继承上游 `NgramProposerGPU`,`:22-35` 只做了 dummy_run 兼容)。真实可用的是 CPU 版 `ngram`(重写了 `propose`,走 `batch_propose` + input_batch 状态,`ngram_proposer.py:50-64`)。

### 4.2 架构分叉:上游 proposer 自治 vs NPU 执行层接管

这是两侧最深的结构差异,一张对照表说清:

| 维度 | 上游 vllm 0.22.1 | vllm-ascend 0.22.1rc1 |
|---|---|---|
| 调用关系 | GPUModelRunner 基本只调 `drafter.propose()`,drafter 自己管模型/输入/循环 | `NPUModelRunner` 把 spec decode 执行层整个接过来:`execute_model`(`model_runner_v1.py:1900`)只做目标前向并把状态存进 `execute_model_state` 返回 None,采样+草稿在 `sample_tokens`(`:2326`)里做 |
| drafter 前向 | **K-1 次独立调用**(PIECEWISE-only),每次单步 | **`_run_merged_draft` 合并多步**(`llm_base_proposer.py`):一次准备 `multi_steps_attn_metadata`,循环内连续前向,步间零 CPU 同步 |
| 图模式 | drafter 只支持 PIECEWISE(我们调研 #45258/#34880 的结论:FULL 无先例) | drafter 可走 ACL FULL 图(`ACLGraphWrapper` 包 `_runnable`);draft_model 默认禁用图(方案 C),`additional_config draft_model_full_graph=true` 开方案 A——即我们的 PR #14510 |
| 拒绝后处理 | 上游 `prepare_inputs_padded` | 同名方法被 ascend 版覆盖,纯 GPU(triton kernel)算 `token_indices_to_sample`,拒绝 token 留作 padding("算了但扔掉",换零同步) |
| 拒绝采样 | `vllm/v1/sample/rejection_sampler.py` | 模块属性级替换(`patch/worker/patch_rejection_sampler.py`):`apply_sampling_constraints` 加 npu_top_k_top_p 路径,`expand_batch_to_tokens`/`rejection_sample` 换 ascend triton kernel(实现在 `vllm_ascend/sample/rejection_sampler.py`) |

细粒度的端到端时序(token 流转、7-token padding 之谜、KV 回滚)在 `spec_decode_eager_flow.md` 已写透;FULL 图的四层 bug 与方案 A/C 在 `draft-model-full-graph-journey.md`;NPU 侧算子限制(FIA TND 布局 query≤16 → **draft token 数硬上限 15**、block_size 128、开 SD 失去 PA 快速路径)在 `attention_backend_arch.md`。此处只补一句总括:**vllm-ascend 重写执行层的根本动机是 NPU 的 host-device 同步代价**——所有设计(merged draft、padded batch、execute/sample 拆分对 async scheduling 的兼容)都指向"热路径零 `.item()`/零 D2H"(与插件 AGENTS.md 的 NPU 条款一致)。

### 4.3 MTP 模型族大表(`patch_speculative_config.py:15-127`)

上游按 model_type 白名单识别 MTP 草稿模型;vllm-ascend 用 `SpeculativeConfig.hf_config_override` 把一大族国产模型映射到自实现的 MTP 架构:deepseek_v3/v32/v4、glm_moe_dsa、pangu_ultra_moe、mimo、glm4_moe(+lite)、glm_ocr、ernie4_5_moe、nemotron_h、qwen3_next、exaone_moe、qwen3_5(+moe)、longcat_flash、step3p5,外加 MistralLarge3 → Eagle 变体。**对 Phase 3 选型的直接含义:DeepSeek 系/Qwen3-Next/Qwen3.5 的 MTP 在 vllm-ascend 上是一等公民**(D3 候选里的 DeepSeek-V2-Lite 属上一代,用 eagle/mtp 路径;GLM-4.5-Air 走 glm4_moe_mtp)。

### 4.4 我们已在场上的一角

bug1(MRO 委托)/bug2(pattern 跨模型污染)/bug3(draft_model 图模式)三个 PR 正是打在 `llm_base_proposer.py` + `model_runner_v1.py` 这条执行层上——即 §4.2 表格右列的实现。上游迁不动的根因也在这里:上游 drafter 是"K-1 次独立调用 + PIECEWISE-only"的另一种架构,cherry-pick 无从谈起(D5 评估结论)。

---

## 5. 上游 0.26.0 的走向(5080 环境,vllm 0.26.0 + torch 2.11)

供"跟进上游"用,只列与本对照相关的:

- `v1/spec_decode/` 新增 **`dynamic/` 目录**——动态/自适应 spec decode 开始成体系(方向上与 DynamicBatch/ProfilingChunk 的"自适应"思潮合流,值得后续专门看);
- `Scheduler.schedule()` 签名多了 **`throttle_prefills: bool`**(`scheduler.py:425`)——prefill 节流成为上游调度器的显式参数,DynamicBatch 那个"decode-first"思想的官方化苗头;
- spec_decode 新增 `step3p5.py`(NPU 侧无)、`vocab_mapping.py`(草稿/目标词表映射,多词表 drafter 的通用化);
- 调度目录结构(`v1/core/sched/` 七个文件)与 0.22.1 完全一致——fork-母本同步压力暂未爆炸。

---

## 6. 对 Phase 0 / 端侧单 batch latency 的启示

PLAN 原则 4:"端侧优化目标是单 batch latency,需改 scheduler / 采样路径"。这次对照摸到的具体抓手:

1. **chunked prefill 是单请求延迟的第一调度变量**:长 prompt 的 prefill chunk 与 decode 抢同一个 token budget,chunk 越大 decode 越饿(ITL 毛刺)。上游用 `long_prefill_token_threshold` 静态限制,vllm-ascend 给了两个自适应答案(DynamicBatch 查表 / ProfilingChunk 在线拟合)。**端侧外推**:端侧单 batch 时这个问题退化(没有多请求争抢),但"prefill chunk 切分对首 token 延迟的影响"仍成立,且查表/在线拟合在端侧更好做——设备型号固定、负载单一,表小模型稳;
2. **decode-first 重排是免费的延迟手段**:SchedulerDynamicBatch 把 decode 请求挪到队首(`scheduler_dynamic_batch.py:160-163`),一行列表推导的成本换 decode 不被 prefill 挡路。端侧单 batch 下退化为"解码任务永远优先于后台任务"的调度策略;
3. **async scheduling + 占位记账**是上游降低每步调度开销的主线(`num_output_placeholders`),vllm-ascend 的 execute/sample 拆分就是对它的适配。端侧上 CPU(Host)与 NPU 的 overlap 同样成立,且窗口比例更大;
4. **spec decode 的 NPU 执行层是现成的主战场**:我们已在场上(PR 系列),且 §4.3 的 MTP 大表说明国产 MoE+MTP 组合是 vllm-ascend 的投入重点——与 D2 候选 B(MoE + SD)方向天然重合;
5. **改调度器时的姿势**:四个 fork 的教训——要长期维护的改动走 `scheduler_cls` 正门(方法级差异尽量用 override 小函数,别整抄 `schedule()`);一次性验证可以抄,但要在笔记里记母本版本。BalanceScheduler 走的模块属性替换偏门,是最脆的一种(依赖 import 顺序)。

---

## 7. 文件索引

### vllm-ascend(工作仓库 research/main)

| 文件 | 关键位置 | 内容 |
|---|---|---|
| `platform.py` | `:647-657` worker_cls 选择;`:665-704` 四调度器接线 | NPU 平台配置中枢 |
| `patch/platform/patch_balance_schedule.py` | `BalanceScheduler:36`,`balance_flag:282-284`,类替换 `:702-703` | DP 均衡调度 |
| `core/recompute_scheduler.py` | `schedule:158`,`update_from_output:762`,`AsyncRecomputeScheduler:1045`,占位 `:149-152`,EPLB 记账 `:820-836` | PD-disagg 重算式抢占 |
| `core/scheduler_dynamic_batch.py` | `BudgetRefiner:35`,`refine_budget:107`,decode-first `:160-163` | SLO 查表动态 budget |
| `core/scheduler_profiling_chunk.py` + `patch/platform/patch_profiling_chunk.py` | `ProfilingChunkScheduler:46`;跨进程补丁 `:216-229` | profiling 动态分块 |
| `spec_decode/__init__.py` | `:33-51` | proposer 工厂(方法矩阵) |
| `spec_decode/llm_base_proposer.py` | `AscendSpecDecodeBaseProposer:134`,`_propose:687` | 执行层核心(已有三篇笔记) |
| `patch/worker/patch_rejection_sampler.py` | 全文件 9 行 | 拒绝采样 NPU kernel 替换 |
| `patch/platform/patch_speculative_config.py` | `hf_config_override:15-127` | MTP 模型族映射表 |

### 上游 vllm 0.22.1(快照)

| 文件 | 关键位置 | 内容 |
|---|---|---|
| `v1/core/sched/scheduler.py` | `Scheduler:64`,`schedule:329`,spec 附着 `:500-514`,KV 回滚 `:1374-1381` | 母本调度器 |
| `v1/core/sched/async_scheduler.py` | `AsyncScheduler:12`,`_update_after_schedule:18-35` | 异步调度 + 占位记账 |
| `config/scheduler.py` | `scheduler_cls:127`,`get_scheduler_cls:168-188` | 调度器插件化钩子 |
| `v1/engine/core.py` | `:135` | get_scheduler_cls 调用点 |
| `config/speculative.py` | `:534-582` | spec decode method 解析(全集) |
| `v1/spec_decode/` | 目录 | custom_class / gemma4 / step3p5 为 NPU 侧缺失项 |
