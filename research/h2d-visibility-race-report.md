# H2D 可见性竞态技术报告:pinned non_blocking H2D 拷贝后同流 kernel 读到旧数据

> ⚠️ **2026-08-28 终审状态:本报告核心结论已被推翻,保留为历史推理记录。**
> 后续审计(归因文档 §14 第 1-5 轮)证明:①本文 §3 全部 repro 级数据(六格
> 矩阵/窗口谱/栅栏阶梯/加压闭环/时间线)的 miss 质量经 alt2 阳性对照证实为
> **探针槽覆盖伪影**(host 在 ~2048 任务 FIFO 提交环内领先 100-200+ 步,双槽
> 被改写,晚拷贝送 future 值;同二进制 unique 槽 = 0%);②纯 CANN 探针
> (libascendcl+libopapi,含满环压力,direct/threaded)**全绿** = 同流
> 拷贝→kernel 排序与可见性在纯 CANN 层正确;③"栅栏盲/流外拷贝"叙事待
> unique 槽复跑,大概率同为伪影。**仍然成立的只有 engine 级经验事实**:
> eagle3 16K 无 fix 确定性崩溃、唯一逃逸值 -1(三计数器 Run 4)、fix 实效。
> 缺陷机制重开(候选:host 单缓冲 pinned 改写 × 深环 / torch_npu 引擎态提交序 /
> 图重放互作),现行有效推理史见 `offstream-copy-attribution.md` §14。
> **勿将本文任何 repro 级数据用作 issue/PR 素材。**
>
> ─────────────────────────────────────────────────────────────────
>
> 2026-08-27 调查终版报告。上游背景:vllm-ascend #14922(eagle 系 SD 16K 崩溃,
> fix = 拷贝改 blocking,已验证正确)。本文档回答该 bug 背后的**平台层问题**:
> 为什么 async H2D 拷贝与同流消费 kernel 之间没有顺序保证,缺陷在哪一层,
> 什么配置安全/不安全。
>
> 姊妹篇:`research/h2d-visibility-race-journey.md`(人话版);
> `research/offstream-copy-attribution.md`(完整推理史/审计日志,含全部被
> 推翻的中间结论;本文只保留终态结论与关键证据);
> `notes/upstream/ascend-pytorch-issue-off-stream-copy.md`(issue 草稿,待按
> 本文重写)。
>
> 证据标注:**[M]** 服务器实测;**[S]** 源码静态阅读(Ascend/pytorch tag
> `v26.1.0-pytorch2.10.0` = pip torch-npu 2.10.0.post4;cann/runtime master)。

## 0. TL;DR

在 Ascend 910B3 / CANN 9.1.0 / torch-npu 2.10.0.post4 上:

1. **[M] 同一条流上,`aclrtMemcpyAsync`(H2D, pinned)的 SDMA task 完成之后
   才启动的同流 AI core kernel,仍可能读到拷贝之前的旧值**。执行顺序完全
   正确(profiling 实证:拷贝先完成、kernel 后启动、区间串行),缺的是
   **数据可见性**:SDMA 写→AI core 读这条跨引擎路径没有被流序覆盖。
2. **[M] 这是一个概率性时序窗口**,开度由"SDMA 积压量 vs 算子提交延迟"的
   赛跑决定,实测谱系从 0%(碰巧关)到 99.94%(64MiB/步积压)。**没有任何
   torch_npu 提交路径是安全的**——包括观测中 miss=0 的配置(加压后 99.94%)。
3. **[M] 唯一安全形态是同步拷贝**(`copy_(non_blocking=False)` → 同步
   `aclrtMemcpy`,不经 SDMA 异步通道)。#14922 的 fix 因此正确且必要。
4. **[M] 栅栏全部无效**:wait_event / event.synchronize / stream.synchronize
   都无法观察到该拷贝或建立顺序(wall 时间证明等待未发生)。
5. 缺口**特定于 AI core 读者**:同样的纯 ACL 拓扑下,SDMA 读者(async D2H)
   100% 读到新值——所以这不是"拷贝挂错流/乱序",是**跨引擎可见性**。

## 1. 背景与环境

- 起点:#14922,`prepare_next_token_ids_padded` 的 `backup_next_token_ids`
  pinned H2D(non_blocking)与同流 `torch.where` 消费之间竞态,stale `-1`
  哨兵逃逸进词表 gather → aivec 崩溃。fix(blocking)已合并验证。
- 平台问题的调查动机:①上游维护者问"为什么不保序";②该行为若为平台
  语义缺口,所有走 CUDA 惯用法(pinned + non_blocking + 同流消费)的
  vllm-ascend / 用户代码都在暴露面;③确认 issue 应报给谁。

环境 [M]:

| 组件 | 版本 |
|---|---|
| NPU | Ascend 910B3(服务器 8 卡容器) |
| CANN | 9.1.0(`ascend-toolkit` 链接至 `cann-9.1.0/`) |
| driver | 25.3.rc1.1 |
| torch / torch-npu | 2.10.0 / 2.10.0.post4(= Ascend/pytorch tag `v26.1.0-pytorch2.10.0`) |

## 2. 方法论

- **判别实验驱动**:每个假说配一个最小实验,预期结果预先写成矩阵;推翻即
  记录,不恋战。全天 13 轮,7 个中间结论被推翻(§5)。
- **三个仪器**(全部在 fork `research/v0.23.0`):
  - `repro_h2d_order.py`:torch_npu 层 repro(pinned 64B 被测拷贝 + bg 大拷贝
    积压 + 设备侧零同步 miss 计数;`--copy-mode` 解耦拷贝提交通道;
    `--profile` 采集 msprof Text 导出)
  - `cann_memcpy_order.py`:纯 CANN 层探针(ctypes 直调 libascendcl;读者为
    同流 async D2H;`--dispatch direct|threaded` 解耦提交线程拓扑)
  - `stream_audit.py`:msprof CSV 审计(stream 归属 / TIMELINE 区间重叠 /
    ORDER 派发序逆序 / ARRIVAL 到达序块模式)
- **证据等级**:[M]/[S]/[I],推断必须升级为 [M] 才可入终判。

## 3. 全部实验数据

### 3.1 行为层(R0,[M],08-26/27)

torch_npu repro(2000 步,64B 被测,读者=同流算子):

| miss 谱系 | bg=0 | bg=4×1MiB | bg=8×1MiB | bg=16×1MiB |
|---|---|---|---|---|
| racy(non_blocking=True) | 0.70% | 1.50% | 99.90% | 99.85% |
| fix(non_blocking=False) | — | — | **0%** | — |

栅栏阶梯(racy, bg=8;wall 对照:fix 同步代价 996us/步):

| 栅栏 | wall/步 | 结果 |
|---|---|---|
| `stream.wait_event(ev)` | — | ✗ |
| `ev.record(s)` + `ev.synchronize()` | 740us | ✗(等待从未发生) |
| `copy_stream.synchronize()` | 748us | ✗(miss 99.9%) |

### 3.2 源码层(R1,[S])

torch copy_(H2D, pinned, non_blocking)发射链:

```
Tensor.copy_(non_blocking=True)
└─ CopyKernelOpApi.cpp:43   copy_between_host_and_device_opapi
   ├─ ASYNC_MEMCPY 消息 → host task queue(TASK_QUEUE_ENABLE 默认 1)
   │   consumer 线程(StartConsume 先 SetDevice,NPUQueue.cpp:795)
   │   → aclrtMemcpyAsync(dst, src, kind, paramStream=发射流)   # 带流参数
   └─ process_non_blocking_copy(CachingHostAllocator.cpp:1352)
       └─ pinned → aclrtPointerGetAttributes + record_event
           (record_event 是纯记账,不调 aclrtRecordEvent,:689-719)
```

- 算子下发同样经队列(`EXEC_NPU_CMD` → `OpCommand::RunOpApiV2` 入队,
  op_api_common.h:358);`nm -D libopapi.so | grep -c aclnnInplaceCopy` = 2,
  opapi 路径生效 [M](老路径回落已排除)。
- blocking 路径 = 先 `aclrtSynchronizeStream` 再同步 `aclrtMemcpy`,
  不经 SDMA 异步通道。

### 3.3 profiling 层(R2,[M],TQ=1 + torch,bg=8,50 步,同 run miss=94%)

| 分析 | 结果 |
|---|---|
| stream 归属 | MEMCPY 450 与计算 kernel 350 **同一条流**(655);fix 模式同构(659) |
| fix 的 task 计数 | memcpy=400=8×50(**同步被测拷贝不产生 async task**) |
| TIMELINE 区间重叠 | MEMCPY×kernel **零重叠**(约 20 对重叠全为 memcpy×memcpy,SDMA 多通道自家并行) |
| ORDER 派发序逆序 | **0**(800 tasks;start 严格随 task_id 单调) |
| ARRIVAL 到达序块模式 | **9M 7K 完全交替**(拷贝按 Python 发射序到达并先执行) |

**合取推论(本调查最关键的一条证据)**:执行序正确(拷贝先完成 → kernel
后启动,串行)+ 同 run miss=94% ⇒ **拷贝 task 的完成不蕴含数据对后续 kernel
可见**。注意 task_id 是 runtime 到达序而非 Python 发射序——"零逆序"与
"到达序颠倒"并存曾被误判,ARRIVAL 分析排除了后者。

### 3.4 纯 CANN 层(R3/R4,[M],500 步,bg=8×1MiB;读者=同流 async D2H)

| dispatch | 拓扑 | 结果 |
|---|---|---|
| direct + fence=none | 全主线程直发 | **0%** |
| direct + stream_sync / event_sync / event_wait | | 0% / 0% / 0% |
| direct + plain(pageable 源) | | 0% |
| sync(负对照) | | 0% |
| threaded(worker 线程 FIFO 下发 H2D+D2H,主线程排空后栅栏;忠实镜像 torch_npu 拓扑,worker 自带 SetDevice) | | **0%** |

结论:**libascendcl 在上述拓扑下,SDMA 读者(async D2H)对同流先行的 H2D
100% 有序**。此前"CANN 层流序缺陷"的直发形态被证伪;且可见性缺口被收窄到
**AI core 读者**。

(实现注记:ACL context 为 per-thread,worker 线程须先 `aclrtSetDevice`,
否则一切调用报 107002 `ACL_ERROR_RT_CONTEXT_NULL`——torch_npu consumer
同款处理。)

### 3.5 torch_npu env 对照(R4,[M],repro,读者=算子)

| env | miss |
|---|---|
| `TASK_QUEUE_ENABLE=0`(关队列,一切直发) | **0%** |
| `TASK_QUEUE_ENABLE=1`(默认) | 99.90% |
| `TASK_QUEUE_ENABLE=2`(V2/poll) | 99.90% |
| `PER_STREAM_QUEUE=1`(per-stream 队列) | 99.85% |

### 3.6 通道矩阵(R5,[M],repro `--copy-mode`,被测拷贝提交通道 × 算子通道)

| # | 被测拷贝 | 算子(TQ) | query epilogue | miss |
|---|---|---|---|---|
| 1 | torch copy_(队列) | 队列(1) | torch 内置 | 99.90% |
| 2 | torch copy_(直发) | 直发(0) | torch 内置 | **0%** |
| 3 | acl-direct(主线程 ctypes) | 队列(1) | 无 | 99.90% |
| 4 | acl-direct(主线程 ctypes) | 直发(0) | 无 | 46.95% |
| 5a | acl-direct(主线程 ctypes) | 直发(0) | 有(ctypes) | 52.65% |
| 5b | acl-direct(主线程 ctypes) | 队列(1) | 有(ctypes) | **4.65%** |

- 5a vs 4:`aclrtPointerGetAttributes` **无屏障效果**;
- 5b 的 4.65% = query 阻塞 + consumer 拖延造成的时序红利,非语义保证;
- 格 2 与格 4/5a 无 profile 时 wall 几乎相同(415 vs 438us/步)而 miss
  0% vs 47%——差异只剩 epilogue 微差。

### 3.7 观察者效应与 task 计数(R6,[M])

| 配置(TQ=0) | 无 profile | 加 profiler(119ms/步税) |
|---|---|---|
| torch copy_(格 2) | 0% | 0%,memcpy task = **450** |
| acl-direct(格 4) | 46.95% | **0%(!)**,memcpy task = **450** |

- 两路 64B 被测拷贝**都产生 SDMA task**(450=9×50)→ 通道分流
  (PCIE_BAR vs SDMA)假说出局;
- profiler 税把算子提交拖慢到远超 SDMA 消化时间 → 窗口关闭 → miss=0。
  **0% 的脆弱性直接可见**。

### 3.8 加压闭环(R7,[M])

`TQ=0 + torch copy_ + bg=16×4MiB(64MiB/步)×5000 步`:
**miss = 4997/5000 = 99.94%**。

"格 2(TQ=0+torch)的 0% 是低负载巧合"定谳——64MiB 积压淹没一切 epilogue
微差,窗口全开。**没有任何 torch_npu 提交路径是安全的**。

## 4. 统一竞态窗口模型(终判)

**缺陷**:同流 `aclrtMemcpyAsync`(H2D, pinned)的 SDMA task 完成后,数据
对**同流后续 AI core 任务**的可见性不被流序保证。窗口开度由
**SDMA 积压消化速度 vs 算子提交延迟**的赛跑决定:

```
窗口开度谱(全部 [M]):
99.94%  TQ=0 + torch, 64MiB/步积压(R7)
99.90%  TQ=1(算子经 consumer 异步提交,窗口常态全开)
52.65%  TQ=0 + ctypes + query(R5a)
46.95%  TQ=0 + ctypes(R4 格 4)
 4.65%  TQ=1 + ctypes + query(R5b,时序红利)
 0.00%  TQ=0 + torch 低负载(R4 格 2,R7 证伪其安全性)
 0.00%  任意路径 + profiler 税(R6,观察者效应)
 0.00%  纯 CANN 层 + SDMA 读者(D2H)——不受影响
```

**责任划分**:

| 层 | 定性 |
|---|---|
| CANN runtime / 硬件 | **根因**:SDMA 写→AI core 读的跨引擎可见性未纳入流序;aclrtMemcpyAsync 的完成语义对后续 kernel 不蕴含数据可见(acl_rt.h 只承诺"用 synchronizeStream 确保完成",未承诺可见性序,且实测栅栏对该拷贝失明) |
| torch_npu | 暴露面放大者:task queue 拓扑把窗口开到必然级(99.9%),但非根因;其 CachingHostAllocator recordEvent 复用保护(CUDA 同款设计)同样建立在被违反的假设上 |
| 用户代码 | 唯一安全写法 = `copy_(non_blocking=False)`(payload 小时阻塞开销可忽略);或确认读者为 SDMA(D2H)类 |

## 5. 中间结论推翻史(防止后人重走弯路)

| # | 中间结论(曾判"定谳") | 推翻实验 |
|---|---|---|
| 1 | 拷贝是"流外操作"(不挂流) | R2:MEMCPY task 在发射流上 |
| 2 | 同流乱序/并发执行 | R2:零重叠+零逆序 |
| 3 | CANN runtime 直发流序缺陷 | R3:纯 CANN direct 全绿 |
| 4 | host task queue 是肇因(TQ=0 治愈) | R5 格 4:TQ=0+ctypes 仍 47% |
| 5 | 跨线程提交拓扑触发(H-dispatch) | R4:纯 CANN threaded 全绿 |
| 6 | 物理通道分流 PCIE_BAR vs SDMA(H-channel) | R6:两路均 450 task |
| 7 | recordEvent / PointerGetAttributes 隐式屏障 | R1 源码(record 纯记账)+ R5a(query 无效) |

方法论教训:①profiling 时间戳/ID 的**语义**(task_id=到达序而非发射序)必须
独立验证,否则"零逆序"可自洽于两种相反现实;②**miss=0 不构成安全性证据**,
只有窗口模型能同时解释 0% 与 99.94%;③观察者效应(profile 税)可以完全
翻转结论,判别实验必须评估仪器的扰动。

## 6. 影响面

- 所有"pinned host buffer + `copy_(non_blocking=True)` + 同流 AI core 消费"
  的代码(CUDA 标准惯用法)在本栈上不安全。#14922 的引擎崩溃只是最响的
  表现;静默数据损坏同样可能。
- torch_npu 的 `CachingHostAllocator` host buffer 复用保护(recordEvent 门控)
  建立在同一被违反的假设上:pinned 块可能在拷贝数据真正可 consumed 前被
  复用(第二重数据撕裂风险,独立于用户侧任何栅栏)。
- 正确性问题与性能取舍:blocking 拷贝对 <1KB payload 的额外开销可忽略
  (#14922 实测 ITL 无回归);大 payload 应评估双缓冲 + 事件同步之外的安全
  模式(待 CANN 侧澄清,见 issue 三问)。

## 7. issue 素材清单(下会话写作用)

- 标题方向:probabilistic stale read by same-stream kernels after
  `aclrtMemcpyAsync` H2D(pinned)——SDMA completion not visibility-ordered
  for AI-core consumers; window modulated by submission timing。
- 核心证据:六格矩阵(§3.6)+ 加压闭环(§3.8)+ 观察者效应(§3.7)+
  执行序正确的时间线(§3.3)+ 纯 CANN D2H 全绿(§3.4,缺口 AI-core 特定)。
- 三问:①acl_rt.h 的"完成"语义是否蕴含对后续同流任务的可见性?②有没有
  正确的 async + 流序 + 可见性的写法(API/flag)?③若为 runtime 缺陷可否
  转交 CANN 团队?
- **落仓决策依赖第七轮探针(§14 of offstream-copy-attribution.md)**:
  `cann_aicore_visibility.py`(纯 CANN + aclnnAdd AI core 消费者)红 →
  自包含 repro,落 **gitcode cann/runtime**(最近近亲 #873 亦在该仓,
  重复检索见笔记 `notes/upstream/issue-duplicate-search-20260828.md`);
  绿(加压后仍绿)→ 落回 Ascend/pytorch(torch_npu)。红绿两路都定仓。
- 附件:`repro_h2d_order.py`(含 --copy-mode/--profile)、`cann_memcpy_order.py`
  (direct/threaded)、`cann_aicore_visibility.py`(纯 CANN AI core 消费者)、
  `stream_audit.py`;fork research/v0.23.0(红/绿定仓后回填 commit)。

## 8. 工件索引

| 工件 | 位置(research/v0.23.0) |
|---|---|
| torch_npu repro(三模式 + copy-mode + profile) | `research/repro_h2d_order.py` |
| 纯 CANN 探针(direct/threaded,fence 矩阵) | `research/cann_memcpy_order.py` |
| 纯 CANN AI core 消费者探针(aclnnAdd + mock 自测) | `research/cann_aicore_visibility.py`(+ `_mock.c`) |
| msprof 审计(stream/TIMELINE/ORDER/ARRIVAL) | `research/stream_audit.py` |
| 推理史/审计日志(13 轮原始记录) | `research/offstream-copy-attribution.md` |
| 人话版 | `research/h2d-visibility-race-journey.md` |
| issue 草稿(待重写) | 笔记仓库 `notes/upstream/ascend-pytorch-issue-off-stream-copy.md` |
| 重复检索结论(2026-08-28,gitcode/Gitee/GitHub 全扫) | 笔记仓库 `notes/upstream/issue-duplicate-search-20260828.md` |
| 上游 PR(fix) | vllm-ascend #14922 |
