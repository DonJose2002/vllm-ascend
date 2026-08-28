# Off-stream copy 归因文档:当前判断、依据来源、薄弱点与判别实验

> 2026-08-27,#14922 流外拷贝调查的归因存档。本文件的直接动因:维护者/读者对
> "问题在 CANN 层"表示怀疑。**怀疑已被第一轮判别实验证实一半:纯 CANN 直发
> 全绿(§8),CANN 的单线程流序无辜;嫌疑收窄到 torch_npu 的下发拓扑(跨线程
> 提交)。第二轮实验(--dispatch threaded)已备好,见 §9。**
> 本文档保留完整推理史(包括被推翻的中间结论),以供复盘;**当前有效结论
> 以 §8/§9 为准,§1-§7 中与 §8 冲突处已被 §8 勘误覆盖**。
>
> 证据标注约定:**[M]**=服务器实测(measured);**[S]**=源码静态阅读(static);
> **[I]**=推断(inference,未实测/未逐字节验证)。

## 1. 当前判断(一句话)

pinned non_blocking H2D `copy_` 在 torch_npu 2.10.0.post4 + CANN 9.1.0 上的
乱序,不是 torch_npu 把拷贝发错了流(源码全链带 stream,profiling 显示 task
在发射流上),而是拷贝的"完成"没有被接入流的依赖序——但**这个断裂点的精确
层次(CANN runtime 库 / driver / SDMA 硬件管线,或 torch_npu 的下发时序)+
profiling 时间戳的可信度,现有证据无法定谳**,需要 §5 的纯 CANN 实验分辨。

## 2. 三层证据链与依据来源

### 2.1 行为层(全部 [M],仪器 `research/repro_h2d_order.py`,2026-08-26/27 服务器)

| # | 观测 | 数据 |
|---|---|---|
| A1 | racy(pinned + non_blocking H2D)与同流消费 kernel 乱序 | bg=0/4/8/16 → miss 0.70%/1.50%/99.90%/99.85%(随 SDMA 积压单调饱和) |
| A2 | fix(blocking)全压力 miss=0 | 同 bg=8,miss=0 |
| A3 | 栅栏阶梯三级全失效 | wait_event ✗;record+event.synchronize ✗(wall 740us/step);copy_stream.synchronize ✗(wall 748us/step,miss 99.9%)——均 < fix 同步代价 996us/step,**等待从未发生** |
| A4 | ASCEND_LAUNCH_BLOCKING=1 治愈(08-24 引擎侧) | 同配置必崩→通过 |

来源:STATUS 2026-08-27 行;`notes/upstream/pr-14922-ordering-reply.md` §2/§3。

### 2.2 源码层(全部 [S],Ascend/pytorch tag `v26.1.0-pytorch2.10.0` 浅 clone)

torch_npu 发射链(每步带行号):

```
Tensor.copy_(src, non_blocking=True)                    # H2D, contiguous, same dtype
└─ CopyKernelOpApi.cpp:169  NPUNativeOpApiFunctions::copy_
   └─ :43-49  copy_between_host_and_device_opapi, non_blocking 分支
      ├─ CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch     (CalcuOpUtil.cpp:267)
      │  └─ AsyncTaskQueueInterface.cpp:86  AsyncCopyTask::LaunchCopyTask
      │     ├─ TASK_QUEUE_ENABLE 默认 1 (OptionsManager.cpp:583)
      │     │  → ASYNC_MEMCPY 消息 enCurrentNPUStream 入 host 环形队列
      │     │    (NPUStream.cpp:748 记录 paramStream = 发射时的 current stream)
      │     │    PER_STREAM_QUEUE 默认 0 (:601) → 全设备单条 FIFO
      │     └─ consumer 线程 ReadQueue→Call (NPUQueue.cpp:448,同步执行后才推进读指针)
      │        → OpParamMaker.cpp:704 AsncExecFunc
      │        → :708 + MemcopyAsyncFunc:467-479
      │          aclrtMemcpyAsync(dst, dstLen, src, srcLen, kind, paramStream)
      │                                              # ← 带流参数下发 [S]
      └─ process_non_blocking_copy                     (CachingHostAllocator.cpp:1352)
         └─ pinned/registered 内存 → 仅 CachingHostAllocator_recordEvent(ptr, stream)
                                            # ← CUDA 同款设计,假设流序安全 [S]
```

栅栏设施([S]):`NPUStream::synchronize()`(NPUStream.cpp:407)= 先经 `stream()`
(:417-445,排空 host 队列,即拷贝的 aclrtMemcpyAsync **已被 consumer 线程
下发进 libascendcl**)再 `AclrtSynchronizeStreamWithTimeout`。Event record/wait
同样经 task queue(RECORD_EVENT/WAIT_EVENT 消息)。

blocking 路径([S]):CopyKernelOpApi.cpp:50-54 = 先 `aclrtSynchronizeStream`
再同步 `aclrtMemcpy`——不走 aclrtMemcpyAsync,是其天然安全的原因。

ASCEND_LAUNCH_BLOCKING 治愈机制([S]):OptionsManager.cpp:578
`CheckBlockingEnable()` → task queue 关闭 + 调用线程同步下发。

官方契约([S],cann/runtime 仓 `include/external/acl/acl_rt.h:2295-2301`):

> "Asynchronous memory replication between Host and Device … After calling
> this interface, be sure to call the aclrtSynchronizeStream interface to
> **ensure that the task of memory replication has been completed**"

即官方文档承诺 synchronizeStream 保证 memcpy 完成。

### 2.3 profiling 层([M],msprof Text 导出 + `research/stream_audit.py`,2026-08-27)

| # | 观测 | 数据 |
|---|---|---|
| C1 | MEMCPY task 记录在发射流 | racy:memcpy 450 个与计算 kernel 350 个同在 stream 655;fix:659 同构;引擎侧 eagle3/ngram 交叉验证同流存在 |
| C2 | 被测同步拷贝不产生异步 task 记录 | fix 的 memcpy 计数 400=8bg×50,无被测拷贝(两路径分叉佐证) |
| C3 | 执行顺序 = 派发顺序 | start 逆序 = 0(racy 800 tasks / fix 750 tasks,按 task_id 对 start 时间) |
| C4 | MEMCPY × AI-kernel 区间零重叠 | racy 20 对重叠全为 memcpy×memcpy(SDMA 多通道自家并行) |
| C5 | 消费 kernel 在拷贝 task_stop 之后执行仍读旧值 | C3+C4 ⇒ kernel.start ≥ copy.stop;同 run miss=94% |

## 3. 推理链的薄弱点(诚实清单,按杀伤力排序)

**W1(最致命):msprof 的 MEMCPY task 时间戳语义未独立验证。**
C3/C4/C5 全部依赖 task_time.csv 的 `task_start/task_stop` 反映**真实 SDMA
执行时刻**。如果时间戳记录的是 aclrtMemcpyAsync 的**下发时刻**(task queue
consumer 调用时刻),那么"串行且有序"是假象:真实执行可能远晚于消费 kernel。
此时存在一个**完全不需要 CANN 缺陷的复合假说**:
> torch_npu host task queue 的 consumer 线程把拷贝下发得比消费 kernel 晚
> (例如两类消息的处理速度/优先级不同,或拷贝消息在 consumer 内部再排队),
> 而 msprof 的 memcpy 时间戳恰好掩盖了这一点。
此假说同时解释 A1(乱序)、A3(栅栏无效——栅栏等的是"已下发的流内任务",
拷贝还没进 libascendcl,自然没人等它)与 C1-C4(时间戳假象)。**纯 CANN
脚本(§5)不经过 task queue、不依赖时间戳(用 D2H 读数据本身),是分辨
W1 的唯一干净实验。**

**W2:实际 dispatch 路径未运行时实证。**
[S] 链路假设走 `CopyKernelOpApi.cpp`(opapi 路径)。`copy_` 入口有
`DO_COMPATIBILITY(aclnnInplaceCopy, NPUNativeFunctions::copy_(...))` 回落机制
(CopyKernelOpApi.cpp:171),老路径 `CopyKernel.cpp` 未逐行排除。旁证:C1
(memcpy 在发射流)与 C2(同步路径无 task 记录)与 opapi 链路的两个分叉
一致;但 H2D 最终都汇聚到 aclrtMemcpyAsync(带 stream),对归因结论影响
有限——对"何层乱序"无影响,对"torch_npu 哪个组件参与"有影响。

**W3:pip wheel ↔ git tag 映射靠 version.txt 内容,非构建级确证。**
torch-npu==2.10.0.post4 ↔ tag `v26.1.0-pytorch2.10.0`(version.txt 字面
2.10.0.post4,且相邻 tag 为 post2/post3,序列吻合)。wheel 构建源可能与
tag 有后续 cherry-pick。缓解:§5 脚本绕开 torch_npu 源码逐行阅读的必要性。

**W4:"CANN 层"是笼统称呼。**
即使 W1 被排除(纯 CANN 复现),证据也只能支撑:
**libascendcl 的 {aclrtMemcpyAsync(H2D, pinned) + 同流后续任务} 组合不满足
acl_rt.h 自己声明的契约**。断裂点在 runtime 库实现、driver 还是 SDMA 硬件
管线的可见性语义,外部证据无法分辨——那是华为侧排查的事。issue 措辞应
停在"runtime 契约违反",不应更具体。

**W5(次要):C5 是"miss 与 C3/C4 同 run 发生"的联合推断。**
miss=94% 与 ORDER=0 出自同一次 50 步 run(profiler 税下 miss 略低于无
profile 的 99.9%)。严格说 C5 断言的是"同 run 内两种观测并存",由
C3+C4 推出 kernel 在拷贝后执行——推论依赖 C3/C4 的时间戳(W1)。

## 4. 已排除的解释(及其排除依据)

| 假说 | 排除依据 |
|---|---|
| 拷贝挂错流(专用拷贝流) | C1 [M] + 源码 paramStream [S] |
| 同流任务并发执行 | C4 [M](含 W1 保留:若时间戳为真) |
| 派发序颠倒(后发先至) | C3 [M](含 W1 保留) |
| host 写与在途读混淆 | repro 双缓冲 + 每步写真值设计 |
| 预期值污染(miss 判定自身) | 预期值(counter)在计算流设备侧生成,无 H2D |
| torch_npu 没传 stream | 源码 [S](带 W2/W3 保留) |
| **拷贝被 host task queue 延迟下发,真实执行晚于消费 kernel** | **未排除!即 W1 复合假说,§5 判别** |

## 5. 判别实验:纯 CANN 层最小脚本

`research/cann_memcpy_order.py`:**零 torch / 零 torch_npu**,ctypes 直接
dlopen `libascendcl.so`,只用 ACL C API:

```
每步: host 写 step id 到 pinned 缓冲(aclrtMallocHost)
      [可选 bg] K 个 1MiB aclrtMemcpyAsync H2D 同流积压
      被测拷贝: aclrtMemcpyAsync(dev, host, 64B, H2D, stream)     [racy]
               或 aclrtMemcpy(...) 同步版                          [sync 对照]
      [fence 变体] none | stream_sync | event_sync | event_wait
      消费:    aclrtMemcpyAsync(host_back, dev, 64B, D2H, stream)  ← 设备内存
               的天然读者(无裸 kernel 可用,D2H 即读取)
      收尾:    aclrtSynchronizeStream → host 比较 host_back[0] == step?
```

判别逻辑:消费侧是 **D2H 拷贝的数据本身**,不依赖任何时间戳/profiler;
且全程不经 torch_npu 的 task queue(调用线程直接下发)。

### 预期结果矩阵(核心)

| 结果组合 | 定性 |
|---|---|
| racy miss>0 且 sync miss=0 | **纯 CANN 复现** → torch_npu 完全出局;libascendcl 契约违反实锤(W1 排除) |
| racy fence=stream_sync 仍 miss>0 | 更强:官方文档点名的 synchronizeStream 兜底也失效 → 数据物理未落地/完成语义假 |
| racy fence=none miss=0,但 torch_npu repro miss>0 | **W1 复合假说成立**:问题在 torch_npu 层(task queue 下发时序),CANN 无辜——当前判断需推翻重写 |
| 全 miss=0(与 bg 无关) | 同上,torch_npu 嫌疑回升;下一步排查 task queue consumer 时序 |

矩阵第二、三行是本实验的分辨力所在:**无论哪个结果,怀疑都被回答**。

### 脚本的辅助对照

- `--host-mem plain`(普通 malloc 的 pageable 源)vs `pinned`(aclrtMallocHost):
  CANN 对 pageable async 拷贝的行为(torch_npu 源码声称未注册内存降级同步
  [S],此处独立验证);
- `--stale-stats`:miss 时打印读到的值的分布(落后 1 步 vs 落后多步,
  区分浅管线与深积压形态);
- 每步 wall 时间(A3 的纯 CANN 对照:栅栏是否真的在等)。

## 6. 归因边界(结论能说到哪)

- 能说:乱序不在 torch_npu 的"流选择/参数传递"层([S]+C1 [M]);
- 若 §5 复现:能说 libascendcl 的 aclrtMemcpyAsync(H2D, pinned)+流同步
  组合违反其头文件契约([M],W1 排除);
- 不能说:缺陷在 runtime so 还是 driver 还是 SDMA 硬件可见性(需华为侧);
- 若 §5 不复现:当前"CANN 层"判断**作废**,转向 torch_npu task queue
  下发时序(届时 repro 加 TASK_QUEUE_ENABLE=2 / PER_STREAM_QUEUE=1 对照,
  以及 bypass task queue 的直发路径实验)。

## 7. 工件索引

- 行为层仪器:`research/repro_h2d_order.py`(fork research/v0.23.0@4c01ab16b)
- profiling 分析器:`research/stream_audit.py`(同上)
- **判别脚本:`research/cann_memcpy_order.py`**(direct 模式 @870c7aa13;threaded 模式 @90a4b4791)
- issue 草稿(终稿待判别结果修正):`notes/upstream/ascend-pytorch-issue-off-stream-copy.md`
- 上游 PR:#14922(其 fix = blocking,无论归因如何都正确且必要)

## 8. 第一轮判别结果(2026-08-27 晚,[M]):CANN 直发无辜,W1 部分证实

`cann_memcpy_order.py --dispatch direct` 服务器全矩阵(500 步,bg=8×1MiB):

| 配置 | miss |
|---|---|
| racy + fence=none + pinned | **0** |
| racy + fence=stream_sync | 0 |
| racy + fence=event_sync | 0 |
| racy + fence=event_wait | 0 |
| racy + fence=none + plain(pageable) | 0 |
| sync(负对照) | 0 |

**判读**:libascendcl 的 `aclrtMemcpyAsync`(H2D, pinned, 64B)在**调用线程
直发**时,与同流 D2H 消费完全有序,SDMA 积压下也如此。§1 的"CANN 层"
判断**在直发拓扑下被证伪**;缺陷层次回到 torch_npu 内部。用户怀疑成立。

**同时勘误 §2.3-C5(判读错误,被 W1 命中)**:ORDER 分析假设 task_id =
Python 发射序,推出"消费 kernel 在拷贝后执行"。实际上 **task_id 是任务
到达 runtime 的序**:若消费 kernel 先于拷贝到达 runtime(跨通道下发时序),
task_id(kernel) < task_id(copy),执行序跟随 task_id,全程"有序"——
但相对 Python 语义序是颠倒的。profile 的"串行+有序"与"kernel 先到达、
拷贝后到达"完全自洽,C5 推断不成立。task_time.csv 的 memcpy 时间戳语义
问题(W1 后半)也随之不再是必要假设——需要的只是 task_id 语义修正。

## 9. 第二轮判别:下发线程拓扑(`--dispatch threaded`,@90a4b4791)

**候选机制(H-dispatch)**:torch_npu 的 host task queue 拓扑中,memcpy 与
算子由 **consumer 线程**下发(`AsyncCopyTask::LaunchCopyTask` 入队 ASYNC_MEMCPY;
算子 `OpCommand::RunOpApiV2` 也入队 EXECUTE_OPAPI_V2——EXEC_NPU_CMD V1 宏
尾部同样走 RunOpApiV2,op_api_common.h:358),栅栏由**用户线程**执行
(`NPUStream::synchronize` 先 MakeSureQueueEmpty 再 aclrtSynchronizeStream)。
若 CANN runtime 的流序/流同步对"**由不同线程提交的任务**"记账不同
(per-thread stream submission list 形态),则:consumer 提交的 memcpy 与
(consumer 或用户线程提交的)消费任务/栅栏之间无顺序保证 → 全部观测
(A1 乱序/A3 栅栏空转/时间线"有序但颠倒")可解,且与纯 CANN 直发全绿
(§8)不矛盾。

**实验**:`--dispatch threaded` 忠实镜像该拓扑——worker 线程按严格 FIFO
执行 H2D 们 + D2H 消费(= host queue),主线程 queue.join(= 排空)后执行
栅栏与最终同步;fence 前同样先 join(= torch_npu fence 的 MakeSureQueueEmpty)。

预期矩阵:

| 结果 | 定性 |
|---|---|
| threaded miss>0 而 direct miss=0 | **跨线程提交是触发条件**:runtime 的流序对提交线程敏感 → 缺陷形态="CANN runtime 对跨线程提交的流序/流同步语义"+ torch_npu 的队列设计踩中它(责任两侧:文档未言明 + 设计假设);issue 面向两者 |
| threaded 也 miss=0 | CANN 拓扑无关 → 嫌疑收窄到 torch_npu 队列内部(消息顺序/复制时序/paramStream 传递),下一步 repro + TASK_QUEUE_ENABLE=0/2、PER_STREAM_QUEUE=1 对照 |

辅助实验(与 threaded 同批跑,均为一条 env):

```bash
TASK_QUEUE_ENABLE=0 python3 research/repro_h2d_order.py --mode racy --bg 8   # 关队列:若 miss=0,队列是肇因
TASK_QUEUE_ENABLE=2 python3 research/repro_h2d_order.py --mode racy --bg 8   # V2/poll 路径
PER_STREAM_QUEUE=1 python3 research/repro_h2d_order.py --mode racy --bg 8    # per-stream 队列
```

TASK_QUEUE_ENABLE=0 治愈 ⇒ 队列机制肇因实锤(与 ASCEND_LAUNCH_BLOCKING
治愈同机制不同步,分辨率更高);三种 env 的组合结果可与 threaded 结果
交叉验证 H-dispatch。

## 10. 第三/四轮:通道矩阵 + query 屏障(2026-08-27 深夜,[M])

六格矩阵(bg=8,被测 64B pinned H2D,消费=同流算子):

| # | 被测拷贝 | 算子通道(TQ) | query epilogue | miss |
|---|---|---|---|---|
| 1 | torch copy_(队列) | 队列(=1) | torch 内置 | 99.90% |
| 2 | torch copy_(直发) | 直发(=0) | torch 内置 | **0%** |
| 3 | acl-direct(主线程) | 队列(=1) | 无 | 99.90% |
| 4 | acl-direct(主线程) | 直发(=0) | 无 | 46.95% |
| 5a | acl-direct(主线程) | 直发(=0) | 有(ctypes) | 52.65% |
| 5b | acl-direct(主线程) | 队列(=1) | 有(ctypes) | **4.65%** |

判读:
- **query 非屏障**(4 vs 5a 无差);"TQ=0 治愈"= 假象(4 仍 47%);
- 5b 的 4.65% 候选机制:query 阻塞主线程 → 算子消息入队延迟 → consumer(被
  bg 消息拖住)发算子时数据已落地 —— 时序窗口效应,非语义保证;
- **格 2 vs 5a 无解矛盾**:同为 TQ=0、bg 同 torch、epilogue 同 query,仅被测
  拷贝 torch vs ctypes,0% vs 52.65%。剩余唯一系统性解释:**torch copy_
  运行时未走 opapi 路径**(DO_COMPATIBILITY 回落老 CopyKernel.cpp,其
  H2D non_blocking 行为不同,可能含额外同步)—— W2(dispatch 路径未运行时
  实证)复活并升级为头号;
- 纯 CANN direct(D2H 读者)全绿 与 格 4(AI core 读者)47% 并存 →
  "SDMA 写→AI core 读可见性窗口"仍是缺陷核心形态的候选,但格 2 的自愈
  机制必须在定稿前钉死(否则任何归因都可能被该隐藏变量污染)。

## 11. 第五轮(待执行):路径实证两条

1. libopapi 符号检查(判 DO_COMPATIBILITY 回落):
   `nm -D <libopapi.so> | grep -c aclnnInplaceCopy`(0=回落老路径实锤)
2. dispatch 日志(ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=1,
   --steps 3):直接看 copy_ 的 op 名与队列消息序。

## 12. 第五轮结果 + 第六轮设计(2026-08-27 深夜续)

- `nm -D libopapi.so | grep -c aclnnInplaceCopy` = **2** → opapi 路径生效,
  "老路径回落"(W2)出局;dispatch 日志被 init 噪声淹没(需精准 grep)。
- 日志金矿线索:`H2DCopyMgr: alloc h2d copy buff success, policy 2/1`
  (runtime h2d_copy_mgr.cc)——CANN runtime 的 H2D 搬运有**多物理通道**:
  PCIE_BAR(CPU 直写,`size <= PCIE_BAR_COPY_SIZE` 时)/ ASYNC_PCIE_DMA(SDMA)
  / SYNC / UB(h2d_copy_mgr.hpp:27-32, Init :99-104)。该 Mgr 本身是算子
  参数搬运池(ut 名 argAllocator),但揭示了通道分流的实现模式。
- **格 2 vs 4/5a 的新候选(H-channel)**:torch copy_ 的 64B 与 ctypes 的
  64B 走了**不同物理通道**(torch→BAR/CPU直写=立即可见=0%;ctypes→SDMA
  task=完成≠可见=47%)。裁决手段 = **profile 数 memcpy task**:fix 模式已
  证同步路径无 async task(400=8×50);若格 2(TQ=0+torch)的 memcpy 计数
  也是 400(缺被测 50 个)→ 通道分歧实锤;若 450 → 通道相同,另寻差异。
- 第六轮命令:
  ```
  TASK_QUEUE_ENABLE=0 python3 research/repro_h2d_order.py --mode racy \
      --steps 50 --bg 8 --profile /tmp/reprof_g2
  python3 research/stream_audit.py /tmp/reprof_g2 | grep -E "MEMCPY|ARRIVAL|tasks="
  TASK_QUEUE_ENABLE=0 python3 research/repro_h2d_order.py --mode racy \
      --steps 50 --bg 8 --copy-mode acl-direct --profile /tmp/reprof_g4
  python3 research/stream_audit.py /tmp/reprof_g4 | grep -E "MEMCPY|ARRIVAL|tasks="
  ```

## 13. 第六轮 + 终判(2026-08-27 夜,[M])

### 第六轮数据

- 格 2(TQ=0+torch,profile):miss=0,**memcpy=450(被测 64B 有 task)**;
- 格 4(TQ=0+acl-direct,profile):**miss=0(!)**,memcpy=450;
- 两者 ARRIVAL 均 9M7K、零逆序、零 mem-kernel 重叠。

### 判读

1. **H-channel 出局**:两条路 64B 都产生 SDMA task(450=9×50),无通道分流;
2. **观察者效应实锤**:无 profile 的格 4 miss=47%,加 profiler(119ms/步税)
   → 0%——算子提交被拖慢远超 SDMA 消化时间,窗口关闭;
3. 格 2 与格 4 无 profile 时 wall 几乎相同(415 vs 438us/步),miss 0% vs 47%
   ——差别只在 memcpy 提交后的 epilogue 微差(query+记账 vs 直接返回)。

### 终判:统一竞态窗口模型(六格全收敛)

**缺陷 = SDMA memcpy 完成后,数据对同流后续 AI core 任务的可见性不被流序
保证;窗口开度由提交时序微差决定(连续谱,非语义屏障)。**

| 配置 | miss | 窗口 |
|---|---|---|
| TQ=1(算子 consumer 异步提交) | 99.90% | 全开 |
| TQ=1+ctypes+query | 4.65% | 微开(query 阻塞+consumer 拖延) |
| TQ=0+ctypes(最快提交) | 46.95% | 半开 |
| TQ=0+ctypes+query | 52.65% | 半开(query 非屏障) |
| TQ=0+torch(epilogue 略长) | 0% | 碰巧关 |
| 任意路径+profiler 税 | 0% | 强关 |

支撑证据:
- **reprof_racy(TQ=1,同 run miss 94%)时间线:零逆序+零重叠+串行** = 执行序
  正确(拷贝先完成、算子后启动)但读到旧值 → **可见性缺口的直接时间线证据**;
- 0%↔47% 随 profiler 开关/epilogue 微差摆动 → 不是语义保证;
- 纯 CANN D2H(SDMA 读)全绿 → 缺口特定于 **AI core 读路径**(SDMA 写→
  AI core 读的一致性窗口);
- TQ/env 的全部效应均可归因于提交时序(窗口开度),无一处需要"队列重排/
  通道分流/屏障"假设。

### 归因定稿

- **层次**:CANN runtime/硬件——SDMA 写→AI core 读的跨引擎可见性未纳入
  流序(aclrtMemcpyAsync 完成语义对后续 kernel 不蕴含数据可见);
  torch_npu 各路径均为受害者(不同路径改变窗口开度,不提供保证);
- **torch_npu 侧次要问题**:TQ=1 拓扑(算子由 consumer 异步提交)放大窗口
  至必然级(99.9%),是暴露面而非根因;
- **#14922 fix(blocking)正确且必要**:同步 aclrtMemcpy 不经 SDMA 异步通道;
- ~~仍开放(可选):格 2 无 profile 加压验证~~ **✅ 已实证(2026-08-27 夜)**:
  `TQ=0 + torch copy_ + bg=16×4MiB(64MiB/步积压), steps=5000` →
  **miss=4997/5000 = 99.94%**——"TQ=0+torch 的 0% 是巧合"定谳,统一模型
  完全闭环:**没有任何 torch_npu 提交路径是安全的**,窗口开度唯一由
  SDMA 积压量 vs 算子提交延迟的赛跑决定(64MiB 积压 >> epilogue 微差)。

### issue 方向(重写)

标题形态:probabilistic stale read by same-stream kernels after
aclrtMemcpyAsync H2D (SDMA completion not visibility-ordered for AI-core
consumers; window modulated by submission timing)。证据 = 六格矩阵 +
profiler 开关效应 + reprof_racy 时间线(执行序正确+可见性缺失)。

## 14. 第七轮(2026-08-28):纯 CANN AI core 消费者复现探针

**动机**:终判指向 CANN runtime 层,但当时的纯 CANN 探针
(`cann_memcpy_order.py`)消费者是 D2H 读回 = SDMA 引擎读 = 终判中
"本就不受影响"的路径——全绿与终判自洽,却留下一块短板:**没有自包含
的纯 CANN 红 repro**(红的最小复现依赖 torch_npu 提供消费 kernel)。
据此落 issue 到 cann/runtime 时会被"请先隔离 torch"一轮打回。另:重复
检索(2026-08-28,`notes/upstream/issue-duplicate-search-20260828.md`)
确认 gitcode cann/runtime#873(D2H/event 方向)是最近近亲、无重复 issue,
而该仓受理的正是 runtime 语义类问题。

**探针**:`research/cann_aicore_visibility.py`(+ mock
`cann_aicore_visibility_mock.c`)——零 torch/零 torch_npu:
libascendcl(ctypes)+ libopapi 两段式 `aclnnAdd`(真 AI core/vector
kernel,API 形态逐项镜像 op-plugin 生产用法:9 参 `aclCreateTensor`、
create → GetWorkspaceSize → run → destroy 顺序、ACL_DT_INT32=3/
ACL_FORMAT_ND=2 经 cann-runtime 头文件核实)。证据协议同
`repro_h2d_order.py`:消费者把 `out = src + 1` 落进 device 侧历史缓冲,
结尾一次 sync D2H 判读(**零逐步 host 同步**,规避观察者效应);esc 计
初始 -1 哨兵读(原 bug 逃逸值族);lag 直方图。

时间线(每步):host 写 slot → K×bg 1MiB async H2D(SDMA 积压)→ 被测
64B async H2D → [fence: none|stream_sync|event_sync|event_wait] →
同流 `aclnnAdd` 消费 → 末尾统一 synchronize + D2H。

**判读矩阵**(脚本自打 verdict):
- racy miss>0 → **纯 CANN AI core 复现**(defect 完全低于 torch 层);
- +stream_sync 仍 miss>0 → STRONGEST(sync 返回但后续 kernel 读旧值);
- +event_* 仍 miss>0 → FENCE-BLIND;
- mode=sync / bg=0 = 负对照,应全绿;
- 全绿 → 加压配方 `--bg 16 --bg-elems 1048576 --steps 5000`
  (torch 侧 99.94% 的同款)+ `--dispatch threaded`。

**本地验证**:`--selftest` 用 gcc 编译 mock(单 .so 同时导出两库子集;
pending-落地内存模型,ACLMOCK_LANDING/SYNC_BLIND/EVENT_BLIND 三旋钮)
跑 6 分支全过:clean×2 / lag1(racy miss=100%、esc=1、lag=1 数值精确)/
fence 救 / event 盲 / sync 盲——判读逻辑与计数管线端到端核实;
ruff check+format 全绿。

**意义(判读后)**:
- 红 → issue 证据链升级为自包含纯 CANN repro,落 cann/runtime 无软肋;
- 绿(加压后仍绿)→ 终判需修正:AI core 消费经 aclnn 直发拓扑安全,
  窗口需要 torch_npu 提交拓扑参与 → issue 落回 Ascend/pytorch
  (torch_npu),按暴露面定性。**红绿两路都是决定性信息。**

**服务器命令**(容器内,无 torch 依赖,~1-2 分钟/条):
```bash
python3 research/cann_aicore_visibility.py                     # 主格:racy bg=8
python3 research/cann_aicore_visibility.py --mode sync         # 负对照
python3 research/cann_aicore_visibility.py --fence stream_sync # 栅栏格
python3 research/cann_aicore_visibility.py --dispatch threaded # 跨线程格
# 若全绿,加压配方:
python3 research/cann_aicore_visibility.py --bg 16 --bg-elems 1048576 --steps 5000
```

### 外部审查回应(2026-08-28,红绿判读前)

另一 AI 对脚本提 4 条"数据失真"质疑,逐条核查(不照单全收):

1. **"join() 不保证硬件入队→假阴性"——方向误**。join() 保证的是
   aclrtMemcpyAsync 调用序先于 kernel launch(**必要前提;缺失才是假
   正向**——kernel 先于拷贝提交)。若拷贝硬件入队被推迟:同流执行序仍
   保证拷贝先于 kernel,而缺陷恰是"拷贝已完成仍读旧值"(reprof_racy
   时间线),推迟制造不出干净通过,只会移动提交时序谱上的点(窗口模型
   §4 本义)。防御:文档钉死 **direct=裁决主格**(无队列中介),threaded
   =探索格(继承报告格 3 的混合线程混淆警示)。
2. **"as_i32 越界写→假阳性"——不成立**。两处调用点的 ctypes 视图长度
   ==分配长度(payload×4 == payload_b)且只写 [0];无越界路径。防御:
   assert payload≥1 + 注释钉死构造等式。
3. **"worker 初始化失败静默丢弃→esc=100% 误判"——半误**。copy_errors
   在 verdict 链第一优先(原版即 ABORTED,对方漏看);但数字行确实会打
   误导性 100%。防御:threaded 每步 join 后检测即中断 + 分析只覆盖
   steps_done(未测步不再入统计)。
4. **"selftest 环境传递缺漏"——不成立**。find_lib/find_opapi_lib 均
   env 优先(insert(0));且误跑真硬件会让 lag1 分支响亮 FAIL 而非静默
   通过。防御:case 1 加 mock 路径正向断言。

修复后 selftest 6/6 复验通过,ruff 全绿。

### 第一轮服务器结果(2026-08-28,910B3 容器):负对照红 → 判读全面修正

| 配置 | miss | lag 形态 | 初版 verdict | 修正后定性 |
|---|---|---|---|---|
| **sync(负对照)** | **99.90%** | **future(-227)** | UNEXPECTED | **PROBE-INVALID:负对照必须绿** |
| racy direct bg=8 | 99.85% | future(-204) | (声称复现) | **无效:dispatch lag 伪影** |
| racy threaded | 99.75% | future(-204) | (声称复现) | 无效(同上) |
| fence=event_wait | 99.80% | future(-170) | FENCE-BLIND | 无效(同上) |
| fence=stream_sync bg=8 | 0% | — | — | 绿(及时+有序) |
| fence=stream_sync 64MiB×5000 | 0% | — | — | 绿 |
| fence=event_sync 64MiB×5000 | 0% | — | — | 绿 |

**修正逻辑**:①阻塞拷贝按定义有序,sync 格红 = 该形态下管线被扭曲,
全部"复现"作废(外部审查"负对照必须绿"原则在此应验);②lag 全负且
巨大 = kernel 读到**未来 ~200 步的值**(step2 读到 src=4)——可见性缺口
只能产生**旧的**值(stale,lag>0),future 不可能由"拷贝不可见"产生。

**新模型(全 8 格自洽)**:纯 CANN 两段式 aclnn launch 的 AI core 任务
在忙碌提交线程下**延迟派发 ~150-200ms**( ramp 后稳定 ~204 步定深管线;
sync 格 -227/racy -204/event_wait -170/threaded -204 同量级);SDMA
拷贝即时派发;**host 阻塞类调用(SynchronizeStream/Event)强制清空派发
管线**(故 fence 格全绿:kernel 及时且有序);event_wait(设备侧)与阻塞
memcpy 都不清空(故红)。这本身是重大机制发现——**同一条流上
"拷贝→kernel"的执行序在忙碌直发拓扑下不成立**,方向与 torch_npu 侧
(kernel 及时、拷贝落地晚→stale)相反,是同一"跨引擎序列化缺失"硬币
的两面;且可能与 ASCEND_LAUNCH_BLOCKING=1 治愈原 eagle 竞态的机制直接
相关(launch 同步化=派发及时化)。

**探针已修**(同日):stale/future/esc/garbage 四分类 + verdict 分支
(future-only → KERNEL-DISPATCH LAG,明确"不可裁决可见性";sync 红 →
PROBE-INVALID);mock 增 ACLMOCK_EXEC_DELAY(排队派发模型),selftest
7 分支全过。

**决定性待跑格子**(初版命令漏了最关键的一格——压力+无栅栏):
```bash
python3 research/cann_aicore_visibility.py --mode racy --bg 16 --bg-elems 1048576 --steps 5000
# SDMA 延迟(64MiB/步,数百 ms)必须压过 kernel 派发延迟(~180ms):
#   stale>0  → 纯 CANN 排序/可见性缺陷实锤(kernel 读到流序在前的拷贝之前的旧值)
#   miss=0   → 依赖已插入且可见 → 纯 CANN 直发安全 → 缺陷需 torch_npu 拓扑 → 落回 Ascend/pytorch
#   future 仍主导 → 再加大 bg-elems
python3 research/cann_aicore_visibility.py --mode racy --bg 16 --bg-elems 1048576 --steps 5000 --dispatch threaded
# 机制探针(可选,每条 ~1 分钟):
python3 research/cann_aicore_visibility.py --mode racy --bg 0 --steps 2000   # 无拷贝时 kernel 是否仍滞后
python3 research/cann_aicore_visibility.py --mode racy --steps 200          # lag 爬坡形状(定深 vs 定时延)
ASCEND_LAUNCH_BLOCKING=1 python3 research/cann_aicore_visibility.py --mode racy --steps 200  # 同步 launch=派发及时化?
```

### 第二轮服务器结果(2026-08-28 续):"派发延迟"模型也被推翻 → 2048 任务 FIFO 提交环 + 探针槽复用 bug

| 配置 | miss | 主导 lag | 每步 ring 任务数 | 2048/任务数 预测 |
|---|---|---|---|---|
| racy bg=8×1MiB(一轮) | 99.85% | **-204** | 8bg+1copy+1add=10 | **204.8** ✓ |
| sync bg=8(一轮) | 99.90% | **-227** | 8bg+1add=9(阻塞拷贝不入环) | **227.6** ✓ |
| event_wait bg=8(一轮) | 99.80% | **-170** | 10+record+wait=12 | **170.7** ✓ |
| **racy bg=16×4MiB 5000 步** | 99.94% | **-114** | 16+1+1=18 | **113.8** ✓ |
| threaded 同上 | 99.94% | **-114** | 18 | **113.8** ✓ |
| **racy --bg 0** | **0%** | — | — | 环浅,无覆盖 ✓ |
| ASCEND_LAUNCH_BLOCKING=1 steps=200 | 98.5% | 平坦 ~-130s | — | 环行为与 launch 模式无关 ✓ |

**五形态 ±1 精确命中 → 终模型**:CANN 提交管线 = **~2048 任务的 FIFO 环形队列**,host 领先跑满环后被背压限速(64MiB/步时 host 速率 = SDMA 线速 ~20GB/s)。没有"kernel 特殊延迟派发"——一切任务 FIFO 有序。

**future 读的真因 = 探针 v1/v2 自身的槽复用 bug**:双槽交替(改写周期 2),host 领先 114-227 步,被测拷贝**晚执行时读到的 host 槽已被未来步覆盖**,把未来值搬上设备。铁证 = **奇偶指纹**:racy 系全部主导 lag 为偶数(204/114/170,周期 2 的签名),sync 模式(阻塞拷贝不经槽)是奇数 227。前两轮全部"复现"判读作废(含我 08-28 写入本文档的"派发延迟"模型——连同 §14 首轮小节一并按本节为准);**纯 CANN 直发目前零缺陷证据**。

**v3 修复**(`cecbdb49c` 后续 commit):每步独占槽(steps×64B pinned,写一次永不改写)——拷贝无论何时执行,搬运的就是自己那步的值。判读分支改名 FUTURE-READ ANOMALY(v3 后出现 = kernel 侧真滞后,仍非可见性判读)。mock find_alloc 改范围匹配(块内偏移指针),selftest 7/7。

**连带警示(入 issue 前必须审计)**:`repro_h2d_order.py`(torch 侧)的 miss 计数同样**从未区分 stale/future**,且用同款双槽(host 领先时同样可能被覆盖)——其 racy miss 数字引用前需加 stale/future 分解复核;**仍站得住的 torch 侧证据**:engine 三计数器 esc=[-1,-1](真·旧值,-1 只存在于设备初始/门关步,不可能由槽覆盖产生)+ msprof 时间线(拷贝 task_stop < kernel start 仍读旧值)+ clamp 消毒实验。

**~2048 环本身的工程意义**(独立于本案,值得单独记档):host 可领先设备 2000 任务意味着任何"host 写 pinned 缓冲 → async 拷贝"模式都有 ~2000 任务深的覆盖窗口——正是 CUDA CachingHostAllocator record_event 保护、而 torch_npu 侧(见 §源码链 CachingHostAllocator 分析)假设流序成立才安全的那类风险的定量版。

**v3 重跑清单**(全矩阵,共 ~1 分钟):
```bash
python3 research/cann_aicore_visibility.py                                # racy bg=8:预期 miss=0(环有序+独占槽)
python3 research/cann_aicore_visibility.py --mode sync                    # 负对照:预期绿
python3 research/cann_aicore_visibility.py --bg 0                         # 平静对照:预期绿
python3 research/cann_aicore_visibility.py --fence event_wait             # 设备侧栅栏:stale>0 才是缺陷
python3 research/cann_aicore_visibility.py --bg 16 --bg-elems 1048576 --steps 5000  # 满环压力:stale>0 才是缺陷
python3 research/cann_aicore_visibility.py --bg 16 --bg-elems 1048576 --steps 5000 --dispatch threaded
```
