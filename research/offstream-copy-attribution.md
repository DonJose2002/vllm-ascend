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

### 第三轮服务器结果(2026-08-28 终):纯 CANN 直发全绿 → 归因层改判

| 配置 | 结果 |
|---|---|
| racy bg=8 / bg=0 / event_wait | miss=0 全绿 |
| **racy 满环压力 64MiB×5000(direct + threaded)** | **miss=0, stale=0 —— 决定性格绿** |
| sync 负对照 | future=-227 红 = **控制组设计伪影**(见下) |

**结论**:独占槽下,**纯 CANN(libascendcl async 拷贝 + libopapi 两段式算子,同流,单/跨线程)排序与可见性全部正确,零缺陷证据**。08-27 终判的归因层"CANN runtime/硬件 SDMA 写→AI core 读可见性缺口"**被推翻**——该缺口在纯 CANN 直发拓扑下不存在;缺陷必须依赖 torch_npu 层(其任务队列 consumer 拓扑/拷贝路径/pinned 管理)才暴露。engine 级证据(-1 哨兵逃逸+崩溃、#14922 fix 实效)不受影响,但机制归属回到 torch_npu 侧待裁。

**sync 伪影机制**:阻塞拷贝绕过 ring 立即执行,host 领先 ~227 步(=2048/9,又一指纹命中)时,后续阻塞拷贝把**共享 dev_src** 覆盖成未来值,ring 里排队的 Add_k 晚读到——设备侧缓冲复用隐患(与 host 槽覆盖同类,CUDA 下同样属于用户责任)。

**v4(同日)**:被测拷贝与消费者改**每步独占设备切片**(初始化 SENTINEL,单拷贝写、单 kernel 读)——跨步值流彻底归零:miss 只能以 esc(拷贝未落地/不可见)呈现,future 结构性不可能;负对照恢复必须绿。mock lag1 用例改 esc=50 断言;EXEC_DELAY 用例改"晚 kernel 必须仍然全绿"断言;selftest 7/7。

**repro_h2d_order.py 审计补丁(v2,同日)**:①独占 host 槽(steps×payload,写一次);②device 侧 future 计数器(`fut += gpu[0] > counter`)+ 输出分解 stale/future/esc——**torch 侧 99.9%/47%/99.94% 的"窗口谱"数字在此次审计前全部冻结不可引用**(其 miss 从未分方向,双槽同款覆盖隐患;msprof 时间线"拷贝完成<kernel 启动+读错值"同样不区分 stale/future)。设备缓冲保持共享(镜像 engine 的共享 backup 缓冲,这是 bug 的原始形态)。

**torch 层裁决格**(审计后重跑):
```bash
python3 research/repro_h2d_order.py --mode racy                      # TQ=1 默认:stale 还是 future 主导?
python3 research/repro_h2d_order.py --mode racy --bg 0               # 平静对照
python3 research/repro_h2d_order.py --mode racy --bg 16 --bg-elems 1048576 --steps 5000   # 满环压力
TASK_QUEUE_ENABLE=0 python3 research/repro_h2d_order.py --mode racy --copy-mode acl-direct --bg 8   # 格4复刻
```
判读:**stale/esc 主导 = torch_npu 层缺陷实锤**(issue 落 Ascend/pytorch,机制=其提交拓扑);**future 主导 = repro 数字此前全是 host-run-ahead 伪影**,缺陷只剩 engine 级 -1 逃逸单点,issue 重写为窄口径。

### 第四轮(2026-08-28 续):torch 审计首跑全绿 → 三读法与 alt2 阳性对照

审计四格(TQ=1 默认 / bg=0 / 满环 64MiB×5000 / TQ=0+acl-direct)**全部 miss=0 stale=0 future=0 esc=0**(wall 320/302/2499/513us/步)。全绿有三种读法,未裁决前不得采信任何一种:

- **A(槽覆盖伪影,模型预言)**:旧 miss 质量本就是"host 领先 ~204 步 + 双槽被覆盖 → future 计 miss",独占槽一改归零——纯 CANN 两轮的同款剧本。若 A 成立:**torch repro 层面竞态不存在**,缺陷收窄到 engine 级 -1 逃逸单点(旧 99.9%/47%/99.94% 全部作废)。
- **B(补丁致哑,必须排除)**:`cpu_slots[step-1]` 是大 pinned 张量的**行视图**——若 torch_npu 视其为非 pinned,`copy_(non_blocking=True)` 静默退化阻塞拷贝,竞态被结构性消灭,全绿无意义。
- **C(环境漂移,必须排除)**:当日日志出现 `LD_PRELOAD detected` 警告(此前未见记录);旧红可能根本不可复现。

**判别设计(已实现)**:`--slot-mode alt2` = 原样复刻旧双槽交替(阳性对照,同二进制配对跑);启动时打印 `pinned_whole/pinned_row`(B 的直接观测)。判读矩阵:

| alt2 bg=8 结果 | 定性 |
|---|---|
| ~99.9% 且 future 主导,unique 同参 0% | **A 实锤**:伪影机制成立 + async 路径完好 + 环境稳定,三读法一次全闭 |
| 0% | B 或 C——先查 pinned_row 输出;仍绿则 git 回退旧版脚本复跑二分 |
| unique 出现 esc/stale(任何格) | 真缺陷信号(意外,升格处理) |

```bash
python3 research/repro_h2d_order.py --mode racy --slot-mode alt2            # 阳性对照主格
python3 research/repro_h2d_order.py --mode racy --slot-mode alt2 --bg 0     # 对照的平静格
python3 research/repro_h2d_order.py --mode racy --slot-mode alt2 --bg 16 --bg-elems 1048576 --steps 5000  # 99.94% 锚点复刻
```

### 第五轮(2026-08-28 终审):alt2 阳性对照闭案 → 证据体系重写

| 结果 | 判读 |
|---|---|
| alt2 bg=8:**99.85%,future=1997** | 与旧 99.9% 精确复现,全 future |
| alt2 满环:**99.94%,future=4997** | 与旧 99.94% 逐位复现,全 future |
| alt2 bg=0:0.35% future | 旧 0.70% 同量级(启动瞬态:host 领先的头几步) |
| `pinned_whole=True pinned_row=True` | 读法 B(行视图失 pinned→copy_ 静默阻塞)死亡 |
| 同二进制/同环境复现旧红 | 读法 C(环境漂移)死亡 |

**读法 A 定谳:repro 全部 miss 质量 = 槽覆盖伪影**(host 领先 + 双槽改写 → 晚拷贝送 future 值)。同二进制配对(unique 0% vs alt2 99.9%)= 单变量闭案。

#### 证据清单重审(2026-08-28 终)

**死亡(勿再引用)**:
- repro 的六格"窗口谱"(0.7%/1.5%/99.9%/99.94%/47%……)——全部 future 伪影;
- **msprof 时间线证据(§3.3"拷贝 task_stop < kernel start 仍读旧值")**——该 run 的 miss 本身是伪影:拷贝确实先完成(送的是被覆盖槽的未来值),时间线自洽于伪影机制,与可见性缺口无关;
- **栅栏阶梯"sync 不等拷贝"(wall 740us<996us 推断)**——wall 比较是弱推断(64B 拷贝等待本可微秒级);且 event 格旧 miss 同为伪影候选,待 unique 槽复跑裁决;
- 08-27 报告 §4 终判归因层("CANN runtime SDMA 写→AI core 读可见性缺口")——纯 CANN v3/v4 全绿已推翻;
- "torch_npu 拓扑放大窗口"叙事——repro 层两种 manifestation 均为伪影。

**存活(engine 级经验事实)**:
- eagle3 16K 无 fix 确定性崩溃 / fix 后 9/9 全绿(多轮);
- 三计数器 Run 4:esc=[-1,-1],c3=1/843(唯一逃逸值 = -1,device 侧计数,零 host 同步);
- 算术线索:c1=837 ≈ 门关步 836 + **1**——**逃逸步的拷贝位置(c1 检查点)就已经是 -1**,即 -1 在"拷贝之后第一个 kernel"处已然可见,不是 where 与 c1 之间才出现。

#### 引擎缺陷机制:剩余假设空间(重开)

- **(b) host 侧 pinned 单缓冲改写 × 深环**:engine 的 backup.cpu 是**单缓冲每步改写**(比 repro 的双槽更激进);host 领先时,在途拷贝晚执行会读到**后续门关步写入的 -1**→拷贝亲自把 -1 送上设备→where 消费→逃逸。与 c1=836+1 算术吻合(c1 位置已是 -1 = 拷贝送来的或未落地)。CUDA 不崩的解释:同构 hazard 但 host 实际跑前量被逐步 D2H 依赖(采样结果)压住+队列语义差异 → 窗口不落于 -1 内容上。**若 (b) 成立:这是 upstream CpuGpuBuffer 使用模式在深提交队列下的固有风险,#14922 的 blocking fix 恰好对症(同步拷贝 = 内容被即时捕获,host 改写不再影响在途拷贝)——fix 跨机制稳健。**
- **(a) torch_npu TQ=1 提交序破坏**(拷贝与算子在真实引擎 op 混合/图重放下环序被破)——与纯 CANN FIFO 有序矛盾,需引擎级证据才可主张;
- **(c) 引擎特有**(ACL graph 重放与逐 token 拷贝互作)。

**判别实验(engine 级,下一步)**:env 门控给 backup.cpu 做逐步双缓冲(改写隔步)——崩溃消失 = (b) 定谳;仍崩 = (a)/(c)。另:fence ladder 用 `--slot-mode unique` 复跑 event 格 2 条(裁决"栅栏盲"叙事死活)。

#### 连带行动

- **#14922 机制叙事需更正评论**:已交维护者的回复(中/英)构建于被推翻的"流外拷贝/栅栏盲/可见性缺口"故事上;**fix 本身跨机制稳健,不受影响**,但机制段必须更正——诚实义务,尽快补评。
- **外部 issue 计划冻结**:目前没有任何层被证明存在可外报缺陷;重复检索/两份 issue 草稿归档备用。~2048 环行为学(深度、背压、run-ahead 定量)可作独立的文档型贡献候选(低优先级)。

### 第六轮(2026-08-28 收尾):栅栏叙事终审闭环 + 更正评论定稿

验证三命令结果(M]:①`event + unique` = **miss=0/2000**(脚本自打 verdict:
"stream sync covered the copy")——**"栅栏盲/流外拷贝"叙事终审死亡**,
旧的 wall 时间推断(740us<996us ⇒ "没等")作废;②`event + alt2` =
miss 0.4%,**全 future 方向**(启动瞬态 host 跑前,与同步行为无关);
③`fix + alt2` = miss=0(fix 对 host 槽设计不敏感,跨机制稳健再证)。

**顺带修复(探针卫生)**:该轮 event+alt2 出现 miss=8 < future=11 的
计数矛盾——三个谓词各自重读 `gpu[0]`,拷贝在两次读之间落地即不一致;
已改 `sample = gpu[0].clone()` 单次快照后再谓词;event 模式 verdict 分支
同步改为按 stale/future 分向(旧分支 miss>0 即打 "OFF-STREAM COPY
CONFIRMED",已失效)。CPU 自测 2 模式×2 槽 4/4 绿。

**#14922 更正评论终稿**:`notes/upstream/pr-14922-mechanism-correction.md`
(中英双版,撤回栅栏叙事/归因/六格证据,保留 engine 事实 + fix 跨机制稳健
论证,含第 6 条栅栏复核实测),交用户粘贴。

### 第七轮(2026-08-28):staged-copy 引擎判别实验(机制 (b) vs (a)/(c))

**设计**:拷贝点(env `VLLM_ASCEND_SD_STAGED_COPY=<N≥2>`,仅在
`VLLM_ASCEND_SD_REVIVE_RACE=1` 下生效):每步先把单页
`backup.cpu[:num_reqs]` 快照进**私有 pinned 环形分页**(host memcpy,<1KB),
异步拷贝只读自己的页——host 要过 N 步才会再碰该页;**其余竞态时序形状原样
保留**(仍是 non_blocking H2D + 同流 where)。`_sd_stage_next_page` 惰性分配
(inference_mode(False) 镜像 CpuGpuBuffer),engagement 行镜像 counters 风格;
本地 UT `research/test_staged_copy.py`(ast 抽真实方法 + pin-free torch stub,
验证轮转/单次分配/状态持久)通过;ruff/mypy 零新增。

**判读矩阵**(L2082 拷贝点,`run_baseline_npu.sh eagle3` 16K 复现配置):

| 结果 | 定性 |
|---|---|
| 单页崩 + 分页绿(计数格 c3=0) | **(b) 定谳**:毒经 host 源页改写进入,流序全程无辜 |
| 分页也崩(先试 N=64) | (b) 出局:来源正确的拷贝仍送出 -1 → (a)/(c) 引擎层设备侧问题 |
| 单页对照不崩 | 复现条件漂移,停止判读 |

```bash
# A 阳性对照:复活竞态,单页(预期崩;不崩=条件漂移停)
VLLM_ASCEND_SD_REVIVE_RACE=1 TIERS=16384 CONCS=1 NPUS=<id> \
  bash research/run_baseline_npu.sh eagle3 8021
# B 判别:复活竞态 + 8 私有页
VLLM_ASCEND_SD_REVIVE_RACE=1 VLLM_ASCEND_SD_STAGED_COPY=8 TIERS=16384 CONCS=1 NPUS=<id> \
  bash research/run_baseline_npu.sh eagle3 8022
# C 判别+计数:绿则读数直接可读(预期 c2~7、c3=0;SUMMARY 自动带 counters 行)
VLLM_ASCEND_SD_REVIVE_RACE=1 VLLM_ASCEND_SD_STAGED_COPY=8 VLLM_ASCEND_SD_COUNTERS=1 \
  TIERS=16384 CONCS=1 NPUS=<id> bash research/run_baseline_npu.sh eagle3 8023
# B/C 崩则加深复跑:VLLM_ASCEND_SD_STAGED_COPY=64(端口递增)
```

### 第八轮(2026-08-28 终):staged-copy 判别实验定谳 → 全案闭合,机制 (b) 确认

| 格 | 配置 | 结果 |
|---|---|---|
| A | REVIVE=1 单页(阳性对照) | **ok=0/8 崩**——复现成立 |
| B | REVIVE=1 + STAGED=8 | **ok=8/8 绿**(ITL 25.8ms / out/s 83.78 / accept 2.48) |
| C | B + COUNTERS | **绿;`steps=847 c1=840 c2=7 c3=0 c1x=0 esc=none`** |

对照单页时代同款读数(Run 2'/4,2026-08-25):`steps=843 c1=837 c2=7 c3=1
esc=[-1,-1]`。**唯一变量 = 拷贝源是私有快照页还是被逐步重写的单页;c2=7
不变(门开次数与页设计无关),c3 1→0、esc [−1,−1]→none。**

**判读**:
- **(b) 定谳**:-1 经"晚执行的异步拷贝读到 host 已重写的单页源"进入——
  16K 边界步 SDMA 积压使 64B 小拷贝晚执行,eagle µs 级步循环让 host 抢先
  把单页改写回门关步的 -1,拷贝如实送达 -1,边界步 where 消费 → gather(-1)
  → aivec 故障。**流序与可见性全程无辜**。
- **(a)/(c) 出局**:若提交序/图重放破坏了拷贝→消费者顺序,私有页救不了
  (拷贝会正确送达自己的页,where 仍会读旧设备值逃逸)——c3=0 直接排除。
- **#14922 blocking fix 的机制解释定稿**:同步拷贝当场定格单页内容,
  host 后续改写不再影响传输;staged 实验证实同族有效(异步+私有页同样
  c3=0)——fix 跨机制稳健 + 机制现已实锤。
- **附带发现(待同日对照确认)**:staged 保持异步,ITL 25.8ms,可能优于
  blocking 路径(历史 9/9 矩阵 16K 档 ~31ms)——若同日对照坐实,**双页
  替代 blocking 是潜在上游优化**(省同步税),可选 run D:默认 fix 同配置
  复跑一条即知。

**案件状态:CLOSED**(08-24 崩溃 → 08-25 修复+三计数器 → 08-26/27 平台层
调查(两次定性被推翻)→ 08-28 六轮审计翻案 + 引擎机制判别定谳)。

### Run D(2026-08-28 设计,待执行):staged vs blocking 同日对照

目的:B 轮 staged ITL 25.8ms vs blocking 历史 ~31ms 是**跨日数据**(已知
跨日漂移 ~2ms 量级,T1 教训);同日同卡背靠背各跑一条,坐实或否掉
"staged 保住异步管线、优于 blocking"这一上游优化素材。

#### 安全锚点(最小修复回退记录,先读后跑)

- **默认路径 = 最小修复,无需任何操作即处于修复态**:`llm_base_proposer.py`
  的 `prepare_next_token_ids_padded` 拷贝点,不带 env 时走
  `non_blocking=False` blocking 拷贝(commit `62533dafa`,2026-08-24;
  9/9 全量矩阵 + 其后全部运行验证)。
- **staged 只在双 env 下激活**:`VLLM_ASCEND_SD_REVIVE_RACE=1` 且
  `VLLM_ASCEND_SD_STAGED_COPY>=2`,二者缺一即回落默认 blocking。默认字节
  不因本实验改变——**回退 = 不带 env 重启 serve,无代码回滚、无分支切换**。
- **Phase 2 及后续研究基线一律不带这两个 env** = 确切可运行版本。
- serve log 自证路径:`[SD-staged-copy] engaged` 行存在 = staged;不存在
  且无 `[SD-counters]` 行 = 默认 blocking。
- commit 锚点:本手册落盘 commit(research/v0.23.0,已推 myfork);staged
  实现 = `55ae3ba57`,判别数据 = `4bac35396` §14 第八轮。

#### 命令(容器内,同卡同日背靠背;D1 先跑——缺的数据点优先)

```bash
# D1: 默认 blocking fix(不带任何研究 env)= 安全锚点本尊
TIERS=16384 CONCS=1 NPUS=<id> bash research/run_baseline_npu.sh eagle3 8024
cp experiments/out/serve-npu-bf16-eagle3-k5.log  experiments/out/serve-rund-fix-blocking.log
cp experiments/out/baseline-npu-qwen3-8b-npu-bf16-eagle3-k5.json experiments/out/rund-fix-blocking.json
grep -cF "[SD-staged-copy] engaged" experiments/out/serve-rund-fix-blocking.log || true   # 预期 0(实际日志行带右括号,须 -F 整串匹配)

# 排水(脚本不杀 serve;共享服务器纪律,防双引擎重叠):
pkill -TERM -f "vllm serve.*--port 8024"; sleep 30
# 确认钉卡 HBM 回落(npu-smi info,空闲底噪 ~3.4GB)再跑 D2

# D2: staged(与 B 轮同配置,作同日第二锚点)
VLLM_ASCEND_SD_REVIVE_RACE=1 VLLM_ASCEND_SD_STAGED_COPY=8 \
  TIERS=16384 CONCS=1 NPUS=<id> bash research/run_baseline_npu.sh eagle3 8025
cp experiments/out/serve-npu-bf16-eagle3-k5.log  experiments/out/serve-rund-staged8.log
cp experiments/out/baseline-npu-qwen3-8b-npu-bf16-eagle3-k5.json experiments/out/rund-staged8.json
grep -F "[SD-staged-copy] engaged" experiments/out/serve-rund-staged8.log   # 预期 "8 private pinned pages" 行
```

注意:两 run 的 TAG 同为 `npu-bf16-eagle3-k5`,serve log/JSON 会互相覆盖,
**copy-aside 是强制步骤**(命令已含);SUMMARY 不记录研究 env,靠
engaged 行 + 归档文件名区分。

#### 判读矩阵

| 结果 | 定性与行动 |
|---|---|
| D2 绿且 ITL(D2) < ITL(D1) − 2ms | staged 优坐实(差值跨过日漂移量级)→ 上游优化素材成立:follow-up PR/评论提案(生产形态=写入时轮转或 CpuGpuBuffer 环,见 §14 第八轮) |
| D2 绿且差 <2ms | blocking 已够,staged 降级为"可选",不推进;B 轮 25.8 记为日间漂移 |
| D2 崩 | staged N=8 生产化不安全(拷贝在途寿命可超 8 步)→ 可选加深 `STAGED_COPY=64` 复跑一次再判;仍崩则 staged 出局,上游素材只保留 blocking fix。**对研究主线零影响** |
| D1 崩 | 重大异常(默认修复路径理论不可崩,9/9 验证过)→ 立即停,贴 serve log tail;排查方向=环境漂移/条件变化,而非代码 |
| 两 run accept 差 >0.1 | 数据可疑(机制上只差拷贝方式,不影响采样),复跑确认 |

数据回传:两份 SUMMARY 块(含 ITL/accept/out-s)+ 两行 grep 结果即可。

#### Run D 结果(2026-08-28 执行,@a357c7cc2):第一格命中,staged 优坐实

| 格 | 配置 | ITL p50 | out/s | accept(C/B) | TTFT p50 | ok |
|---|---|---|---|---|---|---|
| D1 | 默认 blocking(NPU3,grep=0 自证默认路径) | 33.7ms | 66.13 | 2.4824 / 2.4734 | 209.3ms | 8/8 |
| D2 | REVIVE+STAGED=8(NPU2,engaged 行在) | **25.7ms** | **84.29** | 2.4824 / 2.4645 | 195.9ms | 8/8 |

- **判读 = 第一格**:差 8.0ms ≫ 2ms 阈值;out/s **+27.5%**;accept(counters)
  逐位一致 2.4824(机制自洽:只差拷贝方式,不触采样)。
- **卡间差异排除**:首对 D1/D3 落在不同卡(auto-pick 3 vs 2,用户漏带
  NPUS 参数),用户同卡补测结果一致。
- **跨日复现**:staged 三跑 25.8(B 轮 08-28 晨)/ 25.7(D2)高度稳定;
  blocking 今日 33.7 vs 历史矩阵 ~31 同量级。**blocking 同步税 ≈8ms/步**,
  与机制预言吻合(每步 aclrtSynchronizeStream 排空管线,run-ahead 流水
  化被拆掉)。
- **上游素材成立**:follow-up 方向 = 写入时轮转(省掉快照 memcpy)或
  CpuGpuBuffer 层环(模式级修复覆盖所有调用点);N 安全界仍为开放问题
  (N=8 三跑全绿为实证,生产建议大余量 N 或页复用前 event 守卫)。
- 执行注记:手册原 grep 模式漏了 `]`(实际日志行 `[SD-staged-copy]
  engaged: ...`),已改 `-F` 整串匹配。

### Run E(2026-08-28 设计,待执行):事件协议变体——第三个修法同日三臂对照

背景:维护者提及 upstream `GPUModelRunner.synchronize_input_prep()`
(vllm gpu_model_runner.py:3809-3822,pin v0.23.0 亦有)。研究结论
(2026-08-28,代码级):该协议防的正是本案权害类(其注释原文"don't
overwrite pinned memory while a prior non_blocking H2D DMA is still
reading"),但 ①窗口只包 execute_model 顶部输入 prep,backup 改写在
`propose_draft_token_ids`(model_runner_v1.py:1788/1821)窗口外;
②upstream 自己的 GPU drafter backup(llm_base_proposer.py:1065-1070,
单页改写+copy_to_gpu)同样无保护;③`prepare_inputs_event` 仅在
use_async_scheduling 时创建,我们的配置下为 None=no-op。

**变体实现**(`_SD_EVENT_COPY`,research 线,默认 off,仅 REVIVE 下
生效;与 STAGED 互斥,EVENT 优先):backup 专用 event(`torch.npu.Event
(blocking=True)`,cuda 回退,无 kwarg 回退),**入口半**在 host 改写
`backup.cpu` 之前 `event.synchronize()`(等上一步异步拷贝落地),
**出口半**在原路径 `copy_to_gpu()`(保持 non_blocking)之后
`event.record()`。拷贝全程异步,只栅栏源复用。

判读前提:正确性 = 挺过 16K 崩溃配置(counters 复核 c3=0);性能 =
同日三臂(blocking/event/staged)ITL 对照。

```bash
# E1: blocking 默认(安全锚点本尊,不带研究 env)
TIERS=16384 CONCS=1 NPUS=<id> bash research/run_baseline_npu.sh eagle3 8026
cp experiments/out/serve-npu-bf16-eagle3-k5.log  experiments/out/serve-rune-fix-blocking.log
cp experiments/out/baseline-npu-qwen3-8b-npu-bf16-eagle3-k5.json experiments/out/rune-fix-blocking.json
grep -cF "[SD-event-copy] engaged" experiments/out/serve-rune-fix-blocking.log || true   # 预期 0

# 排水后:
pkill -TERM -f "vllm serve.*--port 8026"; sleep 30   # 等 HBM 回落再跑 E2

# E2: 事件协议(拷贝保持异步 + 源复用栅栏)
VLLM_ASCEND_SD_REVIVE_RACE=1 VLLM_ASCEND_SD_EVENT_COPY=1 \
  TIERS=16384 CONCS=1 NPUS=<id> bash research/run_baseline_npu.sh eagle3 8027
cp experiments/out/serve-npu-bf16-eagle3-k5.log  experiments/out/serve-rune-event.log
cp experiments/out/baseline-npu-qwen3-8b-npu-bf16-eagle3-k5.json experiments/out/rune-event.json
grep -F "[SD-event-copy] engaged" experiments/out/serve-rune-event.log   # 预期 engaged 行

# E3: staged(同日第三臂,与 run D 的 D2 互为复现锚点)
VLLM_ASCEND_SD_REVIVE_RACE=1 VLLM_ASCEND_SD_STAGED_COPY=8 \
  TIERS=16384 CONCS=1 NPUS=<id> bash research/run_baseline_npu.sh eagle3 8028
cp experiments/out/serve-npu-bf16-eagle3-k5.log  experiments/out/serve-rune-staged8.log
cp experiments/out/baseline-npu-qwen3-8b-npu-bf16-eagle3-k5.json experiments/out/rune-staged8.json

# (可选,正确性复核)E2 配置 + 计数器:预期绿 + c3=0 esc=none
VLLM_ASCEND_SD_REVIVE_RACE=1 VLLM_ASCEND_SD_EVENT_COPY=1 VLLM_ASCEND_SD_COUNTERS=1 \
  TIERS=16384 CONCS=1 NPUS=<id> bash research/run_baseline_npu.sh eagle3 8029
```

注意:三臂 TAG 相同,copy-aside 强制;同卡执行;SUMMARY 不记研究 env,
靠 engaged 行(`[SD-event-copy]`/`[SD-staged-copy]`)与归档名区分。

#### 判读矩阵

| 结果 | 定性与行动 |
|---|---|
| E2 崩 | 引擎级 event 配对(record 于当前流 + host synchronize)在 NPU 深环下不足——变体出局,blocking/staged 不受影响;记录后不再追 |
| E2 绿 + (可选)c3=0 | 正确性成立;进入性能判读 |
| ITL(E2) ≈ ITL(E1) | 入口等待把 host run-ahead 压到设备节奏,与 blocking 同量级 → 变体无吞吐优势,仅剩"upstream 习语"评审价值(自用线无用) |
| ITL(E1) > ITL(E2) > ITL(E3) | 中间态:量化"拷贝异步化收益 − 入口等待代价"的分解 |
| ITL(E2) ≈ ITL(E3) | 事件协议与 staged 打平 → 事件协议成为等价但更简的形态(无 N 安全界问题)——**自用线最优候选** |
| accept 差 >0.1 于任一臂 | 数据可疑,复跑确认 |

**服务对象注记(2026-08-28 用户定调)**:PR #14922 合入前景不抱
期望(维护者意向偏保守);三修法对比数据服务自用研究线——为 Phase 2+
的引擎改造(staged 或事件协议择优)提供选型依据。默认路径恒为 blocking
最小修复(安全锚点不变),实验路径全部 env 门控。

#### Run E 结果(2026-08-31 执行,@7d2c5486b,NPU2 同日四跑):E2≈E3 → 事件协议定为自用线选定形态

| 臂 | 配置 | ITL p50 | out/s | accept(C) | ok | 自证 |
|---|---|---|---|---|---|---|
| E1 | blocking 默认 | 33.4ms | 66.18 | 2.4824 | 8/8 | engaged 行 = 0 |
| E2 | REVIVE+EVENT | **25.9ms** | **83.62** | 2.4824 | 8/8 | `[SD-event-copy] engaged` 行在 |
| E3 | REVIVE+STAGED=8 | 25.8ms | 83.87 | 2.4824 | 8/8 | 跨日三跑 25.8/25.7/25.8 复现 |
| E2' | E2+COUNTERS | 25.8ms | 83.68 | 2.4824 | 8/8 | `steps=847 c1=840 c2=7 c3=0 c1x=0 esc=none` |

- **判读 = 判读矩阵末行命中**:ITL(E2)≈ITL(E3)(差 0.1ms,噪声级)
  → **事件协议 = staged 的等价形态且更简**(无 N 安全界、无页环管理、
  upstream `synchronize_input_prep` 同源习语)——**自用线选定形态**,
  Phase 2+ 引擎改造采用此修法(判定记录)。
- **正确性定谳**:E2' 计数器与 staged 时代(Run C,08-28)逐位一致
  (steps=847/c1=840/c2=7/c3=0/esc=none;门开 7 次与页/事件设计无关,
  逃逸归零)——事件栅栏在引擎级完整阻断 -1 逃逸;counters 臂 ITL
  不受扰动(25.8ms,插桩纪律保持)。
- **blocking 同步税机理修正(重要)**:E2 每步同样要等"上一步拷贝
  落地"(entry `event.synchronize()`),ITL 却与零等待的 staged 打平
  → **run D 的 ~8ms 税不是"等拷贝"的代价,而是 torch_npu blocking
  拷贝路径自身的开销**(aclrtSynchronizeStream 全排空 + 同步 memcpy
  慢路径)。host 在 propose 点本就不跑前(被步内结构天然压住),等
  一个"步内早已落地"的事件 ≈ 免费;blocking 贵在路径,不在等待。
- 执行注记:08-31 执行(设计 08-28);E1→E2 间 30s 排水不足触发一次
  PREFLIGHT-FAIL(E1 引擎 HBM 22GB 未排空),等待后重试通过——共享
  服务器排水间隔应 ≥60s 或轮询 npu-smi 确认回落。
- **默认路径未动**:blocking 仍为默认(安全锚点);事件协议转正
  (改默认)是独立决策,留待 Phase 2 引擎改造时一并执行。
