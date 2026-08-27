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
