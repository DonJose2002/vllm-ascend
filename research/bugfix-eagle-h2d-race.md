# 复盘:eagle 系投机解码 16K 崩溃(pinned H2D 异步竞态)

> 2026-08-24 | 修复 commit:research 线 `62533dafa`(v0.23.0 容器验证)/ PR 分支 `pr/bugfix-eagle-h2d-race`(`56f262aa1`,基于 origin/main `1a086ce8a`)| 环境:vllm-ascend v0.23.0 = fixes,910B3,Qwen3-8B + Tengyunw/qwen3_8b_eagle3(eagle3 头,1 层,draft_vocab 32000),K=5
> 姊妹篇:`eagle-h2d-race-journey.md`(人话版,给同事)

## 0. 一句话

`prepare_next_token_ids_padded` 把一批"兜底 token id"用 pinned **非阻塞** H2D 拷贝送上 NPU;torch_npu 上这类拷贝走 SDMA 引擎,与发射流上后续计算 kernel 的先后顺序**没有保证**——输掉竞态时 `torch.where` 读到未初始化内存(0xa5a5a5a5,CANN 毒化标记),垃圾 id 混进 drafter 的 input_ids,词表规模的 embedding gather 拿垃圾当索引,越界读 GM,aivec 硬件异常,引擎死亡。

## 1. 现象与误判

1. **误判"16K 特异"**(与 SD 收缩崩溃同款教训):首跑 4K/c1 全绿、16K/32K 全灭。真相:4K 也中(同日 B 实验 4K 6/8,2 条"干净空流"),只是概率低;16K 深异步队列下近 100%。
2. **误判"eagle3 头权重有问题"**:首跑即出 NPU 首个 SD 正收益(4K ITL 22.9ms、out/s 104),不像坏权重;但崩溃签名(vector core 异常)容易让人先怀疑硬件/权重。三张物理卡(NPU 1/2/4/5)复现 → 排除单卡。
3. **"干净空流"的迷惑性**:失败请求的流是"正常关闭、0 token、无异常"——不是 HTTP 500。引擎死在更深处,APIServer 只是优雅收尾。若 harness 的 ok 判定只看"连接正常关闭",这种失败会被静默吞掉(bench_baseline.py 的 [DONE] 判定救了我们)。

## 2. 定位过程(时间线)

### 2.1 现场与第一轮假设

首跑(4K/c1 8/8 绿,ITL 22.9ms / accept 2.55 / out/s 104),16K/32K 全灭。serve log traceback 漂在 `shutdown` 时的 `AclrtSynchronizeDevice`(EZ9999 → vector core exception)——**崩溃点与肇事点分离**,这是异步执行故障的典型特征。scheduler dump:16K 请求 chunked prefill 第 6 块(11904/2048),单请求,非 SD 步。`num_common_prefix_blocks=[110]` 引出 H1(前缀缓存交互)。

### 2.2 三个判别实验

| 实验 | 结果 | 结论 |
|---|---|---|
| B:`--no-enable-prefix-caching` 16K | 仍全灭 | **H1(前缀缓存)驳倒** |
| C:`--max-num-batched-tokens 32768`(单块 prefill) | 启动 dummy_run 即崩 | 与 chunk 边界无关;单次前向 token 数相关 |
| eager:`--enforce-eager` 16K | 仍崩(同签名) | **图重放排除** |

### 2.3 plog 锁定故障 kernel

```
~/ascend/log/debug/plog/plog-<pid>_*.log
fault kernel_name = GatherV2_..._high_precision
args[12] = 0x25180 = 151936 = Qwen3 词表大小   → embedding 类查表
args[18] 高位 = 0xa5a5a5a5                      → CANN 未初始化内存毒化标记
errcode 0x800000: MTE accesses an invalid GM address
```

结论:某个 token-id/索引 buffer 未初始化就被消费,垃圾 id 越界读词表。

### 2.4 决定性翻转:ASCEND_LAUNCH_BLOCKING=1

`ASCEND_LAUNCH_BLOCKING=1`(每次 kernel launch 同步等待完成)+ eager → **16K 通过**(accept 2.66)。去掉 env → 必崩。唯一能被这样翻转的故障类别:**异步竞态**。这同时解释了 4K 偶发(窗口窄)与 16K 必现(深队列窗口宽)。

### 2.5 二分探针:找到垃圾的 carrier

静态分析锁定两条候选链(所有写入路径表面流内有序,纯读码无法再二分):

- 链 A(target):`target_token_ids = input_ids.gpu[token_indices]`(索引链)
- 链 B(next):`backup_next_token_ids` pinned H2D(`CpuGpuBuffer.copy_to_gpu`,`non_blocking=True`)→ `torch.where` 消费

插桩(env 门控,`VLLM_ASCEND_SD_DEBUG`,默认 off):**保持时序**的单张量 clamp(一次 elementwise,不加 host 同步):

| 探针 | 结果 | 结论 |
|---|---|---|
| clamp(双钳) | 过 | 垃圾确在两个 id 张量之一 |
| sync_check(min/max 打印,host 同步) | 过 + 216 次全 ok + 零 BAD | 观察治愈竞态 → 写入是异步的、消费先跑 |
| clamp_target(只钳 A) | **崩** | A 无辜 |
| clamp_next(只钳 B) | **过** | **垃圾 carrier = 链 B 实锚** |

### 2.6 根因链闭合

```
prepare_next_token_ids_padded (llm_base_proposer.py):
  backup.np[:n] = <host 侧 get_token_id 兜底值>       # host 写 pinned
  copy_to_gpu(n) → gpu.copy_(cpu, non_blocking=True)  # ← pinned H2D,SDMA 引擎,无流序保证
  next_token_ids = torch.where(cond, selected, backup.gpu[:n])   # 竞态消费点
      ↓
set_inputs_first_pass eagle 分支:
  self.input_ids[token_indices_to_sample] = next_token_ids      # 垃圾 scatter 进 drafter 输入
      ↓
eagle 头 embedding gather(GatherV2,表 151936×4096 bf16 = 1244659712 字节,与 plog dump 吻合;
                          32000 缩减词表只在输出侧 lm_head,输入侧无映射)
      ↓ 垃圾 id(0xa5a5...) 越界索引
MTE invalid GM address → aivec 异常 → EE9999 → 引擎死亡
```

时序细节(此前未强调):`seq_lens_list` 的 `.tolist()` 在拷贝前**排空了设备队列**——竞态不是"深队列插队",而是**每步在空闲设备上的 photo finish**:SDMA 拷贝(µs 级)与 host 随后几 µs 内发射的 where→scatter→gather 链几乎同时起跑。消费链长度(§3.2)= 给 SDMA 的落地缓冲垫:eagle 头 1 层 = 无垫;draft_model/dflash 的重前向 = 毫秒级垫。上下文长度两条作用:①chunked prefill 下 16K=8 块 vs 4K=2 块,= 4 倍次数的重复试验(实证);②每块伴随大量元数据 H2D,SDMA 侧可能拥塞(假设,未测)。

### 2.7 开放问题:stale-value 悖论(诚实边界)

上述模型有一个未闭合的逻辑洞:**`backup.gpu` 是 `torch.zeros_like` 初始化的持久 buffer**——若 where 只是"读在拷贝落地前",读到的应是零或上一步旧值(均为合法 id),不应是 0xa5a5(CANN 对**新分配**内存的毒化标记),也就不应越界。同 stream 下 where 的新分配输出张量也会被 where 完整写入,消费方同样不该看到毒。

两个候选微观机制(均未证实):
- **(i) SDMA 微型 pinned 传输的异常/撕裂落地**,把毒化中间态写入目的 buffer;
- **(ii) 乱序对根本不是 copy-vs-where**:真正失序的是别的一对(疑涉及新分配张量的相邻小 kernel,驱动级 launch 合并边角),而 blocking 拷贝恰好充当流上**全屏障**,把任何一对都串行化——统一解释修复有效、launch-blocking 有效、clamp 时序扰动有效。

修复对两者皆对症(依赖缺失 + 确定性屏障,9/9 实证),不依赖分辨 (i)/(ii)。

**判别实验(待做)**:device 侧 OOB 计数器——where 之后插 clamp+计数 kernel(纯 device,不加 host 同步,躲开海森堡),步末一次性 D2H 汇总。计数 >0 = 垃圾实流经 next_token_ids 值(支持 i);=0 = 垃圾另有产地、屏障才是修复本质(支持 ii)。

## 3. 三个"为什么"

### 3.1 为什么上游 vllm(CUDA)没有这个 bug?

上游 GPU 的同型代码也用 `CpuGpuBuffer.copy_to_gpu`(non_blocking=True),但 **CUDA 的 `cudaMemcpyAsync` 入队到 stream 后,与同 stream 的后续 kernel 严格有序**——异步只是"不阻塞 host",顺序由流语义保证。torch_npu 的 pinned H2D 走 SDMA 引擎,与发射流(launch stream)上后续计算 kernel 的先后**没有等价保证**。一句话:**代码相同,后端流语义不同;NPU 移植不能默认继承 CUDA 的流序直觉**。

### 3.2 为什么只有 eagle 系触发?(暴露矩阵)

调用归属(按上游 `SpeculativeConfig`:`use_eagle()` = `{"eagle","eagle3","mtp","dflash","dspark"}`,`uses_draft_model()` = `{"draft_model"}`;`model_runner` 的 `use_padded_batch` 分派):

| drafter | 走 `prepare_next_token_ids_padded`(含竞态拷贝) | next_token_ids 的消费路径 | 结果 |
|---|---|---|---|
| eagle / eagle3 / mtp(`pass_hidden_states=True` → `net_slots=0`) | ✅ `use_eagle()` | **分支 A**:eagle 分支直接 scatter 进 `self.input_ids`(裸 tensor op)→ 1 层头多步合并前向 → embedding gather | eagle3 16K 必崩,4K 偶发 |
| dflash / dspark(`parallel_drafting`,`net_slots=K`) | ✅ `use_eagle()` | **分支 B**:CopyAndExpandEagleInputs(AscendC 算子)→ 5 层头单 pass 并行 | 4K/16K 探针未复现 |
| draft_model(`net_slots=1`) | ✅ `uses_draft_model()` | **分支 B**:CopyAndExpandEagleInputs → 0.6B × K+2 步串行(即 plan A 图) | 27 cell 从未复现 |
| ngram / suffix(host 提议器) | ❌(bookkeeping 后走 CPU list 路径,函数不被调用) | 无设备 buffer | **唯一结构性免疫** |

**准确结论**:竞态拷贝被除 ngram/suffix 外的全部 padded 方法共享;draft_model/dflash 是"**共享毒源、消费链更长(B 分支多一跳算子 + 更重的头)、实测未爆**",不是免疫。修复作用于拷贝本身,同时拆掉 A/B 两条分支脚下的雷——这是本修复覆盖面大于表象证据(只有 eagle3 爆)的原因。

### 3.3 为什么 16K 必现、4K 偶发?(机制假设,非实证)

**已实证**:竞态存在(翻转实验)、carrier 是该拷贝(clamp 二分)、16K 近必现/4K 偶发(多轮复现)。

**机制解释为最佳拟合假设**,两种可并存:
1. **消费链密度(解释方法间差异)**:eagle 系从拷贝到 gather 只有几个轻 kernel + 1 层头,device 侧很快追上 SDMA;B 分支多一跳 AscendC 算子且头模型重,窗口天然宽。
2. **SDMA 队列拥塞(解释长度间差异)**:16K/32K chunked prefill 每块产生大量 metadata H2D,backup 拷贝在 SDMA 队列里被排到后面、延迟拉大;4K 拷贝少,大概率在消费前落地。注意这与"计算队列深所以窗口宽"的直觉相反——计算队列深反而给拷贝更多时间,**拥塞必须发生在 SDMA 侧**才能解释观察。
   (我们未直接测量 SDMA 延迟,该条为推断;修复不依赖此假设成立。)

## 4. 修复

最小改动:该处 H2D 改 **blocking** 拷贝:

```python
self.backup_next_token_ids.gpu[:num_reqs].copy_(
    self.backup_next_token_ids.cpu[:num_reqs], non_blocking=False
)
```

- payload = `num_reqs × int32`(<1KB),阻塞开销微秒级,换正确性稳赚;
- **不**全局改 `CpuGpuBuffer.copy_to_gpu`(大量大缓冲依赖 non_blocking 性能,如 hidden_states);
- 备选方案(未采用):拷贝后补 event/record_stream 依赖——正确但复杂,1KB 载荷不值得。

### 验证(修复 `62533dafa`,工作区遮蔽零重装生效)

1. 16K 单请求 × 4 种配置(裸跑/两种探针/全量矩阵)全过——此前该配置 4/4 必崩;
2. eagle3 **全量矩阵 9/9 cell**(4K/16K/32K × c1/c4/c16)全绿,accept 2.34-2.66 双口径一致;
3. 修复前 4K 偶发空流(B 实验 2/8)同根同愈,后续多轮 4K 全 8/8。

## 5. 教训清单

1. **崩溃点 ≠ 肇事点**:异步栈的 traceback 漂在 sync 点,先看 plog 的 fault kernel 名再谈代码定位;
2. **`ASCEND_LAUNCH_BLOCKING=1` 翻转 = 竞态实锤**:同步启动是 NPU 竞态的"活检工具",一翻即知类别——但绝不能当修复(性能不可接受,且掩盖问题);
3. **0xa5a5a5a5 是通用诊断锚**:任何 NPU kernel 故障 args 里出现它 = 读到未初始化设备内存,顺藤摸"谁没写完就被读";
4. **保持时序的探针是竞态二分的唯一姿势**:clamp(一次 elementwise)保时序,host 同步打印(sync_check)会治好竞态——只证明"异步写入存在",不能定位 carrier;两者配合,单张量 clamp 完成二分;
5. **CUDA 直觉在 NPU 上要逐条验证**:non_blocking 拷贝、event、stream 语义是最容易踩的三件套(本项目已集齐:本 bug + D5 的 stream 上限);
6. **竞态 bug 无法 UT 回归**:可复现配置 + 证据链 + 修复翻转就是验收物,PR 描述要写全(同 SD 收缩崩溃先例);
7. **"干净空流"是深层故障的礼貌面具**:ok 判定必须看 [DONE] 与 token 数,harness 早已修对,这次直接受益。

## 6. 文件索引

| 文件 | 位置 | 角色 |
|---|---|---|
| `vllm_ascend/spec_decode/llm_base_proposer.py` `prepare_next_token_ids_padded` | 修复点(`62533dafa` research 线 / `56f262aa1` PR 线) | blocking H2D |
| 上游 `vllm/v1/utils.py` `CpuGpuBuffer.copy_to_gpu` | 根因侧(未改) | non_blocking=True 的来源;CUDA 安全/NPU 无保证 |
| 同文件 `set_inputs_first_pass` eagle 分支 | 消费链 | scatter 进 drafter input_ids |
| plog(`~/ascend/log/debug/plog/plog-<pid>_*.log`) | 证据 | fault kernel=GatherV2,0xa5a5 标记 |
| 插桩 `VLLM_ASCEND_SD_DEBUG`(research 线保留,PR 线不含) | 工具 | clamp/clamp_target/clamp_next/sync_check |
| `experiments/out/serve-npu-bf16-eagle3-k5.log` + SUMMARY 块 | 证据 | 9/9 全绿矩阵 |
