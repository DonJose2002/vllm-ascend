# 复盘:eagle 系投机解码 16K 崩溃(pinned H2D 异步竞态)

> 2026-08-24 | 修复 commit:research 线 `62533dafa`(v0.23.0 容器验证)/ PR 分支 `pr/bugfix-eagle-h2d-race`(`56f262aa1`,基于 origin/main `1a086ce8a`)| 环境:vllm-ascend v0.23.0 = fixes,910B3,Qwen3-8B + Tengyunw/qwen3_8b_eagle3(eagle3 头,1 层,draft_vocab 32000),K=5
> 姊妹篇:`eagle-h2d-race-journey.md`(人话版,给同事)

## 0. 一句话

`prepare_next_token_ids_padded` 把一批"兜底 token id"用 pinned **非阻塞** H2D 拷贝送上 NPU;torch_npu 上这类拷贝走 SDMA 引擎,与发射流上后续计算 kernel 的先后顺序**没有保证**——输掉竞态时 `torch.where` 读到未初始化内存(0xa5a5a5a5,CANN 毒化标记),垃圾 id 混进 drafter 的 input_ids,词表规模的 embedding gather 拿垃圾当索引,越界读 GM,aivec 硬件异常,引擎死亡。

> **终判(2026-08-25,同日二次修正)**:三计数器实验闭合此案为**值故事**——逃逸门是 `where` 的 backup 选择门(c2=7 次/843 步),racy 配置下恰有 1 次读到未落地旧值而流出 1 个 OOB id(c3=1),即引擎死亡事件;clamp_next 的治愈实为值消毒而非时序屏障。**c1 的"毒常驻"解读已被 Run 3 勘误**:那 837/840 个 OOB 是上游 `get_token_id` 的 **-1 哨兵**(设计如此,门关步 backup 本就是 -1),逃逸值实为 **stale 哨兵 -1**(待 Run 4 esc 值捕获定谳)。详见 §2.7.1 与 §7.4-7.6。

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
| clamp_target(只钳 A) | **崩** | A 排除 |
| clamp_next(只钳 B) | **过** | 链 B 与崩溃强关联——**注:这只证明"在 B 处插入时序扰动能救",不必然证明垃圾在 B 的值里**(值 carrier vs 时序屏障两种解释,见 §2.7;证据倾向后者) |

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

时序勘误(2026-08-24 二次核查):此前"`tolist()` 排空队列 → 空闲设备 photo finish"的说法**不成立**——`gpu_input_batch.num_tokens_no_spec` 是 **CPU numpy**(`num_tokens_no_spec_cpu_tensor.numpy()`),对它 `.tolist()` 是纯 CPU 操作,不同步任何设备队列。真实时序:`where` 排在采样尾巴的 kernel 之后,SDMA 名义上有这段尾巴的时间落地拷贝——简单 copy-vs-where 模型下竞态反而应该**很难**输,与 16K 近 100% 必崩的事实矛盾(→ §2.7)。上下文长度两条作用不变:①chunked prefill 下 16K=8 块 vs 4K=2 块 = 4 倍次数的重复试验(实证);②每块伴随大量元数据 H2D,SDMA 侧可能拥塞(假设,未测)。

### 2.7 开放问题:值故事的三重矛盾(诚实边界)

"垃圾经 `next_token_ids` 值传播"这条路径,累计有三个未闭合的逻辑洞:

1. **stale-value 悖论**:`backup.gpu` 是 `torch.zeros_like` 初始化的持久 buffer——若 where 只是"读在拷贝落地前",读到的应是零或上一步旧值(均为合法 id),不应是 0xa5a5(CANN 对**新分配**内存的毒化标记),也就不应越界。
2. **值传播与链长无关(用户连环追问引出)**:垃圾 int32 作为**值**穿过 where/scatter/CopyAndExpand 都不会出事,直到被当作**索引**(gather)才成为非法访存——"崩溃点(gather)≠竞态点(copy-vs-where)"。因此若垃圾真在 `next_token_ids` 值里,draft_model(Qwen3-0.6B embed 151936×1024)和 dflash(151936×4096)的 embedding gather 一样会越界崩——**"消费链长"不可能构成值免疫**。更进一步,`where` 是 `backup.gpu` 的唯一读者,且在**所有** padded 方法中的发射位置完全相同(都在采样尾巴后)——若竞态是 copy-vs-where,三方法的竞态窗口一模一样,"只有 eagle 崩"(§3.2)在该模型下**无法解释**。
3. **选择条件稀有性**:`where` 仅对 `cond=false`(该行本步无任何有效采样 token,即被丢弃的请求)的行**选用** backup 值。正常跑完的 bench 里这种行即使有也极少;若垃圾必须经此选择路径传播,16K 的近 100% 必崩难以成立。

两个候选微观机制(均未证实):
- **(i) SDMA 微型 pinned 传输的异常/撕裂落地**,把毒化中间态写入目的 buffer——但需同时解释洞 2/3(选择路径稀有却必崩);
- **(ii) 乱序对根本不是 copy-vs-where**:真正失序的是别的一对(身份未定),而 blocking 拷贝、clamp_next 的插入 kernel、launch-blocking 的共同点是**在流上插入串行化/时序屏障**,把不管哪一对都隔开——统一解释全部翻转实验。"只有 eagle 崩"来自各方法调度形态差异(eagle 每步 µs 级密集发射 vs 重 drafter 拉长步间节奏),而非值传播路径长短。

**当前证据倾向 (ii)**(三重矛盾都指向值路径可能从未发生),但 (i) 不能排除。修复对两者皆对症(依赖缺失处补确定性屏障 + 9/9 实证),不依赖分辨 (i)/(ii)。

**判别实验(三计数器版,零 host 同步,run 结束一次读走)**:
- c1:copy 之后立即数 `backup.gpu` OOB(毒在不在 buffer 里;>0 支持 (i) 撕裂落地);
- c2:数 `cond=false` 行数(backup 到底有没有被选中过;≈0 = 值传播路径不通,值故事出局);
- c3:where 之后数 `next_token_ids` OOB(垃圾有没有流出去;c1>0 ∧ c2>0 ∧ c3>0 = 值故事全链闭合)。
跑全绿 + 三计数全零 → 屏障故事定性;此后才值得猎"真正的乱序对"(候选:跨流 buffer 复用如 draft_token_ids 侧流、scatter-vs-gather 与图参数 update_stream 的交互)。

### 2.7.1 终判:值故事闭合(2026-08-25,三计数器实验执行完毕;同日 Run 3 勘误一次)

**读数**(Run 2',NPU2,commit `ac2e4ab83`,revive+counters+clamp 保活,SIGUSR1 活体读数):

```
[SD-counters] sigusr1 steps=843 c1=837 c2=7 c3=1
```

**c2>0 ∧ c3>0 → 值故事胜出,屏障假说出局**:门开 7 次,racy 配置下 1 次读到未落地旧值、1 个 OOB id 流出 where——**这 1 次逃逸就是无 clamp 时的引擎死亡事件**。clamp 值消毒救下(Run 2/2' 绿),fix 下 c3=0(Run 3)。

**勘误(同日,Run 3 触发)**:我最初把 c1=837 解读为"毒近乎常驻"——**错**。Run 3(纯 fix+counters)读数 `steps=847 c1=840 c2=7 c3=0`:fix 下拷贝已同步落地,c1 仍 ~99%。真相(源码核实):

- 上游 `CachedRequestState.get_token_id(idx)`(vllm/v1/worker/gpu_input_batch.py)对**未提交位置返回 -1 哨兵**;
- 正常 decode 步里 `num_tokens_no_spec-1` 恰指向"刚采样未提交"的位置 → **门关步的 backup 恒为 -1**(设计如此:门关时该值永不被 where 选中,是死数据);
- 门开步(请求边界/discarded,7 次)idx 落在已提交 token 上 → backup 是**真 token**;
- 数值自洽的铁证:**847-840=7=c2**(两次 run 均完美重合)——c1 数的就是哨兵底噪,它从一开始就看不见竞态;
- `CpuGpuBuffer` 是 `torch.zeros_like` 初始化(utils.py)——buffer 里从来没有 0xa5a5。

**修正后的完整机制(stale 哨兵故事)**:

```
正常步:门关,backup 落地 -1(哨兵,无害死数据)
门开步(≈1%):host 写真 token → non_blocking 拷贝在途 → where 抢先读
             → 读到该行上一次落地的旧值 = -1 哨兵(OOB!)
             → next_token_ids 流出 -1(c3=1)
无 clamp:-1 scatter 进 drafter input_ids → embedding gather 以 -1 为索引
        → GM 越界 → aivec → 引擎死亡(Run 0/1 的崩溃,亦是修前 16K 必崩)
```

三重矛盾的最终消解:
1. **stale-value 悖论**:旧值不是"零或合法旧 token"而是 **-1 哨兵本身**——它作为值 OOB,只是上游设计里永远不该被消费;竞态打破的正是这个契约。0xa5a5 无需入场(plog args[18] 的 0xa5a5 是 kernel 参数区某字段,并非索引值本身;索引实际值待 Run 4 esc 捕获定谳)。
2. **"只有 eagle 崩"**:门开频率(≈1/120 步,请求边界)对所有 padded 方法相同,差异在单位时间的试验次数与 launch-ahead 深度(eagle µs 级密集发射把 host 推得更靠前,窗口更宽)。
3. **稀有性**:门开 7 次 × 竞态命中 1/7 = 每 run 恰 1 次逃逸。4K 偶发/16K 必现 = 步数即试验次数。

**对 §2.5 判读表的追溯修正**:clamp_next"过"的机制是**值消毒**(把流出的 -1/垃圾钳进合法词表域),不是时序屏障——直接证据:同样插在 where 后、同样流上 elementwise/reduction 的 c3 计数 kernel(Run 1)**不**治愈(只数不改值)。教训 4 的"保时序探针"框架仍然成立,但"插入即屏障"的隐含假设被证伪。

**修复语义**:blocking 拷贝 = 门开步的 where 之前拷贝必然已落地 → 门开必读到真 token → 值路径枯竭(c3=0,Run 3 实证;哨兵底噪 c1 留存属正常)。9/9 实证一致。

**剩余开放点(仅一个)**:逃逸值的直接捕获——esc 探针(Run 4)将给出 min/max:预期 **-1**(哨兵故事);若测得 0xa5a5a5a5(int32 = -1515870811)则"毒落地"路径复活,需再议。

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

**准确结论(2026-08-24 修正)**:竞态拷贝被除 ngram/suffix 外的全部 padded 方法共享。但"消费链长 → draft_model/dflash 未爆"的旧解释**不成立**:①值传播与链长无关(垃圾值会崩任何方法的词表 gather,见 §2.7 洞 2);②`where` 作为 backup 的唯一读者,在三方法中发射位置相同,链长连首读时序也影响不了。方法间差异在简单 copy-vs-where 模型下**无法解释**,是支持 §2.7 假设 (ii)(乱序对另有其人、修复实为屏障)的关键证据之一;方法差异更可能来自调度形态(eagle µs 级步节奏 vs 重 drafter 毫秒级)。修复作用于拷贝/屏障,对全部 padded 方法的保护不变。

### 3.3 为什么 16K 必现、4K 偶发?(机制假设,非实证)

**已实证**:竞态存在(翻转实验)、blocking 该拷贝即愈(9/9)、16K 近必现/4K 偶发(多轮复现)。

**机制解释(均为假设)**:
1. **掷骰子次数(最扎实)**:chunked prefill 下 16K=8 块 vs 4K=2 块 = 4 倍次数的重复试验;decode 阶段 16K 的元数据 H2D 也更大更多。
2. **SDMA 队列拥塞**:每块 prefill 伴随大量元数据 H2D,backup 拷贝在 SDMA 队列被排后(未测;注意计算队列深反而给拷贝更多时间,拥塞必须在 SDMA 侧才解释得通)。
3. **方法间差异(eagle-only)**:无已证机制;调度形态假设见 §3.2/§2.7。
   修复不依赖以上任何一条成立。

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
7. **"干净空流"是深层故障的礼貌面具**:ok 判定必须看 [DONE] 与 token 数,harness 早已修对,这次直接受益;
8. **崩溃点 ≠ 竞态点,值传播 ≠ 时序传播**(用户连环追问引出):垃圾作为**值**穿过多少中间计算都无害,直到被当**索引**才炸(plog 指 gather 是崩溃点,不是竞态点);反过来,"在 X 插一个 kernel 就治好"只证明 X 处的**时序**与崩溃相关,不证明垃圾在 X 的值里——二分探针给的是**时序定位**,值定位需要无扰动的计数器。

## 6. 文件索引

| 文件 | 位置 | 角色 |
|---|---|---|
| `vllm_ascend/spec_decode/llm_base_proposer.py` `prepare_next_token_ids_padded` | 修复点(`62533dafa` research 线 / `56f262aa1` PR 线) | blocking H2D |
| 上游 `vllm/v1/utils.py` `CpuGpuBuffer.copy_to_gpu` | 根因侧(未改) | non_blocking=True 的来源;CUDA 安全/NPU 无保证 |
| 同文件 `set_inputs_first_pass` eagle 分支 | 消费链 | scatter 进 drafter input_ids |
| plog(`~/ascend/log/debug/plog/plog-<pid>_*.log`) | 证据 | fault kernel=GatherV2,0xa5a5 标记 |
| 插桩 `VLLM_ASCEND_SD_DEBUG`(research 线保留,PR 线不含) | 工具 | clamp/clamp_target/clamp_next/sync_check |
| `experiments/out/serve-npu-bf16-eagle3-k5.log` + SUMMARY 块 | 证据 | 9/9 全绿矩阵 |

## 7. 三计数器判别实验(runbook,2026-08-25 实现)

§2.7 设计的实现与执行手册。**实现**(research 线 commit `d978d01c9`,`llm_base_proposer.py`):两个独立 env + `_RaceCounters`(int64 device 累加,零 host 同步,run 末 atexit/`__del__` 一次 D2H 读走)+ `_oob_count` helper + CPU UT `research/test_race_counters.py`(ast 抽取真实源码节点验证:scenario A 模拟值故事端到端 c1=c2=c3=1 精确恢复;scenario B 修复路径只 c2 增;scenario C 全关零计数)。

| env | 语义 |
|---|---|
| `VLLM_ASCEND_SD_REVIVE_RACE=1` | 拷贝点复活原始竞态路径(`CpuGpuBuffer.copy_to_gpu()`,整缓冲 non_blocking);默认关 = blocking 修复生效 |
| `VLLM_ASCEND_SD_COUNTERS=1` | 启用 c1/c2/c3 计数(steps 为 host 侧纯计数,计数 kernel 与 clamp 探针同为保时序 device 归约) |

两开关独立是**故意的**:Run 0 只复活竞态不加计数,对照"计数 kernel 自身治愈竞态"的混淆(clamp_next 教训)。

**执行顺序**(16K 复现配置,eagle3 K=5,修前 4/4 必崩;工作区遮蔽 → 宿主机 pull 即生效,serve 期间禁切分支):

```bash
# 服务器容器内(先宿主机 git pull):
# Run 0 对照:预期仍崩(复现成立)。不崩 = 复现条件漂移,停下重估,勿继续解读
VLLM_ASCEND_SD_REVIVE_RACE=1 TIERS=16384 CONCS=1 \
  NPUS=<id> bash research/run_baseline_npu.sh eagle3 8021
cp experiments/out/serve-npu-bf16-eagle3-k5.log experiments/out/serve-race-run0.log

# Run 1 实验:跑 ≥2 次(换卡/重跑),判读矩阵见 §2.7
VLLM_ASCEND_SD_REVIVE_RACE=1 VLLM_ASCEND_SD_COUNTERS=1 TIERS=16384 CONCS=1 \
  NPUS=<id> bash research/run_baseline_npu.sh eagle3 8022
cp experiments/out/serve-npu-bf16-eagle3-k5.log experiments/out/serve-race-run1.log

# 证据收集(两个 run 都要):
grep -E "SD-counters" experiments/out/serve-race-run*.log
```

**读数语义**:serve 进程退出时(atexit 或 proposer `__del__` 双保险)打一行
`[SD-counters] <origin> steps=N c1=X c2=Y c3=Z`。硬崩(aivec 原生故障)两路都不触发——**崩本身即数据点**(计数 kernel 未治愈竞态,对照 clamp_next 先例解读)。SUMMARY 块照常贴回。

**判读速查**(详矩阵见 §2.7):

| 观测 | 结论 |
|---|---|
| Run 1 绿 + 三计数全零 | **屏障故事定性**:垃圾从未走值路径;下一步猎真正乱序对(候选:draft_token_ids 侧流 / update_stream / scatter-vs-gather 与图参数交互) |
| Run 1 绿 + c1>0 ∧ c2>0 ∧ c3>0 | **值故事闭合**(撕裂落地→被选中→流出,三环齐证) |
| Run 1 绿 + 混合(如 c1>0 ∧ c2=0) | 毒落地但从未被选中 → 值路径不通,倒向屏障故事;c1 单独成立则"撕裂落地"机制局部成立 |
| Run 1 崩 | 计数归约未能如 clamp_next 般治愈 → 插入点时序不足以隔开乱序对;结合 Run 0 是否崩一起入档 |

### 7.1 首轮结果与判读(2026-08-25,Run 0/1)

| Run | 配置 | 卡 | 结果 |
|---|---|---|---|
| 0 | REVIVE_RACE 单开 | NPU2 | **崩**:8/8 空流,aivec MTE @ prefill ~12-14K(num_computed_tokens=12288+2048,单请求),签名与修前一致;证据 `experiments/out/serve-race-run0.log` |
| 1 | REVIVE_RACE + COUNTERS | NPU4 | **崩**:同签名;无 `[SD-counters]` 行(原生故障杀进程,atexit 不触发);证据 `serve-race-run1.log` |

判读:
1. **Run 0 = 复现成立 + revive 生效**(site-packages 是 blocking 修复版,能崩的唯一路径就是 revive 分支——兼为工作区代码活跃的旁证)。
2. **Run 1 = 计数 kernel 未治愈竞态**,这是重要反向证据:c3 的归约 kernel 插在 where 后、scatter 前,与 clamp_next 的插入位置几乎相同、同为流上 elementwise/reduction kernel——屏障假说预言它应同样隔开乱序对,但没有 → **clamp_next 的治愈更可能是"值消毒"(把垃圾钳进合法词表域)而非时序屏障,证据流向值故事**(§2.7 假设 i 方向)。
3. **洞 3(cond=false 稀有)出现解法候选**:两次崩溃步都在 chunked prefill 期间(num_output_tokens=0,尚未生成任何 token)——预填充步可能根本没有已采样 token → cond=false 或许并不稀有,值故事的"选择路径稀有却必崩"矛盾可能在预填充场景消解。待 c2 实测。
4. 歧义消除跟进:首版无 engagement 日志,"计数器是否真跑过"不可证 → 已补一次性 engagement 行(`_RaceCounters.__init__`,host-only 零同步)。

### 7.2 Run 2:clamp 组合保活取读数(决定性)

思路:**clamp 治愈保住引擎(绿跑 → atexit 触发读数),计数器在 clamp 之前计数**——时序链:`prepare_next_token_ids_padded`(c1 → where → c2 → c3)返回 → `_propose` 入口才施加 clamp。故 c3 数的是 where 原始输出,垃圾是否流出不受 clamp 影响。

```bash
# 宿主机先 git pull(含 engagement 日志),容器内:
VLLM_ASCEND_SD_REVIVE_RACE=1 VLLM_ASCEND_SD_COUNTERS=1 VLLM_ASCEND_SD_DEBUG=clamp_next \
  TIERS=16384 CONCS=1 bash research/run_baseline_npu.sh eagle3 8023
cp experiments/out/serve-npu-bf16-eagle3-k5.log experiments/out/serve-race-run2.log
grep -E "SD-counters" experiments/out/serve-race-run2.log
```

| 观测(Run 2) | 结论 |
|---|---|
| 绿 + c1>0 ∧ c2>0 ∧ c3>0 | **值故事闭合**:撕裂落地→被选中→流出,三环齐证(clamp 只救不改证) |
| 绿 + 全零 | 垃圾从未走值路径,但无 clamp 必崩(Run 1)→ 悖论入档,屏障故事复位且需解释"为何计数 kernel 不是屏障而 clamp kernel 是" |
| 绿 + 混合 | 按 §2.7 矩阵部分解读(如 c1>0∧c2=0:毒落地但未被选中,崩因另寻) |
| 崩 | clamp 在新代码下未能治愈(回归异常),贴 traceback |

### 7.3 Run 2 结果:绿跑但读数丢失 → 读数机制改造(2026-08-25)

**Run 2 结果**(NPU2,commit `23c1eece0`):**绿跑 8/8**(TTFT 194.7ms / ITL 25.8ms / accept 2.4945 双口径一致)——clamp 保活成立,计数器 engaged 行在(06:23:52);**但无读数行**:atexit 与 `__del__` 都没能产出输出。两个候选原因无法区分:①退出路径根本没触发(vllm EngineCore 的 shutdown 链可能绕过 Python atexit);②触发了但 `counts.tolist()`(NPU D2H)或 logger 在 shutdown 阶段已不可用,被 `except: pass` 静默吞掉。数据本身安全地累积在 device buffer 里,只是没取回。

**机制改造**(三层,重跑即见效):
1. **SIGUSR1 活体读数**:`_RaceCounters.__init__` 注册 SIGUSR1 handler——serve 存活期间任何时刻 `kill -USR1 <EngineCore pid>` 即刻 dump 计数(logger 保证健康);彻底绕开退出路径的全部不确定性;
2. **文件双写**:report 同时 append `/tmp/sd_counters_<pid>.txt`;logger 死了文件还在;
3. **异常显形**:D2H 失败不再静默,`READ FAILED: <异常>` 写入同一文件——重跑一次即可区分"没触发"vs"触发但失败"。

harness 联动(`run_baseline_npu.sh` on_exit):**kill serve 之前自动向 EngineCore 发 USR1**(pid 从 serve log 的 `EngineCore pid=` 提取,sleep 2 落盘),SUMMARY 块新增 `counters:` 行(engaged 行除外,末 2 行)。计数未 engage 时 grep 无命中,无副作用。

**重跑指令**(Run 2',与 Run 2 同配置):

```bash
# 宿主机先 git pull(含 USR1 机制),容器内:
VLLM_ASCEND_SD_REVIVE_RACE=1 VLLM_ASCEND_SD_COUNTERS=1 VLLM_ASCEND_SD_DEBUG=clamp_next \
  TIERS=16384 CONCS=1 bash research/run_baseline_npu.sh eagle3 8024
# SUMMARY 块应含 counters: 行;或手工核对:
grep "SD-counters" experiments/out/serve-npu-bf16-eagle3-k5.log
cat /tmp/sd_counters_*.txt   # 兜底文件(含 READ FAILED 诊断)
```

判读矩阵不变(§7.2 表);USR1 在 bench 结束后触发,其后不再产生新数据,读数扰动无影响。

### 7.4 Run 2' 读数与终判(2026-08-25)

**Run 2'**(NPU2,commit `ac2e4ab83`,与 Run 2 同配置 + USR1 机制):**绿跑 8/8**(TTFT 201.3 / ITL 25.8 / accept 2.4945,与 Run 2 一致),SIGUSR1 活体读数成功(serve log 与 `/tmp/sd_counters_<pid>.txt` 双写一致):

```
[SD-counters] sigusr1 steps=843 c1=837 c2=7 c3=1
```

**按 §7.2 矩阵判读:c1>0 ∧ c2>0 ∧ c3>0 → 值故事闭合**(详判与三重矛盾的消解见 §2.7.1)。要点(**注:第 1 条的"毒近乎常驻"解读已被 Run 3 勘误为 -1 哨兵底噪,见 §2.7.1/§7.5;c2/c3 两条不受影响**):

1. ~~c1=837/843:毒近乎常驻~~ → 实为 `get_token_id` 的 -1 哨兵(门关步 backup 恒为 -1,设计如此);
2. c2=7:backup 选择门(cond=false)稀有但非零——洞 3 的实测答案;
3. c3=1:7 次开门 1 次撞上未落地旧值;**无 clamp 时这 1 次即崩溃**(Run 0/1 必崩),clamp 值消毒救下(Run 2/2' 绿);
4. Run 1 的"计数 kernel 不治愈"与"clamp 治愈"合并成钳:治愈靠改值不靠时序 → 屏障假说出局。

### 7.5 Run 3(可选收官):fix 状态下 c1 应归零

值故事预测:默认 fix(blocking)下拷贝同步落地 → buffer 恒合法 → **c1=0**(c2 仍可能有罕见非零,c3 应为 0)。这是"修复消灭源头"的直接读数:

```bash
# 不带 REVIVE_RACE、不带 clamp,纯 fix + counters:
VLLM_ASCEND_SD_COUNTERS=1 TIERS=16384 CONCS=1 \
  bash research/run_baseline_npu.sh eagle3 8025
```

预期绿跑 + `c1=0 c3=0`(c2 允许罕见非零——门本身与竞态无关)。若 c1>0 则说明毒源不止这一处拷贝,需再追(意外但高价值)。

**Run 3 结果(2026-08-25,NPU2,commit `acb66ee80`)**:绿跑 8/8(TTFT 214.7/ITL 33.6/accept 2.4824),读数 `steps=847 c1=840 c2=7 c3=0`——**c3=0 符合预期(fix 阻断逃逸),c1=840 违背预期**。这触发 §2.7.1 的同日勘误:c1 的"毒常驻"解读是错的,837/840 是上游 `get_token_id` 的 -1 哨兵底噪(数值铁证:847-840=7=c2,门开步恰好是 backup 合法的步)。fix 的实证判据由此修正为 **c3=0**(而非 c1=0)。

### 7.6 Run 4:逃逸值捕获(esc 探针,最后一个开放点)

c1 语义修正后,唯一悬而未决的是**逃逸值本身**:stale 哨兵 -1(源码推理的主选)vs CANN 毒 0xa5a5(SDMA 撕裂落地,需重新入场的暗牌)。实现(`esc` 探针,device 侧 min/max 归约,零 host 同步):计数行新增 `c1x`(OOB 排除 -1 哨兵 = 真垃圾计数)与 `esc=[min, max]`(where 后 next_token_ids 中 OOB 元素的值域;无逃逸显示 `esc=none`)。

```bash
# 宿主机 git pull 后,容器内:
# Run 4(主):revive + counters + clamp 保活 —— 复刻 Run 2' 的逃逸条件
VLLM_ASCEND_SD_REVIVE_RACE=1 VLLM_ASCEND_SD_COUNTERS=1 VLLM_ASCEND_SD_DEBUG=clamp_next \
  TIERS=16384 CONCS=1 bash research/run_baseline_npu.sh eagle3 8026
```

| 观测(Run 4) | 结论 |
|---|---|
| 绿 + `esc=[-1, -1]`(或含 -1) | **哨兵故事定谳**:逃逸值 = stale -1,plog 的 0xa5a5 与索引无涉;机制链全闭合 |
| 绿 + `esc=[-1515870811, ...]` | **毒落地复活**:0xa5a5 真在值路径里,SDMA 撕裂/毒写机制需重开调查 |
| 绿 + `esc=none` | 本 run 门开 7 次未撞竞态(1/7 概率,正常)——重跑 1-2 次 |
| 崩 | clamp 回归异常,贴 traceback |

预期解读注:c1x 在 racy 配置下若 >0(排除哨兵后仍有真垃圾)则说明 buffer 里除哨兵外还有别的脏东西——那才是"毒常驻"的真正残部;预期 c1x≈0(哨兵故事下 buffer 只有哨兵与真值)。
