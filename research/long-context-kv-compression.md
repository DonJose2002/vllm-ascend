# 长序列 KV 压缩两条路线:hamming 动态基线 vs 静态真释放(技术版)

> 2026-09-15 整理。范围 = Phase 2 迄今(C 线 ✅ / B 线 B1+B2+B1.5 ✅,B3 待跑)。
> 数据出处:C 线 = `experiments/phase2-cline-report.md`(笔记仓);B 线 =
> `experiments/phase2-kv-compression-design.md` §4 + 本文 §5-§7;判读原始记录 =
> `experiments/server_logs/`。人话版见 `long-context-kv-compression-journey.md`。

## 0. 问题:长序列的成本全部汇聚在 KV

decode 每步要为每个请求重读全部历史 KV,于是长出三本账:

- **显存**:线性增长。5080 51K token 池在 16K/c16 已溢出排队;NPU 285K 才全接
  (Phase 0 "KV 压力交叉点":16K/c16 NPU ITL 31.0 反超 GPU 34.3);
- **带宽**:ITL 斜率 = KV 扫描价签。Phase 0 实算 dense 4K/c1 每步 16.6GB≈11ms,
  32K 89GB≈60ms(SD 的 drafter 再乘 ~6×);
- **质量代价**:任何压缩都用精度换,需要一条"紧度×质量"曲线才知道换得值不值。

Phase 2 的任务 = 把压缩落到 NPU 引擎,量出 accuracy/memory/latency 三维账,
为 Phase 4(SpecKV:草稿压缩+验证恢复)供弹药。两条路线先后落地:
vllm-ascend 内置的 hamming_sparse(C 线,当基线评)与自研的静态真释放
static_kv_compact(B 线)。

## 1. hamming_sparse:查询感知动态压缩(C 线基线)

### 1.1 理论

SnapKV 族思想的"在线动态"变体:**不训练、不改动 attention 算子,靠每步重写
attention 元数据来选择可见 KV 子集**。

- prefill 期:每 token 的 key 经 HashEncoder(默认**随机投影** → 128 bit 指纹,
  LSH 式,无需任何训练产物)算出指纹,存每层的 hashk_cache;
- decode 期:当前 query 同编码,与全部指纹算**海明距离**,按距离选 top-k
  chunk(默认 k=4096 token = 32 chunk;chunk=block=128),外加 sink(首块)/
  recent(末块)锚点;
- 选块结果直接**重写 block_table 与 seq_lens** 交给 FIA——attention 算子读的
  就是"被压缩的序列",自身零改动。

直觉:海明距离小的指纹 ⇒ key 与 query 方向相近 ⇒ 该块注意力权重大 ⇒ 值得保留。
随机投影是这个方法的"免训练"卖点,也是它的精度天花板来源。

### 1.2 实现(行号级核读,2026-08-31)

| 环节 | 代码 |
|---|---|
| prefill 指纹缓存 | `attention_v1.py:817`,`reshape_and_cache_kvcomp` |
| decode 每步重选 | `attention_v1.py:819-826` / `attention_utils.py:97-142`,自定义算子 `npu_hamming_dist_top_k` |
| 配置面(**双键**,缺一 KeyError) | `enable_hamming_sparse` bool(attention 层门,attention_utils.py:149)+ `hamming_sparse{enabled,sparse_json_location}`(runner 门,ascend_config.py:344-347);json 强制 |
| 紧度主旋钮 | `vllm_hash_attention_topk`(默认 4096);`seq_len_threshhold` 2048 以下不触发 |

三个关键定性(全部为 Phase 2 的靶子):
1. **在线动态**:每步按当前 query 重选——精度上限高(查询感知),但每步都要付
   一次选择成本;
2. **只省带宽不省显存**:kvcomp 代码零 free/evict,被排除 block 仍占 HBM,
   hashk_cache 反倒贴 ~0.4%——它是 latency 优化,不是 memory 优化;
3. **与 SD 互斥 = 静默守卫**(attention_utils.py:151 / model_runner_v1.py:613,
   连 warning 都没有),结构性根因四条:单 query 选块 vs R×(K+1) query /
   压缩 seq_lens 破坏 SD 记账契约 / hashk 不懂多 token+回滚 / drafter 未接线。

### 1.3 埋雷(未文档化 + 零 e2e 的代价)

- `vllm_hash_attention_skip_layers` 被两处不兼容语义消费(层号列表 `in` 读法 vs
  逐层 mask `[idx]` 读法),上游默认 `[]` 在**任何模型首个 decode 前向必崩**
  IndexError;唯一双读法安全值 = `[null]×36`(修复 `514b86b2e`;上游一行修复
  候选 = PR #6 素材);
- FULL 图 capture 期 kvcomp decode 路径 kernel 越界(aivec)——该特性在
  上游 #12049(07-18)已被整体移除,v0.23.0 里的是未修完的孤儿初版;
  → C 线全部 cell 被迫 eager 口径。

## 2. C 线效果(2026-09-03 判读闭环)

12+2 cell(topk 2048/4096/8192 × 16K/32K × c1/c16 + 互斥确认),全部 eager。

### 2.1 latency:eager 遮蔽 + host 税,带宽收益物理测不到

- **eager 遮蔽(核心发现)**:eager 的 host 开销 ~72ms/步成为 ITL 地板
  (dense 四 cell 平坦 72.6/72.4/79.8/79.7),Phase 0 图模式同 cell 有斜率
  (18.4→44.9,tier 斜率 13.9ms 被抹成 −0.1)——**一切低于地板的 device 侧
  差异(含压缩省下的 KV 扫描)不可见**;唯一漏出痕迹 = 4096@32K/c16 +17.1;
- **hamming host 税 +17~24ms(1.2-1.3×)**:每层 ~7-9 次算子发射 × 36 层 +
  每步 `tolist()` 同步,与 topk/tier/conc 全无关的常数;
- **topk8192 判死**:32K/c1 引擎死(64 chunk 越过 nightly 算子唯一测点
  k=48 的覆盖边界);其 digest NIAH 0.000 = 引擎先死、NIAH 全灭的**失败记账
  伪影**,非实测垃圾(needle_eval 失败按 miss 记)。

### 2.2 accuracy:topk4096 甜点 + 绝对预算下限 ~4K

| 配置 | 16K NIAH | 32K NIAH | 保留率 |
|---|---|---|---|
| topk4096 | **1.0(20/20)** | **1.0(20/20)** | 26% / 13% |
| topk2048 | 0.9 | 0.2(塌) | 13% / 6.6% |

- 同 13% 保留率下绝对 2048(0.9)vs 4096(1.0)分化 → 质量不由比值唯一决定,
  **绝对预算下限 ~4K 规则**(选型按 max(4096, 15%));Phase 4 草稿侧激进区间
  2048-4096 有据;
- 随机投影指纹在 ≥4K 预算已满分——**选择器在动态形态下不是瓶颈**。

### 2.3 memory:零释放确认

KV 池自报 285,696 tokens 三配置不变;hashk 倒贴 +0.4%。预期坐实 = B 线立靶完成。

### 2.4 互斥:行为级证实

hammingsd(hamming+ngram)≡ 纯 ngram(ITL Δ≤2.7ms、accB 逐位一致)vs
hamming-only +22ms——静默关闭,零 warning(文档级发现,Phase 4 引用)。

### 2.5 C 线裁决

hamming 被证伪的**不是精度**(4096 预算动态选择满分)而是**工程形态**:
每步重写 = host 税 +22ms + 图捕获必崩 + eager 地板锁死带宽收益。B 线因此定案
"prefill 末一次性静态选择 + 真释放",不带动态。

## 3. B 线:static_kv_compact 静态真释放(我们的驱逐策略)

### 3.1 理论

SnapKV 正统形态:**一个请求的 prefill 一结束,一次性决定保留哪些 KV 块,把其余
块物理释放回 KV 池;之后整个 decode 只读保留块组成的"视图",零选择成本**。

与 hamming 的三点对立:静态一次性(零每步税)vs 每步动态;真释放(显存+容量
两本账都省)vs 零释放;选择器可换(机制与策略解耦)vs 绑定 hash 基建。

### 3.2 设计约束(每条绑定 C 线证据)

1. **图兼容**:选择只在 prefill 完成后做一次;decode 路径零 python、零 tensor
   改写 → 可捕获(hamming 每步重写 = capture 必崩的反面);
2. **真释放**:被汰 block 回 KVCacheManager free → 池等效容量 topk4096 档
   ~8× @32K;
3. **紧度默认 max(4096, 15%)**:绝对预算下限规则;
4. **收益叙事锚定 KV 压力角落**(高并发×长上下文):图模式反事实 32K/c16 ~2×;
5. ~~选择器起步从简~~(**2026-09-10 B2 实证修正**):静态选择器 NIAH = 保留率
   数学期望(0.20),hamming 满分是"每步重选"的性质不遗传——**选择器升级为
   主矛盾** → B1.5 观测窗投票(§4.2)。

### 3.3 实现:三件套(`52e7c7efe`)

- **协调器** `vllm_ascend/worker/static_kv_compact.py`:记录注册表 + 选择器 +
  manager 手术 + 视图装配(纯 stdlib+torch,CPU 可测);
- **调度器钩子** `patch/platform/patch_static_kv_compact.py`:包
  `Scheduler.update_from_output`(压缩决策,严格步间)+ 包 `KVCacheManager.free`
  (请求结束遗忘记录);
- **引擎钩子** `model_runner_v1.py _initialize_attn_metadata`:有活跃记录时用
  scratch 视图覆盖 block_table_tensor + seq_lens;**无记录时 dense 路径字节
  不变(零开销)**。

三个关键实现机制(经 pinned v0.23.0 源码核读定稿):

1. **null 保位释放**(不是"从列表删除"):被汰位置置 `_null_block`(镜像 SWA
   `remove_skipped_blocks`)——列表长度与全部位置算术(追加/准入/增长)保持
   完整;uncached 块 `free_blocks(prepend=True)` 立即可复用 → **物理池真释放,
   admission 记账不变**;
2. **引擎行留洞 + 视图 gather**(不是"重写引擎行"):slot mapping 按全量
   position 索引行(pos//128),行不可紧凑化;引擎行原样保留,attention 元数据
   层做 gather(keep 位置 ++ 压缩后追加的尾块,尾部零填充);seq_lens 视图 =
   `optimistic_seq_lens − dropped_tokens`,**自动继承乐观 +1 语义,无 off-by-one**;
   positions/num_computed_tokens/is_prefilling 保持全量真值;
3. **同进程前提 + 结构性门禁六条**:world_size==1(调度器与 runner 同进程,
   否则手术与视图永不相遇)/ async scheduling off(vllm 0.23.0 默认 async-ON
   级联,09-09 零事件之谜根因)/ SD off / prefix caching off / 单 full-attention
   KV 组 / 无 kv-connector。

**图兼容赌注**(B2 第一验证点):ACL graph 对 forward args 按地址捕获,但
block_table/seq_lens 走 `update_attn_params` 元数据通道逐层更新(参数本就每步
变值);scratch 缓冲一次分配跨步稳定 → 图模式可行,eager 无条件正确。

## 4. 选择器两代

### 4.1 stride 占位(B1/B2):机制验证载体,质量判死

- 算术:sink 1 块 + recent 4 块 + 等距 stride 补足 max(4096, 15%)——刻意保守,
  用于先把释放/视图/图兼容机制跑通;
- **B2 效果(2026-09-10 四跑定案)**:机制 3/3 全过——图兼容 ✓(PIECEWISE+FULL
  各 35/35,bench+20 NIAH 全程无崩)、**真释放 ✓**(28 事件:16K 保 32 释 91/
  32K 保 37 释 210,kept_tokens 3994/4688 ≈ budget 自洽)、**视图生效 ✓**
  (NIAH 1.0→0.20 即铁证,反证此前满分 = 视图未装);
- **NIAH 0.20 = stride 的数学期望,不是机制问题**:单块存活概率 = 保留率
  (16K≈26%/32K≈15%),命中按深度聚簇(16K@0.9 近尾窗 2/2、32K@0.25 踩中
  stride 2/2,其余全 0)→ stride 判死,选择器升级 = B1.5;
- 副产物:async 调度税 2.9ms@16K/c1(dense async-off 21.2 vs async-on 18.3)、
  compact −1.4ms 露头(19.8 vs 21.2)。

### 4.2 dwvote(B1.5):decode 窗口注意力投票

- **理论**:查询感知的唯一来源是"某个 token 的真实 q 过一遍模型"。prefill 窗口
  挂钩被 PIECEWISE 图 replay 杀死(无 python),临时前向需手建 ~20 字段 NPU
  元数据——排除法后选定 **decode 窗口投票**:请求 prefill 完成后头 D=8 个
  decode 步强制 eager,每层 self_attn 挂 forward hook,用该步 decode token 的
  真实 q(复刻 qwen3.py qkv_proj→q_norm→rotary 精确路径)× 该层 K cache
  (GQA 分组)算 softmax 注意力,按块 max 聚合、跨层跨步累加;D 步后取
  top-预算块 + sink/recent 锚定手术。零票/异常自动回落 stride(正确性兜底);
- **NIAH 可救的机理**:针类问题的答案在 prompt 内,模型生成头几个 token 时
  会强 attend 针所在块 → 针块赢投票;
- **实现**:`kv_compact_voting.py`(vllm-free,CPU UT 可导)+ static_kv_compact
  的 PENDING_VOTES 状态机 + model_runner 两个调用点(needs_eager_step 覆写
  cudagraph_mode=NONE + raw_model_forward 替换 _model_forward)。

### 4.3 排障史(run 5-9,编译包裹链四根因,全部钉死)

| run | 根因 | 修复 |
|---|---|---|
| 5 | FULL 图 `splitting_ops=[]` → attention 在编译区内,CUDAGraphMode.NONE 只跳 replay 不跳编译体,hook 永不触发 | 包 `torch.compiler.disable()` |
| 6 | torch≥2.6 `compiler.disable()` 作 ctx manager 直接抛 → 引擎死 | 换 `set_stance("force_eager")` |
| 7 | stance 对 `TorchCompileWithNoGuardsWrapper.__call__`(guard 全弃/bytecode 直执)无效,永远到不了 python 前向 | 直调 `.forward` |
| 8 | 无 error/无 vote(旧日志探针未覆盖,取证定界) | 四角探针(4fe748beb/a62ad75a1) |
| 9 | **装饰器在内层**:`@support_torch_compile` 挂在 Qwen3Model 非 ForCausalLM 外壳,外壳 forward 第一句 `self.model(...)` 即撞进内层编译派发,36 层 hook 构造性不可能触发(raw start/end OK 62ms eager 步 + hook 零触发 + ITL 钉死 eager 地板) | `b4f6c4304`:下沉一层直调内层原始 `.forward`(`original_code_object` 混入标记)+ `with_kwargs=True`(Qwen3 关键字调 self_attn 第二颗雷) |

## 5. 效果总账(hamming vs dwvote,同 NPU/同模型/同 harness)

| 维度 | hamming sparse(C 线) | static_kv_compact + dwvote(B 线,run 10) |
|---|---|---|
| 选择时机 | 每步动态重选(查询感知) | prefill 末一次性静态(decode 窗投票) |
| **accuracy(NIAH)** | topk4096 **1.0**(16K/32K) | **1.0(20/20,vs stride 0.20)** |
| **memory** | **零释放**(池 285,696 不变,hashk +0.4%) | **真释放**:16K 释 91/123 块、32K 释 210/247 块;用量轨迹 5.5%→1.5%→0% 可见 |
| **latency** | eager 遮蔽测不到收益;自身 host 税 **+17~24ms** 常数 | **ITL 19.6 vs dense 21.6(−2.0ms,物理可见)**;代价 = 头 8 步 eager 窗(itl99 110ms 尾巴) |
| **图兼容** | eager-only(FULL capture 期 aivec 崩) | PIECEWISE+FULL 各 35/35,全程无崩 |
| SD 互斥 | 静默守卫 | 门禁排除(研究范围一致) |
| 净收益公式 | 带宽省(理论)− host 常数税 = 净负(eager 口径) | **带宽省 − 头 D 步 eager 窗口税**,B3 c16 分 cell 量化 |

## 6. 遗留与下一步

1. **B3 全矩阵**(16K/32K × c1/c16,dwvote/stride/dense 同口径,正式记录前干净
   全量重装):量化净收益公式,重点 32K/c16 KV 压力角落(图模式反事实 ~2×);
2. B4 B 线报告:三维账 vs hamming 收口;
3. 端侧外推:静态一次性形态天然适配端侧(无每步税、池浅时释放收益放大);
   dwvote 的 D 步 eager 窗在端侧 host 更弱时税更重,D 调参与窗内并发是端侧变量;
4. Phase 4 接口:静态压缩与 SD 的互斥是方向 A 靶子(压缩 seq_lens 破坏 SD 记账
   契约等四条,见 §1.2);紧度×质量曲线(绝对预算下限 ~4K)是草稿侧选型输入。
