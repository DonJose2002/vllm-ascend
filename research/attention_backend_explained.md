# Attention Backend 通俗版:vllm-ascend 是怎么在昇腾上做注意力的

> 本文是 `attention_backend_arch.md`(严谨版,带文件:行号)的叙事版。适合快速建立直觉;查细节、写代码时请回看严谨版。
> 基于版本:vllm-ascend v0.22.1rc1 + vllm 0.22.1。

---

## 1. 从一个大问题说起:GPU 代码怎么跑到 NPU 上?

vllm 出生时满脑子都是 CUDA:attention 调 FlashAttention 算子,显存管理问 CUDA 要,图捕获用 CUDA Graph。

昇腾 NPU 没有这些东西,但它有自己的一套家当:torch_npu 提供的官方算子、自己的内存分配器、自己的图机制(ACL Graph)。

vllm-ascend 的做法不是"把 vllm 重写一遍",而是当**翻译官 + 器官移植医生**:

- 对 vllm 说:"你看到的'平台'还是那个平台,接口都一样"(这就是 `NPUPlatform`,继承了上游的 `Platform` 基类);
- 背后把每个关键器官换成 NPU 版:attention 换成 torch_npu 算子、worker 换成 `NPUWorker`、CUDA Graph 换成 ACL Graph;
- 换不了的就打补丁(`patch/` 目录,直接替换上游类的方法)。

所以读这份代码的正确姿势是:随时问"这一步,上游 vllm 原来是怎么做的,昇腾换成了什么?"

---

## 2. 一个请求进来,谁来做 attention?

vllm-ascend 不只有一个 attention backend,而是一个**小家族**,按模型特征自动分派。你可以把它想象成医院分诊台:

| 病人(模型特征) | 分诊结果 | 一句话 |
|---|---|---|
| 普通模型(Qwen/Llama 等 GQA) | `AscendAttentionBackend` | 全科医生,90% 的请求到这 |
| DeepSeek 系(MLA) | `AscendMLABackend` | 专科:处理 MLA 的吸收/不吸收两种形态 |
| DeepSeek V3.2 稀疏注意力(配置里有 `index_topk`) | `AscendSFABackend` | 专科:NSA 稀疏注意力 |
| DSA 压缩 KV 的模型 | `AscendDSABackend` | 专科:压缩KV的注意力 |
| 训练-推理要数值完全一致 | `AscendFABackend` | 保守方案:慢但对得齐(要额外装 flash_attn_npu_v3) |

分诊逻辑在 `platform.py` 的 `get_attn_backend_cls`,就三个判断:用不用 MLA?用不用稀疏?用不用压缩?三维坐标一查表,完事。

还有个特殊角色:**上下文并行(CP)**。当长序列被切到多卡上时,不换 backend,而是把 impl 换成 `AscendAttentionCPImpl`——同一个壳,换了个发动机。

---

## 3. KV Cache:仓库里的固定货架

KV cache 就是"已经算过的 token 的记忆",存在一个巨大张量里。形状是:

```
(2, num_blocks, 128, num_kv_heads, head_size)
      ↑      ↑
   key/value  每个货架固定放 128 个 token
```

**为什么是 128?** 因为底层 NPU attention 算子只认 128。`get_supported_kernel_block_sizes()` 返回的就是 `[128]`,没有商量余地。上游 vllm 常见的 block 16/32 在这里会被强制刷新成 128。

请求的 KV 不是连续存放的,哪个 token 在哪,记在 `block_tables` 里(每行是一个请求的"货架编号清单")。**记住这张表,后面讲 KV 压缩时会看到:改这张表 = 改 attention 看到什么记忆。**

---

## 4. 五种状态:请求的一生

每个调度步,batch 会被打上一个状态标签(`AscendAttentionState`),决定 attention 走哪条路:

- **PrefillNoCache**:全新请求,从头算 prefill。K/V 就是现算的,不用翻仓库。
- **PrefillCacheHit**:前缀命中(比如系统提示词之前算过),prefill 时要读仓库。
- **ChunkedPrefill**:长 prefill 被切块,逐块算。这是默认状态。
- **DecodeOnly**:每个请求只算 1 个新 token。最常见、最热路径。
- **SpecDecoding**:投机解码的验证步——目标模型一口气算 1+K 个 token。

有个精妙的细节:**"算几个 token"才算 decode?** 阈值不是 1,而是 `1 + draft_token 数`。也就是说投机解码的验证步(一次算多个 query)也被当作"decode 类"处理。但这个数有天花板:**16**。NPU 的 FIA 算子(TND 布局)单请求最多 16 个 query,所以 draft token 最多配 15 个。这是硬件级约束,不是配置能绕开的。

---

## 5. 两条计算路径:大路和小路

真正算 attention 时,backend 在两条路里选:

**大路(默认):FIA(Fused Infer Attention)**
算子 `torch_npu.npu_fused_infer_attention_score`,一个算子把 Q·K·softmax·V 全干完,支持 paged KV(通过 block_table)、causal mask、滑动窗口。几乎所有场景都走它。它还有个 v2 变体,多了个"可学习的 sink"(Qwen3-next 这类模型需要)。

**小路(特定条件):PA(Paged Attention)**
算子 `torch_npu._npu_paged_attention`,decode 专用快车道。但上这条路要同时满足一堆苛刻条件:
1. 没开投机解码
2. 不是 950(A5)芯片
3. 图模式恰好是"只给 decode 建图"(FULL_DECODE_ONLY)
4. 当前 batch 大小在配置的白名单里

**推论:一开投机解码,PA 就没了,必然回 FIA。** 做 Phase 1 实验时要记得这个联动,别把性能变化归因错了。

---

## 6. ACL Graph:录好的计算怎么应对每步都变的数据?

CUDA 的痛点:decode 每步的 seq_lens、block_table 都在变,纯录制的图没法直接重放。

GPU 上 vllm 的做法是把 attention 排除在图外。**昇腾的方案更激进也更优雅:attention 也录进图里,但每次重放前"打补丁"。**

流程像这样:

1. **捕获时**:每层 attention 的参数张量(用弱引用存,省内存)登记在册,算子调用被 `graph_task_group_begin/end` 包住录进图;
2. **重放时**:每步在一条更新流上,对每个登记过的任务执行"开始打补丁 → 用最新的 seq_lens/block_table 重发一遍算子 → 结束打补丁",运行时会把这些新参数塞进图里的对应位置。

效果 = 图的重放开销 + 数据的动态性,两头都要。

代价也有:更新时用的是 **CPU 上的 list**(seq_lens_list),意味着每步有 D2H 拷贝。这是 NPU 算子 API 设计带来的税。

已知的坑(严谨版 §4 有细节):滑动窗口(SWA)模型在全图重放时不能刷新 block_tables,会输出错乱——代码里直接写死了"保持捕获时的表"。

---

## 7. 意外之喜:KV 压缩已经有了半成品!

Phase 2 要做的 KV 压缩,其实 0.22.1 已经内置了一套(叫 hamming sparse,配置项 `enable_hamming_sparse`),思路相当聪明:

1. **prefill 时**:每层把 key 做个哈希编码(把每个 chunk 的 key 压成一个短指纹),存进并行的指纹仓库;
2. **decode 时**:自定义算子 `npu_hamming_dist_top_k` 拿 query 的指纹去和所有 chunk 指纹比距离(汉明距离),**选出最相关的 top-k 个 chunk**;
3. 拿这 k 个 chunk **重新拼一张 block_table、改写 seq_lens**——attention 算子完全不知道发生了压缩,它只是"看到"了一个短一点的记忆。

这个设计的精华在第 3 步:**压缩 = 改表,不动算子**。任何"挑选哪些 KV 保留"的策略(SnapKV、H2O……)都可以套这个壳。这直接给 Phase 2 铺了路。

但要泼一盆冷水:**它和投机解码是互斥的**(代码里显式判断 speculative_config 就关掉)。如果 Phase 4 想做"草稿激进压缩 + 验证恢复"(SpecKV 思路),要打破的就是这个互斥。

---

## 8. INT8 KV:已经落地,不用从零开始

另一条已铺好的路:INT8 KV cache(`AscendC8AttentionBackendImpl`,配 QuaRot 类模型用)。KV 在写进仓库前量化成 INT8,attention 算子用反量化参数直接算。还有配套的 NZ 格式(昇腾特有的内存排布,尾维固定 32)。

对 Phase 3 的"drop + INT8 hybrid"来说,这意味着 INT8 这半边是现成的,只需要做 drop 那半边。

---

## 9. 对我们三个 Phase 的白话结论

**Phase 1(投机解码):**
- draft token 数量上限 15(FIA TND 限制),配置再大也没用;
- 开了投机解码就告别 PA 快道,全走 FIA——测加速比时要意识到你的对照组也在变化;
- 拒绝采样(accept/reject 的核心)在 `patch/worker/patch_rejection_sampler.py`,是 NPU 特有实现;
- draft 模型的图和目标模型的图是两套独立管理的,还处理了 DFlash 打乱层序的 hack——说明这块已经很复杂,改进空间和坑都不少。

**Phase 2(KV 压缩):**
- 别改 attention 算子,改 block_table + seq_lens 就够了(hamming sparse 已验证此路通);
- block_size = 128,设计 eviction 策略时 chunk 粒度天然对齐到 128;
- 已有 INT8 KV 可以叠加;
- block 的生命周期管理在 `patch_kv_cache_*.py`,实现前先读。

**Phase 3(MoE):**
- 已有 `patch_routed_experts_capture.py`(捕获专家路由的钩子),benchmark 专家分布可以直接从这下手;
- 模型级魔改的合法姿势是往 `patch/worker/` 加文件,参考 deepseek/qwen3 现有 patch。

---

## 10. 一图流总结

```
请求进入
   │
   ▼
platform.py 分诊 ──→ 选 backend(MLA?稀疏?压缩?普通?)
   │
   ▼
batch 打状态标签(5 种,prefill 系 / decode 系)
   │
   ▼
decode 系 + 满足全部苛刻条件? ──是──→ PA 小路(_npu_paged_attention)
   │否
   ▼
FIA 大路(npu_fused_infer_attention_score / v2)
   │
   ├─ 开了 hamming sparse? → 先改写 block_table(top-k chunk),再进 FIA
   ├─ C8 量化模型? → INT8 KV + NZ 视图进 FIA
   └─ 图捕获中? → 参数登记 + graph_task_update 打补丁重放
```

下次读码路线图(按性价比排序):`mla_v1.py` 等专科 backend(做 MoE 再看)→ `_EXTRA_CTX` 全局态(理解 draft/target 怎么切换)→ `patch_kv_cache_coordinator.py`(Phase 2 前置)。
