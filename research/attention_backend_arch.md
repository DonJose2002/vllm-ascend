# vllm-ascend 0.22.1rc1 Attention Backend 与 Plugin 机制源码分析

> 适用版本:vllm-ascend v0.22.1rc1(= 本仓库 `research/main` 基底)+ vllm 0.22.1
> 姊妹篇:`spec_decode_eager_flow.md`(投机解码 eager 流程)
> 本文所有 `文件:行号` 基于 `vllm_ascend/` 包内路径。

---

## 0. 一句话总览

vllm-ascend 不重写 attention,而是通过 **hardware plugin 接口**替换平台层,再把所有 attention 计算收敛到 **torch_npu 官方算子**(FIA = `npu_fused_infer_attention_score`,PA = `_npu_paged_attention`),用 **AscendMetadata 的 5 状态机**统一 prefill/decode/spec-decode 的路径分派;ACL graph(对应 CUDA graph)通过 **graph_task_update** 机制解决动态 shape 重放问题。

---

## 1. Backend 家族与选型(谁在什么条件下被选用)

选型入口 `platform.py:784 get_attn_backend_cls`,key = `(use_mla, use_sparse, use_compress)`:

| 条件 | Backend | 文件 | 用途 |
|---|---|---|---|
| FA3 被选 + `use_batch_invariant` + 装了 `flash_attn_npu_v3` | `AscendFABackend` | `attention/fa3_v1.py` | 训练-推理一致性场景(数值对齐),牺牲性能 |
| (T,F,F) | `AscendMLABackend` | `attention/mla_v1.py`(1804 行) | DeepSeek MLA(Q 吸收/不吸收两路) |
| **(F,F,F)** | **`AscendAttentionBackend`** | **`attention/attention_v1.py`(1783 行)** | **通用 GQA/MQA——90% 模型走这里** |
| (T,T,F) | `AscendSFABackend` | `attention/sfa_v1.py` | DeepSeek V3.2 NSA 稀疏注意力(hf config 有 `index_topk` 即触发) |
| (T,F,T) | `AscendDSABackend` | `attention/dsa_v1.py` | DSA(compress KV,`DSAAttentionImpl` 接口在 `attention/abstract.py`) |
| 310P | `AscendAttentionBackend310` | `_310p/attention/` | 310P 专用阉割版 |

注册方式:`attention_v1.py:73` `@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")`。
名字 HACK:`get_name()` 在 v2 runner 下伪装成 `"FLASH_ATTN"` 绕过上游断言(attention_v1.py:82)。

**CP(context parallel)不单列 backend**:`get_impl_cls()` 在 `enable_cp()`(PCP 或 DCP > 1)时换 `AscendAttentionCPImpl`(attention_v1.py:86-90),builder 同理。

---

## 2. 核心数据结构

### 2.1 KV Cache 布局(attention_v1.py:101)

```
(2, num_blocks, block_size=128, num_kv_heads, head_size)   # key 与 value 分离存放
```

- `get_supported_kernel_block_sizes() → [128]`(attention_v1.py:139)——**只支持 block 128**,这是 NPU 算子的硬约束(刷新逻辑在 `utils.py refresh_block_size`)。
- cache 是**逻辑 view**:FIA 调用时按 `key_cache.view(num_block, block_size, -1)` 拍平(num_kv_heads*head_size 合并,attention_v1.py:1015-1020),NZ 格式(C8 路径)用 `_nz_5d_view` 重排为 `(blocks, kv_heads, head_size//32, block, 32)`(attention_v1.py:1441,NZ 尾维固定 32)。

### 2.2 五状态机 `AscendAttentionState`(attention_v1.py:143)

```python
PrefillNoCache=0   # 无前缀,纯 prefill,K/V 直接用新算的(不读 cache)
PrefillCacheHit=1  # 前缀命中,prefill 要读 cache
DecodeOnly=2       # 纯 decode(query_len==1,或 spec=1)
ChunkedPrefill=3   # 分块 prefill(默认状态)
SpecDecoding=4     # 投机解码验证(target 一次吃 1+draft 个 token)
```

状态决定 `_get_fia_params`(attention_v1.py:985)返回的 key/value/block_table:
- PrefillNoCache:不需要 cache,直接截断 `key[:num_tokens]`(attention_v1.py:1082)
- 其余:从 `self.key_cache` view 出来 + `attn_metadata.block_tables`

### 2.3 AscendMetadata(attention_v1.py:151)

每层共享一份。关键字段:
- `attn_state` / `attn_mask`(mask 来自单例 `AttentionMaskBuilder`,attention_mask.py)
- `seq_lens` / `seq_lens_list` / `actual_seq_lengths_q`(FIA 要 **CPU list**,这是 NPU 算子 API 决定的——注意 H2D/D2H 开销)
- `block_tables` / `slot_mapping` / `query_start_loc`
- `kvcomp_metadata`(KV 压缩,见 §6)
- `reshape_cache_event`(PD 分离 producer 场景的 cache 写入事件)

### 2.4 decode 阈值与 spec decode 耦合(attention_v1.py:242-253)

```python
decode_threshold = 1 + num_speculative_tokens   # query_len ≤ 此值都算 decode
assert decode_threshold <= 16   # FIA TND layout 的硬上限!
```
**端侧注意**:NPU FIA 算子单请求 query 长度上限 16——draft token 数(K+1)不能超过 16,这是 spec decode 方法在 NPU 上的一个物理约束。

`split_decodes_and_prefills`(utils.py:273)按阈值把 batch 前段 decode / 后段 prefill 分开(reorder_batch_threshold 保证 decode 排前面)。

---

## 3. 两条计算路径与分派

### 3.1 分派逻辑 `forward_impl`(attention_v1.py:1258)

```python
if attn_state == DecodeOnly and using_paged_attention(num_tokens, cfg) and 无SWA:
    forward_paged_attention()      # PA 路径
else:
    forward_fused_infer_attention()  # FIA 路径(几乎一切场景)
```

### 3.2 PA 路径条件(utils.py:44)

`using_paged_attention` 要同时满足:
1. **无 spec decode**(speculative_config is None)
2. **非 A5(950)** 芯片
3. `cudagraph_mode == FULL_DECODE_ONLY` 且 runtime batch 在 `ascend_config.pa_shape_list`

→ PA 是"decode 专用图"的优化路径,算子 `torch_npu._npu_paged_attention`(attention_v1.py:1174)。**开 spec decode 就必然回 FIA**——Phase 1 实验时注意这个联动。

### 3.3 FIA 路径(attention_v1.py:1045)

三个算子变体:

| 算子 | 触发 | 特性 |
|---|---|---|
| `npu_fused_infer_attention_score` | 默认 | `input_layout="TND"`;`sparse_mode`: 0=mask 显式, 3=causal, 4=slidingWindow(pre_tokens/next_tokens);SWA 用 `pre_tokens=window` |
| `npu_fused_infer_attention_score_v2` | `sinks is not None`(Qwen3-next 类) | 支持 `learnable_sink`(attention_v1.py:1093) |
| `npu_fusion_attention` | encoder-only/pooling | 纯 FlashAttention,无 paged cache(attention_v1.py:1196) |

workspace 管理:按 num_tokens 缓存(`_npu_fused_infer_attention_score_get_max_workspace`,attention_v1.py:741)。

---

## 4. ACL Graph(全图捕获)机制——与 CUDA graph 的关键差异

目标:decode 全图捕获(FULL mode)时,attention 算子被录进 graph,但 **seq_lens/block_table 每步都变**。NPU 方案(attention_v1.py:680 `full_graph_fia` / attention_v1.py:400 `update_graph_params`):

1. **捕获时**:每层 attention 参数张量以 **weak_ref** 存进 `graph_params.attn_params[num_tokens]`(attention_v1.py:771-798),用 `torch.npu.graph_task_group_begin/end` 包住算子调用;
2. **重放时**:每步在 update stream 上,对每个 task `graph_task_update_begin` → 重发算子(带最新 seq_lens/block_table,来自当步 attn_metadata)→ `graph_task_update_end`,由 runtime 替换 graph 内的参数(attention_v1.py:419-464)。

要点:
- 重放更新用的是 **CPU list**(`seq_lens_list`/`actual_seq_lengths_q`,attention_v1.py:518-519)——每步有 D2H;
- draft model 有独立的 graph params(`get_draft_graph_params` / `_prefill`),DFlash 等 spec 方法会打乱 target 模型 attn_keys 层序,用正则重排(attention_v1.py:557-572);
- **SWA 模型在全图重放时不能刷新 block_tables(会产出错乱,attention_v1.py:627-636)**——已知坑;
- piecewise 模式下 attention 层(`vllm::mla_forward`/`vllm::dsa_forward`)加入 splitting_ops(platform.py:611)。

---

## 5. Plugin / Patch 机制(改上游不 fork 上游)

### 5.1 平台注册(platform.py:126)

`NPUPlatform(Platform)`,`_enum=OOT`,入口链:
- `pre_register_and_update`(platform.py:182)→ `adapt_patch(is_global_patch=True)`(全局 patch,utils.py:511)+ 往 `--quantization` choices 注入 "ascend"
- `check_and_update_config`(platform.py:448):巨量 NPU 特有配置修正(enforce_eager→CompilationMode.NONE、block_size 刷新、cudagraph sizes 重算、worker_cls 替换为 `vllm_ascend.worker.worker.NPUWorker`,platform.py:647-657)
- 编译后端:`AscendCompiler`(platform.py:174,`use_inductor=False`),图包装 `ACLGraphWrapper`(platform.py:861)
- `import_kernels`(platform.py:768):lazy 注册 `vllm_ascend_C` 自定义算子(RL 场景可见性问题)

### 5.2 patch 目录(platform.py:182 之外的按需 patch)

- `patch/platform/`:调度/KV cache 管理/分布式级——`patch_scheduler.py`、`patch_kv_cache_coordinator.py`、`patch_kv_cache_interface.py`、`patch_speculative_config.py`(Phase 1 关注)、`patch_rejection_sampler` 在 worker 侧
- `patch/worker/`:模型级——`patch_deepseek_mtp.py`、`patch_qwen3_dflash.py`、`patch_draft_quarot.py`、`patch_rejection_sampler.py`(**Phase 1 核心:拒绝采样在 NPU 上的实现**)、`patch_routed_experts_capture.py`(MoE 专家捕获,Phase 3 参考)

改动上游行为的合法姿势 = 新 patch 文件(猴子补丁目标类方法),见仓库根 AGENTS.md 的 Patching Requirement。

---

## 6. KV Compression 已有实现:Hamming Sparse(Phase 2 直接相关!)

`additional_config.enable_hamming_sparse` 开启(kvcomp_attn/attention_utils.py:145),流程:

1. **prefill/chunked**:每层对 key 做 hash 编码,`npu_reshape_and_cache_bnsd` 存入并行 hash cache(attention_v1.py:692 → attention_utils.py:72-94)
2. **decode**:`npu_hamming_dist_top_k` 自定义算子(query hash 与 cache hash 求 Hamming 距离)**选 top-k chunk,重建压缩后的 block_table + seq_lens**(attention_v1.py:694 → attention_utils.py:97-142),attention 本体不变——**只算选中的 chunk**
3. 支持 sink + recent 窗口;每请求独立 topk
4. **与 spec decode 互斥**(attention_utils.py:151)——Phase 4 若做 Spec+KV 压缩融合,这就是要打破的点

**启示**:改写 `block_tables + seq_lens` 是 NPU 上实现 KV eviction 的低侵入路径(attention 算子无需改),SnapKV 式策略可复用这套钩子;hash 编码 + hamming top-k 全部走自定义算子 `torch.ops._C_ascend.*`(`_cann_ops_custom/`)——triton-ascend 之外还有 C++ 算子注册通道。

## 7. INT8 KV Cache(C8/QuaRot)路径

`AscendC8AttentionBackendImpl`(attention_v1.py:1346,由 `quantization/methods/kv_c8.py` 做类替换激活):
- KV 写入前量化为 INT8(静态 per-channel scale)
- FIA 反量化参数 `key_antiquant_scale/mode=0` + `inner_precise=1`,NZ 格式 BNSD view(attention_v1.py:719-738)
- **NPU INT8 KV 已是落地特性**——Phase 3 的 "drop + INT8 hybrid" 有现成底座

---

## 8. 对三个 Phase 的直接影响清单

| Phase | 结论 |
|---|---|
| **P1 Spec** | FIA TND 限制 draft≤15;开 spec 即失去 PA 路径;拒绝采样在 `patch/worker/patch_rejection_sampler.py`;draft/target 全图捕获已支持但复杂(双 graph params + DFlash 层序 hack) |
| **P2 KV** | block_size 硬编码 128;eviction 低侵入点 = 改 block_table+seq_lens(hamming sparse 已验证此路);已有 INT8 KV(C8)可叠加;`patch_kv_cache_*` 管理 block 生命周期 |
| **P3 MoE** | `patch_routed_experts_capture.py` 已有专家路由捕获钩子;EPLB 目录存在(`vllm_ascend/eplb/`);模型级 MoE 优化参考 `patch/worker/` 现有 deepseek/qwen3 patch |

## 9. 待深入(下次读)

- [ ] `mla_v1.py` / `sfa_v1.py` / `dsa_v1.py`(DeepSeek 系;做 MoE 方向再读)
- [ ] `ascend_forward_context.py` 的 `_EXTRA_CTX`(is_draft_model 等全局态如何传递)
- [ ] `AttentionMaskBuilder` 的 mask 复用策略(长序列 mask 显存)
- [ ] `patch_kv_cache_coordinator.py` 与上游 KVCacheManager 差异(Phase 2 前置)
- [ ] PA/FIA 的 shape 限制全集(`pa_shape_list` 从哪来:`ascend_config.py`)
