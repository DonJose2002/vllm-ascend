# 复盘:SD batch 收缩崩溃(draft_model × FULL 图 × R=5 陷阱)

> 2026-08-21 | 修复 commit `44a213fad`(research/v0.23.0,待 PR #4 化) | 环境:vllm-ascend v0.23.0 = fixes,910B3,Qwen3-8B/0.6B,K=5
> 姊妹篇:`bugfix-draft-model-full-graph.md`(bug3,plan A/C)。那个是"怎么把功能修对",这个是"一个潜伏很深的崩溃 bug 怎么被 baseline 数据暴露、定位、修复"。

## 0. 一句话

并发请求从 8 排水到 5 的那一步,5×(K+1)=30 个 token 落在 FULL 图捕获尺寸表的空洞里(表从 24 直接跳 36),填充逻辑补出一个幽灵请求;而投机解码的注意力元数据在"去填充"时只切了 query_start_loc、故意保留了带幽灵行的 block_table——两者对"batch 里到底有几个请求"各执一词,上游算 slot mapping 的 `repeat_interleave` 当场崩掉。

## 1. 现象与三次误判(教训部分)

第一现场很脏,连续错了三次才看清:

1. **误判"16K 特异"**:bench 矩阵里 4K 三档全过、16K 起全灭,最自然的解读就是"长 prompt+SD 崩了"。真相:崩溃发生在 **4K/c16 cell 中途**,16K/32K 只是排队等一个已经死了的引擎。教训:**bench 矩阵的"哪个 cell 失败"不等于"什么条件触发"**——cell 内部还有时间维度(8 个请求先后完成,batch 在收缩)。
2. **误判"ok=8"**:引擎死时 5 条在途流被截断,但 harness 的成功判定只看"收到过 token"(token_ts 非空),把截断流也计成成功——所以崩溃那个 cell 的表显 ok=8,数据看似正常。真相是 3 完成 + 5 截断的混合体。教训:**流式 benchmark 的 ok 必须以 `[DONE]` 为准**,截断要显式记入 errors(已修进 `bench_baseline.py`)。
3. **误判"容器密封"**:此前我们刚把容器从 editable 安装的坑里救回来,改用非 editable `pip install .`,我据此断言"checkout 不再影响运行时"。但 traceback 的文件路径显示引擎加载的是**工作区**代码而非 site-packages 副本——断言被当场证伪(机制未完全查明,疑为 spawn 子进程继承 cwd 搜索路径)。教训:非 editable 的"密封性"要在目标进程上验证,不能只看安装形态。**副作用是好的**:改 .py 后 `git pull` 即生效,这次插桩和修复都靠它零重装落地。

## 2. 定位过程(按时间线)

### 2.1 从 traceback 拿到崩溃点

serve log 的关键帧:

```
llm_base_proposer.py:1579 set_inputs_first_pass
  → vllm/v1/spec_decode/utils.py:252 compute_new_slot_mapping
    → RuntimeError: repeats must have the same size as input along dim
```

读上游 0.23.0 的 `compute_new_slot_mapping`(0.26.0 同构):

```python
batch_size, n_blocks_per_req = cad.block_table_tensor.shape   # ← batch 从 bt 行数来
req_indices = torch.arange(batch_size)
req_indices = torch.repeat_interleave(
    req_indices,
    cad.naive_query_lens() + num_new_tokens,   # ← 长度从 qsl 来(5 个)
    output_size=len(new_positions),
)
```

`repeat_interleave(input=arange(A), repeats=长度 B)` 报这个错,只有一种可能:**A ≠ B,即 bt 行数 ≠ qsl 隐含的请求数**。这份 cad 内部自相矛盾。

### 2.2 静态读码走到死胡同

顺着 cad 的构造追:`_build_attention_metadata` → cm_base(qsl 和 bt 都按 `num_reqs_padded` 切,自洽)→ 尾部突然有一刀:

```python
if spec_decode_common_attn_metadata is not None and (
    num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
):
    spec_decode_common_attn_metadata = spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
```

而 `AscendCommonAttentionMetadata.unpadded()`(attention/utils.py:256)干了件注释自己都承认奇怪的事:

```python
query_start_loc=self.query_start_loc[: num_actual_reqs + 1],   # 切到真实请求数
...
# NOTE: keep all tokens for block_table_tensor and slot_mapping otherwise
# there will be error about shape mismatch during reshape and cache.
# This is really strange since vLLM slices them as well
block_table_tensor=self.block_table_tensor,                     # ← 故意不切!
```

qsl 被切、bt 保留——**"半去填充"**。对注意力内核无害(它们都按 qsl 驱动,多出的 bt 行没人读);对 `compute_new_slot_mapping` 致命(它拿 bt 行数当 batch_size)。

但静态读到这里还不够:为什么大多数时候不崩?需要解释触发条件。

### 2.3 插桩拿到决定性数据

在崩溃点前埋了一个形状打印(research 插桩,`b8d27f049`),单 cell 复现(4K/c16,~6 分钟):

```
[SD-debug] cad self-inconsistent: qsl numel=6 (naive len=5) vs block_table rows=6;
           total_num_output_tokens=35 net_num_new_slots=1; qsl head=[0,6,12,18,24,30]; bt shape=(6,320)
```

同时 scheduler dump 显示崩溃步:5 个在途请求、每个 scheduled 6 token(=1+5 草稿)、~220-250/256 输出进度——正是 8 请求排水到 5 的中间态。

### 2.4 数学闭环:R=5 陷阱

FULL 图捕获尺寸表(serve log 里原样打印):

```
[6, 12, 18, 24, 36, 42, 48, 60, 66, 72, ...]   ← 注意:没有 30
```

- R=1..4:R×6 = 6/12/18/24,命中桶,无需填充 → qsl 与 bt 本来就一致,不崩;
- **R=5:5×6=30,落在 24 与 36 之间的空洞里** → 填充到 36,多出 6 个 token;
- 填充逻辑(`_pad_query_start_loc_for_fia` 的 mixed-batch 分支)往 qsl 末尾**插一个 dummy 请求**来表达这 6 个多余 token → `num_reqs_padded=6`,bt 切到 6 行;
- 随后 `unpadded(30, 5)` 把 qsl 切回 5、bt 留 6 行 → 自相矛盾的 cad 传进 drafter;
- `arange(6)` vs `naive_query_lens()` 长度 5 → 崩。

为什么 D5 从没撞上:D5 的 bench 是 30 并发(180 token,在表里)、且请求同步完成,排水经过 5 的窗口极窄;我们的 harness 是 8 请求错峰完成,**必然**经过 R=5。

## 3. 修复

最小改动,在 `set_inputs_first_pass`(llm_base_proposer.py)里,仅对 slot mapping 这一次计算对齐两边的请求视角:

```python
qsl_num_reqs = cad.query_start_loc.numel() - 1
bt_num_rows = cad.block_table_tensor.shape[0]
if bt_num_rows > qsl_num_reqs:
    slot_cad = cad.replace(block_table_tensor=cad.block_table_tensor[:qsl_num_reqs])
else:
    slot_cad = cad
new_slot_mapping = compute_new_slot_mapping(cad=slot_cad, ...)
```

要点:
- 用 `cad.replace()`(dataclasses.replace 的便捷方法,上游 `CommonAttentionMetadata` 自带)造一个**临时视图**,不改原 cad;
- `unpadded()` 的语义原样保留——它保留填充 bt 是有理由的(切了会坏 reshape_and_cache,plan A 图重放也依赖填充视图);
- phantom 行的 block id 是填充时清零的(`blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)`),反正 slot_cad 已切掉,不会被索引到。

### 验证

1. **CPU 单元验证**(本机,vllm 0.26 的上游实现,5 请求 qsl + 6 行 bt + 35 新位置):原始形态**逐字复现服务器报错**;切片修复后输出 35 个 slot,逐个核对请求归属、块号、块内偏移**全对**;
2. **服务器端到端**:单 cell(4K/c16)通过 → 全量 9/9 cell 通过(4K/16K/32K × 1/4/16),accept 2.80-3.02 与修复前 4K 档一致(修复不影响数值路径,只修崩溃)。

## 4. 修复后的完整 plan C 数据(顺带归档)

ITL ~293-330ms 横跨所有档位(plan C drafter eager 主导,上下文长度几乎不影响——drafter 开销淹没 KV 扫描);对照 dense(16.5-44.9ms)SD 在 plan C 下是 5.7-18× 恶化,**plan C 只是止血,plan A 才是可用形态**——这组数据反向坐实了 D5 做 plan A 的必要性。完整表格见 `experiments/phase0-baseline-report.md` §3。

## 5. 教训清单

1. **cell 失败位置 ≠ 触发条件**:bench 有 cell 内时间维度(batch 排水形态),归因前先看 scheduler dump 里崩溃步的真实 batch 组成;
2. **流式 ok 判定必须看 `[DONE]`**:截断流是静默毒药,尤其是"引擎中途死"这种场景;
3. **"半去填充"是元数据一致性的天敌**:qsl 与 bt 必须对"请求数"同观;任何从 bt 行数推导 batch_size 的代码都是雷(上游 `compute_new_slot_mapping` 就这么写);
4. **捕获表空洞是结构性风险**:uniform dispatch 假设 num_tokens 是桶的倍数,batch 收缩经过空洞时就触发填充;空洞位置随捕获表配置变化,换 K/换表就可能换一批 R 值踩雷(本例 K=5 → R×6,空洞在 30);
5. **运行时加载路径要在目标进程验证**:traceback 的文件路径比 pip 元数据诚实;"非 editable=密封"在 spawn 多进程模型下未必成立;
6. **插桩-单cell复现-数学闭环**是深 bug 的高效打法:6 分钟一轮的服务器复现,配合 CPU 上的上游实现复现,把"猜测"变成"算术"。

## 6. 文件索引

| 文件 | 位置 | 角色 |
|---|---|---|
| `vllm_ascend/spec_decode/llm_base_proposer.py` `set_inputs_first_pass` | 修复点(`44a213fad`) | slot_cad 对齐 |
| `vllm_ascend/attention/utils.py` `unpadded()` | 根因侧(未改) | qsl 切/bt 留的"半去填充" |
| `vllm_ascend/worker/model_runner_v1.py:3361` | 调用点(未改) | spec cad 过 unpadded 的那一刀 |
| `_pad_query_start_loc_for_fia` | 填充来源(未改) | mixed-batch 分支插 dummy 请求 |
| 上游 `vllm/v1/spec_decode/utils.py` `compute_new_slot_mapping` | 崩溃点(上游) | bt 行数当 batch_size |
| `experiments/npu-sd-fix-verify-2026-08-20.txt` | 笔记仓库 | 修复后 9/9 cell 证据 |
