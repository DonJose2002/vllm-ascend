# 复盘:draft_model + FULL 图模式,一夜打通四层 bug 的完整历程

> 写给后来复盘的同事。目标读者:了解 vllm/vllm-ascend 基本架构,但不熟悉 spec decode drafter 内部实现的工程师。
> 基线:vllm-ascend v0.22.1rc1 + vllm 0.22.1,分支 `research/main`。
> 严谨版细节(带完整行号、验证清单)在姊妹篇:`bugfix-draft-model-update-stream.md` / `bugfix-qknorm-pattern-cross-model.md` / `bugfix-draft-model-full-graph.md`。

---

## 故事背景

我们想在昇腾 NPU 上验证一件事:Qwen3-8B(目标模型)+ Qwen3-0.6B(草稿模型)的投机解码,把 `cudagraph_mode` 从 NONE 打开成 FULL,能不能用图模式加速 decode。

这个配置(`method="draft_model"`)在 vllm-ascend 0.22.1 上从没人跑通过。原因很快会看到:不是一处 bug,而是**四层叠着的 bug**——每修一层,暴露下一层。像剥洋葱。

另外有个重要背景:profiling 数据显示,旧配置(eager)下草稿模型**从来没走过图模式**。所以我们不是在"修复退化",而是在**开垦一条新路**。

---

## 第一层:部署即崩 —— update_stream 不存在

**现象**:`vllm serve` 阶段直接 `AttributeError: 'AscendDraftModelProposer' object has no attribute 'update_stream'`。

**排查思路**:一个属性明明在基类里有赋值,为什么子类实例上没有?顺着报错找 `_update_full_graph_params` 用到 `self.update_stream`,往上追赋值点,发现在 `AscendSpecDecodeBaseProposer._maybe_share_lm_head` 的末尾(vllm_ascend/spec_decode/llm_base_proposer.py)。

**根因:Python 多继承的方法解析顺序(MRO)遮蔽**。

```
class AscendDraftModelProposer(DraftModelProposer, AscendSpecDecodeBaseProposer):
```

上游 vllm 的 `DraftModelProposer._maybe_share_lm_head` 是个**空方法**(vllm/v1/spec_decode/draft_model.py,设计意图:独立草稿模型不共享 lm_head)。因为上游类排在 MRO 第一位,这个空方法**遮蔽**了 Ascend 版实现。而 Ascend 版对这个 method 而言虽然不动 lm_head,但末尾承担着 ACL 全图设置的两件大事:

- `self.update_stream = torch.npu.Stream()` —— 报错的那个属性
- `self._runnable = ACLGraphWrapper(...)` —— 草稿模型的图捕获引擎

所以真实伤害比报错深:**图捕获从未发生,草稿模型一直在静默跑 eager**。

**为什么 EAGLE 不中招**:上游 `EagleProposer` 没有 override 这个方法,MRO 正常解析到 Ascend 版。全项目只有 draft_model 被上游空方法挡住。

**修复**(vllm_ascend/spec_decode/draft_proposer.py):子类显式委托一行:

```python
def _maybe_share_lm_head(self, model):
    AscendSpecDecodeBaseProposer._maybe_share_lm_head(self, model)
```

不动 MRO、不动基类、对 lm_head 语义零影响。**教训:多继承 + 两边同名方法 = MRO 踩雷高发区;上游"空方法"的遮蔽最隐蔽,因为它看起来什么都没做。**

---

## 第二层:编译期崩 —— 6144 与 4096 之战

**现象**:修完第一层,部署推进到编译阶段,报
`ValueError: Split sizes add up to 6144 but got the tensor's size of 4096`
(6144 = 目标模型 qkv 宽度 4096+1024+1024;4096 = 草稿模型的 qkv 总宽)。

**排查思路**:谁在拿 6144 的尺寸切 4096 的张量?沿着 traceback 反推到 QKNormRope 融合模式的 pattern 函数——它的 split 尺寸是**烧死在闭包里的**。

**根因(比表象深三层)**:

1. FULL 模式下,torch.compile 全部走 npugraph_ex 后端,融合算子的注册表是**进程级全局的**——目标模型和草稿模型共用一张表;
2. `BasePattern.register`(vllm_ascend/compilation/passes/base_pattern.py)的防重复注册用 `类名+eps` 做键,**不含模型身份**。目标模型先注册了 6144 尺寸的模式,草稿模型来注册时被认为"重复",**直接跳过**——4096 尺寸的变体从未注册;
3. 于是全局表里只有 6144 的模式,编译草稿图时它被应用上去,split 当场爆炸。

**第一版修复为什么不够**:我最初在 extra_check 里加了形状守卫,但服务器日志证明守卫根本没机会执行——崩溃点在 torch pattern_matcher 自己的**验证性重 trace**里(torch/_inductor/pattern_matcher.py 的 check_fn):它用真实 shape 重新执行 pattern 函数来验证烧死的常量,split 抛 ValueError。而 check_fn 的保护网**只 catch RuntimeError**,ValueError 逃逸,整个部署崩。

**最终修复**(全在 base_pattern.py,惠及全部 6 个融合 pass):
- 防重键加入输入形状签名 → 两个模型的变体各自注册
- pattern 函数包一层宽度守卫 wrapper(`functools.wraps` 保持签名):宽度**可证明不匹配时主动抛 RuntimeError**——正好落进 torch 设计好的优雅拒绝路径(只丢融合,不崩编译)
- npugraph_ex 重复注册容错 + 闭包按形状改名

**教训:修 bug 要区分"崩溃点"和"根因层";利用组件自带的失败路径(让它抛对类型的异常)比在外面加拦截更优雅。**

---

## 第三层:运行期崩 —— 幽灵请求

**现象**:两层修完,部署成功、开始生成,第一步就报
`RuntimeError: The expanded size of the tensor (1) must match the existing size (2)... Target sizes: [1, 5]. Tensor sizes: [2, 5]`
(草稿输出 [2,5] vs CPU 缓冲 [1,5])。

**排查思路**(用户自己打了大量 print 定位到一半,我们接手验证):为什么图模式认为有 2 个请求?

**根因:token 数的整除失配**。草稿模型每步要吃 `R×(K+2)` 个 token(R 个请求 × (K+1 个验证 token + 1 个额外种子槽,后者来自 `set_inputs_first_pass` 的 extra-slots 逻辑)。而 dispatch 的 FULL-uniform 分支**硬性假设** token 数是 `(K+1)` 的倍数,并按整除推算请求数:

```
num_reqs = 12 // 6 = 2   ← 凭空造出第二个"幽灵请求"
```

图按 2 个请求烘焙,输出 [2,5];真实只有 1 个请求,拷贝炸。

**为什么 EAGLE/MTP/DFlash 不中招**:它们的额外槽净增加为 0(token 数保持 (K+1) 倍数)。**全项目唯独 draft_model 是 1**。上游 vllm 的 drafter 只用 PIECEWISE(relaxed key,无整除断言),从设计上回避了这个问题。

**两步走**:
- **方案 C(止血,先跑通)**:`use_cuda_graph` 排除 draft_model 方法 → 草稿 eager(≈上游语义),target 的 FULL 图不受影响。服务器验证通过,证明 bug1/2/3 修复链有效;
- **方案 A(正解)**:给 drafter 建专属捕获表 `R×(K+2)`(如 [6,12]→[7,14]),dispatch 按新表 pad,幽灵请求零长度序列会被 attention 自然跳过,超尺寸优雅回 eager。

**教训:接口两端的隐式契约(整除假设)在新增使用方时最容易碎;先用降级方案验证修复链,再做正式方案,每步都有可回退点。**

---

## 第四层:不崩了,但草稿变笨了 —— off-by-one

**现象**:方案 A 部署成功、正常出 token,但平均接受长度从 >3 崩到 1.3(每步几乎只接受 1 个 token)。

**排查思路**:接受率崩 = 草稿 token 全错 = 草稿 hidden states 错 = attention 元数据错。逐字段对照 eager 路径,锁定 seq_lens。

**根因**:方案 A 的 FULL padding 分支里,`seq_lens` 抄了 EAGLE 分支的 `runner.seq_lens`。但两个方法的语义不同:

- draft_model 的 `set_inputs_first_pass` 走 extra-slots 分支,末尾 `extend_all_queries_by_N` 会把 **seq_lens 每行 +1**(让 attend 范围覆盖 step0 新写入的种子 KV);
- EAGLE 不走 extend,`runner.seq_lens` 对它恰好正确;
- **dflash(和 draft_model 一样走 extend)的分支用的是 extend 后的值——我当时抄错了对象**。

off-by-one 使所有 query 的 causal attend 边界整体左移一位 → 每个 draft hidden 都错位 → 草稿基本全被拒。第 1 个 token 偶尔对,正好解释 1.3 而不是 1.0。

**修复**:改用 extend 后的 `common_attn_metadata.seq_lens`(对齐 dflash 先例与已验证的 eager 语义)。

**教训:跨方法抄分支时,先确认该方法走的是哪条 set_inputs 路径——函数返回值才是语义基准,不是长得像的兄弟分支。**

---

## 尾声:接受率在 3 附近波动 —— 虚惊一场

修复后接受长度回到 ~3 但围绕原值双向波动。分析:双向波动是图编译(static kernel tiling/累加顺序差异)在 BF16 + greedy argmax 下的数值噪声特征,属良性;若单边压低才是系统性问题。逐项复检后确认无系统偏差,结案。

期间顺手造了工具 `research/bench_sd.py`(纯客户端、零依赖):双口径接受长度(metrics counters + 流式突发到达估计)、ITL/TTFT 分位、per-position 接受率、A/B compare。中途还发现服务器 `/metrics` 没暴露 spec counters,以及 vllm 的 `validate_environ` 会对所有 `VLLM_ASCEND_*` 环境变量报 warning(无害,flag 正常生效)。这个工具留给后续回归验证。

顺带一提:排查"接受率"问题时曾怀疑我改过引擎代码,git 历史证明修复 1.3 之后 `vllm_ascend/` 一行未动(只有 research/ 脚本)——**git log 是这类争论的最终裁判**。

---

## 改动全景(代码定位速查)

| 层 | 文件 | 改动 | commit |
|---|---|---|---|
| bug1 | `vllm_ascend/spec_decode/draft_proposer.py` | `_maybe_share_lm_head` 显式委托基类实现 | `413cfe180`(重写后 hash) |
| bug2 | `vllm_ascend/compilation/passes/base_pattern.py` | shape 签名去重键 + search_fn 宽度守卫 wrapper + nge 容错 | `826291dd1` + `82926eef0` |
| bug3-C | `vllm_ascend/spec_decode/llm_base_proposer.py` | `use_cuda_graph` 排除 draft_model(现为默认行为) | `d5d6ef0c7` |
| bug3-A | `vllm_ascend/envs.py` | `VLLM_ASCEND_DRAFT_MODEL_FULL_GRAPH` 开关 | `40b85c309` |
| | `vllm_ascend/worker/model_runner_v1.py` | drafter 捕获表派生(`_check_and_update_cudagraph_mode` 末尾)+ `_dummy_run` 捕获翻译 | 同上 |
| | `vllm_ascend/spec_decode/llm_base_proposer.py` | `__init__` flag / `dummy_run` K+2 元数据 / `_propose` K+2 dispatch | 同上 |
| A-1 | 同上 `_propose` FULL padding 分支 | seq_lens 改用 extend 后值 | `33205ca2d` |
| 工具 | `research/bench_sd.py` | bench/compare/check 子命令 | `713539a78`→`4f7747017` |

**验证状态**:方案 C(默认)与方案 A(flag=1)均经服务器端到端验证,接受长度正常。

## 给复盘同事的三个带走结论

1. **多继承插件体系里,上游空方法是最隐蔽的遮蔽源**——加新 proposer 方法时,把上游同名的所有 override 列一遍;
2. **融合 pattern 的注册表是全局的,而 pattern 尺寸是模型相关的**——任何"多模型同进程编译"的场景都要检查注册键是否含形状身份;
3. **drafter 的 token 预算 = R×(K+1) + R×extra_slots**——`net_num_new_slots_per_request` 决定了一个 spec 方法与 dispatch 整除假设的兼容性,新增 spec 方法先算这笔账。
