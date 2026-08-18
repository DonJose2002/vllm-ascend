# Debug 笔记:FULL 模式下 draft_model 编译崩溃 "Split sizes add up to 6144 but got the tensor's size of 4096"

> 日期:2026-08-17 | 基线:vllm-ascend v0.22.1rc1 + vllm 0.22.1 | 状态:**修复已提交,待服务器验证**
> 前序:bug 1(update_stream MRO 遮蔽)见 `bugfix-draft-model-update-stream.md`——那是本 bug 的前置(修好 1 才走到这里)。

## 现象

`cudagraph_mode=FULL` + `method=draft_model`:部署(编译/捕获)阶段报
```
ValueError: Split sizes add up to 6144 but got the tensor's size of 4096
```
6144 = 目标模型 qkv 宽(4096+1024+1024,head_dim=128);4096 = 自定义 draft 的 qkv 总宽。

## 根因(2026-08-17 二次修正:拿到服务器 traceback 后)

**崩溃点的精确定位**(服务器日志):不是替换闭包执行时崩,而是 **torch pm `check_fn` 的 shape 验证 re-trace**:

```
nge pattern_pass_manager.apply_pass → torch PatternMatcherPass.apply:2051
  → check_fn:1517  specific_graph = trace_fn(search_fn_new, sym_args + args)
  → 执行 pattern 函数体 qkv.split([6144,...]) 对 4096 fake tensor
  → _refs.split_with_sizes → torch._check_with(ValueError) → ValueError 逃逸
```

机制链:
1. torch pm 的**初始结构匹配忽略 int 常量**(check_fn 注释原文 "our initial match ran with ignore_types=(int, ...)")——所以 6144 尺寸的模式能结构匹配 4096 的图
2. check_fn 的设计:用**真实 shape 重新 trace search_fn** 验证烧进去的尺寸,失败应走优雅拒绝——但它**只 catch RuntimeError**(pattern_matcher.py:1517-1521 两个 try 都是 `except RuntimeError`)
3. `split` 尺寸不匹配抛的是 **ValueError** → 逃逸 → nge `error_code.py` 的 wapper 重新 raise → 整个部署崩
4. **extra_check 在 re-trace 之后才被调用**——任何挂在 extra_check 上的守卫都来不及救

**触发条件**:FULL 模式下 `enable_npugraph_ex=True` → 所有 torch.compile 走 nge 后端,其模式注册表**进程级全局**(target + draft 共享);PIECEWISE/eager 走 `GraphFusionPassManager`,但 `_registered_patterns` 全局去重(键=`类名_eps`)让 draft 的注册被静默跳过、draft 自己的 pass 里没有模式 → 不崩(但 draft 也从未融合)。

**波及面**:全部 6 个 pass 文件的模式(qknorm_rope / norm_quant / allreduce_rmsnorm / muls_add / SP)都走 `BasePattern.register`;其中带烧死尺寸的模式(qkv split)是重灾区。

## 修复(四层防线,全部在 base_pattern.py)

1. **去重键加入 example-inputs shape 签名**:target/draft 各自注册自己的变体;同模型重复 configure 仍去重
2. **search_fn 宽度守卫 wrapper(关键修复,二次修正新增)**:`_wrap_search_fn_with_width_guard` 用 `functools.wraps`(保持签名,`inspect.signature` 经 `__wrapped__` 解析不受影响)包裹 pattern 函数——主输入(最宽 ≥2 维输入)末维宽度**可证明不匹配时主动抛 RuntimeError**,正好落进 check_fn 的 `except RuntimeError → log_trace_failure → return False` 优雅拒绝路径;宽度是 SymInt/不可判定时不干预,交给原 re-trace 机制
3. **extra_check 形状守卫**(re-trace 之后、替换应用之前的双保险):`match.kwargs[主输入].meta["val"]` 宽度校验,不可判定拒绝(reject-on-unknown)
4. **nge 注册容错 + 闭包按 shape 改名**:`nge.register_replacement` 遇 "Duplicate pattern" RuntimeError 降级 warning;闭包名加形状哈希后缀

效果:FULL 模式下 nge 全局注册表同时持有 target/draft 两套变体;对各自图,错误变体经守卫优雅跳过、正确变体正常融合——**两个模型都保住融合**,优于"关掉 nge 注册"的 workaround(那会两者都不融合)。

## 本地验证(CPU + 真实 torch pm,忠实复现服务器路径)

用 `cat(split(...))` 模式镜像 QKNormRope(初始 trace 合法、retrace 才在 split 崩),经 `make_fx(tracing_mode="real")` 建图、真实 `pm.register_replacement` + `PatternMatcherPass.apply` 走完整 check_fn 路径:

- **旧路径**:`ValueError: Split sizes add up to 64 but got the tensor's size of 40` 逃逸(与服务器 6144/4096 完全同构,连 check_fn 前的 E 级 meta 日志都一致)✓ 复现
- **新路径**:宽度 40 vs 注册 64 → applied=0 **无异常**;宽度匹配 → applied=1,融合后图只剩替换算子(`aten.mul.Tensor`)✓
- **FULL 模式模拟**(双变体同一全局 pass):target 图 fused=1 + draft 图 fused=1,双方都融合 ✓
- wrapper 签名保持 ✓(argnames_static 解析不受影响)

## 服务器验证清单

```bash
git pull   # research/main
# 配置: --speculative-config '{"method":"draft_model",...}' + cudagraph_mode FULL
# 期望:
#   1. 不再出现 Split sizes ValueError
#   2. debug 级可见 "declining to apply"(宽守卫拒绝错误变体的证据)
#   3. 部署完成后正常 serve;若出现新的运行期错误,属于用户预告的下一阶段问题
```

## 遗留 / 下一阶段

- [x] 服务器验证(上述清单,0.22.1rc1)
- [x] **v0.23.0 复现证据(2026-08-18)**:`logs/v0.23.0-baseline-bug2-split-crash.txt`——官方 tag 基线,traceback 与 0.22.1rc1 逐帧同构(nge apply_pass → check_fn:1517 re-trace → qknorm_rope_fusion_pass.py:60 split → ValueError 逃逸)
- [x] **v0.23.0 PR2 生效证据(2026-08-18)**:`logs/v0.23.0-baseline-pr2-bug1-update-stream.txt`——baseline+PR2 后编译越过 split 崩溃点,推进到图捕获阶段(说明 pattern 守卫生效)
- [ ] 上游 PR 提交(文案就绪,待 5b/5c/5d 完成后回填验证结果)
