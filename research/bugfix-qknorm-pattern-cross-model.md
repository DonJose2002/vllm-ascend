# Debug 笔记:FULL 模式下 draft_model 编译崩溃 "Split sizes add up to 6144 but got the tensor's size of 4096"

> 日期:2026-08-17 | 基线:vllm-ascend v0.22.1rc1 + vllm 0.22.1 | 状态:**修复已提交,待服务器验证**
> 前序:bug 1(update_stream MRO 遮蔽)见 `bugfix-draft-model-update-stream.md`——那是本 bug 的前置(修好 1 才走到这里)。

## 现象

`cudagraph_mode=FULL` + `method=draft_model`:部署(编译/捕获)阶段报
```
ValueError: Split sizes add up to 6144 but got the tensor's size of 4096
```
6144 = 目标模型 qkv 宽(4096+1024+1024,head_dim=128);4096 = 自定义 draft 的 qkv 总宽。

## 根因(三层,已从源码逐层证实)

**触发条件**:FULL 模式下 `enable_npugraph_ex=True`(platform.py 只在 NONE/PIECEWISE 关它)→ 所有 torch.compile 走 `npugraph_ex_compile`(compiler_interface.py:283);该分支**完全忽略** `compiler_config` 里的 pass key → `GraphFusionPassManager` 的 FX passes 不运行,**融合完全依赖 `nge.register_replacement` 的进程级全局注册表**。

**直接原因**:`BasePattern.register`(base_pattern.py)的全局去重集 `_registered_patterns` 键 = `类名_eps`,不含模型身份:
1. target 先编译 → 注册 6144 尺寸模式(进 nge 全局表 + 全局去重集)
2. draft 编译(自己的 VllmBackend/eagle_head tag/PassManager)→ `register()` 命中去重 → **直接 return,4096 尺寸模式从未注册**
3. nge 编译 draft 图 → 注册表里只有 6144 变体,替换闭包烧死 `q_size=4096/kv=1024` → 在 draft 的 4096 宽 qkv 上 split → ValueError

**为什么 eager/PIECEWISE 不崩**:那条路走 `fusion_pass_compile` → torch pm。torch pm 的 check_fn(pattern_matcher.py:1525-1575)用**真实 shape 重 trace search_fn** 验证,trace 失败被 catch 后优雅拒绝(`log_trace_failure → return False`);replacement 只在 shape 一致后才 trace。nge 路径缺这层保护,错误直接逃逸。

**波及面**:全部 6 个 pass 文件的模式(qknorm_rope / norm_quant / allreduce_rmsnorm / muls_add / sequence_parallel*)都走 `BasePattern.register`——同样的跨模型隐患(draft hidden 与 target 不同即中招),本修复一并覆盖。

## 修复(三层防线,全部在 base_pattern.py,不碰各模式类)

1. **去重键加入 example-inputs shape 签名**(根本修复):`pattern_id = 类名_eps_形状:类型,...`。target/draft 各自注册自己的变体;同模型重复 configure 仍被去重。
2. **extra_check 加形状守卫**(安全网):`register` 时取"最宽 ≥2 维输入"(即主激活,如 qkv)的 argname 和宽度,组合进 `get_extra_check`。守卫经 `match.kwargs[主输入].meta["val"].shape[-1]` 判定:
   - 宽度匹配 → 放行
   - 宽度不匹配 → 拒绝(模式是给别的模型注册的)
   - **不可判定 → 拒绝**(reject-on-unknown):最坏退化成"该图不融合但正确"——等价于用户此前 workaround 的效果,绝不会崩。torch pm 侧 `match.kwargs` 含全部参数名是硬保证(check_fn 前置断言)。
3. **nge 注册容错 + 闭包按 shape 改名**:闭包 `__name__` 加形状哈希后缀(按名去重的注册表可容纳两个变体);`nge.register_replacement` 遇 `RuntimeError("Duplicate pattern")` 降级为 warning(镜像 npugraph_ex_utils_check.py:70-73 既有处理),其他异常照常抛。

不采用"直接去掉 nge 注册"(用户旧 workaround):那会让 FULL 模式下 target 也失去融合。本修复在保住 target 融合的同时让 draft 也能正确融合;若服务器验证发现 nge 的 Match 无 kwargs 导致守卫全部失效(日志会有 "main input width not verifiable"),fallback 是对 QKNormRope* 关闭 nge 注册(一行改动,见下)。

## 本地验证(无 NPU,CPU 模拟 + 真实 torch pm)

- **旧逻辑复现**:模拟 class+eps 去重 + 无守卫 nge → 对 draft 张量应用 target 模式,精确复现报错文本 `Split sizes add up to 6144 but got the tensor's size of 4096`
- **新逻辑**:target/draft 均注册(nge 表 2 条目)、同 shape 重复注册去重、两个宽度各自正确应用、不可判定 match 全拒绝
- **真实文件集成**(mock npugraph_ex,import 真实 base_pattern.py + 真实 torch pm):shape 键去重 ✓、guard 判定 ✓、nge 重复容错 ✓、非重复异常仍抛 ✓、闭包改名不破坏 pm 注册/trace ✓
- ruff 本机不可用,服务器跑 `ruff check vllm_ascend/compilation/passes/base_pattern.py`

## 服务器验证清单

```bash
git pull   # research/main
# 配置: --speculative-config '{"method":"draft_model",...}' + cudagraph_mode FULL(或默认非 NONE)
# 期望:
#   1. 不再出现 Split sizes ValueError
#   2. 日志出现两次 "Wrapping draft model with ACLGraphWrapper"(bug1 修复的延续)
#   3. debug 级可看到 "Rejecting ...Pattern: main input width != N"(守卫工作正常的证据)
#   4. 若见 "main input width not verifiable from match" 频繁出现 → nge Match 无 kwargs,
#      执行 fallback:get_extra_check 里对 QKNormRope* 返回 None 走纯 stream check,
#      并在 register 中跳过 nge 注册(即用户旧 workaround 的受控版)
```

## 遗留 / 下一阶段

- [ ] 服务器验证(上述清单)
- [ ] 用户预告:修掉本 bug 后还有后续问题(部署过了之后的运行期问题)——届时另开条目
- [ ] 可考虑向上游报 issue(与 bug 1 可合并成 draft_model+FULL 模式的系列修复)
