# Phase 1.5 税探针 runbook(服务器执行手册)

> 设计正本:笔记仓库 `experiments/phase1.5-tax-probe-design.md`(2026-08-26 收官决策)
> 本手册 = 服务器侧执行 + 回传 + 判读速查;工具链已就绪(本分支 research/)

## 0. 工具链(全部本地验证过,2026-08-26)

| 部件 | 文件 | 验证状态 |
|---|---|---|
| 窗口触发器 | `research/profile_window.py` | 本地伪 SSE 服务器 E2E 通过(2 轮、start@tok5、stop 每轮、退出码) |
| trace 聚合器 | `research/profile_step_breakdown.py` | `selftest` 全过:类别预算精确恢复、gzip、未归类榜、graph 不透明告警、差分精确 |
| serve 集成 | `run_baseline_npu.sh`(PROFILER/PROFILE_ONLY env) | bash -n;API 链路人工核实(见 §4) |
| 批驱动 | `run_phase1.sh p15` | bash -n |

## 1. 一键执行(容器内,repo 根目录)

```bash
git pull   # 宿主机;确认 research/v0.23.0 @ 本次 commit
bash research/run_phase1.sh p15          # ~10 个 run,一夜
```

顺序 = 设计 §8:**T2 smoke(dense+profiler)→ T1(ngram K∈{1,3,8},6 cell)→ T2 全量(ngram/eagle3)**。
每个 run 结束各打一个 SUMMARY 块;末尾 DIGEST 含 T1 kregress + T2 三配置类别差分。

## 2. 回传协议(T3,不变式:trace 不离服务器)

每个 SUMMARY 块(含 `prof:` 段 = 服务器端聚合后的类别 TSV)+ 末尾 DIGEST,整块复制贴回。
无需拷任何 trace/日志文件。若某 run 失败,连同 "profiler chain broken" 诊断行一起贴。

## 3. 判读速查(贴回前自查)

| 观察到 | 含义 | 动作 |
|---|---|---|
| SUMMARY 含 `prof:` 类别表 + `wall_ms_per_step` | 链路通 | 对比 wall_ms_per_step vs T1 同 cell ITL(profiler 税 sanity,应同量级) |
| `OPAQUE-GRAPH WARNING` | FULL 图 replay 把逐算子遮成单事件 | 用 `EXTRA_SERVE_ARGS=--enforce-eager` 重跑该配置(p15-prof 目录别覆盖:`PROFILER_DIR` 指新目录) |
| `PROFILE-FAIL: /start_profile HTTP 404` | profiler router 未挂(vllm v0.23.0 门在 profiler_config 后) | 检查 serve 命令里 `--profiler-config` 是否带上(脚本自动加;被 EXTRA_SERVE_ARGS 覆盖会丢) |
| `prof: NO trace files` + serve log 无 profiler 行 | TorchNPUProfilerWrapper 或 torch_npu profiler 崩 | 贴 serve log 中 profil 相关行;T2 降级 T1-only(设计 §7 预案) |
| T1 kregress 的 b′ 异常大(>3ms/K) | ngram host 提议成本泄漏进 ITL | 先查 host 侧(ngram 匹配),勿直接归因税 |
| eagle3 trace 无采样/记账类别增量 | eagle 分支算子名不匹配类别表 | 把 `top-unclassified` 榜贴回,扩 RULES 表(纯数据驱动,无需重跑) |

## 4. 已核实的 API 链路(防服务器现场考古)

- pin 的 vllm = **v0.23.0**(Dockerfile `VLLM_TAG` + `.github/vllm-release-tag.commit`)
- 该版本 `/start_profile` **门在 `--profiler-config.profiler != None` 之后**(entrypoints/serve/profile/api_router.py:attach_router);无旧式 `--profile` 顶层 flag
- `ProfilerConfig`:`max_iterations`(engine 自动停,worker.py 每 execute_model 调 `profiler.step()`)、`torch_profiler_with_stack`(默认 true,NPU wrapper 映射到 with_modules、开销大 → 我们显式关)、`ignore_frontend`(跳过 AsyncLLM CPU 侧 trace)
- NPU 侧:`vllm_ascend/profiler/torch_npu_profiler.py` 的 `TorchNPUProfilerWrapper`(Level1,`with_stack=False` 硬编码,data_simplification,`tensorboard_trace_handler` 导出 chrome trace)
- 二次 /start_profile 语义:wrapper 自动停后仍 active,**每轮必须显式 /stop_profile** 复位(profile_window.py 已内置)

## 5. 参数默认值(均可 env 覆盖)

`PROFILER_STEPS=40`(每轮记录的 engine 步数)、`PROFILER_ROUNDS=2`、`PROFILER_START_TOKENS=24`(跳过 prefill/首图税)、`PROFILE_TIER=4096`、K=5(T2 与 E1/E2 口径一致)。

## 6. 判读矩阵(设计 §5,回填结论用)

| ②采样+③记账 合计 | 动作 |
|---|---|
| > 3ms | 列工程候选清单(H2D 合并/记账融合/采样路径),穿插上游 PR,不阻塞 Phase 2 |
| 1.5-3ms | 入档进 Phase 2 |
| < 1.5ms | 税不可压缩 = D2-B 天花板数据,反向支撑评估 D2-C |
