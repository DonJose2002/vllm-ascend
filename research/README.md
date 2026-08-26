# research/ — 研究产物(不进上游)

> **署名警示(AI 助手必读)**:本仓库根目录的上游 `AGENTS.md` 中 Commit Messages
> 一节的 trailer 示例(Copilot/Claude/gemini)是上游社区的举例,**不要照抄字面**。
> 本项目的 AI 助手是 GLM-5.3,commit trailer 一律使用 `Assisted-by: GLM-5.3`。
> 详见笔记仓库 `AGENTS.md` §1(2026-08-17 事故记录)。

本目录存放基于 v0.22.1rc1 的研究笔记与实验脚本,与上游代码隔离,同步上游时不会冲突。

## 工作流

1. 本机在 `research/main` 分支上修改代码/脚本,push 到 GitHub
2. 昇腾服务器 `git pull` 后验证(NPU 环境只有服务器有)
3. 验证通过的改动保留在分支历史中,commit message 注明验证状态

## 文件说明

- `spec_decode_eager_flow.md` — 投机解码(draft_model)eager 模式完整流程源码分析(基于 v0.22.1rc1 + vllm 0.22.1)
- `attention_backend_arch.md` — attention backend 家族/选型、KV cache 布局、FIA/PA 路径、ACL graph 机制、plugin/patch 体系、KV compression(hamming sparse)现状(严谨版,带文件:行号)
- `attention_backend_explained.md` — 同上内容的通俗叙事版,快速建立直觉用
- **draft_model + FULL 图 debug 系列(✅ 已解决,2026-08-17)**:
  - `draft-model-full-graph-journey.md` — **自然语言复盘**:四层 bug(update_stream MRO / 模式跨模型污染 / 幽灵请求 / off-by-one seq_lens)全程与教训,含代码定位速查表,给同事复盘首选
  - `bugfix-draft-model-update-stream.md` — bug1 严谨版(MRO 遮蔽)
  - `bugfix-qknorm-pattern-cross-model.md` — bug2 严谨版(pattern 注册表跨模型污染)
  - `bugfix-draft-model-full-graph.md` — bug3 严谨版(方案 C/A 设计、上游调研、事故 A-1)
- `bench_sd.py` — SD 量化基准(bench/compare/check):双口径接受长度(metrics + 突发估计)、ITL/TTFT 分位、per-position 接受率;纯 stdlib 客户端
- `test_SD.sh` — 服务器投机解码实验启动脚本(Qwen3-8B + Qwen3-0.6B draft,单卡,eager,带 profiler)
- **Phase 1.5 税探针(2026-08-26,设计在笔记仓库 experiments/phase1.5-tax-probe-design.md)**:
  - `p15-runbook.md` — 服务器执行手册(一键命令、回传协议、判读速查、已核实 API 链路)
  - `profile_window.py` — T2 窗口触发器:单流请求 + 第 N token 处 arm /start_profile,engine 侧 max_iterations 自动停
  - `profile_step_breakdown.py` — T2 服务器端 trace 聚合器:chrome trace(json/gz)→ 每步类别 TSV + 未归类榜 + graph 不透明告警 + --diff 差分;`selftest` 子命令自带数值精确恢复测试
  - `run_baseline_npu.sh` 新 env:`PROFILER=1 PROFILE_ONLY=1`(serve 带 --profiler-config、跳过 bench 防 profiler 污染测量、SUMMARY 内嵌聚合 TSV)
