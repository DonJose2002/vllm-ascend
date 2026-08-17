# research/ — 研究产物(不进上游)

本目录存放基于 v0.22.1rc1 的研究笔记与实验脚本,与上游代码隔离,同步上游时不会冲突。

## 工作流

1. 本机在 `research/main` 分支上修改代码/脚本,push 到 GitHub
2. 昇腾服务器 `git pull` 后验证(NPU 环境只有服务器有)
3. 验证通过的改动保留在分支历史中,commit message 注明验证状态

## 文件说明

- `spec_decode_eager_flow.md` — 投机解码(draft_model)eager 模式完整流程源码分析(基于 v0.22.1rc1 + vllm 0.22.1)
- `attention_backend_arch.md` — attention backend 家族/选型、KV cache 布局、FIA/PA 路径、ACL graph 机制、plugin/patch 体系、KV compression(hamming sparse)现状(严谨版,带文件:行号)
- `attention_backend_explained.md` — 同上内容的通俗叙事版,快速建立直觉用
- `test_SD.sh` — 服务器投机解码实验启动脚本(Qwen3-8B + Qwen3-0.6B draft,单卡,eager,带 profiler)
