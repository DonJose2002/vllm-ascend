# Debug 笔记:draft_model 模式 + cudagraph 下 AttributeError: no attribute 'update_stream'

> 日期:2026-08-17 | 基线:vllm-ascend v0.22.1rc1 + vllm 0.22.1 | 状态:**修复已提交,待服务器验证**

## 现象

部署自定义模型(draft_model 投机解码 + cudagraph 打开)时报:
```
AttributeError: 'AscendDraftModelProposer' object has no attribute 'update_stream'
```
仅 cudagraph 模式出现(eager 正常,因为 eager 不走 `_update_full_graph_params`)。

## 根因(MRO 遮蔽)

```
vllm_ascend/spec_decode/draft_proposer.py:8
class AscendDraftModelProposer(DraftModelProposer, AscendSpecDecodeBaseProposer)
                                ↑ 上游排 MRO 第一位
```

- 上游 `DraftModelProposer._maybe_share_lm_head`(`vllm/v1/spec_decode/draft_model.py:86`)是**空方法**:draft model 设计上不共享 lm_head
- Ascend 版 `_maybe_share_lm_head`(`vllm_ascend/spec_decode/llm_base_proposer.py:410`)对 method=="draft_model" 也不动 lm_head,但**末尾(line 449-463)承担 ACL 全图设置**:
  - `self.update_stream = torch.npu.Stream()`(line 456)
  - `self._runnable = ACLGraphWrapper(self._run_merged_draft, ..., CUDAGraphMode.FULL)`(line 457-463)
- MRO 使上游空方法胜出 → 这两步全部跳过

**后果是两层的**:
1. 表象:`_update_full_graph_params`(llm_base_proposer.py:2000→2005)访问 `self.update_stream` → AttributeError
2. 隐藏层:即使只补 `update_stream` 属性,`_runnable` 仍是裸 `_run_merged_draft`——**draft model 图捕获从未发生**,silently eager 跑 + 空 update。治标不治本。

## 为什么只有 draft_model 中招

- EAGLE:上游 `EagleProposer` **没有** override `_maybe_share_lm_head` → MRO 正常解析到 Ascend 版 ✓
- draft_model:上游 `DraftModelProposer` 有空 override → 遮蔽 ✗

## 上游状态(2026-08-17 调研)

- 最新 main(v0.26.0rc)`draft_proposer.py` 类定义未变,`update_stream` 仍在 `_maybe_share_lm_head` 内(main 上改为 `= None`,因 #13600 统一 main/draft stream)→ **该 bug 在最新版依然存在**
- 无相关 issue(#13600 是 eagle3/MRV2 fullgraph 死锁,不同问题)。可考虑向上游报 issue/PR

## 修复(最小侵入:恢复被遮蔽的实现)

`draft_proposer.py` 中 override `_maybe_share_lm_head` 并显式委托:

```python
def _maybe_share_lm_head(self, model: nn.Module) -> None:
    AscendSpecDecodeBaseProposer._maybe_share_lm_head(self, model)
```

- 语义零损失:对 draft_model,Ascend 版等价于"不动 lm_head + ACL 图设置",与上游空方法对 lm_head 的意图一致
- 不改 MRO、不改基类、不影响 EAGLE/MTP 等其他方法
- 纯 Python MRO 模拟验证:修复前 update_stream 缺失 + wrapper 未武装(bug 复现);修复后两者恢复

## 服务器验证清单

```bash
git pull   # research/main
# 跑 test_SD.sh 的配置但去掉 cudagraph_mode=NONE(即开 FULL 或默认):
#    --speculative-config '{"method": "draft_model", ...}'
# 期望:部署过 update_stream 报错点;日志应出现
#   "[spec_decode/base] Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL..."
# 若 capture 阶段报新错(shape/pad 类),属下一阶段问题,与本 bug 无关
```

## 遗留 / 下一阶段

- [x] 服务器实际验证(部署 + 跑通 generate,0.22.1rc1)
- [x] **v0.23.0 复现证据(2026-08-18)**:`logs/v0.23.0-baseline-pr2-bug1-update-stream.txt`——官方 tag + PR2 的分支上,编译越过 bug2 后在图捕获阶段崩于 `_update_full_graph_params → self.update_stream → AttributeError`,与 0.22.1rc1 同一 bug(触发顺序不同:0.23.0 上 bug2 先挡路,需先修 bug2 才暴露 bug1)
- [ ] 上游 PR 提交(文案就绪,待验证链完成后回填)
