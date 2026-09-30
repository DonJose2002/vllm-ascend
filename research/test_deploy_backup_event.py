#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU unit test for the deploy-branch backup-page H2D fence (2026-09-30).

Branch deploy/v0.25.1rc1-fixes ports the #14922 use-after-rewrite fix to the
v0.25.1rc1 stack as an UNCONDITIONAL event-protocol fence (run E form,
25.9ms vs blocking 33.4ms on 910B3): in prepare_next_token_ids_padded the
pinned backup page rewrite is bracketed by event.synchronize() (entry) and
event.record() (exit), the copy itself staying non_blocking.

The proposer module cannot be imported without torch_npu, so the node under
test (_backup_h2d_fence) is extracted from the REAL source via ast and exec'd
into a stub namespace with a fake Event recording calls. A second wiring check
asserts source order at the call site: synchronize BEFORE the np-page rewrite,
record AFTER copy_to_gpu.

Run: python3 research/test_deploy_backup_event.py
"""

import ast
import sys
import types
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "vllm_ascend" / "spec_decode" / "llm_base_proposer.py"

HELPER = "_backup_h2d_fence"
SITE_FN = "prepare_next_token_ids_padded"


def _extract(path: Path, names: tuple[str, ...]):
    tree = ast.parse(path.read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            assert node.name not in found
            found[node.name] = node
    assert set(names) <= set(found), f"missing nodes: {set(names) - set(found)}"
    return found, tree


class _FakeEvent:
    calls: list[str] = []

    def __init__(self, blocking: bool = False) -> None:
        self.blocking = blocking
        self.calls = _FakeEvent.calls
        self.calls.append(f"ctor(blocking={blocking})")

    def synchronize(self) -> None:
        self.calls.append("synchronize")

    def record(self) -> None:
        self.calls.append("record")


def _load(torch_stub):
    (nodes, _), _ = _extract(SOURCE, (HELPER,)), None
    ns = {
        "torch": torch_stub,
        "__builtins__": __builtins__,
    }
    fn_src = ast.unparse(nodes[HELPER])
    code = "class _P:\n    " + fn_src.replace("\n", "\n    ")
    exec(compile(code, str(SOURCE), "exec"), ns)  # noqa: S102 - test harness
    return ns["_P"]


def _torch_with(fake_cls, *, npu=True):
    t = types.SimpleNamespace()
    if npu:
        t.npu = types.SimpleNamespace(Event=fake_cls)
    t.cuda = types.SimpleNamespace(Event=fake_cls)
    return t


def main() -> int:
    # lazy single creation, npu preferred, blocking passthrough
    _FakeEvent.calls.clear()
    cls = _load(_torch_with(_FakeEvent, npu=True))
    p = cls.__new__(cls)
    e1 = p._backup_h2d_fence()
    e2 = p._backup_h2d_fence()
    assert e1 is e2, "event must be created lazily exactly once"
    assert e1.blocking is True, "blocking=True must be passed through"
    assert _FakeEvent.calls == ["ctor(blocking=True)"], _FakeEvent.calls

    # idiom drive: entry synchronize (before rewrite) -> exit record (after copy)
    e1.synchronize()
    e1.record()
    assert _FakeEvent.calls == ["ctor(blocking=True)", "synchronize", "record"], _FakeEvent.calls

    # cuda fallback when torch.npu is absent
    _FakeEvent.calls.clear()
    cls2 = _load(_torch_with(_FakeEvent, npu=False))
    p2 = cls2.__new__(cls2)
    e3 = p2._backup_h2d_fence()
    assert isinstance(e3, _FakeEvent) and e3.blocking is True
    assert _FakeEvent.calls == ["ctor(blocking=True)"], _FakeEvent.calls

    # TypeError on blocking kwarg -> plain ctor fallback
    class _NoKwargEvent(_FakeEvent):
        def __init__(self) -> None:  # noqa: D107 - deliberately no kwargs
            _FakeEvent.calls.append("ctor(plain)")

    _FakeEvent.calls.clear()
    cls3 = _load(_torch_with(_NoKwargEvent, npu=True))
    p3 = cls3.__new__(cls3)
    e4 = p3._backup_h2d_fence()
    assert isinstance(e4, _NoKwargEvent)
    assert _FakeEvent.calls == ["ctor(plain)"], _FakeEvent.calls

    # wiring: at the real call site the fence brackets the rewrite+copy
    tree = ast.parse(SOURCE.read_text())
    site = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == SITE_FN:
            site = node
            break
    assert site is not None, f"{SITE_FN} not found"
    seg = ast.get_source_segment(SOURCE.read_text(), site)
    assert seg is not None
    i_sync = seg.index("self._backup_h2d_fence().synchronize()")
    i_write = seg.index("self.backup_next_token_ids.np[:num_reqs] =")
    i_copy = seg.index("self.backup_next_token_ids.copy_to_gpu(num_reqs)")
    i_record = seg.index("self._backup_h2d_fence().record()")
    assert i_sync < i_write < i_copy < i_record, (
        "fence order broken: synchronize must precede the np-page rewrite, "
        "record must follow copy_to_gpu"
    )
    # single rewrite site only (a second unp fenced writer would break the protocol)
    assert seg.count("backup_next_token_ids.np[") == 1

    print(
        "deploy backup-fence OK: lazy single creation, npu>cuda resolution,"
        " blocking passthrough + plain fallback, idiom drive, wiring order"
        " (sync < rewrite < copy < record), single-writer assertion"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
