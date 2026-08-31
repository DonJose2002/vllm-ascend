#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU unit test for the event-protocol fence (research-only, 2026-08-28, run E).

The proposer module cannot be imported on a box without torch_npu, so the
node under test (_sd_backup_event) is extracted from the REAL source file
via ast and exec'd into a stub namespace whose torch.npu.Event is a fake
recording every synchronize()/record() call - the only device-dependent
bit; the lazy-creation + resolution logic itself is device-free.

Checks:
  1. lazy creation: exactly one event, persisted on the instance;
  2. torch.npu.Event preferred when present, torch.cuda.Event fallback;
  3. blocking=True passed through; TypeError falls back to the plain ctor;
  4. callers can drive the upstream idiom: synchronize() (entry, before the
     host rewrite) then record() (exit, after the async copy).

Run: python3 research/test_event_copy.py
"""

import ast
import sys
import types
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "vllm_ascend" / "spec_decode" / "llm_base_proposer.py"

WANTED = ("_sd_backup_event",)


def _extract(path: Path, names: tuple[str, ...]):
    tree = ast.parse(path.read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            assert node.name not in found
            found[node.name] = node
    assert set(names) <= set(found), f"missing nodes: {set(names) - set(found)}"
    return found


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
    nodes = _extract(SOURCE, WANTED)
    ns = {
        "torch": torch_stub,
        "logger": types.SimpleNamespace(info=lambda *a, **k: None),
        "__builtins__": __builtins__,
    }
    fn_src = ast.unparse(nodes["_sd_backup_event"])
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
    # 1+2+3: npu preferred, blocking passthrough, lazy single creation
    _FakeEvent.calls.clear()
    cls = _load(_torch_with(_FakeEvent, npu=True))
    p = cls.__new__(cls)
    e1 = p._sd_backup_event()
    e2 = p._sd_backup_event()
    assert e1 is e2, "event must be created lazily exactly once"
    assert e1.blocking is True, "blocking=True must be passed through"
    assert _FakeEvent.calls == ["ctor(blocking=True)"], _FakeEvent.calls

    # 4: the upstream idiom drives cleanly (entry sync before rewrite,
    #    exit record after the async copy)
    e1.synchronize()
    e1.record()
    assert _FakeEvent.calls == ["ctor(blocking=True)", "synchronize", "record"], _FakeEvent.calls

    # 2: cuda fallback when torch.npu is absent
    _FakeEvent.calls.clear()
    cls2 = _load(_torch_with(_FakeEvent, npu=False))
    p2 = cls2.__new__(cls2)
    e3 = p2._sd_backup_event()
    assert isinstance(e3, _FakeEvent) and e3.blocking is True
    assert _FakeEvent.calls == ["ctor(blocking=True)"], _FakeEvent.calls

    # 3: TypeError on blocking kwarg -> plain ctor fallback
    class _NoKwargEvent(_FakeEvent):
        def __init__(self) -> None:  # noqa: D107 - deliberately no kwargs
            _FakeEvent.calls.append("ctor(plain)")

    _FakeEvent.calls.clear()
    cls3 = _load(_torch_with(_NoKwargEvent, npu=True))
    p3 = cls3.__new__(cls3)
    e4 = p3._sd_backup_event()
    assert isinstance(e4, _NoKwargEvent)
    assert _FakeEvent.calls == ["ctor(plain)"], _FakeEvent.calls

    print(
        "event-protocol fence OK: lazy single creation, npu>cuda resolution,"
        " blocking passthrough + plain fallback, idiom drive verified"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
