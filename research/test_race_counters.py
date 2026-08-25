#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU unit test for the three-counter race probe (research-only).

The full module cannot be imported on a box without torch_npu/triton-ascend,
so the nodes under test (_RaceCounters, _oob_count, _env_flag,
_get_race_counters, prepare_next_token_ids_padded) are extracted from the
REAL source file via ast and exec'd into a stub namespace. This validates
the actual source text, not a copy of it.

Scenario A simulates the value story end to end through the real function:
REVIVE_RACE on + a no-op copy_to_gpu (SDMA copy "has not landed") + poisoned
backup.gpu -> c1 must see the poison, c2 must count the backup-selected row,
c3 must count the garbage flowing out of the where().

Scenario B is the fix active (blocking copy lands): c1/c3 stay flat, c2 still
counts the cond-false row.

Run: python3 research/test_race_counters.py
"""

import ast
import atexit
import os
import signal
import sys
import types
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / "vllm_ascend" / "spec_decode" / "llm_base_proposer.py"

WANTED = (
    "_RaceCounters",
    "_oob_count",
    "_oob_count_hard",
    "_env_flag",
    "_get_race_counters",
    "prepare_next_token_ids_padded",
)


def _extract_nodes(path: Path, names: tuple[str, ...]) -> dict[str, ast.AST]:
    found: dict[str, ast.AST] = {}
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            assert node.name not in found, f"duplicate definition of {node.name}"
            found[node.name] = node
    missing = set(names) - set(found)
    assert not missing, f"missing nodes in {path}: {sorted(missing)}"
    return found


class _Recorder:
    def __init__(self) -> None:
        self.records: list[str] = []

    def info(self, msg: str, *args: object) -> None:
        self.records.append(msg % args if args else msg)

    def warning(self, msg: str, *args: object) -> None:
        self.records.append(msg % args)


class _DeviceOperatorStub:
    @staticmethod
    def index_fill(t: torch.Tensor, dim: int, index: torch.Tensor, value: int) -> torch.Tensor:
        return t.index_fill(dim, index.long(), value)


def _make_namespace() -> dict:
    ns: dict[str, object] = {
        "torch": torch,
        "np": np,
        "os": os,
        "signal": signal,
        "suppress": suppress,
        "_INT64_MAX": torch.iinfo(torch.int64).max,
        "_INT64_MIN": torch.iinfo(torch.int64).min,
        "atexit": atexit,
        "logger": _Recorder(),
        "DeviceOperator": _DeviceOperatorStub,
        "_SD_COUNTERS": False,
        "_SD_REVIVE_RACE": False,
        "_RACE_COUNTERS": None,
    }
    return ns


def _make_input_batch(num_reqs: int, vocab: int, seq_lens: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_tokens_no_spec=np.array(seq_lens, dtype=np.int64),
        req_ids=[f"r{i}" for i in range(num_reqs)],
        vocab_size=vocab,
    )


def _make_backup(cpu_vals: list[int], gpu_vals: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        np=np.zeros(len(cpu_vals), dtype=np.int32),
        cpu=torch.tensor(cpu_vals, dtype=torch.int32),
        gpu=torch.tensor(gpu_vals, dtype=torch.int32),
        copy_to_gpu=lambda: None,  # scenario A: SDMA copy never lands
    )


def test_oob_count(ns: dict) -> None:
    oob = ns["_oob_count"]
    t = torch.tensor([0, 5, 99, 100, -1, 42], dtype=torch.int64)
    assert int(oob(t, 100)) == 2, "expected 2 OOB elements (100 and -1; 99/5/0/42 in-vocab)"
    assert int(oob(torch.tensor([0, 50], dtype=torch.int32), 100)) == 0
    assert oob(t, 100).dtype == torch.int64


def test_env_flag(ns: dict) -> None:
    env_flag = ns["_env_flag"]
    real = dict(os.environ)
    try:
        os.environ.pop("VLLM_ASCEND_TEST_FLAG", None)
        assert env_flag("VLLM_ASCEND_TEST_FLAG") is False
        os.environ["VLLM_ASCEND_TEST_FLAG"] = "0"
        assert env_flag("VLLM_ASCEND_TEST_FLAG") is False
        os.environ["VLLM_ASCEND_TEST_FLAG"] = "1"
        assert env_flag("VLLM_ASCEND_TEST_FLAG") is True
    finally:
        os.environ.clear()
        os.environ.update(real)


def test_counters_accumulate_and_report(ns: dict) -> None:
    ctr = ns["_get_race_counters"]()
    assert ctr is ns["_RACE_COUNTERS"], "singleton must be cached"
    assert any(r.startswith("[SD-counters] engaged") for r in ns["logger"].records), "engagement line"
    ctr.steps += 2
    ctr.bump(0, torch.tensor(1, dtype=torch.int64))
    ctr.bump(0, torch.tensor(2, dtype=torch.int64))
    ctr.bump(1, torch.tensor(3, dtype=torch.int64))
    ctr.bump(2, torch.tensor(4, dtype=torch.int64))
    ctr.bump(3, torch.tensor(5, dtype=torch.int64))
    assert ctr.counts.tolist() == [3, 3, 4, 5], ctr.counts.tolist()

    # SIGUSR1 live readout: handler registered in __init__ must fire on the
    # real signal and produce a logger line without setting `reported`
    # semantics for later exit-time reports.
    import time

    os.kill(os.getpid(), signal.SIGUSR1)
    for _ in range(50):
        if any("sigusr1" in r for r in ns["logger"].records):
            break
        time.sleep(0.02)
    line = next(r for r in ns["logger"].records if r.startswith("[SD-counters] sigusr1"))
    assert "steps=2 c1=3 c2=3 c3=4 c1x=5 esc=none" in line, line

    # file fallback: same line must land in /tmp/sd_counters_<pid>.txt
    fpath = f"/tmp/sd_counters_{os.getpid()}.txt"
    assert "[SD-counters] sigusr1 steps=2 c1=3 c2=3 c3=4 c1x=5 esc=none" in Path(fpath).read_text(), fpath

    ctr.report(origin="unit")
    line = ns["logger"].records[-1]
    assert line.startswith("[SD-counters] unit steps=2 c1=3 c2=3 c3=4 c1x=5 esc=none"), line
    n = len(ns["logger"].records)
    ctr.report(origin="again")  # idempotent
    assert len(ns["logger"].records) == n, "report must be idempotent"
    # fresh singleton for the functional test
    ns["_RACE_COUNTERS"] = None


def test_prepare_functional(ns: dict) -> None:
    VOCAB = 100
    # backup gpu: row0 = -1 (upstream sentinel, "OOB" for c1 but not c1x),
    # row1 = 9999 (real garbage), row2 = -1, row3 clean. cpu rows [:2] clean.
    backup = _make_backup([10, 11, 12, 13], [-1, 9999, -1, 13])
    stub_self = SimpleNamespace(backup_next_token_ids=backup)
    prepare = types.MethodType(ns["prepare_next_token_ids_padded"], stub_self)  # type: ignore[arg-type]
    gib = _make_input_batch(num_reqs=2, vocab=VOCAB, seq_lens=[5, 6])
    requests = {
        "r0": SimpleNamespace(get_token_id=lambda n: 42),
        "r1": SimpleNamespace(get_token_id=lambda n: 43),
    }
    sampled = torch.tensor([[1, 2, 3], [7, 7, 7], [7, 7, 7]], dtype=torch.int64)
    discard = torch.tensor([1], dtype=torch.int64)

    # --- scenario C: everything off (default) -> no counters, fix path taken
    ns["_SD_COUNTERS"] = False
    ns["_SD_REVIVE_RACE"] = False
    nxt, cnt = prepare(sampled, requests, gib, discard, 1)
    assert ns["_RACE_COUNTERS"] is None, "counters must stay uncreated when off"
    # scenario C's blocking copy just cleaned gpu[:2]; re-poison for scenario A
    backup.gpu.copy_(torch.tensor([-1, 9999, -1, 13], dtype=torch.int32))

    # --- scenario A: value story simulated (revive + un-landed copy + poison)
    ns["_SD_COUNTERS"] = True
    ns["_SD_REVIVE_RACE"] = True
    nxt, cnt = prepare(sampled, requests, gib, discard, 1)
    ctr = ns["_RACE_COUNTERS"]
    assert ctr is not None and ctr.steps == 1
    assert ctr.counts.tolist() == [2, 1, 1, 1], (
        f"scenario A expected c1=2 (sentinel+9999) c2=1 c3=1 c1x=1, got {ctr.counts.tolist()}"
    )
    assert nxt.tolist() == [3, 9999, 7], nxt.tolist()  # row1 took the garbage backup
    assert cnt.tolist() == [3, 0, 3], cnt.tolist()
    assert int(ctr.esc_min) == 9999 and int(ctr.esc_max) == 9999, (
        f"esc range should capture the escaped 9999, got [{int(ctr.esc_min)}, {int(ctr.esc_max)}]"
    )

    # --- scenario B: fix active (blocking copy lands over [:num_reqs])
    ns["_SD_REVIVE_RACE"] = False
    nxt, cnt = prepare(sampled, requests, gib, discard, 1)
    assert ctr.steps == 2
    assert ctr.counts.tolist() == [2, 2, 1, 1], f"B: only c2 grows, got {ctr.counts.tolist()}"
    assert nxt.tolist() == [3, 11, 7], nxt.tolist()  # row1 now reads the landed cpu value
    assert int(ctr.esc_min) == 9999 and int(ctr.esc_max) == 9999, "esc unchanged in B"

    ctr.report(origin="functional")
    line = ns["logger"].records[-1]
    assert "steps=2 c1=2 c2=2 c3=1 c1x=1 esc=[9999, 9999]" in line, line


def main() -> int:
    nodes = _extract_nodes(SOURCE, WANTED)
    ns = _make_namespace()
    for name in WANTED:
        exec(compile(ast.Module(body=[nodes[name]], type_ignores=[]), str(SOURCE), "exec"), ns)

    test_oob_count(ns)
    print("PASS oob_count")
    test_env_flag(ns)
    print("PASS env_flag")
    test_counters_accumulate_and_report(ns)
    print("PASS counters accumulate/report/idempotent")
    test_prepare_functional(ns)
    print("PASS prepare_next_token_ids_padded functional (A: value story, B: fix, C: off)")
    print("all race-counter tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
