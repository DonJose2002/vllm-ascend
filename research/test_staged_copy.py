#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU unit test for the staged-copy ring (research-only, 2026-08-28).

The proposer module cannot be imported on a box without torch_npu, so the
node under test (_sd_stage_next_page) plus the _SD_STAGED_COPY constant are
extracted from the REAL source file via ast and exec'd into a stub namespace
(torch.zeros forced to pin_memory=False - the only device-dependent bit; the
rotation logic itself is device-free).

Checks:
  1. engagement allocates exactly depth pages, once;
  2. rotation visits pages round-robin 0,1,...,depth-1,0,... across steps;
  3. repeated calls never re-allocate (turn state persists).

Run: python3 research/test_staged_copy.py
"""

import ast
import sys
import types
from pathlib import Path

import torch as real_torch

SOURCE = Path(__file__).resolve().parents[1] / "vllm_ascend" / "spec_decode" / "llm_base_proposer.py"

WANTED = ("_sd_stage_next_page",)


def _extract(path: Path, names: tuple[str, ...]):
    tree = ast.parse(path.read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            assert node.name not in found
            found[node.name] = node
    # module-level assignment _SD_STAGED_COPY = int(...)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_SD_STAGED_COPY":
                    found["_SD_STAGED_COPY_expr"] = node.value
    assert set(names) <= set(found), f"missing nodes: {set(names) - set(found)}"
    return found


class _PinFreeTorch(types.SimpleNamespace):
    """Stand-in for the torch module: zeros() drops pin_memory (CPU box)."""

    @staticmethod
    def zeros(*a, **k):
        k.pop("pin_memory", None)
        return real_torch.zeros(*a, **k)

    @staticmethod
    def inference_mode(enabled=True):
        return real_torch.inference_mode(enabled)


def _load(depth: int):
    nodes = _extract(SOURCE, WANTED)
    ns = {
        "torch": _PinFreeTorch(),
        "logger": types.SimpleNamespace(info=lambda *a, **k: None),
        "_SD_STAGED_COPY": depth,
        "__builtins__": __builtins__,
    }
    mod = ast.Module(body=list(nodes.values()), type_ignores=[])
    ast.fix_missing_locations(mod)
    # wrap the plain function node as a method via a def under a dummy class
    fn_src = ast.unparse(nodes["_sd_stage_next_page"])
    code = "class _P:\n    " + fn_src.replace("\n", "\n    ")
    exec(compile(code, str(SOURCE), "exec"), ns)  # noqa: S102 - test harness
    return ns["_P"]


def main() -> int:
    depth = 4
    cls = _load(depth)
    proposer = cls.__new__(cls)
    # fake backup buffer: 16 int32 slots (shape only; contents irrelevant)
    ns_buf = types.SimpleNamespace(cpu=real_torch.zeros(16, dtype=real_torch.int32))
    proposer.backup_next_token_ids = ns_buf

    picked = [id(proposer._sd_stage_next_page()) for _ in range(2 * depth + 3)]
    pages = list(dict.fromkeys(picked))
    assert len(pages) == depth, f"expected {depth} distinct pages, got {len(pages)}"
    # round-robin: first `depth` picks are all distinct, then the cycle repeats
    assert picked[:depth] == picked[depth : 2 * depth], "ring does not repeat after one cycle"
    assert picked[0] == picked[2 * depth], "cycle length != depth"
    # no re-allocation: state persists on the instance
    assert proposer._sd_stage_state["turn"] == len(picked)
    assert len(proposer._sd_stage_state["pages"]) == depth
    print(f"staged-copy ring OK: depth={depth}, steps={len(picked)}, rotation verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
