#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU unit test for the Phase 2 kvcomp jsons + hamming shell wiring (research-only).

kvcomp_utils.py cannot be imported on a box without torch/torch_npu/vllm, so
the KVCompConfig dataclass fields+defaults are extracted from the REAL source
via ast (same trick as test_race_counters.py). This validates the actual
upstream schema, not a copy.

Checks:
  1. schema: each research/kvcomp/*.json key set == KVCompConfig field set
     exactly (from_json does cls(**dict): unknown key -> TypeError, missing
     key silently falls back to default - both are drift we must catch).
  2. values: every field equals the upstream default except the intentionally
     varying pair (model_name, vllm_hash_attention_topk).
  3. runtime invariants: chunk_size % 128 == 0, hash_bits % 8 == 0,
     topk % 128 == 0 (runtime consumes topk // block_size), per-layer lists
     have len == num_hidden_layers, random weights => hash_weight is None.
  4. cross-file: the three jsons differ ONLY in vllm_hash_attention_topk and
     it matches the topk in each filename.
  5. shell wiring: run_baseline_npu.sh hamming case - simulate SPEC_ARGS for
     each topk, json.loads the embedded --additional-config fragment, assert
     the dual gate (enable_hamming_sparse bool + hamming_sparse dict with
     BOTH enabled and sparse_json_location - a partial dict KeyErrors in
     AscendConfig), and assert the mutual-exclusion/graph-mode/tiers-default
     guard lines exist.

Run: python3 research/test_kvcomp_json.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KVCOMP_SOURCE = REPO / "vllm_ascend" / "worker" / "kvcomp_utils.py"
RUN_SCRIPT = REPO / "research" / "run_baseline_npu.sh"
KVCOMP_DIR = REPO / "research" / "kvcomp"
TOPKS = (2048, 4096, 8192)

# Fields we intentionally vary away from upstream defaults (see gen_kvcomp_json.py).
VARYING_FIELDS = {"model_name", "vllm_hash_attention_topk"}
# Fields whose upstream default is BROKEN at runtime (not merely unsuitable):
# vllm_hash_attention_skip_layers default [] is read BOTH as a layer-id list
# (kvcomp_utils.py:601, fine) AND as a per-layer mask (attention_utils.py:115,
# IndexError on any index). We ship [None]*36 which satisfies both readings.
LANDMINE_FIXED_FIELDS = {"vllm_hash_attention_skip_layers"}


def dataclass_defaults(path: Path) -> dict[str, object]:
    """Field name -> evaluated default for the KVCompConfig dataclass."""
    tree = ast.parse(path.read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "KVCompConfig")
    out: dict[str, object] = {}
    for node in cls.body:
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        name = node.target.id
        if node.value is None:
            out[name] = "<no-default>"  # would be a required ctor arg - none exist today
            continue
        val = node.value
        if isinstance(val, ast.Call) and ast.unparse(val.func) == "field":
            factory = next(kw for kw in val.keywords if kw.arg == "default_factory")
            lam = factory.value
            assert isinstance(lam, ast.Lambda), f"{name}: unsupported default_factory"
            # eval the lambda BODY (eval'ing the lambda itself returns a function)
            out[name] = eval(ast.unparse(lam.body), {"__builtins__": {}})  # noqa: S307 - pure list literals
        else:
            out[name] = ast.literal_eval(val)
    return out


def load_jsons() -> dict[int, dict]:
    return {t: json.loads((KVCOMP_DIR / f"qwen3-8b-topk{t}.json").read_text()) for t in TOPKS}


def test_schema_and_values(defaults: dict[str, object], jsons: dict[int, dict]) -> None:
    want_fields = set(defaults)
    for topk, cfg in jsons.items():
        got = set(cfg)
        assert got == want_fields, (
            f"topk{topk}: json keys != KVCompConfig fields\n"
            f"  extra (TypeError at from_json):   {sorted(got - want_fields)}\n"
            f"  missing (silent default fallback): {sorted(want_fields - got)}"
        )
        for k, v in cfg.items():
            if k in VARYING_FIELDS or k in LANDMINE_FIXED_FIELDS:
                continue
            assert v == defaults[k], f"topk{topk}: field {k}: json={v!r} != upstream default {defaults[k]!r}"


def test_invariants(jsons: dict[int, dict]) -> None:
    for topk, cfg in jsons.items():
        assert cfg["num_hidden_layers"] == 36, "Qwen3-8B has 36 layers"
        assert cfg["is_mla"] is False, "Qwen3-8B is GQA"
        assert cfg["hash_weight_type"] == "random", "no trained hash weights exist for Qwen3-8B"
        assert cfg["hash_weight"] is None, "random weights must not carry a hash_weight matrix"
        assert cfg["chunk_size"] % 128 == 0
        assert cfg["hash_bits"] % 8 == 0, "HashEncoder packs bits into uint8"
        assert cfg["vllm_hash_attention_topk"] % 128 == 0, "runtime uses topk // block_size"
        assert cfg["vllm_hash_attention_topk"] == topk, "topk must match the filename"
        assert len(cfg["top_k_ratio_per_layer"]) == cfg["num_hidden_layers"]
        assert len(cfg["top_k_index_reuse"]) == cfg["num_hidden_layers"]
        assert cfg["must_select_blocks"] == [0, -2, -1], "sink + recent anchors (design §2.2)"
        # Dual-semantics landmine guard (smoke 2026-09-01 IndexError):
        # attention_utils.py:115 does skip_layers[layer_index] -> needs len>=36
        # AND falsy per entry; kvcomp_utils.py:601 does `layer_index in list`
        # -> no entry may equal any int layer index. [None]*36 is the only
        # value class satisfying both; [] crashes, [0]/[False]-masks snare
        # layer 0 via the membership read.
        skip = cfg["vllm_hash_attention_skip_layers"]
        assert len(skip) == cfg["num_hidden_layers"], "skip_layers must be mask-length for the [idx] read"
        assert all(v is None for v in skip), "skip_layers entries must be None (falsy AND != any int)"
        assert all(not skip[i] for i in range(cfg["num_hidden_layers"])), "no layer may read as skipped"
        assert all(i not in skip for i in range(cfg["num_hidden_layers"])), "membership read must match nothing"


def test_cross_file(jsons: dict[int, dict]) -> None:
    base = {k: v for k, v in jsons[4096].items() if k != "vllm_hash_attention_topk"}
    for topk, cfg in jsons.items():
        diff = {k for k in base if cfg[k] != base[k]}
        assert not diff, f"topk{topk} differs from topk4096 outside the knob: {sorted(diff)}"


def _hamming_case_block(script: str) -> str:
    m = re.search(r"^  hamming\|hammingsd\)\n(.*?)^  \*\)\n", script, re.MULTILINE | re.DOTALL)
    assert m, "hamming case not found in run_baseline_npu.sh"
    return m.group(1)


def test_shell_wiring() -> None:
    script = RUN_SCRIPT.read_text()
    block = _hamming_case_block(script)

    # The dual-gate additional-config fragment, simulated for each topk.
    m = re.search(r'SPEC_ARGS="--additional-config (.*?)"\n', block)
    assert m, "hamming SPEC_ARGS assignment not found"
    frag_raw = m.group(1)
    for topk in TOPKS:
        fake = f"/abs/research/kvcomp/qwen3-8b-topk{topk}.json"
        frag = frag_raw.replace('\\"', '"').replace("$KVCOMP_JSON", fake)
        cfg = json.loads(frag)
        assert cfg["enable_hamming_sparse"] is True, "attention-layer gate must be on"
        hs = cfg["hamming_sparse"]
        assert set(hs) == {"enabled", "sparse_json_location"}, (
            f"runner-gate dict keys {sorted(hs)}: AscendConfig indexes both - a "
            "partial dict KeyErrors at startup (ascend_config.py:345-346)"
        )
        assert hs["enabled"] is True
        assert hs["sparse_json_location"] == fake, "json path must be absolute (serve CWD independence)"

    # Guard lines that must exist (crash-avoidance + experiment hygiene).
    for needle in (
        "KVCOMP_JSON=",  # json existence pre-check before serve
        'if [ ! -f "$KVCOMP_JSON" ]',  # fail fast on missing json
        "readlink -f",  # absolutize
        "seq_len_threshhold",  # docs anchor
        "TIERS_EXPLICIT",  # 4K-tier drop only when caller did not pin TIERS
        "CONCS_EXPLICIT",  # same for CONCS
        "not speculative_config",  # mutual-exclusion evidence note
        "GRAPH_MODE",  # FULL-graph risk fallback hook
    ):
        assert needle in block, f"hamming case lost guard line: {needle!r}"

    # SUMMARY self-evidence emission exists (kvcomp logs nothing itself).
    assert 'echo "hamming: topk=' in script, "SUMMARY lost the hamming: self-evidence line"
    # NIAH plumbing exists and is off by default.
    assert 'NIAH="${NIAH:-0}"' in script
    assert "needle_eval.py run" in script and "needle_eval.py curve" in script


def main() -> int:
    defaults = dataclass_defaults(KVCOMP_SOURCE)
    jsons = load_jsons()
    test_schema_and_values(defaults, jsons)
    test_invariants(jsons)
    test_cross_file(jsons)
    test_shell_wiring()
    n_fields = len(defaults)
    print(f"PASS: {len(jsons)} kvcomp jsons x {n_fields} fields == upstream schema (ast-verified), wiring ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
