#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the Phase 2 hamming-sparse KVComp config jsons (research-only).

The runtime loader is ``KVCompConfig.from_json`` (vllm_ascend/worker/
kvcomp_utils.py), which does ``cls(**config_dict)``: every json key must
EXACTLY match a dataclass field name (note the upstream field is spelled
``seq_len_threshhold`` - double h), and unknown keys raise TypeError. The
schema below therefore mirrors ``asdict(KVCompConfig)`` field-for-field.
Cross-checked against the real dataclass via ast by test_kvcomp_json.py -
if a field is added/renamed upstream, that test fails and this generator
must be updated in the same commit.

Values = upstream defaults (already aligned with Qwen3-8B: 36 layers,
head_dim 128, chunk 128, "random" hash weights need no training artifact)
with only ``model_name`` (traceability) and ``vllm_hash_attention_topk``
(the tightness knob under study) changed. Topk is in TOKENS; the runtime
consumes topk // block_size (= topk // 128 chunks), so topk must stay a
multiple of 128.

Usage:
    python3 research/kvcomp/gen_kvcomp_json.py           # write all three
    python3 research/kvcomp/gen_kvcomp_json.py --check   # verify written files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

NUM_HIDDEN_LAYERS = 36  # Qwen3-8B (== upstream default)

# Sorted the same way as the dataclass body in kvcomp_utils.py to make
# visual diff against asdict() output trivial.
SCHEMA: dict = {
    "model_name": "Qwen3-8B",
    "is_mla": False,  # Qwen3 is GQA, not MLA
    "hash_weight_type": "random",  # random projection LSH; no trained weights
    "num_hidden_layers": NUM_HIDDEN_LAYERS,
    "seq_len_threshhold": 2048,  # upstream field name (double h) - do NOT "fix"
    "chunk_size": 128,  # == block_size 128 (hard NPU constraint)
    "chunk_repre_method": "max",
    "head_dim": 128,  # Qwen3-8B
    "hash_bits": 128,
    "top_k_ratio_per_layer": [0.3] * NUM_HIDDEN_LAYERS,
    "top_k_index_reuse": [-1] * NUM_HIDDEN_LAYERS,
    "must_select_blocks": [0, -2, -1],  # sink(first) + recent(last two)
    "hash_weight": None,  # unused when hash_weight_type == "random"
    "kv_lora_rank": 72,  # MLA-only fields, kept at upstream defaults
    "qk_rope_head_dim": 72,
    "hash_bits_kv_lora": 8,
    "hash_bits_qk_rope": 8,
    "hash_weight_kv_lora": None,
    "hash_weight_qk_rope": None,
    "vllm_hash_attention_topk": 4096,  # THE tightness knob (tokens)
    "vllm_hash_attention_reduction_head_num": None,
    "vllm_hash_attention_rollback_layers": [],  # consumed ONLY as layer-id list
    # LANDMINE (smoke 2026-09-01): consumed with TWO incompatible semantics -
    #   kvcomp_utils.py:601  `layer_index in <list>`       (layer-id list)
    #   attention_utils.py:115 `<list>[layer_index]`       (per-layer mask)
    # Upstream's own default [] crashes the mask read with IndexError on the
    # first decode forward (any model). [None]*36 satisfies BOTH readings
    # with no behavior change: None is falsy (no skip in decode) and no int
    # layer_index equals None (no skip layer at init). [0]*36 would silently
    # un-compress layer 0 via the `in` read (0 in [0]*36); [False]*36 too
    # (0 == False in Python). Upstream fix candidate (PR material): make
    # attention_utils.py:115 use `layer_index in ...` like the init path.
    "vllm_hash_attention_skip_layers": [None] * NUM_HIDDEN_LAYERS,
}

TOPKS = (2048, 4096, 8192)

# Fields that intentionally differ from upstream defaults, per topk.
VARYING_FIELDS = ("model_name", "vllm_hash_attention_topk")


def config_for_topk(topk: int) -> dict:
    cfg = {k: (list(v) if isinstance(v, list) else v) for k, v in SCHEMA.items()}
    cfg["vllm_hash_attention_topk"] = topk
    return cfg


def out_path(topk: int) -> Path:
    return Path(__file__).resolve().parent / f"qwen3-8b-topk{topk}.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify committed files instead of writing")
    args = ap.parse_args()

    for topk in TOPKS:
        cfg = config_for_topk(topk)
        if cfg["vllm_hash_attention_topk"] % 128 != 0:
            raise SystemExit(f"topk {topk} not a multiple of 128 (chunk granularity)")
        text = json.dumps(cfg, indent=4) + "\n"
        path = out_path(topk)
        if args.check:
            on_disk = json.loads(path.read_text())
            if on_disk != cfg:
                print(f"CHECK-FAIL: {path} does not match the generator schema")
                return 1
            print(f"ok {path.name} topk={topk} sha1={hashlib.sha1(text.encode()).hexdigest()[:12]}")
        else:
            path.write_text(text)
            print(f"wrote {path} topk={topk} sha1={hashlib.sha1(text.encode()).hexdigest()[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
