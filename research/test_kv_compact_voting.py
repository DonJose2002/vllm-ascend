# SPDX-License-Identifier: Apache-2.0
"""B1.5 dwvote selector tests (CPU, no NPU).

Covers: vote-selector arithmetic, pending->finalize state machine, stride
fallback, vote-hook math on synthetic modules (a planted needle block must
win), eager-step gating, fail-open paths. Run:
    python research/test_kv_compact_voting.py            # standalone
    pytest research/test_kv_compact_voting.py --noconftest -q
"""

import sys
import types
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "vllm_ascend" / "worker"))
sys.path.insert(0, str(_HERE))

import static_kv_compact as skc  # noqa: E402,I001
import kv_compact_voting as kcv  # noqa: E402,I001
from test_static_kv_compact import (  # noqa: E402
    FakeKVCacheManager,
    FakeManager,
    FakeRequest,
    FakeScheduler,
    FakeSchedulerOutput,
)


class FakeRotary:
    def __call__(self, positions, q, k):
        return q, k


class FakeQKVProj:
    """q part = identity over hidden, k/v parts zeroed."""

    def __init__(self, hidden_size: int, out_size: int):
        self.weight = torch.zeros(out_size, hidden_size)
        self.weight[:hidden_size, :hidden_size] = torch.eye(hidden_size)

    def __call__(self, x):
        return x @ self.weight.t(), None


class FakeAttn:
    def __init__(self, hidden_size: int = 8, num_heads: int = 2, num_kv_heads: int = 1, head_dim: int = 4):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = head_dim**-0.5
        self.qkv_proj = FakeQKVProj(hidden_size, self.q_size + 2 * self.kv_size)
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()
        self.rotary_emb = FakeRotary()


class FakeRunner:
    def __init__(self, k_caches, np_row, qsl):
        self.kv_caches = k_caches
        self.query_start_loc = types.SimpleNamespace(cpu=torch.tensor(qsl, dtype=torch.int64))
        bt = types.SimpleNamespace(np=np_row)
        self.input_batch = types.SimpleNamespace(
            block_table=[types.SimpleNamespace(block_table=bt)],
            req_id_to_index={"r1": 0},
        )
        self.device = torch.device("cpu")


def reset_all():
    skc.RECORDS.clear()
    skc.PENDING_VOTES.clear()
    skc._CHECKED.clear()
    skc._DISABLED_REASON = None
    skc.ENABLED = True
    skc.SELECTOR = "dwvote"
    kcv._RUNNER = None
    kcv._HOOKS.clear()
    kcv._HOOKS_FAILED = False
    kcv._FIRST_VOTE_LOGGED = False


try:
    import pytest

    @pytest.fixture(autouse=True)
    def _reset_each():
        reset_all()

except ImportError:  # standalone main() resets explicitly
    pass


# ---------------------------------------------------------------------------
# pure selector arithmetic
# ---------------------------------------------------------------------------


def test_select_keep_by_votes_arithmetic():
    votes = torch.zeros(16)
    votes[5] = 10.0  # needle block
    keep = skc.select_keep_by_votes(votes, num_prompt_blocks=16, budget_blocks=6)
    assert keep == [0, 5, 12, 13, 14, 15], keep  # sink + top-vote + recent4


def test_select_keep_by_votes_noop():
    votes = torch.ones(4)
    assert skc.select_keep_by_votes(votes, num_prompt_blocks=4, budget_blocks=4) is None


def test_select_keep_by_votes_anchors_excluded_from_ranking():
    # top votes on sink/recent blocks must not waste budget slots
    votes = torch.zeros(16)
    votes[0] = 100.0  # sink
    votes[15] = 99.0  # recent
    votes[7] = 50.0
    keep = skc.select_keep_by_votes(votes, num_prompt_blocks=16, budget_blocks=6)
    assert keep == [0, 7, 12, 13, 14, 15], keep


# ---------------------------------------------------------------------------
# hook vote math
# ---------------------------------------------------------------------------


def _make_voting_fixture():
    """8 prompt blocks of 4 tokens; needle at prompt-block index 4 (block id 11)."""
    k_cache = torch.zeros(16, 4, 1, 4)
    k_cache[11, :, 0, :] = torch.tensor([2.0, 0.0, 0.0, 0.0])
    kv_pair = torch.stack([k_cache, torch.zeros_like(k_cache)])  # (2, blocks, bs, Hkv, D)
    np_row = np.array([[2, 5, 7, 9, 11, 13, 14, 15] + [0] * 8], dtype=np.int32)
    runner = FakeRunner([kv_pair], np_row, qsl=[0, 1])
    attn = FakeAttn()
    hidden = torch.zeros(1, 8)
    hidden[0, 0] = 1.0  # q head 0 -> [1,0,0,0]
    hidden[0, 4] = 1.0  # q head 1 -> [1,0,0,0]
    positions = torch.tensor([10])
    return runner, attn, positions, hidden


def test_vote_layer_needle_block_wins():
    runner, attn, positions, hidden = _make_voting_fixture()
    skc.PENDING_VOTES["r1"] = skc.PendingVote(request_id="r1", prompt_len=32, num_prompt_blocks=8)
    kcv._vote_layer(runner, attn, 0, positions, hidden, skc.PENDING_VOTES)
    pv = skc.PENDING_VOTES["r1"]
    assert pv.vote_steps == 1
    assert pv.votes_dev is not None and pv.votes_dev.shape == (8,)
    assert int(pv.votes_dev.argmax()) == 4, pv.votes_dev
    assert float(pv.votes_dev[4]) > 2 * float(pv.votes_dev[0])


def test_vote_layer_second_layer_accumulates_without_step_increment():
    runner, attn, positions, hidden = _make_voting_fixture()
    runner.kv_caches = [runner.kv_caches[0], runner.kv_caches[0].clone()]
    skc.PENDING_VOTES["r1"] = skc.PendingVote(request_id="r1", prompt_len=32, num_prompt_blocks=8)
    kcv._vote_layer(runner, attn, 0, positions, hidden, skc.PENDING_VOTES)
    kcv._vote_layer(runner, attn, 1, positions, hidden, skc.PENDING_VOTES)
    pv = skc.PENDING_VOTES["r1"]
    assert pv.vote_steps == 1  # only layer 0 counts steps
    assert float(pv.votes_dev[4]) > 0.2  # two layers accumulated


def test_hook_fail_open_on_exception():
    runner, attn, positions, hidden = _make_voting_fixture()
    kcv._RUNNER = runner  # hooks use the installed runner reference
    skc.PENDING_VOTES["r1"] = skc.PendingVote(request_id="r1", prompt_len=32, num_prompt_blocks=8)

    class BadQKV:
        def __call__(self, x):
            raise RuntimeError("boom")

    attn.qkv_proj = BadQKV()
    hook = kcv._make_hook(0)
    hook(attn, (positions, hidden), None)
    assert kcv._HOOKS_FAILED
    assert skc.PENDING_VOTES["r1"].vote_steps == skc.VOTE_STEPS  # force-finalized


# ---------------------------------------------------------------------------
# eager gate + finalize state machine
# ---------------------------------------------------------------------------


def test_needs_eager_step_gating():
    assert not kcv.needs_eager_step()  # empty pending
    skc.PENDING_VOTES["r1"] = skc.PendingVote("r1", 32768, 256, vote_steps=0)
    assert kcv.needs_eager_step()
    skc.PENDING_VOTES["r1"].vote_steps = skc.VOTE_STEPS
    assert not kcv.needs_eager_step()
    skc.PENDING_VOTES["r1"].vote_steps = 0
    skc.SELECTOR = "stride"
    assert not kcv.needs_eager_step()
    skc.SELECTOR = "dwvote"
    skc.ENABLED = False
    assert not kcv.needs_eager_step()


def _dwvote_scheduler():
    manager = FakeManager(num_blocks=256, block_size=128, req_id="r1")
    sched = FakeScheduler(
        requests={"r1": FakeRequest("r1", prompt_len=32768, num_computed_tokens=32768)},
        kv_cache_manager=FakeKVCacheManager([manager]),
    )
    return sched, manager


def test_finalize_with_votes_keeps_needle():
    sched, manager = _dwvote_scheduler()
    votes = torch.zeros(256)
    votes[100] = 42.0
    skc.PENDING_VOTES["r1"] = skc.PendingVote("r1", 32768, 256, vote_steps=skc.VOTE_STEPS, votes_dev=votes)
    skc.maybe_compact_batch(sched, FakeSchedulerOutput({"r1": 1}))
    assert "r1" not in skc.PENDING_VOTES
    rec = skc.RECORDS["r1"]
    assert 100 in rec.keep_positions
    assert 0 in rec.keep_positions and 255 in rec.keep_positions  # sink + recent tail
    assert rec.freed_blocks == 256 - len(rec.keep_positions)
    assert rec.num_prompt_blocks == 256


def test_finalize_zero_votes_stride_fallback():
    sched, _ = _dwvote_scheduler()
    skc.PENDING_VOTES["r1"] = skc.PendingVote("r1", 32768, 256, vote_steps=skc.VOTE_STEPS, votes_dev=torch.zeros(256))
    skc.maybe_compact_batch(sched, FakeSchedulerOutput({"r1": 1}))
    rec = skc.RECORDS["r1"]
    assert rec.keep_positions == skc.select_keep_positions(256, 32768, 128)


def test_finalize_request_gone():
    sched, _ = _dwvote_scheduler()
    sched.requests.clear()
    skc.PENDING_VOTES["r1"] = skc.PendingVote("r1", 32768, 256, vote_steps=3)
    skc.maybe_compact_batch(sched, FakeSchedulerOutput({}))
    assert "r1" not in skc.PENDING_VOTES and "r1" not in skc.RECORDS


def test_window_not_closed_no_finalize():
    sched, _ = _dwvote_scheduler()
    skc.PENDING_VOTES["r1"] = skc.PendingVote("r1", 32768, 256, vote_steps=2)
    skc.maybe_compact_batch(sched, FakeSchedulerOutput({"r1": 1}))
    assert "r1" in skc.PENDING_VOTES and "r1" not in skc.RECORDS


def test_dwvote_pending_created_and_stride_path_intact():
    sched, _ = _dwvote_scheduler()
    skc.maybe_compact_batch(sched, FakeSchedulerOutput({"r1": 1}))
    assert "r1" in skc.PENDING_VOTES and "r1" not in skc.RECORDS
    skc.PENDING_VOTES.clear()
    skc.SELECTOR = "stride"
    skc.maybe_compact_batch(sched, FakeSchedulerOutput({"r1": 1}))
    assert "r1" in skc.RECORDS  # stride commits immediately


def test_runner_wiring_assertions():
    runner_src = (_HERE.parent / "vllm_ascend" / "worker" / "model_runner_v1.py").read_text()
    assert "kv_compact_voting.needs_eager_step()" in runner_src
    assert "kv_compact_voting.maybe_install(self)" in runner_src
    assert 'set_stance("force_eager")' in runner_src
    assert "first vote recorded" in (_HERE.parent / "vllm_ascend" / "worker" / "kv_compact_voting.py").read_text()
    base = (_HERE / "run_baseline_npu.sh").read_text()
    assert "VLLM_ASCEND_KV_COMPACT_SELECTOR:-dwvote" in base
    assert "VLLM_ASCEND_KV_COMPACT_VOTE_STEPS:-8" in base


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            reset_all()
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
