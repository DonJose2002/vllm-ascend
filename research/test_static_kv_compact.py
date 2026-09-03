"""UT for static KV compaction (Phase 2 B1). Runs on plain CPU torch.

Covers: selector arithmetic, kept-token math, manager surgery (null-block
trick + free ordering), scheduler eligibility/gates, runner view assembly
(gather + zero pad + seq_lens override), and wiring presence in
model_runner_v1.py / patch files (source-text asserts, same style as
test_kvcomp_json.py).

Run:  python3 research/test_static_kv_compact.py
"""

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "vllm_ascend" / "worker"))

import static_kv_compact as skc  # noqa: E402,I001


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeBlock:
    def __init__(self, block_id: int, cached: bool = False):
        self.block_id = block_id
        self.block_hash = "h" if cached else None


class FakeBlockPool:
    def __init__(self):
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def free_blocks(self, blocks, prepend: bool = False):
        self.calls.append((tuple(b.block_id for b in blocks), prepend))


class FakeManager:
    def __init__(self, num_blocks: int, block_size: int = 128, cached_from: int | None = None, req_id: str = "r1"):
        self.block_size = block_size
        self._null_block = FakeBlock(-1)
        self.block_pool = FakeBlockPool()
        self.req_to_blocks = {
            req_id: [FakeBlock(i, cached=cached_from is not None and i < cached_from) for i in range(num_blocks)]
        }

    def blocks_of(self, req_id: str = "r1"):
        return self.req_to_blocks[req_id]


class FakeRequest:
    def __init__(self, request_id: str, prompt_len: int, num_computed_tokens: int):
        self.request_id = request_id
        self.num_prompt_tokens = prompt_len
        self.num_computed_tokens = num_computed_tokens
        self.prompt_token_ids = [0] * prompt_len


class FakeSingleManagers:
    def __init__(self, managers):
        self.single_type_managers = managers


class FakeKVCacheManager:
    def __init__(self, managers, enable_caching: bool = False):
        self.enable_caching = enable_caching
        self.coordinator = FakeSingleManagers(managers)


class FakeParallelConfig:
    def __init__(self, world_size: int = 1):
        self.world_size = world_size


class FakeSchedulerConfig:
    def __init__(self, async_scheduling: bool = False):
        self.async_scheduling = async_scheduling


class FakeVllmConfig:
    def __init__(self, parallel=None, sched=None, speculative=None, additional=None, kv_transfer=None):
        self.parallel_config = parallel or FakeParallelConfig()
        self.scheduler_config = sched or FakeSchedulerConfig()
        self.speculative_config = speculative
        self.additional_config = additional
        self.kv_transfer_config = kv_transfer


class FakeScheduler:
    def __init__(self, requests, kv_cache_manager, vllm_config=None):
        self.requests = requests
        self.kv_cache_manager = kv_cache_manager
        self.vllm_config = vllm_config or FakeVllmConfig()


class FakeSchedulerOutput:
    def __init__(self, num_scheduled_tokens: dict):
        self.num_scheduled_tokens = num_scheduled_tokens


class FakeBlockTableObj:
    def __init__(self, np_table, num_blocks_per_row):
        self.block_table = type("Buf", (), {"np": np_table})()
        self.num_blocks_per_row = num_blocks_per_row


class FakeInputBatch:
    def __init__(self, np_table, num_blocks_per_row, num_tokens, req_id_to_index):
        self.block_table = [FakeBlockTableObj(np_table, num_blocks_per_row)]
        self.num_tokens = num_tokens
        self.req_id_to_index = req_id_to_index


class FakeRunner:
    def __init__(self, input_batch, optimistic_seq_lens, device="cpu"):
        self.input_batch = input_batch
        self.optimistic_seq_lens_cpu = optimistic_seq_lens
        self.device = torch.device(device)
        self.use_cp = False


def reset_module():
    skc.RECORDS.clear()
    skc._CHECKED.clear()
    skc._DISABLED_REASON = None
    skc.ENABLED = True


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_selector_noop_within_budget():
    # 16K prompt, 122 blocks; budget blocks = ceil(max(4096, 0.15*15642)/128)=32 -> compact
    keep = skc.select_keep_positions(122, 15642, 128)
    assert keep is not None
    # 4K prompt, 32 blocks; budget = max(4096, 617)=4096 -> 32 blocks >= 32 -> no-op
    assert skc.select_keep_positions(32, 4096, 128) is None
    # one block over budget (33 blocks vs 32 budget blocks) -> compacts
    keep33 = skc.select_keep_positions(33, 4224, 128)
    assert keep33 is not None and len(keep33) == 32


def test_selector_shape():
    keep = skc.select_keep_positions(245, 31233, 128)
    assert keep == sorted(set(keep))
    assert len(keep) == math.ceil(max(4096, math.ceil(31233 * 0.15)) / 128)
    assert keep[0] == 0  # sink
    assert keep[-4:] == [241, 242, 243, 244]  # recent incl tail
    kept = skc.kept_tokens_of(keep, 31233, 245, 128)
    assert kept == 31233 - (245 - len(keep)) * 128
    assert kept >= 4096


def test_manager_surgery():
    mgr = FakeManager(num_blocks=100)
    keep = [0, 1, 96, 97, 98, 99]
    freed = skc.compact_manager_blocks(mgr, "r1", keep)
    assert freed == 94
    blocks = mgr.blocks_of()
    for i in range(100):
        if i in keep:
            assert blocks[i].block_id == i
        else:
            assert blocks[i] is mgr._null_block
    assert mgr.block_pool.calls == [(tuple(range(2, 96)), True)]
    # idempotent
    assert skc.compact_manager_blocks(mgr, "r1", keep) == 0
    assert len(mgr.block_pool.calls) == 1


def test_manager_surgery_cached_first():
    mgr = FakeManager(num_blocks=10, cached_from=4)
    keep = [0, 9]
    freed = skc.compact_manager_blocks(mgr, "r1", keep)
    assert freed == 8
    kinds = mgr.block_pool.calls
    # cached (ids 1-3) freed first without prepend; uncached (4-8) with prepend
    assert kinds[0] == ((1, 2, 3), False)
    assert kinds[1] == ((4, 5, 6, 7, 8), True)


def test_maybe_compact_eligibility():
    reset_module()
    prompt_len = 15642
    mgr = FakeManager(num_blocks=122)
    sched = FakeScheduler(
        {"r1": FakeRequest("r1", prompt_len, num_computed_tokens=prompt_len)},
        FakeKVCacheManager([mgr]),
    )
    skc.maybe_compact_batch(sched, FakeSchedulerOutput({"r1": 1}))
    assert "r1" in skc.RECORDS
    rec = skc.RECORDS["r1"]
    assert rec.prompt_len == prompt_len
    assert rec.num_prompt_blocks == 122
    assert rec.freed_blocks == 122 - len(rec.keep_positions)
    # second step: already recorded, no rescan
    skc.maybe_compact_batch(sched, FakeSchedulerOutput({"r1": 1}))
    assert len(mgr.block_pool.calls) == 1

    # prefill incomplete -> no record, and not marked checked (recheck later)
    reset_module()
    mgr2 = FakeManager(num_blocks=122, req_id="r2")
    sched2 = FakeScheduler(
        {"r2": FakeRequest("r2", prompt_len, num_computed_tokens=8000)},
        FakeKVCacheManager([mgr2]),
    )
    skc.maybe_compact_batch(sched2, FakeSchedulerOutput({"r2": 2048}))
    assert "r2" not in skc.RECORDS and "r2" not in skc._CHECKED
    sched2.requests["r2"].num_computed_tokens = prompt_len
    skc.maybe_compact_batch(sched2, FakeSchedulerOutput({"r2": 1}))  # finished later
    assert "r2" in skc.RECORDS

    # short prompt -> checked-once skip, no record
    reset_module()
    mgr3 = FakeManager(num_blocks=32, req_id="r3")
    sched3 = FakeScheduler(
        {"r3": FakeRequest("r3", 4096, num_computed_tokens=4096)},
        FakeKVCacheManager([mgr3]),
    )
    skc.maybe_compact_batch(sched3, FakeSchedulerOutput({"r3": 1}))
    assert "r3" not in skc.RECORDS and "r3" in skc._CHECKED


def test_structural_gates():
    reset_module()
    bad_world = FakeVllmConfig(parallel=FakeParallelConfig(world_size=2))
    skc.check_structural_gates(FakeScheduler({}, FakeKVCacheManager([FakeManager(4)]), bad_world))
    assert "world_size" in skc.is_disabled()

    reset_module()
    async_sched = FakeVllmConfig(sched=FakeSchedulerConfig(async_scheduling=True))
    skc.check_structural_gates(FakeScheduler({}, FakeKVCacheManager([FakeManager(4)]), async_sched))
    assert "async" in skc.is_disabled()

    reset_module()
    caching_on = FakeScheduler({}, FakeKVCacheManager([FakeManager(4)], enable_caching=True))
    skc.check_structural_gates(caching_on)
    assert "prefix caching" in skc.is_disabled()

    reset_module()
    hamming_on = FakeVllmConfig(additional={"enable_hamming_sparse": True})
    skc.check_structural_gates(FakeScheduler({}, FakeKVCacheManager([FakeManager(4)]), hamming_on))
    assert "hamming" in skc.is_disabled()

    reset_module()
    multi_group = FakeScheduler({}, FakeKVCacheManager([FakeManager(4), FakeManager(4)]))
    skc.check_structural_gates(multi_group)
    assert "KV cache group" in skc.is_disabled()

    reset_module()
    ok = FakeScheduler({"r": FakeRequest("r", 100, 100)}, FakeKVCacheManager([FakeManager(4)]))
    assert skc.check_structural_gates(ok) is True
    assert skc.is_disabled() is None


def test_forget():
    reset_module()
    skc.RECORDS["x"] = 1
    skc._CHECKED.add("x")
    skc.forget("x")
    assert "x" not in skc.RECORDS and "x" not in skc._CHECKED


def test_prepare_runner_views():
    reset_module()
    rows, width = 4, 130
    np_table = torch.zeros(rows, width, dtype=torch.int32)
    num_blocks_per_row = [0, 122, 5, 0]
    np_table[1, :122] = torch.arange(1000, 1122, dtype=torch.int32)  # r1 engine row
    np_table[2, :5] = torch.arange(2000, 2005, dtype=torch.int32)
    num_tokens = [0, 15642 + 300, 640, 0]
    runner = FakeRunner(
        FakeInputBatch(np_table, num_blocks_per_row, num_tokens, {"r1": 1, "r2": 2}),
        torch.tensor([0, 15942, 640, 0], dtype=torch.int32),
    )

    # no records -> fast path None
    assert skc.prepare_runner_views(runner, 3) is None

    keep = [0, 50, 51, 118, 119, 120, 121]
    dropped = 122 - len(keep)
    skc.RECORDS["r1"] = skc.CompactRecord(
        request_id="r1",
        prompt_len=15642,
        num_prompt_blocks=122,
        keep_positions=keep,
        kept_tokens=15642 - dropped * 128,
        dropped_tokens=dropped * 128,
        freed_blocks=dropped,
    )
    views = skc.prepare_runner_views(runner, 3)
    assert views is not None
    # gathered row: keep positions, then appended tail blocks (none yet), zero pad
    row = views.block_table_device[1].tolist()
    expected = [1000 + i for i in keep] + [0] * (width - len(keep))
    assert row == expected
    # dense row untouched
    assert views.block_table_device[2].tolist() == (np_table[2].tolist())
    # seq_lens override: optimistic(15942) - dropped_tokens
    assert views.seq_lens_cpu[1].item() == 15942 - dropped * 128
    assert views.seq_lens_cpu[2].item() == 640

    # appended blocks after compaction appear after keep
    np_table[1, 122:124] = torch.tensor([3001, 3002], dtype=torch.int32)
    num_blocks_per_row[1] = 124
    views2 = skc.prepare_runner_views(runner, 3)
    row2 = views2.block_table_device[1].tolist()
    assert row2[: len(keep) + 2] == [1000 + i for i in keep] + [3001, 3002]

    # record for request not in batch -> unchanged rows, still active for others
    skc.RECORDS.pop("r1")
    skc.RECORDS["gone"] = skc.CompactRecord("gone", 8192, 64, [0, 63], 8192 - 62 * 128, 62 * 128, 62)
    assert skc.prepare_runner_views(runner, 3) is None


def test_runner_state_buffers_stable():
    reset_module()
    np_table = torch.zeros(2, 10, dtype=torch.int32)
    runner = FakeRunner(
        FakeInputBatch(np_table, [5, 0], [640, 0], {"r1": 0}),
        torch.tensor([640, 0], dtype=torch.int32),
    )
    skc.RECORDS["r1"] = skc.CompactRecord("r1", 640, 5, [0, 4], 640 - 3 * 128, 3 * 128, 3)
    v1 = skc.prepare_runner_views(runner, 1)
    v2 = skc.prepare_runner_views(runner, 1)
    assert v1.block_table_device.data_ptr() == v2.block_table_device.data_ptr()
    assert v1.seq_lens_device.data_ptr() == v2.seq_lens_device.data_ptr()


def test_wiring_presence():
    mr = (REPO_ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py").read_text()
    assert "from vllm_ascend.worker import static_kv_compact" in mr
    assert "static_kv_compact.prepare_runner_views(self, num_reqs_padded)" in mr
    assert "if static_kv_compact.ENABLED" in mr
    assert "kv_compact_views.seq_lens_device" in mr
    assert "kv_compact_views.seq_lens_cpu" in mr
    # dense path preserved verbatim when views inactive
    assert "if kv_compact_views is None" in mr

    patch_init = (REPO_ROOT / "vllm_ascend" / "patch" / "platform" / "__init__.py").read_text()
    assert 'os.getenv("VLLM_ASCEND_STATIC_KV_COMPACT", "0") == "1"' in patch_init
    assert "patch_static_kv_compact" in patch_init

    patch_file = (REPO_ROOT / "vllm_ascend" / "patch" / "platform" / "patch_static_kv_compact.py").read_text()
    assert "Scheduler.update_from_output" in patch_file
    assert "KVCacheManager.free" in patch_file
    assert "static_kv_compact.maybe_compact_batch" in patch_file
    assert "static_kv_compact.forget" in patch_file


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            reset_module()
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
