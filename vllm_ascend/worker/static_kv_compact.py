"""Static KV compaction (Phase 2 B-line, research).

SnapKV-style one-shot static selection with true block release, gated by
``VLLM_ASCEND_STATIC_KV_COMPACT=1`` (default off). Design: prefill 末一次性选择
保留块,decode 零 python / 零 tensor 改写(图兼容赌注经 metadata 通道验证)。

Three cooperating pieces (see experiments/phase2-kv-compression-design.md §4):
1. Scheduler-side hook (patch/platform/patch_static_kv_compact.py): wrapped
   ``Scheduler.update_from_output`` detects prefill completion, runs the
   conservative selector, surgically frees evicted blocks (null-block trick,
   mirroring SWA ``remove_skipped_blocks``) and records a per-request view.
2. Engine-side hook (model_runner_v1.py ``_initialize_attn_metadata``): when
   records are active, builds a gathered block-table view + compacted seq_lens
   into scratch buffers and overrides the common attention metadata. Positions,
   num_computed_tokens, is_prefilling and query_start_loc stay at full ground
   truth (hamming semantics: query RoPE at the true absolute position, keys at
   their original phases, visible set = compacted view).
3. Cleanup: wrapped ``KVCacheManager.free`` forgets records on finish/preempt.

Structural gates (research scope): world_size == 1 (scheduler and runner must
share the process), async scheduling off, speculative decoding off, prefix
caching off (``--no-enable-prefix-caching``), single full-attention KV group,
context parallelism off, hamming sparse off.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

import torch

_log = logging.getLogger(__name__)

ENV_MASTER = "VLLM_ASCEND_STATIC_KV_COMPACT"
ENV_BUDGET_TOKENS = "VLLM_ASCEND_KV_COMPACT_BUDGET_TOKENS"
ENV_MIN_PROMPT_LEN = "VLLM_ASCEND_KV_COMPACT_MIN_LEN"

ENABLED = os.environ.get(ENV_MASTER, "0") == "1"
BUDGET_TOKENS = int(os.environ.get(ENV_BUDGET_TOKENS, "4096"))
MIN_PROMPT_LEN = int(os.environ.get(ENV_MIN_PROMPT_LEN, "8192"))
RATIO_FLOOR = 0.15
SINK_BLOCKS = 1
RECENT_BLOCKS = 4

RECORDS: dict[str, CompactRecord] = {}
_CHECKED: set[str] = set()
_DISABLED_REASON: str | None = None
_ACTIVE_LOGGED = False
_CANDIDATE_LOGGED = False


@dataclass
class CompactRecord:
    request_id: str
    prompt_len: int
    num_prompt_blocks: int
    keep_positions: list[int]
    kept_tokens: int
    dropped_tokens: int
    freed_blocks: int


@dataclass
class RunnerViews:
    block_table_device: torch.Tensor
    seq_lens_device: torch.Tensor
    seq_lens_cpu: torch.Tensor


@dataclass
class _RunnerState:
    block_table_cpu: torch.Tensor | None = None
    block_table_device: torch.Tensor | None = None
    seq_lens_cpu_buf: torch.Tensor | None = None
    seq_lens_device_buf: torch.Tensor | None = None


def forget(request_id: str) -> None:
    RECORDS.pop(request_id, None)
    _CHECKED.discard(request_id)


def disable(reason: str) -> None:
    global _DISABLED_REASON
    if _DISABLED_REASON is None:
        _DISABLED_REASON = reason
        _log.warning("[static-kv-compact] coordinator disabled: %s", reason)
    RECORDS.clear()


def is_disabled() -> str | None:
    return _DISABLED_REASON


def select_keep_positions(
    num_prompt_blocks: int,
    prompt_len: int,
    block_size: int,
    sink_blocks: int = SINK_BLOCKS,
    recent_blocks: int = RECENT_BLOCKS,
    budget_tokens: int = BUDGET_TOKENS,
    ratio_floor: float = RATIO_FLOOR,
) -> list[int] | None:
    """Conservative selector: sink + recent + uniform stride to fill budget.

    Returns sorted engine-row positions to keep, or None when the prompt
    already fits the budget (no-op). The tail block is always kept (it carries
    the partial tokens decode appends into).
    """
    budget = max(budget_tokens, math.ceil(prompt_len * ratio_floor))
    budget_blocks = math.ceil(budget / block_size)
    if num_prompt_blocks <= budget_blocks:
        return None
    keep = set(range(sink_blocks))
    keep.update(range(num_prompt_blocks - recent_blocks, num_prompt_blocks))
    need = budget_blocks - len(keep)
    if need > 0:
        middle_len = num_prompt_blocks - recent_blocks - sink_blocks
        if middle_len > 0:
            for i in range(need):
                pos = sink_blocks + (i * middle_len) // need
                if pos >= num_prompt_blocks - recent_blocks:
                    pos = num_prompt_blocks - recent_blocks - 1
                keep.add(pos)
    return sorted(keep)


def kept_tokens_of(keep_positions: list[int], prompt_len: int, num_prompt_blocks: int, block_size: int) -> int:
    """Dropped positions are always full blocks; the tail stays in recent."""
    dropped = num_prompt_blocks - len(keep_positions)
    return prompt_len - dropped * block_size


def compact_manager_blocks(manager, request_id: str, keep_positions: list[int]) -> int:
    """Surgical free on one SingleTypeKVCacheManager (null-block trick).

    Mirrors SWA ``remove_skipped_blocks``: evicted positions become
    ``manager._null_block`` (list length and all positional arithmetic are
    preserved), cached blocks are freed first, uncached blocks are freed with
    ``prepend=True`` so they are the next allocation candidates.
    """
    blocks = manager.req_to_blocks.get(request_id)
    if blocks is None:
        return 0
    null_block = manager._null_block
    keep = set(keep_positions)
    cached_free: list = []
    uncached_free: list = []
    for i, block in enumerate(blocks):
        if i in keep or block is null_block or block == null_block:
            continue
        if block.block_hash is not None:
            cached_free.append(block)
        else:
            uncached_free.append(block)
        blocks[i] = null_block
    if cached_free:
        manager.block_pool.free_blocks(cached_free)
    if uncached_free:
        manager.block_pool.free_blocks(uncached_free, prepend=True)
    return len(cached_free) + len(uncached_free)


def _request_prompt_len(request) -> int:
    num_prompt_tokens = getattr(request, "num_prompt_tokens", None)
    if num_prompt_tokens is not None:
        return int(num_prompt_tokens)
    return len(request.prompt_token_ids)


def check_structural_gates(scheduler) -> bool:
    """One-time config gates; returns False (and disables) when not eligible."""
    global _DISABLED_REASON
    if _DISABLED_REASON is not None:
        return False
    vllm_config = scheduler.vllm_config
    if vllm_config.parallel_config.world_size != 1:
        disable("parallel_config.world_size != 1 (scheduler/runner must share one process)")
        return False
    if getattr(vllm_config.scheduler_config, "async_scheduling", False):
        disable("async_scheduling enabled (compaction must run strictly between step N and N+1)")
        return False
    if vllm_config.speculative_config is not None:
        disable("speculative decoding enabled (B-line is dense-only; Phase 4 territory)")
        return False
    additional_config = vllm_config.additional_config or {}
    if additional_config.get("enable_hamming_sparse") or additional_config.get("hamming_sparse"):
        disable("hamming sparse enabled (mutually exclusive with static compaction)")
        return False
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is not None and getattr(kv_transfer_config, "kv_connector_module_name", None):
        disable("kv transfer connector configured (connector owns block lifecycle)")
        return False
    manager = scheduler.kv_cache_manager
    if getattr(manager, "enable_caching", False):
        disable("prefix caching enabled (B1 requires --no-enable-prefix-caching)")
        return False
    if len(manager.coordinator.single_type_managers) != 1:
        disable("multi KV cache group (hybrid/SWA) not supported in B1")
        return False
    return True


def maybe_compact_batch(scheduler, scheduler_output) -> None:
    """Scheduler-side entry: called from the wrapped update_from_output."""
    global _ACTIVE_LOGGED, _CANDIDATE_LOGGED
    if not ENABLED or _DISABLED_REASON is not None:
        return
    if not check_structural_gates(scheduler):
        return
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens
    if not num_scheduled_tokens:
        return
    if not _ACTIVE_LOGGED:
        _ACTIVE_LOGGED = True
        _log.info(
            "[static-kv-compact] coordinator active (gates passed, min_len=%d budget=%d)",
            MIN_PROMPT_LEN,
            BUDGET_TOKENS,
        )
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    block_size = manager.block_size
    for request_id in num_scheduled_tokens:
        if request_id in RECORDS or request_id in _CHECKED:
            continue
        request = scheduler.requests.get(request_id)
        if request is None:
            continue
        prompt_len = _request_prompt_len(request)
        if prompt_len < MIN_PROMPT_LEN:
            _CHECKED.add(request_id)
            continue
        if request.num_computed_tokens < prompt_len:
            continue
        blocks = manager.req_to_blocks.get(request_id)
        num_prompt_blocks = len(blocks) if blocks else 0
        if not _CANDIDATE_LOGGED:
            _CANDIDATE_LOGGED = True
            _log.info(
                "[static-kv-compact] first candidate seen: req=%s prompt=%d computed=%d blocks=%d",
                request_id,
                prompt_len,
                request.num_computed_tokens,
                num_prompt_blocks,
            )
        keep = select_keep_positions(num_prompt_blocks, prompt_len, block_size)
        if keep is None:
            _CHECKED.add(request_id)
            continue
        kept_tokens = kept_tokens_of(keep, prompt_len, num_prompt_blocks, block_size)
        freed = compact_manager_blocks(manager, request_id, keep)
        RECORDS[request_id] = CompactRecord(
            request_id=request_id,
            prompt_len=prompt_len,
            num_prompt_blocks=num_prompt_blocks,
            keep_positions=keep,
            kept_tokens=kept_tokens,
            dropped_tokens=prompt_len - kept_tokens,
            freed_blocks=freed,
        )
        _log.info(
            "[static-kv-compact] req=%s prompt=%d blocks=%d keep=%d kept_tokens=%d freed=%d",
            request_id,
            prompt_len,
            num_prompt_blocks,
            len(keep),
            kept_tokens,
            freed,
        )


def prepare_runner_views(runner, num_reqs_padded: int) -> RunnerViews | None:
    """Engine-side entry: gathered block-table view + compacted seq_lens.

    Returns None (fast path, metadata untouched) when no record is active for
    a current batch member. View length is derived from the runner's own
    optimistic seq_lens (``full - dropped``) so the optimistic +1-per-decode
    semantics are inherited automatically.
    """
    if not RECORDS:
        return None
    if getattr(runner, "use_cp", False):
        return None
    input_batch = runner.input_batch
    block_table_obj = input_batch.block_table[0]
    np_table = block_table_obj.block_table.np
    num_blocks_per_row = block_table_obj.num_blocks_per_row

    state = runner.__dict__.get("_kv_compact_state")
    if state is None:
        state = _RunnerState()
        state.block_table_cpu = torch.zeros(np_table.shape, dtype=torch.int32, pin_memory=True)
        state.block_table_device = torch.zeros(np_table.shape, dtype=torch.int32, device=runner.device)
        max_num_reqs = np_table.shape[0]
        state.seq_lens_cpu_buf = torch.zeros(max_num_reqs, dtype=torch.int32, pin_memory=True)
        state.seq_lens_device_buf = torch.zeros(max_num_reqs, dtype=torch.int32, device=runner.device)
        runner.__dict__["_kv_compact_state"] = state

    state.block_table_cpu.zero_()
    state.block_table_cpu[:num_reqs_padded] = torch.as_tensor(np_table[:num_reqs_padded])
    seq_lens_cpu = runner.optimistic_seq_lens_cpu[:num_reqs_padded].clone()

    changed = False
    for request_id, record in list(RECORDS.items()):
        idx = input_batch.req_id_to_index.get(request_id)
        if idx is None or idx >= num_reqs_padded:
            continue
        row_blocks = int(num_blocks_per_row[idx])
        if row_blocks < record.num_prompt_blocks:
            continue
        gather = record.keep_positions + list(range(record.num_prompt_blocks, row_blocks))
        view = np_table[idx, gather]
        state.block_table_cpu[idx, : len(view)] = torch.as_tensor(view)
        state.block_table_cpu[idx, len(view) :] = 0
        seq_lens_cpu[idx] = seq_lens_cpu[idx] - record.dropped_tokens
        changed = True

    if not changed:
        return None
    state.block_table_device[:num_reqs_padded].copy_(state.block_table_cpu[:num_reqs_padded], non_blocking=True)
    state.seq_lens_device_buf[:num_reqs_padded].copy_(seq_lens_cpu, non_blocking=True)
    return RunnerViews(
        block_table_device=state.block_table_device[:num_reqs_padded],
        seq_lens_device=state.seq_lens_device_buf[:num_reqs_padded],
        seq_lens_cpu=seq_lens_cpu,
    )
