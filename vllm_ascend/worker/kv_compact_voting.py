# SPDX-License-Identifier: Apache-2.0
"""Decode-window voting selector (Phase 2 B1.5, research, env-gated).

SnapKV-style observation-window voting, decode-side variant (``dwvote``). The
first ``VOTE_STEPS`` decode steps after a request's prefill are forced eager
(call site rides the ``calculate_kv_scales`` override point in
``NPUModelRunner.execute_model``); a forward hook on each ``self_attn`` module
recomputes the step's real query vector for the request's decode token
(mirroring qwen3.py forward exactly: qkv_proj -> q_norm -> rotary_emb) and
votes it against the paged K cache. Per-block max-pooled attention mass
accumulates across layers and steps into ``static_kv_compact.PENDING_VOTES``;
the scheduler hook finalizes (top-budget blocks + sink/recent anchors) once the
window closes. Stride selector is the fallback when no vote landed.

Why hooks + forced eager instead of hooking the real prefill: PIECEWISE graph
capture replays prefill chunks without running python (hooks dead), and an
ad-hoc voting forward would need a hand-built AscendCommonAttentionMetadata
(~20 NPU-specific fields) - both rejected; see phase2-kv-compression-design
§4.4. All diagnostic lines use WARNING: EngineCore's root logger drops stdlib
INFO (09-09 b2smoke lesson).

Inactive unless static_kv_compact.ENABLED and SELECTOR == "dwvote"; module
imports stay vllm-free so CPU unit tests can import it directly.
"""

from __future__ import annotations

import logging
import sys

import torch

if "static_kv_compact" in sys.modules:
    # Research UT path: test inserted the worker dir into sys.path and imported
    # the module top-level; reuse that exact instance (shared module state).
    import static_kv_compact as skc  # type: ignore[no-redef,import-not-found]
else:
    from vllm_ascend.worker import static_kv_compact as skc  # type: ignore[no-redef]

_log = logging.getLogger(__name__)

_RUNNER = None
_HOOKS: list = []
_HOOKS_FAILED = False
_FIRST_VOTE_LOGGED = False


def needs_eager_step() -> bool:
    """execute_model override input: any request still inside its vote window."""
    if not (skc.ENABLED and skc.SELECTOR == "dwvote"):
        return False
    return any(pv.vote_steps < skc.VOTE_STEPS for pv in skc.PENDING_VOTES.values())


def maybe_install(runner) -> None:
    """Lazy one-time hook install on the (possibly wrapped) model's layers."""
    global _RUNNER, _HOOKS_FAILED
    if _HOOKS or _HOOKS_FAILED:
        return
    model = runner.model
    unwrap = getattr(model, "unwrap", None)  # ACLGraphWrapper
    if callable(unwrap):
        model = unwrap()
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        _fail_open("model layers not found on runner.model")
        return
    _RUNNER = runner
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        _HOOKS.append(attn.register_forward_hook(_make_hook(idx)))
    if not _HOOKS:
        _fail_open("no self_attn modules hooked")
        return
    _log.warning(
        "[kv-compact-voting] vote hooks installed on %d layers (D=%d)",
        len(_HOOKS),
        skc.VOTE_STEPS,
    )


def _fail_open(reason: str) -> None:
    """Voting unavailable: force-finalize every pending record so the
    scheduler hook commits them with the stride selector (correctness kept)."""
    global _HOOKS_FAILED
    _HOOKS_FAILED = True
    for pv in skc.PENDING_VOTES.values():
        pv.vote_steps = skc.VOTE_STEPS
    _log.warning("[kv-compact-voting] %s; dwvote disabled, stride fallback", reason)


def _make_hook(layer_idx: int):
    def _hook(module, args, output):
        pend = skc.PENDING_VOTES
        if not pend or _RUNNER is None or _HOOKS_FAILED:
            return
        try:
            _vote_layer(_RUNNER, module, layer_idx, args[0], args[1], pend)
        except Exception:
            _log.exception("[kv-compact-voting] vote hook failed at layer %d", layer_idx)
            _fail_open(f"hook exception at layer {layer_idx}")

    return _hook


def _vote_layer(runner, attn, layer_idx: int, positions, hidden, pend) -> None:
    """One layer's vote for each pending request's decode token this step.

    positions/hidden are the attention module's real forward inputs; the
    request's token sits at query_start_loc[idx] (decode = 1 token/req).
    """
    input_batch = runner.input_batch
    qsl_cpu = runner.query_start_loc.cpu
    k_cache = runner.kv_caches[layer_idx][0]  # (num_blocks, block_size, Hkv, D)
    block_size = k_cache.shape[1]
    for request_id, pv in pend.items():
        idx = input_batch.req_id_to_index.get(request_id)
        if idx is None:
            continue
        blocks = input_batch.block_table[0].block_table.np[idx, : pv.num_prompt_blocks]
        if pv.votes_dev is None:
            pv.blocks_dev = torch.as_tensor(blocks).to(hidden.device)
            pv.votes_dev = torch.zeros(pv.num_prompt_blocks, dtype=torch.float32, device=hidden.device)
        t0 = int(qsl_cpu[idx])
        pos = positions.reshape(-1)[t0 : t0 + 1]
        qkv, _ = attn.qkv_proj(hidden[t0 : t0 + 1])
        q, k, _v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q = attn.q_norm(q.view(1, attn.num_heads, attn.head_dim)).view(1, attn.q_size)
        k = attn.k_norm(k.view(1, attn.num_kv_heads, attn.head_dim)).view(1, attn.kv_size)
        q, _k = attn.rotary_emb(pos, q, k)
        q = q.view(attn.num_heads, attn.head_dim).float()
        keys = k_cache[pv.blocks_dev].reshape(-1, attn.num_kv_heads, attn.head_dim).float()
        group = attn.num_heads // attn.num_kv_heads
        q_grouped = q.view(attn.num_kv_heads, group, attn.head_dim)
        keys = keys.permute(1, 0, 2)  # (Hkv, L, D)
        probs = torch.softmax(torch.matmul(q_grouped, keys.transpose(1, 2)) * attn.scaling, dim=-1)
        block_votes = probs.sum(dim=(0, 1)).view(pv.num_prompt_blocks, block_size).max(dim=1).values
        pv.votes_dev += block_votes
        global _FIRST_VOTE_LOGGED
        if not _FIRST_VOTE_LOGGED:
            _FIRST_VOTE_LOGGED = True
            _log.warning(
                "[kv-compact-voting] first vote recorded (layer=%d pending=%d)",
                layer_idx,
                len(pend),
            )
        if layer_idx == 0:
            pv.vote_steps += 1
