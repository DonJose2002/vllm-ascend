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
_RAW_FWD_LOGGED = False
_HOOK_ENTERED_LOGGED = False


def _resolve_model(runner):
    """The underlying (possibly compile-wrapped) model, ACLGraphWrapper off."""
    model = runner.model
    unwrap = getattr(model, "unwrap", None)  # ACLGraphWrapper
    if callable(unwrap):
        model = unwrap()
    return model


def raw_model_forward(runner, **model_inputs):
    """Call the model's ORIGINAL python forward for dwvote steps.

    The runnable inside ACLGraphWrapper is a TorchCompileWithNoGuardsWrapper
    mix-in (vllm support_torch_compile monkeypatches it into the model's
    bases): its __call__ always dispatches to the compiled artifact (guards
    dropped / bytecode dispatch), so neither CUDAGraphMode.NONE nor dynamo
    stances (force_eager) ever reach a python-executing forward - b2smoke
    run 6/7 evidence. Calling .forward directly bypasses __call__ entirely
    and runs plain python, so the vote hooks actually fire.
    """
    global _RAW_FWD_LOGGED
    model = _resolve_model(runner)
    fwd = getattr(model, "forward", None)
    if not callable(fwd):
        raise RuntimeError(f"no python forward on {type(model).__name__}; dwvote cannot run")
    if not _RAW_FWD_LOGGED:
        _log.warning("[kv-compact-voting] raw forward start (model=%s)", type(model).__name__)
    out = fwd(**model_inputs)
    if not _RAW_FWD_LOGGED:
        _RAW_FWD_LOGGED = True
        _log.warning("[kv-compact-voting] raw forward end OK")
    return out


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
    model = _resolve_model(runner)
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
        # One-shot entry probe: splits "hooks never fired" (line absent) from
        # "fired but early-returned" (line present, first vote absent). Run-8
        # forensics: neither raw-forward nor first-vote lines appeared.
        global _HOOK_ENTERED_LOGGED
        if not _HOOK_ENTERED_LOGGED:
            _HOOK_ENTERED_LOGGED = True
            _log.warning(
                "[kv-compact-voting] hook entered (layer=%d pend=%d runner_set=%s failed=%s)",
                layer_idx,
                len(pend),
                _RUNNER is not None,
                _HOOKS_FAILED,
            )
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
