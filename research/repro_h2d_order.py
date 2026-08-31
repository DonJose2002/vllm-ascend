#!/usr/bin/env python3
"""Standalone demo of the use-after-rewrite hazard behind PR #14922: a late
async H2D copy reads its pinned source AT EXECUTION TIME on NPU.

Extracted from the PR's investigation, with zero vllm dependencies: pinned
CPU slot(s), one NPU buffer, one copy under test per step, one consumer
comparison on the compute stream, and DEVICE-SIDE counters only - a single
host read at the end (same zero-sync protocol as the original three-counter
experiment, so the instrumentation cannot hide the race).

Terminal mechanism (adjudicated on the engine 2026-08-28; see the correction
comment on PR #14922): the engine rewrites a SINGLE pinned host page every
step (gate-closed steps hold the -1 "no token yet" sentinel) while its tiny
non_blocking H2D copy sits in a ~2048-task-deep FIFO submission ring.
Execution order copy->consumer is PRESERVED; what breaks is the VALUE: the
late-executing copy reads the source as rewritten by a LATER step and
delivers it faithfully - at a request boundary that future value is -1, and
where/gather(-1) faults the vector core. The PR fix (blocking copy)
snapshots the page content synchronously. This script reproduces the HAZARD
PATTERN standalone, not the crash: counters observe delivered values instead
of crashing into a vocab gather.

Per-step timeline (racy mode), mirroring prepare_next_token_ids_padded:

    host   : slot <- step_id                    (the "token id" write)
    [bg]   : K large non_blocking H2D copies    (SDMA backlog amplification)
    under  : gpu <- slot, non_blocking=True     (the copy under test)
    stream : counter += 1                       (consumer-side step tick)
             miss += (gpu[0] != counter)        (delivered-value check)
             fut  += (gpu[0] >  counter)        (future = host-rewritten
             esc  += (gpu[0] == SENTINEL)        source delivered late /
                                                  initial fill never covered)

Amplification rationale: in the real engine the race window is created by the
deep SDMA backlog of a chunked-prefill boundary step (16K = 8 chunks of
metadata H2D ahead of the small copy). --bg reproduces that condition with
inert background copies; --bg 0 is the calm-window control.

Faithfulness map (repro element <-> original prepare_next_token_ids_padded):

  slot.fill_(step)          host write of token ids into the pinned buffer
                            (per-step real value = the ~1% gate-open steps of
                            the bug, promoted to 100% to strip the request-
                            boundary confounder and maximize hits)
  gpu.copy_(slot, nb=True)  backup_next_token_ids.gpu.copy_(cpu, nb=True)
                            - literally the same call, same payload scale
  counter+=1; miss+=(...)   torch.where consuming backup.gpu: same position
                             (first compute-stream op after the copy), same
                             stream, same kind of buffer read -> identical
                             exposure to the DELIVERED VALUE. The read value
                             is compared against the stream-side counter
                             (expected value generated ON the compute stream -
                             host-side expectations would need their own H2D
                             and pollute the timeline). future-dominant miss
                             under a REWRITTEN source (alt2) = the script
                             analogue of the engine escape; stale-side miss
                             and esc would be genuine ordering-failure
                             evidence (never observed at terminal audit).
  --slot-mode unique        per-step private page = the "staged" fix shape:
                             source stable while in flight -> clean
  --slot-mode alt2          the original two-buffer alternation = the
                             single-page rewrite behaviour = positive control

Modes (expectations depend on --slot-mode):
  racy   copy_(non_blocking=True)   unique: miss==0 (source stable, async
                                    path intact). alt2 + --bg 8: ~99.9%
                                    future-dominant miss - the
                                    use-after-rewrite pattern, mirroring the
                                    engine-side 4K-occasional / 16K-
                                    deterministic split via backlog depth.
  fix    copy_(non_blocking=False)  miss==0 on BOTH slot designs (the PR fix:
                                    the synchronous copy snapshots the source
                                    content, so later host rewrites cannot
                                    poison the transfer)
  event  copy on a side stream +    fence re-check: unique -> clean (the sync
         copy_stream.synchronize()  covers the copy properly); alt2 ->
                                    residual future-direction miss only
                                    (startup host-run-ahead, NOT a fence
                                    failure). The earlier "fences are blind /
                                    copy is off-stream" readings from this
                                    mode were artifacts of the old slot
                                    design - retracted in the PR correction.

Under alt2, a miss==0 means the parameters did not open the window -
raise --bg / --steps / --bg-elems. Under unique, miss==0 is the expected
clean result.

NPU-API provenance (everything NPU-specific mirrors vllm-ascend production
usage, so no untested API shapes on the hot path): pinned allocation via the
pin_memory ctor kwarg (upstream CpuGpuBuffer / platform.is_pin_memory_available),
`with torch.npu.stream(...)` (weight_transfer/packed_tensor.py), Event.record()
(kv_transfer mooncake connector); only current_stream().wait_event() lacks an
in-tree precedent and carries a synchronize() fallback. The racy/fix modes use
no NPU-specific API at all beyond the pinned ctor. Locally verified on cpu
(synchronous-copy semantics: both modes must report miss=0); the first npu run
IS the measurement.

Usage (server, inside the v0.23.0 container, no serve needed):
  # the use-after-rewrite pattern (expect ~99.9% future-dominant miss):
  python3 research/repro_h2d_order.py --mode racy --slot-mode alt2 --bg 8
  # stable source -> expect clean (the staged-page principle):
  python3 research/repro_h2d_order.py --mode racy --slot-mode unique --bg 8
  # the PR fix shape -> expect clean on both slot designs:
  python3 research/repro_h2d_order.py --mode fix --slot-mode alt2
  python3 research/repro_h2d_order.py --mode racy --bg 0      # calm control
  python3 research/repro_h2d_order.py --device cpu --mode racy # logic selftest
  python3 research/repro_h2d_order.py --selftest               # all 4 combos, cpu
  python3 research/repro_h2d_order.py --mode racy --steps 50 \
      --profile /tmp/reprof        # torch_npu profiler (msprof Text export),
                                   # then auto-audits memcpy vs compute stream
                                   # ownership via research/stream_audit.py

  # Legacy investigation harness (2026-08-27 attribution rounds, kept for
  # reproducibility only - their interpretations were superseded by the
  # 08-28 terminal audit; full trail in research/offstream-copy-attribution.md
  # in the fork's research branch). --copy-mode acl-direct issues the tested
  # copy via ctypes -> libascendcl aclrtMemcpyAsync from the MAIN thread
  # (bypassing torch_npu's host task queue); combined with TASK_QUEUE_ENABLE
  # env it fills the channel matrix used during that investigation:
    TASK_QUEUE_ENABLE=1 python3 research/repro_h2d_order.py --mode racy \
        --copy-mode acl-direct --bg 8
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

import torch

SENTINEL = -1  # upstream get_token_id()'s uncommitted-position sentinel


def make_profiler(out_dir: str):
    """torch_npu profiler with the SAME construction as vllm-ascend's
    TorchNPUProfilerWrapper (Level1, msprof Text export) so the export layout
    matches what research/profile_step_breakdown.py + stream_audit.py expect."""
    import torch_npu.profiler as prof

    experimental_config = prof._ExperimentalConfig(
        export_type=prof.ExportType.Text,
        profiler_level=prof.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=prof.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )
    return prof.profile(
        activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
        with_stack=False,
        profile_memory=False,
        with_modules=False,
        experimental_config=experimental_config,
        on_trace_ready=prof.tensorboard_trace_handler(out_dir, worker_name="repro_h2d"),
    )


def acl_direct_copy_factory(is_npu: bool, with_query: bool = False):
    """Return f(dst_tensor, src_tensor, nbytes) that submits ONE
    aclrtMemcpyAsync (H2D) on the CURRENT torch.npu stream from this (main)
    thread, bypassing torch_npu's copy_ path entirely. Reuses the lib
    discovery/binding from cann_memcpy_order; returns None if unavailable
    (caller falls back to torch copy_ with a loud notice).

    with_query=True additionally mirrors torch_npu's copy_ epilogue: right
    after the memcpy, process_non_blocking_copy calls aclrtPointerGetAttributes
    on the host pointer (CachingHostAllocator.cpp:1356) before returning.
    Tested 2026-08-27 as a barrier hypothesis (does this incidental runtime
    call flush the submission queue?): NO barrier effect - miss stayed at
    the no-query level under the then-used slot design; the torch path's
    cleanliness was later explained by slot stability, not by any epilogue
    call (record_event itself is pure bookkeeping,
    CachingHostAllocator.cpp:689-719)."""
    if not is_npu:
        return None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import cann_memcpy_order

        acl = cann_memcpy_order.Acl(cann_memcpy_order.find_lib())
        stream_handle = torch.npu.current_stream().npu_stream

        query = None
        if with_query:

            class PtrAttrs(ctypes.Structure):
                _fields_ = [
                    ("id", ctypes.c_uint32),
                    ("type", ctypes.c_int32),  # aclrtMemLocationType enum
                    ("pageSize", ctypes.c_uint32),
                    ("rsv", ctypes.c_uint32 * 4),
                ]

            acl.lib.aclrtPointerGetAttributes.argtypes = [ctypes.c_void_p, ctypes.POINTER(PtrAttrs)]
            acl.lib.aclrtPointerGetAttributes.restype = ctypes.c_int32

            def query(src_ptr: int) -> None:
                attrs = PtrAttrs()
                err = acl.lib.aclrtPointerGetAttributes(ctypes.c_void_p(src_ptr), ctypes.byref(attrs))
                if err != 0:
                    raise RuntimeError(f"aclrtPointerGetAttributes failed: aclError={err}")

        def f(dst, src, nbytes: int) -> None:
            err = acl.lib.aclrtMemcpyAsync(
                dst.data_ptr(),
                nbytes,
                src.data_ptr(),
                nbytes,
                cann_memcpy_order.ACL_MEMCPY_HOST_TO_DEVICE,
                stream_handle,
            )
            if err != 0:
                raise RuntimeError(f"aclrtMemcpyAsync direct submit failed: aclError={err}")
            if query is not None:
                query(src.data_ptr())

        return f
    except Exception as e:  # noqa: BLE001 - fallback path must stay loud
        print(f"note: acl-direct unavailable ({e}); falling back to torch copy_")
        return None


def run(args) -> int:
    dev = torch.device(args.device)
    is_npu = args.device.startswith("npu")
    if is_npu:
        import torch_npu  # noqa: F401

    pin = is_npu  # pinned memory only meaningful on the npu path
    profiling = bool(getattr(args, "profile", None))
    if profiling and not is_npu:
        print("--profile is npu-only (msprof export); dropping it for the cpu selftest")
        profiling = False

    def buf(n: int, fill: int = 0, dtype=torch.int32):
        return torch.full((n,), fill, dtype=dtype, pin_memory=pin)

    # v2 audit (2026-08-28): UNIQUE per-step host slots + stale/future split.
    # The pure-CANN rounds proved the runtime stages ~2048 tasks in a FIFO
    # submission ring, so the host can run 100-200+ steps ahead; the old
    # two alternating slots were being OVERWRITTEN before late copies read
    # them, delivering FUTURE values that the miss counter silently counted
    # as hits. One slot row per step (written exactly once) kills that
    # confounder; the new device-side `fut` counter (gpu[0] > counter)
    # separates future reads (host-run-ahead / shared-buffer artifacts) from
    # genuine stale reads (gpu[0] < counter, incl. the -1 sentinel esc) -
    # only stale+esc constitute defect evidence. The DEVICE buffer stays
    # shared on purpose: it mirrors the engine's shared backup buffer.
    #
    # --slot-mode alt2 is the POSITIVE CONTROL: the exact old double-buffer
    # alternation (two small buffers, rewritten every 2 steps). Paired runs
    # in one binary adjudicate the all-green ambiguity:
    #   alt2 ~99.9% future-dominant + unique 0%  -> old miss mass was the
    #     slot-overwrite artifact, async path intact, environment stable
    #   alt2 also 0% -> either the environment drifted (LD_PRELOAD warning
    #     today) or this patch silenced the repro - bisect via git checkout
    #   unique shows esc/stale -> genuine defect signal (unexpected)
    cpu_slots = None
    cpu_a = cpu_b = None
    if getattr(args, "slot_mode", "unique") == "alt2":
        cpu_a = buf(args.payload)
        cpu_b = buf(args.payload)
    else:
        cpu_slots = torch.full((args.steps, args.payload), SENTINEL, dtype=torch.int32, pin_memory=pin)
    gpu = torch.full((args.payload,), SENTINEL, dtype=torch.int32, device=dev)

    # pinned-ness guard (the silencing risk): if the row views of the big
    # pinned tensor are not treated as pinned, copy_(non_blocking=True)
    # silently degrades toward synchronous behavior and any all-green is
    # meaningless. Print, don't assert - a loud record beats a crash.
    if is_npu:
        try:
            whole = bool(cpu_slots.is_pinned()) if cpu_slots is not None else bool(cpu_a.is_pinned())
            row = bool(cpu_slots[0].is_pinned()) if cpu_slots is not None else whole
            print(f"slots: slot_mode={args.slot_mode} pinned_whole={whole} pinned_row={row}")
        except Exception as e:  # noqa: BLE001 - diagnostic only
            print(f"slots: pinned check unavailable ({e})")

    # Background backlog: inert large copies queued ahead of the tested one.
    bg_cpu = buf(args.bg_elems) if args.bg > 0 else None
    bg_gpu = torch.zeros(args.bg_elems, dtype=torch.int32, device=dev) if args.bg > 0 else None

    counter = torch.zeros((), dtype=torch.int32, device=dev)
    miss = torch.zeros((), dtype=torch.int64, device=dev)
    esc = torch.zeros((), dtype=torch.int64, device=dev)
    fut = torch.zeros((), dtype=torch.int64, device=dev)

    copy_stream = torch.npu.Stream() if (args.mode == "event" and is_npu) else None

    acl_direct = None
    if getattr(args, "copy_mode", "torch") != "torch":
        acl_direct = acl_direct_copy_factory(is_npu, with_query=args.copy_mode == "acl-direct-query")

    import contextlib

    if profiling and args.steps > 200:
        print(f"note: --profile with steps={args.steps} makes a heavy export; consider --steps 50-200")

    t0 = time.monotonic()
    ctx = make_profiler(args.profile) if profiling else contextlib.nullcontext()
    with ctx as prof_obj:
        for step in range(1, args.steps + 1):
            if cpu_slots is not None:
                slot = cpu_slots[step - 1]  # this step's OWN row, written once below
            else:  # alt2 positive control: the exact old alternation
                slot = cpu_a if step % 2 == 1 else cpu_b
            slot.fill_(step)  # host write: this step's "real token id"

            if bg_gpu is not None:
                for _ in range(args.bg):
                    bg_gpu.copy_(bg_cpu, non_blocking=True)

            if args.mode == "racy":
                if acl_direct is not None:
                    acl_direct(gpu, slot, args.payload * 4)
                else:
                    gpu.copy_(slot, non_blocking=True)
            elif args.mode == "fix":
                gpu.copy_(slot, non_blocking=False)
            else:  # event: fence the async copy on a side stream, host-side wait
                with torch.npu.stream(copy_stream):
                    gpu.copy_(slot, non_blocking=True)
                # Fence re-check (terminal audit 2026-08-28): with unique
                # slots this mode is CLEAN - miss=0/2000 measured; the sync
                # covers the copy properly. The historical "fences are
                # blind / the event never observed the copy" misses were
                # artifacts of the old alternating slot design
                # (future-direction reads), retracted in the PR correction.
                # copy_stream.synchronize() is kept as the fence because it
                # mirrors the in-tree production workaround
                # (cpu_offload_connector.py:318 + its TODO); wait_event and
                # recorded-event variants were tried during the investigation
                # and are documented in the attribution doc.
                copy_stream.synchronize()

            # Consumer on the compute stream (zero host sync until the very end).
            # `sample` snapshots gpu[0] ONCE (server lesson 2026-08-28: three
            # separate gpu[0] reads let a copy land between them and produced
            # inconsistent counters - miss=8 vs future=11 in one run).
            counter += 1
            sample = gpu[0].clone()
            miss += (sample != counter).to(torch.int64)
            esc += (sample == SENTINEL).to(torch.int64)
            fut += (sample > counter).to(torch.int64)
            if profiling:
                prof_obj.step()

    wall = time.monotonic() - t0
    miss_n = int(miss)
    esc_n = int(esc)
    fut_n = int(fut)
    stale_n = max(miss_n - esc_n - fut_n, 0)

    bg_desc = f"{args.bg}x{args.bg_elems * 4 // 1024}KiB" if args.bg else "off"
    copy_desc = "torch" if acl_direct is None else "acl-direct(ctypes)"
    print(
        f"mode={args.mode} device={args.device} copy={copy_desc} steps={args.steps} "
        f"payload={args.payload * 4}B bg={bg_desc} wall={wall:.2f}s "
        f"({wall / args.steps * 1e6:.1f}us/step)"
    )
    print(
        f"miss={miss_n} ({100.0 * miss_n / args.steps:.2f}% of steps)"
        f" | stale={stale_n} future={fut_n} sentinel-escapes={esc_n}"
    )
    print(
        "note: only stale+esc are defect evidence (ordering failure - never"
        " observed); future = a late copy delivered a host-rewritten source"
        " value (the use-after-rewrite pattern this demo showcases)"
    )

    if args.device == "cpu":
        expected_zero = True  # cpu copies are synchronous by construction
    else:
        # unique slots keep the source stable while the copy is in flight,
        # so racy/event are expected CLEAN too (async path intact); only
        # alt2 (rewritten source) expects the future-dominant demo miss.
        expected_zero = args.mode == "fix" or getattr(args, "slot_mode", "unique") == "unique"
    if expected_zero:
        ok_clean = miss_n == 0
        verdict = "PASS (clean as expected)" if ok_clean else "UNEXPECTED miss>0"
    elif args.mode == "event":
        if stale_n > 0 or esc_n > 0:
            verdict = "FENCE-BLIND (stale-side): fence completed yet the consumer read older data"
        elif miss_n > 0:
            verdict = (
                "fence held; residual miss is future-direction (host-run-ahead artifact),"
                " not a fence failure - see stale/future split"
            )
        else:
            verdict = "stream sync covered the copy (clean)"
    elif stale_n > 0 or esc_n > 0:
        verdict = "RACE REPRODUCED, stale-side (unordered copy observed: kernel read older data)"
    elif fut_n > 0:
        verdict = (
            "FUTURE-only: use-after-rewrite pattern - a late async copy"
            " delivered a host-rewritten source value; NOT a stale-read"
            " (ordering-failure) verdict"
        )
    else:
        verdict = "no race under these parameters - raise --bg/--steps/--bg-elems"
    print(f"verdict: {verdict}")

    if profiling:
        # on_trace_ready fires on context exit; audit whatever landed (the
        # msprof Text export writes PROF_* subtrees under args.profile).
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import stream_audit

            print(stream_audit.audit([args.profile]))
        except Exception as e:  # noqa: BLE001 - audit is best-effort reporting
            print(f"profile audit failed ({e}); rerun later: python3 research/stream_audit.py {args.profile}")
    return 0 if (miss_n == 0) == expected_zero else 1


def selftest() -> int:
    """Logic check on cpu: both modes and both slot designs must run clean."""
    ok = True
    for mode in ("racy", "fix"):
        for slot_mode in ("unique", "alt2"):
            args = argparse.Namespace(
                device="cpu",
                mode=mode,
                steps=500,
                payload=16,
                bg=2,
                bg_elems=1024,
                profile=None,
                copy_mode="torch",
                slot_mode=slot_mode,
            )
            print(f"--- selftest {mode} slot={slot_mode} ---")
            if run(args) != 0:
                ok = False
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="npu", help="npu | cpu (cpu = logic selftest)")
    ap.add_argument("--mode", choices=["racy", "fix", "event"], default="racy")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument(
        "--payload", type=int, default=16, help="int32 elements of the tested buffer (16 = 64B, bug-site scale)"
    )
    ap.add_argument("--bg", type=int, default=8, help="background large H2D copies per step (0 = calm-window control)")
    ap.add_argument("--bg-elems", type=int, default=262144, help="int32 elements of each background copy (1MiB)")
    ap.add_argument(
        "--profile",
        default=None,
        metavar="DIR",
        help="npu-only: enable torch_npu profiler, msprof Text export into DIR, then auto-audit stream ownership",
    )
    ap.add_argument(
        "--copy-mode",
        choices=["torch", "acl-direct", "acl-direct-query"],
        default="torch",
        help="tested copy channel: torch = Tensor.copy_ (host task queue); acl-direct = ctypes"
        " aclrtMemcpyAsync from the main thread on the current torch.npu stream; acl-direct-query"
        " additionally mirrors torch copy_'s aclrtPointerGetAttributes epilogue (barrier test)",
    )
    ap.add_argument(
        "--slot-mode",
        choices=["unique", "alt2"],
        default="unique",
        help="host slot design: unique = one pinned row per step, written once (artifact-free);"
        " alt2 = the exact old two-buffer alternation (POSITIVE CONTROL for the slot-overwrite"
        " artifact - expect future-dominant miss under backlog if the async path is intact)",
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="cpu logic check: both modes x both slot designs, must run clean",
    )
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.device == "cpu" and args.mode == "event":
        print("event mode is npu-only (cross-stream fence); use racy/fix on cpu")
        return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
