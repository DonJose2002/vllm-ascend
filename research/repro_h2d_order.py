#!/usr/bin/env python3
"""Standalone repro: non_blocking H2D copy vs consumer-kernel ordering on NPU.

Extracted from vllm-ascend PR #14922 (eagle H2D race), with zero vllm
dependencies: one pinned CPU buffer, one NPU buffer, one copy under test per
step, one consumer comparison on the compute stream, and DEVICE-SIDE counters
only - a single host read at the end (same zero-sync protocol as the original
three-counter experiment, so the instrumentation cannot hide the race).

Per-step timeline (racy mode), mirroring prepare_next_token_ids_padded:

    host   : cpu_slot <- step_id                (new value; host write)
    [bg]   : K large non_blocking H2D copies    (SDMA backlog amplification)
    under  : gpu <- cpu_slot, non_blocking=True (the copy in question)
    stream : counter += 1                       (consumer-side step tick)
             miss += (gpu[0] != counter)        (reads whatever has landed)
             esc  += (gpu[0] == SENTINEL)       (stale -1 sentinel escape)

If the copy is not ordered before the consumer kernel, the comparison reads
the PREVIOUS landed value: miss counts exactly the steps whose copy had not
landed yet; esc counts reads of the initial -1 sentinel (the exact value that
escaped into the vocab gather in the original crash).

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
                            sensitivity to "has the copy landed". The read
                            value is compared against the stream-side counter
                            (expected value generated ON the compute stream -
                            host-side expectations would need their own H2D
                            and pollute the timeline); miss = read of a stale
                            previous value, esc = read of the initial -1
                            sentinel = the original escape event, counted
                            instead of crashed into gather(-1).
  --bg / double-buffered    environment reconstruction only: SDMA backlog
  slots                     (the 16K-always condition) / removal of the
                            orthogonal host-write-vs-inflight-read confounder.

Modes:
  racy   copy_(non_blocking=True)                 expected: miss > 0
  fix    copy_(non_blocking=False)                expected: miss == 0 (the PR)
  event  copy on a side stream + event fence      expected: miss == 0, but the
         with a HOST-side wait (synchronize)      fence must be host-waited:
         device-side cross-stream waits do not reliably cover this copy path
         on torch_npu (in-tree cpu_offload_connector.py:317 TODO + its
         synchronize() workaround; an earlier wait_event()-based build of
         this repro measured misses under the fence).

A miss==0 under racy does NOT invalidate the bug (see PR evidence); it means
these parameters did not open the window - raise --bg / --steps / --bg-elems.

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
  python3 research/repro_h2d_order.py --mode racy
  python3 research/repro_h2d_order.py --mode fix
  python3 research/repro_h2d_order.py --mode racy --bg 0      # calm control
  python3 research/repro_h2d_order.py --device cpu --mode racy # logic selftest
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

SENTINEL = -1  # upstream get_token_id()'s uncommitted-position sentinel


def run(args) -> int:
    dev = torch.device(args.device)
    is_npu = args.device.startswith("npu")
    if is_npu:
        import torch_npu  # noqa: F401

    pin = is_npu  # pinned memory only meaningful on the npu path

    def buf(n: int, fill: int = 0, dtype=torch.int32):
        return torch.full((n,), fill, dtype=dtype, pin_memory=pin)

    # Two host slots, alternating per step: the host write of step N must not
    # race the in-flight SDMA read of step N-1 (a host-side concern orthogonal
    # to the device-side ordering under test - the original buffer is written
    # every step too, so double-buffering only removes a confounder).
    cpu_a = buf(args.payload)
    cpu_b = buf(args.payload)
    gpu = torch.full((args.payload,), SENTINEL, dtype=torch.int32, device=dev)

    # Background backlog: inert large copies queued ahead of the tested one.
    bg_cpu = buf(args.bg_elems) if args.bg > 0 else None
    bg_gpu = torch.zeros(args.bg_elems, dtype=torch.int32, device=dev) if args.bg > 0 else None

    counter = torch.zeros((), dtype=torch.int32, device=dev)
    miss = torch.zeros((), dtype=torch.int64, device=dev)
    esc = torch.zeros((), dtype=torch.int64, device=dev)

    copy_stream = torch.npu.Stream() if (args.mode == "event" and is_npu) else None
    fence = torch.npu.Event() if (args.mode == "event" and is_npu) else None

    t0 = time.monotonic()
    for step in range(1, args.steps + 1):
        slot = cpu_a if step % 2 == 1 else cpu_b
        slot.fill_(step)  # host write: this step's "real token id"

        if bg_gpu is not None:
            for _ in range(args.bg):
                bg_gpu.copy_(bg_cpu, non_blocking=True)

        if args.mode == "racy":
            gpu.copy_(slot, non_blocking=True)
        elif args.mode == "fix":
            gpu.copy_(slot, non_blocking=False)
        else:  # event: fence the async copy instead of making it blocking
            with torch.npu.stream(copy_stream):
                gpu.copy_(slot, non_blocking=True)
            fence.record(copy_stream)
            # Device-side cross-stream waits (current_stream().wait_event /
            # wait_stream) do NOT reliably cover this copy path on torch_npu:
            # the in-tree cpu_offload_connector.py wait_for_layer_load carries
            # an explicit TODO to switch to wait_stream "after fixing the bug"
            # and currently host-synchronizes instead - for the same shape of
            # side-stream non_blocking H2D copies. Mirror that production
            # posture (host-side event wait); an earlier build of this repro
            # used wait_event() and showed misses, consistent with that TODO.
            fence.synchronize()

        # Consumer on the compute stream (zero host sync until the very end).
        counter += 1
        miss += (gpu[0] != counter).to(torch.int64)
        esc += (gpu[0] == SENTINEL).to(torch.int64)

    wall = time.monotonic() - t0
    miss_n = int(miss)
    esc_n = int(esc)

    bg_desc = f"{args.bg}x{args.bg_elems * 4 // 1024}KiB" if args.bg else "off"
    print(
        f"mode={args.mode} device={args.device} steps={args.steps} "
        f"payload={args.payload * 4}B bg={bg_desc} wall={wall:.2f}s "
        f"({wall / args.steps * 1e6:.1f}us/step)"
    )
    print(f"miss={miss_n} ({100.0 * miss_n / args.steps:.2f}% of steps) sentinel-escapes={esc_n}")

    if args.device == "cpu":
        expected_zero = True  # cpu copies are synchronous by construction
    else:
        expected_zero = args.mode in ("fix", "event")
    if expected_zero:
        verdict = "PASS (ordered as expected)" if miss_n == 0 else "UNEXPECTED miss>0"
    else:
        verdict = (
            "RACE REPRODUCED (unordered copy observed)"
            if miss_n > 0
            else "no race under these parameters - raise --bg/--steps/--bg-elems"
        )
    print(f"verdict: {verdict}")
    return 0 if (miss_n == 0) == expected_zero else 1


def selftest() -> int:
    """Logic check on cpu: both modes must run clean and report zero misses."""
    ok = True
    for mode in ("racy", "fix"):
        args = argparse.Namespace(device="cpu", mode=mode, steps=500, payload=16, bg=2, bg_elems=1024)
        print(f"--- selftest {mode} ---")
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
    args = ap.parse_args()
    if args.device == "cpu" and args.mode == "event":
        print("event mode is npu-only (cross-stream fence); use racy/fix on cpu")
        return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
