#!/usr/bin/env python3
"""Pure-CANN (libascendcl via ctypes) ordering probe for aclrtMemcpyAsync H2D.

Zero torch / zero torch_npu. This exists to answer ONE question the
torch_npu-level evidence cannot: is the stale-read defect below torch_npu?

    If THIS script reproduces the stale read  -> the defect is in libascendcl/
       driver territory (contract of acl_rt.h itself); torch_npu is exonerated.
    If it never reproduces                    -> the torch_npu layer is back on
       trial (host task-queue dispatch timing); the CANN-attribution is dead.

Design: per step, write a step id into a PINNED host buffer (aclrtMallocHost),
[optionally] queue K large async H2D copies on the same stream (SDMA backlog),
issue the copy under test, optionally apply a fence, then read the device
memory back with an async D2H on the SAME stream and a final
aclrtSynchronizeStream. The consumer is the D2H DATA ITSELF - no profiler
timestamps, no torch_npu task queue, nothing indirect.

Per-step timeline (mode=racy, fence=none):

    host : pin[0] <- step
    [bg] : K x aclrtMemcpyAsync(dev_bg <- pin_bg, 1MiB, H2D, stream)
    test : aclrtMemcpyAsync(dev <- pin, 64B, H2D, stream)     [racy]
           aclrtMemcpy(dev <- pin, 64B, H2D)                  [sync control]
    fence: none | stream_sync | event_sync | event_wait
    read : aclrtMemcpyAsync(pin_back <- dev, 64B, D2H, stream)
    end  : aclrtSynchronizeStream -> compare pin_back[0] == step on host

Expected under CUDA-style stream semantics (and per the acl_rt.h contract
"call aclrtSynchronizeStream to ensure the memory replication task has
completed"): pin_back[0] == step always. A miss means the D2H read the
device memory BEFORE the tested H2D's data was physically there, i.e. the
H2D's completion was not joined into the stream's dependency order.

Verdict matrix (racy vs sync, fence variants) - printed at the end:

  racy miss>0 (fence=none)             -> pure-CANN reproduction: ordering
                                          not enforced on one stream
  + fence=stream_sync still miss>0     -> STRONGEST: the documented
                                          synchronizeStream guarantee itself
                                          is broken (data not landed after a
                                          full stream sync)
  + fence=stream_sync miss==0          -> ordering is opt-in via explicit
                                          sync only (weaker-than-CUDA but
                                          documented); torch_npu's
                                          CUDA-style recordEvent fencing is
                                          built on a semantic that does not
                                          exist here
  racy miss==0 everywhere (bg raised)  -> NOT reproduced at pure CANN level:
                                          suspicion moves back to torch_npu
                                          (task-queue dispatch timing); see
                                          research/offstream-copy-attribution.md W1

Usage (server, inside the v0.23.0 container, no torch needed):
  python3 research/cann_memcpy_order.py                        # racy, bg=8, fence=none
  python3 research/cann_memcpy_order.py --fence stream_sync
  python3 research/cann_memcpy_order.py --mode sync            # negative control
  python3 research/cann_memcpy_order.py --host-mem plain       # pageable source
  python3 research/cann_memcpy_order.py --bg 0                 # calm window
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import os
import sys
import time

# Values verified against cann/runtime include/external/acl/acl_rt.h (enum
# order: H2H=0, H2D=1, D2H=2, D2D=3) and acl_base.h (ACL_ERROR_NONE=0).
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_ERROR_NONE = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0

LIB_CANDIDATES = [
    "/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so",
    "/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/libascendcl.so",
    "/usr/local/Ascend/driver/lib64/common/driver/libascendcl.so",
    "/usr/local/Ascend/driver/lib64/driver/libascendcl.so",
]


class AclError(RuntimeError):
    pass


class Acl:
    """ctypes binding for the handful of aclrt APIs this probe needs."""

    def __init__(self, path: str):
        self.lib = ctypes.CDLL(path)
        self.path = path
        c_i32, c_u64, c_vp = ctypes.c_int32, ctypes.c_uint64, ctypes.c_void_p
        f = self.lib
        f.aclrtSetDevice.argtypes = [c_i32]
        f.aclrtResetDevice.argtypes = [c_i32]
        f.aclrtCreateStream.argtypes = [ctypes.POINTER(c_vp)]
        f.aclrtDestroyStream.argtypes = [c_vp]
        f.aclrtCreateEvent.argtypes = [ctypes.POINTER(c_vp)]
        f.aclrtDestroyEvent.argtypes = [c_vp]
        f.aclrtRecordEvent.argtypes = [c_vp, c_vp]
        f.aclrtSynchronizeEvent.argtypes = [c_vp]
        f.aclrtStreamWaitEvent.argtypes = [c_vp, c_vp]
        f.aclrtSynchronizeStream.argtypes = [c_vp]
        f.aclrtMalloc.argtypes = [ctypes.POINTER(c_vp), c_u64, ctypes.c_int32]
        f.aclrtFree.argtypes = [c_vp]
        f.aclrtMallocHost.argtypes = [ctypes.POINTER(c_vp), c_u64]
        f.aclrtFreeHost.argtypes = [c_vp]
        f.aclrtMemcpy.argtypes = [c_vp, c_u64, c_vp, c_u64, ctypes.c_int32]
        f.aclrtMemcpyAsync.argtypes = [c_vp, c_u64, c_vp, c_u64, ctypes.c_int32, c_vp]
        for name in (
            "aclrtSetDevice",
            "aclrtResetDevice",
            "aclrtCreateStream",
            "aclrtDestroyStream",
            "aclrtCreateEvent",
            "aclrtDestroyEvent",
            "aclrtRecordEvent",
            "aclrtSynchronizeEvent",
            "aclrtStreamWaitEvent",
            "aclrtSynchronizeStream",
            "aclrtMalloc",
            "aclrtFree",
            "aclrtMallocHost",
            "aclrtFreeHost",
            "aclrtMemcpy",
            "aclrtMemcpyAsync",
        ):
            getattr(f, name).restype = ctypes.c_int32

    def chk(self, err: int, what: str) -> None:
        if err != ACL_ERROR_NONE:
            raise AclError(f"{what} failed with aclError={err}")


def find_lib() -> str:
    cands = list(LIB_CANDIDATES)
    if os.environ.get("ACL_LIB"):
        cands.insert(0, os.environ["ACL_LIB"])
    found = glob.glob("/usr/local/Ascend/**/libascendcl.so", recursive=True)

    def rank(p: str) -> tuple[int, str]:
        # lexicographic order would hand us devlib (debug/symbol variants)
        # before the real runtime lib, and possibly a cross-compile x86_64
        # copy on an aarch64 host (server layout: cann-9.1.0/aarch64-linux/...)
        badness = 0
        if "devlib" in p:
            badness += 2
        if "x86_64" in p:
            badness += 1
        return (badness, p)

    cands += sorted(found, key=rank)
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise SystemExit("libascendcl.so not found; set ACL_LIB=/path/to/libascendcl.so")


def as_i32(ptr: int, n: int):
    return (ctypes.c_int32 * n).from_address(ptr)


def run(args) -> int:
    acl = Acl(find_lib())
    print(f"libascendcl: {acl.path}")
    acl.chk(acl.lib.aclrtSetDevice(args.device), "aclrtSetDevice")

    stream = ctypes.c_void_p()
    event = ctypes.c_void_p()
    acl.chk(acl.lib.aclrtCreateStream(ctypes.byref(stream)), "aclrtCreateStream")
    acl.chk(acl.lib.aclrtCreateEvent(ctypes.byref(event)), "aclrtCreateEvent")

    def dev_buf(nbytes: int) -> int:
        p = ctypes.c_void_p()
        acl.chk(acl.lib.aclrtMalloc(ctypes.byref(p), nbytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc")
        return p.value or 0

    def host_pin(nbytes: int) -> int:
        p = ctypes.c_void_p()
        acl.chk(acl.lib.aclrtMallocHost(ctypes.byref(p), nbytes), "aclrtMallocHost")
        return p.value or 0

    payload_b = args.payload * 4
    pin = host_pin(payload_b)
    pin_back = host_pin(payload_b)
    dev = dev_buf(payload_b)
    keep_plain = None
    bg_pin = bg_dev = None
    if args.host_mem == "plain":
        keep_plain = ctypes.create_string_buffer(payload_b)  # pageable malloc
        src_ptr = ctypes.addressof(keep_plain)
        src_view = as_i32(src_ptr, args.payload)
    else:
        src_ptr = pin
        src_view = as_i32(pin, args.payload)
    if args.bg > 0:
        bg_b = args.bg_elems * 4
        bg_pin = host_pin(bg_b)
        bg_dev = dev_buf(bg_b)
        ctypes.memset(bg_pin, 0, bg_b)

    miss = 0
    stale_lags: dict = {}

    # dispatch topology: torch_npu issues BOTH memcpys and kernels from its
    # task-queue consumer thread (OpCommand::RunOpApiV2 enqueues EXECUTE_OPAPI_V2
    # too), while the USER thread performs the fences (stream/event sync - the
    # host queue is drained before them). --dispatch threaded reproduces that
    # topology faithfully: a worker thread executes the H2D copies AND the D2H
    # readback in strict FIFO order; the main thread waits until everything is
    # submitted (queue.join, mirroring the host-queue drain), then
    # synchronizes the stream and compares. A stale read under THIS shape
    # means same-stream ordering depends on which thread submitted the work -
    # a runtime defect form direct submission cannot expose.
    import queue as queue_mod
    import threading

    work_q: queue_mod.Queue = queue_mod.Queue()
    h2d_failed: list = []

    def submit(dst: int, src: int, nbytes: int, kind: int, what: str) -> None:
        if args.dispatch == "direct":
            acl.chk(acl.lib.aclrtMemcpyAsync(dst, nbytes, src, nbytes, kind, stream), what)
        else:
            work_q.put((dst, src, nbytes, kind, what))

    def worker() -> None:
        # ACL contexts are PER-THREAD: torch_npu's queue consumer calls
        # SetDevice first (StartConsume, NPUQueue.cpp:795) - without this the
        # first memcpy returns 107002 ACL_ERROR_RT_CONTEXT_NULL (observed on
        # the server, 2026-08-27). Mirror the production topology exactly.
        try:
            err = acl.lib.aclrtSetDevice(args.device)
            if err != ACL_ERROR_NONE:
                h2d_failed.append(f"worker aclrtSetDevice: aclError={err}")
                # drain the queue and bail out without submitting anything
                while True:
                    item = work_q.get()
                    work_q.task_done()
                    if item is None:
                        return
        except Exception as e:  # noqa: BLE001
            h2d_failed.append(f"worker aclrtSetDevice: {e}")
        while True:
            item = work_q.get()
            if item is None:
                work_q.task_done()
                acl.lib.aclrtResetDevice(args.device)
                return
            dst, src, nbytes, kind, what = item
            try:
                err = acl.lib.aclrtMemcpyAsync(dst, nbytes, src, nbytes, kind, stream)
                if err != ACL_ERROR_NONE:
                    h2d_failed.append(f"{what}: aclError={err}")
            except Exception as e:  # noqa: BLE001
                h2d_failed.append(f"{what}: {e}")
            work_q.task_done()

    worker_thread = None
    if args.dispatch == "threaded":
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

    t0 = time.monotonic()
    for step in range(1, args.steps + 1):
        src_view[0] = step  # host write: this step's value

        if bg_dev is not None:
            for _ in range(args.bg):
                submit(bg_dev, bg_pin, args.bg_elems * 4, ACL_MEMCPY_HOST_TO_DEVICE, "bg aclrtMemcpyAsync")

        if args.mode == "racy":
            submit(dev, src_ptr, payload_b, ACL_MEMCPY_HOST_TO_DEVICE, "test aclrtMemcpyAsync")
        else:  # sync control: synchronous aclrtMemcpy, no stream involved
            acl.chk(
                acl.lib.aclrtMemcpy(dev, payload_b, src_ptr, payload_b, ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy",
            )

        # fences mirror torch_npu: NPUStream::synchronize() drains the host
        # queue BEFORE waiting, so under threaded dispatch the worker must
        # have submitted everything first (join), then the fence runs.
        if args.fence != "none" and args.dispatch == "threaded":
            work_q.join()
        if args.fence == "stream_sync":
            acl.chk(acl.lib.aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")
        elif args.fence == "event_sync":
            acl.chk(acl.lib.aclrtRecordEvent(event, stream), "aclrtRecordEvent")
            acl.chk(acl.lib.aclrtSynchronizeEvent(event), "aclrtSynchronizeEvent")
        elif args.fence == "event_wait":
            acl.chk(acl.lib.aclrtRecordEvent(event, stream), "aclrtRecordEvent")
            acl.chk(acl.lib.aclrtStreamWaitEvent(stream, event), "aclrtStreamWaitEvent")

        # consumer: async D2H on the same stream; under threaded dispatch it
        # goes through the worker AFTER all H2Ds of this step (strict FIFO),
        # exactly like kernels follow the copy in torch_npu's host queue.
        submit(pin_back, dev, payload_b, ACL_MEMCPY_DEVICE_TO_HOST, "readback aclrtMemcpyAsync")

        if args.dispatch == "threaded":
            work_q.join()  # everything submitted (host-queue drain equivalent)
        acl.chk(acl.lib.aclrtSynchronizeStream(stream), "final aclrtSynchronizeStream")

        got = as_i32(pin_back, args.payload)[0]
        if got != step:
            miss += 1
            lag = step - got
            stale_lags[lag] = stale_lags.get(lag, 0) + 1

    wall = time.monotonic() - t0
    if worker_thread is not None:
        work_q.put(None)
        worker_thread.join(timeout=30)
        if h2d_failed:
            print(f"worker-thread ACL errors: {h2d_failed[:3]}")

    # cleanup
    if bg_dev is not None:
        acl.lib.aclrtFree(bg_dev)
        acl.lib.aclrtFreeHost(bg_pin)
    acl.lib.aclrtFree(dev)
    acl.lib.aclrtFreeHost(pin)
    acl.lib.aclrtFreeHost(pin_back)
    acl.lib.aclrtDestroyEvent(event)
    acl.lib.aclrtDestroyStream(stream)
    acl.lib.aclrtResetDevice(args.device)

    bg_desc = f"{args.bg}x{args.bg_elems * 4 // 1024}KiB" if args.bg else "off"
    print(
        f"mode={args.mode} fence={args.fence} dispatch={args.dispatch} host_mem={args.host_mem} "
        f"steps={args.steps} payload={payload_b}B bg={bg_desc} "
        f"wall={wall:.2f}s ({wall / args.steps * 1e6:.1f}us/step)"
    )
    stale = ", ".join(f"lag={k}:{v}" for k, v in sorted(stale_lags.items())[:4])
    print(f"miss={miss} ({100.0 * miss / args.steps:.2f}% of steps)" + (f" stale: {stale}" if stale else ""))

    if h2d_failed:
        verdict = "ABORTED: worker-thread H2D errors (see above)"
    elif args.mode == "sync":
        verdict = "PASS (sync memcpy ordered)" if miss == 0 else "UNEXPECTED: even synchronous memcpy misreads"
    elif miss > 0 and args.dispatch == "threaded":
        verdict = (
            "CROSS-THREAD REPRODUCTION: memcpys submitted from another thread are not ordered"
            " with / not observed by the main thread's same-stream consumer and stream sync -"
            " matches torch_npu's task-queue topology"
        )
    elif miss > 0 and args.fence in ("none", "event_sync", "event_wait"):
        verdict = (
            "PURE-CANN REPRODUCTION: same-stream H2D->D2H read stale data (completion not joined into stream order)"
        )
    elif miss > 0 and args.fence == "stream_sync":
        verdict = (
            "STRONGEST: documented aclrtSynchronizeStream guarantee itself broken (data absent after full stream sync)"
        )
    elif miss == 0:
        verdict = (
            "NOT reproduced at pure-CANN level (raise --bg/--steps, try --dispatch threaded;"
            " if still clean, suspicion returns to torch_npu task queue internals)"
        )
    print(f"verdict: {verdict}")
    print(
        "note: complete the verdict matrix by also running: --fence stream_sync / event_sync"
        " / event_wait, --mode sync, --host-mem plain, --bg 0 (see module docstring)"
    )
    # observational probe: the verdict TEXT carries the reading; the exit code
    # only signals hard ACL errors (raised above), not miss counts - both
    # reproduced and clean outcomes are valid measurement results.
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--dispatch",
        choices=["direct", "threaded"],
        default="direct",
        help="direct: all ACL calls from the main thread; threaded: H2D memcpys from a"
        " worker thread (mirrors torch_npu's task-queue consumer), D2H/sync from main",
    )
    ap.add_argument("--mode", choices=["racy", "sync"], default="racy")
    ap.add_argument("--fence", choices=["none", "stream_sync", "event_sync", "event_wait"], default="none")
    ap.add_argument(
        "--host-mem",
        choices=["pinned", "plain"],
        default="pinned",
        help="H2D source memory: aclrtMallocHost (pinned) vs malloc (pageable)",
    )
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--payload", type=int, default=16, help="int32 elements of the tested buffer (16 = 64B)")
    ap.add_argument("--bg", type=int, default=8, help="background 1MiB async H2D copies per step (0 = calm)")
    ap.add_argument("--bg-elems", type=int, default=262144, help="int32 elements of each background copy")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
