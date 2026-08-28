#!/usr/bin/env python3
"""Pure-CANN (libascendcl + libopapi via ctypes) AI-core consumer visibility probe.

Extends research/cann_memcpy_order.py: the consumer is no longer the D2H
readback (SDMA engine read - the path the unified window model says is
unaffected) but aclnnAdd - a real AI-core (vector) kernel - launched on the
SAME stream immediately after the tested async H2D copy. Zero torch, zero
torch_npu: if THIS reproduces the stale read, the visibility gap of
research/h2d-visibility-race-report.md is demonstrated entirely below the
torch layer, in libascendcl/libopapi/runtime territory, with a self-contained
reproducer suitable for a cann/runtime issue.

Evidence protocol mirrors research/repro_h2d_order.py: the consumer writes
its verdict into a DEVICE history buffer (out = src + 1 per step) and the
host reads everything back exactly ONCE after the final synchronize - no
per-step host sync, so the instrumentation cannot close the window
(observer-effect lesson, report section 3.7).

Per-step timeline (mode=racy, fence=none, dispatch=direct):

    host : slot[0] <- step                     (alternating pinned host slots)
    [bg] : K x aclrtMemcpyAsync(1MiB H2D)      (SDMA backlog amplification)
    test : aclrtMemcpyAsync(dev_src <- slot, 64B, H2D, stream)
    [fence: none | stream_sync | event_sync | event_wait]
    cons : aclnnAdd(dev_src + dev_ones -> hist[step])    <- AI core reads dev_src
    end  : aclrtSynchronizeStream -> single sync D2H of hist -> host verdict

Expected under CUDA-style stream semantics (and the acl_rt.h contract):
hist[step][0] == step + 1. A miss means the AI-core kernel read a STALE
dev_src - the copy's completion was not joined into the stream's
data-visibility order. esc counts reads of the initial -1 sentinel (the
exact value family that escaped into the vocab gather in the original bug).

Verdict matrix (print at the end; complete by also running --fence
stream_sync / event_sync / event_wait, --mode sync, --bg 0):

  stale>0 or esc>0 (fence=none)     -> PURE-CANN AI-CORE REPRODUCTION (the
                                       kernel read data OLDER than its
                                       stream-preceding copy)
  + fence=stream_sync still stale   -> STRONGEST: aclrtSynchronizeStream returned
                                       but the next kernel still saw stale data
  + fence=event_* still stale       -> FENCE-BLIND: event apparatus cannot
                                       observe/inherit the copy's visibility
  future>0 only                     -> FUTURE-READ ANOMALY: the kernel read
                                       values from LATER steps. Two known
                                       mechanisms, neither of which is a
                                       visibility verdict: (a) v1/v2 probes
                                       (2026-08-28): host runs ahead inside a
                                       ~2048-task FIFO submission ring and a
                                       late copy read an OVERWRITTEN host slot
                                       - parity fingerprint: dominant lags even
                                       (slot rewrite period 2); fixed in v3 by
                                       unique per-step slots; (b) consumer
                                       kernel dispatched after later copies.
                                       The ring fingerprint: 2048/tasks_per_step
                                       predicted every server-measured lag to
                                       +-1 across five shapes.
  mode=sync miss==0                 -> negative control (blocking copy ordered);
                                       sync miss>0 = PROBE-INVALID run (pipeline
                                       distorted) - never cite it as evidence
  miss==0 everywhere                -> not reproduced at these parameters

API provenance (no untested shapes on the hot path): the aclnn two-stage
pattern, aclCreateTensor 9-arg form, aclCreateScalar and the
create -> GetWorkspaceSize -> run -> destroy order all mirror torch_npu /
op-plugin production usage (op_api_common_base.h typedefs,
AddKernelNpuOpApi.cpp EXEC_NPU_CMD, AtbCommon.h ConvertTypeV2); enum values
ACL_DT_INT32=3 / ACL_DT_INT64=9 / ACL_FORMAT_ND=2 verified against
cann-runtime include/external/acl/acl_base_rt.h.

Usage (server, inside the v0.23.0 container, no torch needed):
  python3 research/cann_aicore_visibility.py                       # racy, bg=8
  python3 research/cann_aicore_visibility.py --fence stream_sync
  python3 research/cann_aicore_visibility.py --mode sync           # control
  python3 research/cann_aicore_visibility.py --bg 0                # calm control
  python3 research/cann_aicore_visibility.py --dispatch threaded   # cross-thread
  python3 research/cann_aicore_visibility.py --selftest            # local mock run
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import os
import subprocess
import sys
import tempfile
import time

SENTINEL = -1  # upstream get_token_id()'s uncommitted-position sentinel
ACL_DT_INT32 = 3  # acl_base_rt.h aclDataType
ACL_DT_INT64 = 9
ACL_FORMAT_ND = 2  # acl_base_rt.h aclFormat
ACL_MEMCPY_HOST_TO_DEVICE = 1  # acl_rt.h aclrtMemcpyKind (H2H=0, H2D=1, D2H=2, D2D=3)
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_ERROR_NONE = 0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cann_memcpy_order  # noqa: E402  (Acl binding, find_lib, as_i32 reuse)

OPAPI_CANDIDATES = [
    "/usr/local/Ascend/ascend-toolkit/latest/lib64/libopapi.so",
    "/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/libopapi.so",
]


def find_opapi_lib() -> str:
    cands = list(OPAPI_CANDIDATES)
    if os.environ.get("OPAPI_LIB"):
        cands.insert(0, os.environ["OPAPI_LIB"])
    found = glob.glob("/usr/local/Ascend/**/libopapi.so", recursive=True)

    def rank(p: str) -> tuple[int, str]:
        # same ordering discipline as cann_memcpy_order.find_lib: never let
        # devlib (debug) or cross-compile copies win over the real runtime lib
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
    raise SystemExit("libopapi.so not found; set OPAPI_LIB=/path/to/libopapi.so")


class OpApi:
    """ctypes binding for the aclnn two-stage op API + tensor descriptors."""

    def __init__(self, opapi_path: str, acl: cann_memcpy_order.Acl):
        self.lib = ctypes.CDLL(opapi_path)
        self.path = opapi_path
        c_i32, c_i64, c_u64, c_vp = ctypes.c_int32, ctypes.c_int64, ctypes.c_uint64, ctypes.c_void_p
        p_i64, p_u64, p_vp = ctypes.POINTER(c_i64), ctypes.POINTER(c_u64), ctypes.POINTER(c_vp)
        spec = {
            "aclCreateTensor": ([p_i64, c_u64, c_i32, p_i64, c_i64, c_i32, p_i64, c_u64, c_vp], c_vp),
            "aclCreateScalar": ([c_vp, c_i32], c_vp),
            "aclDestroyTensor": ([c_vp], c_i32),
            "aclDestroyScalar": ([c_vp], c_i32),
            "aclnnAddGetWorkspaceSize": ([c_vp, c_vp, c_vp, c_vp, p_u64, p_vp], c_i32),
            "aclnnAdd": ([c_vp, c_u64, c_vp, c_vp], c_i32),
        }
        self.fn = {}
        for name, (argtypes, restype) in spec.items():
            fn = None
            for host in (self.lib, acl.lib):
                try:
                    fn = getattr(host, name)
                except AttributeError:
                    continue
                break
            if fn is None:
                raise SystemExit(
                    f"symbol {name} not exported by {opapi_path} nor libascendcl;"
                    f" verify with: nm -D {opapi_path} | grep {name}"
                )
            fn.argtypes = argtypes
            fn.restype = restype
            self.fn[name] = fn
        self._acl = acl
        self._ws = ctypes.c_void_p()  # cached workspace, grown on demand
        self._ws_bytes = 0

    def tensor(self, dev_ptr: int, n: int, dtype: int = ACL_DT_INT32) -> int:
        dims = (ctypes.c_int64 * 1)(n)
        strides = (ctypes.c_int64 * 1)(1)
        t = self.fn["aclCreateTensor"](dims, 1, dtype, strides, 0, ACL_FORMAT_ND, dims, 1, ctypes.c_void_p(dev_ptr))
        if not t:
            raise RuntimeError("aclCreateTensor returned NULL")
        return t

    def scalar(self, value: int, dtype: int = ACL_DT_INT64) -> int:
        v = ctypes.c_int64(value)
        s = self.fn["aclCreateScalar"](ctypes.byref(v), dtype)
        if not s:
            raise RuntimeError("aclCreateScalar returned NULL")
        return s

    def launch_add(self, src_t: int, other_t: int, alpha_s: int, out_t: int, stream) -> None:
        """Two-stage aclnnAdd: descriptor create -> GetWorkspaceSize -> run ->
        descriptor destroy, the exact order op-plugin's EXEC_NPU_CMD + RAII
        maintainers use per call."""
        ws_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        err = self.fn["aclnnAddGetWorkspaceSize"](
            src_t, other_t, alpha_s, out_t, ctypes.byref(ws_size), ctypes.byref(executor)
        )
        if err != ACL_ERROR_NONE:
            raise RuntimeError(f"aclnnAddGetWorkspaceSize failed: aclnnStatus={err}")
        if ws_size.value > self._ws_bytes:
            if self._ws:
                self._acl.lib.aclrtFree(self._ws)
            p = ctypes.c_void_p()
            self._acl.lib.aclrtMalloc(ctypes.byref(p), ws_size.value, cann_memcpy_order.ACL_MEM_MALLOC_HUGE_FIRST)
            self._ws, self._ws_bytes = p, ws_size.value
        err = self.fn["aclnnAdd"](self._ws, ws_size.value, executor, stream)
        if err != ACL_ERROR_NONE:
            raise RuntimeError(f"aclnnAdd failed: aclnnStatus={err}")
        for name, h in (
            ("aclDestroyTensor", src_t),
            ("aclDestroyTensor", other_t),
            ("aclDestroyTensor", out_t),
            ("aclDestroyScalar", alpha_s),
        ):
            self.fn[name](ctypes.c_void_p(h))

    def cleanup(self) -> None:
        if self._ws:
            self._acl.lib.aclrtFree(self._ws)
            self._ws = ctypes.c_void_p()
            self._ws_bytes = 0


def run(args) -> int:
    acl = cann_memcpy_order.Acl(cann_memcpy_order.find_lib())
    opapi = OpApi(find_opapi_lib(), acl)
    print(f"libascendcl: {acl.path}")
    print(f"libopapi:    {opapi.path}")
    acl.chk(acl.lib.aclrtSetDevice(args.device), "aclrtSetDevice")

    stream = ctypes.c_void_p()
    event = ctypes.c_void_p()
    acl.chk(acl.lib.aclrtCreateStream(ctypes.byref(stream)), "aclrtCreateStream")
    acl.chk(acl.lib.aclrtCreateEvent(ctypes.byref(event)), "aclrtCreateEvent")

    payload_b = args.payload * 4

    def dev_buf(nbytes: int) -> int:
        p = ctypes.c_void_p()
        acl.chk(
            acl.lib.aclrtMalloc(ctypes.byref(p), nbytes, cann_memcpy_order.ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc",
        )
        return p.value or 0

    def host_pin(nbytes: int) -> int:
        p = ctypes.c_void_p()
        acl.chk(acl.lib.aclrtMallocHost(ctypes.byref(p), nbytes), "aclrtMallocHost")
        return p.value or 0

    def h2d_sync(dev: int, host_vals, nbytes: int, what: str) -> None:
        acl.chk(
            acl.lib.aclrtMemcpy(
                dev, nbytes, ctypes.cast(host_vals, ctypes.c_void_p), nbytes, ACL_MEMCPY_HOST_TO_DEVICE
            ),
            what,
        )

    # v4: per-step UNIQUE device slices (round-3 lesson, 2026-08-28). v3 made
    # host slots unique but kept ONE shared dev_src; that residual sharing is
    # why the sync-mode control stayed red: a blocking copy bypasses the ring
    # and executes immediately, so with the host ~227 steps ahead the later
    # blocking copies OVERWROTE dev_src before the ring-queued Add_k of an
    # earlier step ran (future reads, lag -227 = 2048/9 exactly). With one
    # device slice per step - initialized to SENTINEL, written by exactly one
    # copy, read by exactly one kernel - no cross-step value flow exists at
    # all: a miss can only manifest as esc (copy not landed / not visible
    # when its kernel ran) or garbage. future becomes structurally
    # impossible; the negative control must be green again.
    slots_b = args.steps * payload_b
    pin = host_pin(slots_b)
    dev_slices = dev_buf(slots_b)
    dev_ones = dev_buf(payload_b)
    hist = dev_buf(args.steps * payload_b)
    keep_plain = None
    if args.host_mem == "plain":
        # pageable source: one big buffer, same per-step addressing
        keep_plain = ctypes.create_string_buffer(slots_b)
        slots_base = ctypes.addressof(keep_plain)
    else:
        slots_base = pin
    bg_pin = bg_dev = None
    if args.bg > 0:
        bg_b = args.bg_elems * 4
        bg_pin = host_pin(bg_b)
        bg_dev = dev_buf(bg_b)
        ctypes.memset(bg_pin, 0, bg_b)

    # init: ones = 1; every device slice = SENTINEL (a copy that has not
    # landed / not become visible when its kernel runs surfaces as out==0 -
    # the sentinel-escape counter of the original bug)
    ones_host = (ctypes.c_int32 * args.payload)(*([1] * args.payload))
    sent_all = (ctypes.c_int32 * (args.steps * args.payload))(*([SENTINEL] * (args.steps * args.payload)))
    h2d_sync(dev_ones, ones_host, payload_b, "init ones aclrtMemcpy")
    h2d_sync(dev_slices, sent_all, slots_b, "init sentinel aclrtMemcpy")

    import queue as queue_mod
    import threading

    work_q: queue_mod.Queue = queue_mod.Queue()
    copy_errors: list = []

    def submit(dst: int, src: int, nbytes: int, what: str) -> None:
        if args.dispatch == "direct":
            acl.chk(acl.lib.aclrtMemcpyAsync(dst, nbytes, src, nbytes, ACL_MEMCPY_HOST_TO_DEVICE, stream), what)
        else:
            work_q.put((dst, src, nbytes, what))

    def worker() -> None:
        # ACL contexts are PER-THREAD (observed server-side 2026-08-27:
        # first memcpy returns 107002 CONTEXT_NULL without this), mirroring
        # torch_npu's queue consumer calling SetDevice first.
        try:
            err = acl.lib.aclrtSetDevice(args.device)
            if err != ACL_ERROR_NONE:
                copy_errors.append(f"worker aclrtSetDevice: aclError={err}")
                while True:
                    item = work_q.get()
                    work_q.task_done()
                    if item is None:
                        return
        except Exception as e:  # noqa: BLE001
            copy_errors.append(f"worker aclrtSetDevice: {e}")
        while True:
            item = work_q.get()
            if item is None:
                work_q.task_done()
                acl.lib.aclrtResetDevice(args.device)
                return
            dst, src, nbytes, what = item
            try:
                err = acl.lib.aclrtMemcpyAsync(dst, nbytes, src, nbytes, ACL_MEMCPY_HOST_TO_DEVICE, stream)
                if err != ACL_ERROR_NONE:
                    copy_errors.append(f"{what}: aclError={err}")
            except Exception as e:  # noqa: BLE001
                copy_errors.append(f"{what}: {e}")
            work_q.task_done()

    worker_thread = None
    if args.dispatch == "threaded":
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

    t0 = time.monotonic()
    steps_done = 0
    for step in range(1, args.steps + 1):
        # host write into THIS STEP'S OWN slot: the ctypes view spans exactly
        # one payload (payload int32 == payload_b bytes) inside the per-step
        # slot region, and only element [0] is stored - no out-of-bounds path
        # exists by construction; written exactly once, never rewritten.
        assert args.payload >= 1, "payload must cover at least one int32"
        slot_ptr = slots_base + (step - 1) * payload_b
        cann_memcpy_order.as_i32(slot_ptr, args.payload)[0] = step  # host write

        if bg_dev is not None:
            for _ in range(args.bg):
                submit(bg_dev, bg_pin, args.bg_elems * 4, "bg aclrtMemcpyAsync")

        if args.mode == "racy":
            submit(dev_slices + (step - 1) * payload_b, slot_ptr, payload_b, "test aclrtMemcpyAsync")
        else:  # sync control: blocking aclrtMemcpy, no stream involved
            h2d_sync(dev_slices + (step - 1) * payload_b, slot_ptr, payload_b, "test aclrtMemcpy(sync)")

        # threaded: work_q.join() guarantees the aclrtMemcpyAsync CALLS have
        # returned in FIFO order before the kernel launch - the API-level
        # submission-order precondition (without it the kernel could be
        # submitted before the tested copy: a FALSE POSITIVE). It does NOT
        # (and cannot, without a stream sync that would destroy the window)
        # guarantee hardware-level enqueue; deferred enqueue only moves this
        # run along the submission-timing spectrum of the window model
        # (report section 4) - it cannot turn a real defect into a clean
        # pass, because same-stream execution still orders copy before
        # kernel and the defect is precisely "completed copy, stale read".
        # The DECISIVE arm for red/green is --dispatch direct (no queue
        # indirection at all); threaded is the exploratory cross-thread
        # topology point, conflation caveat as in grid 3 of the report.
        if args.dispatch == "threaded":
            work_q.join()
            if copy_errors:
                print(f"worker-thread ACL errors: {copy_errors[:3]}")
                print(f"ABORT at step {step}: copies were dropped, remaining steps not measured")
                break
        if args.fence == "stream_sync":
            acl.chk(acl.lib.aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")
        elif args.fence == "event_sync":
            acl.chk(acl.lib.aclrtRecordEvent(event, stream), "aclrtRecordEvent")
            acl.chk(acl.lib.aclrtSynchronizeEvent(event), "aclrtSynchronizeEvent")
        elif args.fence == "event_wait":
            acl.chk(acl.lib.aclrtRecordEvent(event, stream), "aclrtRecordEvent")
            acl.chk(acl.lib.aclrtStreamWaitEvent(stream, event), "aclrtStreamWaitEvent")

        # consumer: AI-core kernel on the same stream; verdict lands in hist
        src_t = opapi.tensor(dev_slices + (step - 1) * payload_b, args.payload)
        ones_t = opapi.tensor(dev_ones, args.payload)
        out_t = opapi.tensor(hist + (step - 1) * payload_b, args.payload)
        alpha_s = opapi.scalar(1)
        opapi.launch_add(src_t, ones_t, alpha_s, out_t, stream)
        steps_done = step

    wall = time.monotonic() - t0

    if worker_thread is not None:
        work_q.put(None)
        worker_thread.join(timeout=30)
        if copy_errors:
            print(f"worker-thread ACL errors: {copy_errors[:3]}")
    acl.chk(acl.lib.aclrtSynchronizeStream(stream), "final aclrtSynchronizeStream")

    hist_host = (ctypes.c_int32 * (args.steps * args.payload))()
    acl.chk(
        acl.lib.aclrtMemcpy(
            ctypes.cast(hist_host, ctypes.c_void_p),
            args.steps * payload_b,
            hist,
            args.steps * payload_b,
            ACL_MEMCPY_DEVICE_TO_HOST,
        ),
        "final D2H aclrtMemcpy",
    )

    # analyze only the steps whose consumer kernel was actually launched
    # (after an abort, later hist slots hold unwritten device memory)
    #
    # stale-vs-future split is the whole game (server lesson 2026-08-28):
    #  - STALE (src older than its step)  = the visibility-gap signature
    #  - FUTURE (src newer than its step)  = the kernel executed AFTER later
    #    same-stream copies - a launch/dispatch-pipeline artifact that says
    #    NOTHING about copy->kernel visibility and must never be cited as a
    #    reproduction (the first server matrix hit exactly this: ~200-step
    #    future lag in every non-fence run, negative control included)
    miss = esc = garbage = stale = future = 0
    lags: dict[int, int] = {}
    future_lags: dict[int, int] = {}
    examples: list[str] = []
    for k in range(1, steps_done + 1):
        out0 = hist_host[(k - 1) * args.payload]
        if out0 == k + 1:
            continue
        miss += 1
        src_saw = out0 - 1
        if out0 == 0:  # read SENTINEL: src never overwritten for this step's reader
            esc += 1
        elif src_saw == 0 or src_saw > args.steps:
            garbage += 1
        elif src_saw > k:
            future += 1
            future_lags[k - src_saw] = future_lags.get(k - src_saw, 0) + 1
        else:
            stale += 1
            lags[k - src_saw] = lags.get(k - src_saw, 0) + 1
        if len(examples) < 4:
            kind = "FUTURE" if src_saw > k else ("SENTINEL" if out0 == 0 else "STALE")
            examples.append(f"step={k} kernel_saw_src={src_saw} (want {k}) [{kind}]")

    # cleanup
    opapi.cleanup()
    if bg_dev is not None:
        acl.lib.aclrtFree(bg_dev)
        acl.lib.aclrtFreeHost(bg_pin)
    acl.lib.aclrtFree(hist)
    acl.lib.aclrtFree(dev_ones)
    acl.lib.aclrtFree(dev_slices)
    acl.lib.aclrtFreeHost(pin)
    acl.lib.aclrtDestroyEvent(event)
    acl.lib.aclrtDestroyStream(stream)
    acl.lib.aclrtResetDevice(args.device)

    bg_desc = f"{args.bg}x{args.bg_elems * 4 // 1024}KiB" if args.bg else "off"
    steps_desc = f"{steps_done}/{args.steps}" if steps_done != args.steps else f"{args.steps}"
    print(
        f"mode={args.mode} fence={args.fence} dispatch={args.dispatch} host_mem={args.host_mem} "
        f"steps={steps_desc} payload={payload_b}B bg={bg_desc} "
        f"wall={wall:.2f}s ({wall / args.steps * 1e6:.1f}us/step)"
    )
    lag_desc = ", ".join(f"lag={k}:{v}" for k, v in sorted(lags.items())[:4])
    fl = ", ".join(f"lag={k}:{v}" for k, v in sorted(future_lags.items())[:4])
    print(
        f"miss={miss} ({100.0 * miss / max(steps_done, 1):.2f}% of {steps_done} measured steps)"
        f" stale={stale} future={future} sentinel-escapes={esc}"
        + (f" garbage={garbage}" if garbage else "")
        + (f" | stale-lags: {lag_desc}" if lag_desc else "")
        + (f" | future-lags: {fl}" if fl else "")
    )
    for e in examples:
        print(f"  {e}")

    if copy_errors:
        verdict = "ABORTED: worker-thread ACL errors (see above)"
    elif args.mode == "sync":
        verdict = (
            "PASS (sync memcpy ordered)"
            if miss == 0
            else "PROBE-INVALID RUN: sync-mode negative control must be green - the pipeline was"
            " distorted (see stale/future split); never cite this run as evidence"
        )
    elif stale == 0 and esc == 0 and future > 0:
        verdict = (
            "FUTURE-READ ANOMALY (post-v3 = unexpected): the AI-core kernel read values from steps"
            " LATER than its own. With unique per-step slots (this version) a late copy can no longer"
            " deliver overwritten values, so this indicates the consumer kernel executed after later"
            " same-stream copies (kernel-side dispatch lag) - still NOT a copy->kernel visibility"
            " verdict. Compare against the 2026-08-28 ring fingerprint: the runtime stages ~2048"
            " tasks in a FIFO submission ring (2048/tasks_per_step predicted every measured lag"
            " to +-1), so host run-ahead is structural. Decisive cell stays:"
            " --mode racy --bg 16 --bg-elems 1048576 --steps 5000."
        )
    elif stale > 0 or esc > 0:
        extra = f" (mixed with {future} future reads - dispatch lag present, see split)" if future else ""
        if args.fence == "stream_sync":
            verdict = (
                "STRONGEST: aclrtSynchronizeStream returned, yet the next same-stream"
                f" AI-core kernel still read stale data - visibility not covered by full stream sync{extra}"
            )
        elif args.fence in ("event_sync", "event_wait"):
            verdict = f"FENCE-BLIND: event fences completed, yet the AI-core kernel still read stale data{extra}"
        else:
            verdict = (
                "PURE-CANN AI-CORE REPRODUCTION: same-stream aclnn kernel read data older"
                " than its stream-preceding aclrtMemcpyAsync (H2D) - ordering/visibility gap"
                f" reproduced with ZERO torch/torch_npu in the loop{extra}"
            )
    else:
        verdict = (
            "NOT reproduced at these parameters - the decisive pressure cell is"
            " --mode racy --bg 16 --bg-elems 1048576 --steps 5000 (SDMA lag dominating"
            " kernel dispatch lag); keep --mode sync / --bg 0 controls green"
        )
    print(f"verdict: {verdict}")
    print(
        "note: complete the verdict matrix: --fence stream_sync|event_sync|event_wait,"
        " --mode sync, --bg 0, --dispatch threaded (see module docstring)"
    )
    return 0


MOCK_SRC = "cann_aicore_visibility_mock.c"


def selftest() -> int:
    """Compile the local mock (libascendcl+libopapi subset, pending-landing
    memory model) and run the script against it across the semantic branches:
    clean / lag-1 stale / fence-saves / fence-blind / sync-blind / sync-copy."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, MOCK_SRC)
    if not os.path.isfile(src):
        raise SystemExit(f"mock source not found: {src}")
    tmp = tempfile.mkdtemp(prefix="cann_mock_")
    so = os.path.join(tmp, "mock_cann.so")
    subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", so, src], check=True)
    self_py = os.path.abspath(__file__)
    base_env = {
        **os.environ,
        "ACL_LIB": so,
        "OPAPI_LIB": so,
        "ACLMOCK_LANDING": "immediate",
        "ACLMOCK_SYNC_BLIND": "0",
        "ACLMOCK_EVENT_BLIND": "0",
    }

    def run_mock(env_extra: dict, *cli: str) -> str:
        env = {**base_env, **env_extra}
        out = subprocess.run([sys.executable, self_py, *cli], env=env, capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            return f"RC={out.returncode}\n{out.stdout}\n{out.stderr}"
        return out.stdout

    cases = [
        (
            "clean (immediate landing)",
            {},
            ["--steps", "50"],
            # "mock_cann.so" needle proves the subprocess really routed to the
            # mock via ACL_LIB/OPAPI_LIB (guards against a silent hardware run)
            ("mock_cann.so", "miss=0 (0.00%", "sentinel-escapes=0", "NOT reproduced"),
        ),
        (
            "clean (sync mode control)",
            {},
            ["--mode", "sync", "--steps", "50"],
            ("miss=0", "PASS"),
        ),
        (
            "lag-1 stale: racy esc=100% (copy never landed before kernel)",
            {"ACLMOCK_LANDING": "lag1"},
            ["--steps", "50"],
            ("miss=50 (100.00%", "sentinel-escapes=50", "stale=0", "PURE-CANN AI-CORE REPRODUCTION"),
        ),
        (
            "lag-1 + stream_sync fence saves",
            {"ACLMOCK_LANDING": "lag1"},
            ["--fence", "stream_sync", "--steps", "50"],
            ("miss=0",),
        ),
        (
            "lag-1 + event fence blind (miss stays)",
            {"ACLMOCK_LANDING": "lag1", "ACLMOCK_EVENT_BLIND": "1"},
            ["--fence", "event_sync", "--steps", "50"],
            ("miss=50", "FENCE-BLIND"),
        ),
        (
            "lag-1 + sync-blind stream fence (miss stays)",
            {"ACLMOCK_LANDING": "lag1", "ACLMOCK_SYNC_BLIND": "1"},
            ["--fence", "stream_sync", "--steps", "50"],
            ("miss=50", "STRONGEST"),
        ),
        (
            "dispatch-lag with unique slices: late kernels must stay clean",
            {"ACLMOCK_LANDING": "immediate", "ACLMOCK_EXEC_DELAY": "1"},
            ["--steps", "50"],
            ("miss=0", "NOT reproduced"),
        ),
    ]
    ok = True
    for name, env_extra, cli, needles in cases:
        out = run_mock(env_extra, *cli)
        missing = [n for n in needles if n not in out]
        status = "PASS" if not missing else f"FAIL (missing {missing})"
        if missing:
            ok = False
            out_preview = "\n".join(out.splitlines()[-6:])
            print(f"--- {name}: {status}\n{out_preview}")
        else:
            print(f"--- {name}: {status}")
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--mode", choices=["racy", "sync"], default="racy")
    ap.add_argument("--fence", choices=["none", "stream_sync", "event_sync", "event_wait"], default="none")
    ap.add_argument("--host-mem", choices=["pinned", "plain"], default="pinned")
    ap.add_argument("--dispatch", choices=["direct", "threaded"], default="direct")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument(
        "--payload", type=int, default=16, help="int32 elements of the tested buffer (16 = 64B, bug-site scale)"
    )
    ap.add_argument("--bg", type=int, default=8, help="background 1MiB async H2D copies per step (0 = calm window)")
    ap.add_argument("--bg-elems", type=int, default=262144, help="int32 elements of each background copy")
    ap.add_argument(
        "--selftest", action="store_true", help="compile the local mock lib and verify all verdict branches"
    )
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
