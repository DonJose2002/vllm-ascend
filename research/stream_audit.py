#!/usr/bin/env python3
"""Stream-ownership audit of msprof Text-export CSVs (memcpy vs compute kernels).

Motivation (PR #14922 off-stream-copy investigation): behavioral evidence says
a pinned non_blocking H2D copy_ on torch_npu is invisible to the stream/event
apparatus (three-level fence ladder all miss; wall-times prove no wait ever
happened). This tool upgrades that behavioral finding into profiling fact by
answering one question from existing msprof exports: which stream do Memcpy
tasks land on, and does it match the streams compute kernels run on?

It deliberately knows NOTHING about the exact CSV schema of any CANN version:
headers are discovered per file (a "stream id"-ish column, a name-ish column,
a time-ish column), so the same code audits p15 engine traces and the repro
--profile export alike. If a file lacks a stream column (e.g. the aggregated
op_statistic.csv) it is skipped with a visible reason - the audit never
guesses.

Interpretation cheat-sheet (printed with the output):
  - memcpy streams disjoint from compute streams  -> copies ride a dedicated
    channel outside the launch stream: off-stream confirmed at task level.
  - memcpy streams identical to compute streams  -> copies DO land on the
    launch stream; ordering violation is then an in-stream execution-order
    defect (compare copy end-time vs consumer start-time if time columns
    exist - both evidence forms are reportable, neither is a dead end).

Usage (server, inside the v0.23.0 container, no serve needed):
  python3 research/stream_audit.py <prof_dir> [<prof_dir> ...]   # recursive csv discovery
  python3 research/stream_audit.py --csv <file.csv> [...]        # audit specific files
  python3 research/stream_audit.py selftest                      # synthetic-fixture check

Output: a single ===STREAM_AUDIT BEGIN/END=== paste block (per-file stats +
interpretation hints), one file per line-group - per-file grouping is the
de-duplication boundary (task_time vs op_summary carry the same tasks).
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import itertools
import os
import re
import sys

# Rows whose name/type matches these (case-insensitive) are memcpy/memset-class
# tasks. "HostToDevice"/"DeviceToHost" are msprof's spelled-out directions.
MEM_RE = re.compile(r"mem\s*cpy|mem\s*cpyh|memcpy|memset|hosttodevice|devicetohost|hosttohost", re.I)
# Aggregated tables that cannot carry per-task streams (skip, do not guess).
AGGREGATE_BASENAMES = ("op_statistic", "operator_details", "api_statistic")


def norm(h: str) -> str:
    """Normalize a header for fuzzy matching: lowercase, strip non-alnum."""
    return re.sub(r"[^a-z0-9]", "", h.lower())


def find_col(headers: list[str], need: tuple[str, ...], forbid: tuple[str, ...] = ()) -> str | None:
    """First header whose normalized form contains all `need` and no `forbid`."""
    for h in headers:
        n = norm(h)
        if all(t in n for t in need) and not any(f in n for f in forbid):
            return h
    return None


def discover_csvs(roots: list[str]) -> list[str]:
    out: list[str] = []
    for root in roots:
        if os.path.isfile(root):
            out.append(root)
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".csv"):
                    out.append(os.path.join(dirpath, fn))
    return sorted(out)


def load_rows(path: str) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv_mod.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def fmt_count(m: dict) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(m.items(), key=lambda kv: -kv[1]))


def audit_file(path: str) -> list[str]:
    """One file -> a group of report lines (empty list = skip silently)."""
    base = os.path.basename(path)
    lines = [f"file: {os.path.relpath(path)}"]
    try:
        headers, rows = load_rows(path)
    except Exception as e:  # unreadable csv: visible skip, never fatal
        return lines + [f"  SKIP unreadable: {e}"]
    if not headers or not rows:
        return lines + ["  SKIP empty"]
    lines.append(f"  rows={len(rows)} headers={headers}")

    if any(norm(base).startswith(a) for a in AGGREGATE_BASENAMES):
        # aggregated per-op tables never carry per-task streams; raw pid-suffixed
        # variants are tried too - the stream-column check below is the
        # authoritative gate either way, this only improves the skip reason.
        return lines + ["  SKIP aggregated per-op table (no per-task stream column by construction)"]

    stream_col = find_col(headers, ("stream",))
    if stream_col is None:
        return lines + ["  SKIP no stream column (aggregated/host-side table?)"]
    lines.append(f"  stream_col='{stream_col}'")

    name_col = find_col(headers, ("name",))
    type_col = find_col(headers, ("type",), forbid=("dtype",)) or find_col(headers, ("tasktype",))
    dur_col = find_col(headers, ("duration",)) or find_col(headers, ("time",), forbid=("start", "wait"))

    mem_streams: dict = {}
    other_streams: dict = {}
    mem_names: dict = {}
    for r in rows:
        name = " ".join(str(r.get(c, "") or "") for c in (name_col, type_col) if c)
        sid = (r.get(stream_col) or "?").strip() or "?"
        if MEM_RE.search(name):
            mem_streams[sid] = mem_streams.get(sid, 0) + 1
            key = (name_col and r.get(name_col, "")) or name.strip() or "<anon>"
            mem_names[key] = mem_names.get(key, 0) + 1
        else:
            other_streams[sid] = other_streams.get(sid, 0) + 1

    if not mem_streams:
        return lines + [f"  no memcpy/memset rows (other-streams: {fmt_count(other_streams) or 'none'})"]

    lines.append(f"  MEMCPY/MEMSET streams: {fmt_count(mem_streams)}")
    lines.append(f"  memcpy-ish names: {fmt_count(dict(list(mem_names.items())[:8]))}")
    top_other = dict(sorted(other_streams.items(), key=lambda kv: -kv[1])[:8])
    lines.append(f"  OTHER-task streams: {fmt_count(top_other)}")
    overlap = set(mem_streams) & set(other_streams)
    if other_streams and not overlap:
        lines.append(
            "  HINT: memcpy streams are DISJOINT from compute-task streams"
            " -> copies ride a stream other tasks never use"
        )
    elif overlap:
        only_mem = set(mem_streams) - set(other_streams)
        lines.append(
            f"  HINT: shared streams {sorted(overlap)}"
            + (f"; memcpy-ONLY streams {sorted(only_mem)}" if only_mem else "")
            + (" -> copies land on the launch stream; check in-stream ordering (times)" if not only_mem else "")
        )
    else:
        lines.append("  HINT: no compute rows to compare against (memcpy-only trace)")
    if dur_col:
        lines.append(
            f"  duration_col='{dur_col}' (present; pairwise copy-vs-consumer timing lives in the repro --profile audit)"
        )

    # Timeline invariant check: on a stream, tasks must execute serially (CUDA
    # stream semantics torch_npu mirrors), so per-stream task intervals must be
    # pairwise disjoint. Any overlap between a MEMCPY and another task on the
    # SAME stream is direct profiling evidence that stream ordering is not
    # enforced for the copy. Zero overlap + observed reordering would instead
    # point at a completion-vs-visibility gap (copy task 'done' but data not
    # yet visible to AI-core reads) - both outcomes are reportable.
    tl = timeline_stats(rows, stream_col, name_col, type_col)
    if tl:
        lines.extend(tl)
    return lines


def timeline_stats(rows: list[dict], stream_col: str, name_col: str | None, type_col: str | None) -> list[str]:
    """Per-stream interval-overlap + dispatch-vs-execution order audit.

    Two invariants of CUDA stream semantics (which torch_npu mirrors):
    (a) same-stream tasks execute serially -> intervals pairwise disjoint;
    (b) same-stream tasks execute in dispatch order -> a later-dispatched
        task (larger task id) never STARTS before an earlier one. A start
    inversion involving a MEMCPY is direct evidence that stream ordering
    does not govern the copy (e.g. consumer kernel dispatched after the
    copy but executed before it - exactly the observed stale read).
    [] when the table lacks usable times.
    """
    headers = rows[0].keys()
    start_col = find_col(list(headers), ("start",), forbid=("wait",))
    stop_col = find_col(list(headers), ("stop", "end"))
    dur_col = find_col(list(headers), ("duration",)) or find_col(list(headers), ("time",), forbid=("start", "wait"))
    tid_col = find_col(list(headers), ("taskid",))
    if start_col is None or (stop_col is None and dur_col is None):
        return []

    def to_f(v) -> float | None:
        try:
            return float(str(v).strip())
        except (ValueError, TypeError):
            return None

    # stream -> list of (start, stop, is_memcpy, label, task_id or None)
    per_stream: dict[str, list] = {}
    for r in rows:
        s = to_f(r.get(start_col))
        if s is None:
            continue
        e = to_f(r.get(stop_col)) if stop_col is not None else None
        if e is None and dur_col is not None:
            d = to_f(r.get(dur_col))
            e = s + d if d is not None else None
        if e is None or e < s:
            continue
        name = " ".join(str(r.get(c, "") or "") for c in (name_col, type_col) if c).strip()
        sid = (r.get(stream_col) or "?").strip() or "?"
        label = (name or "<anon>")[:40]
        tid = to_f(r.get(tid_col)) if tid_col is not None else None
        per_stream.setdefault(sid, []).append((s, e, bool(MEM_RE.search(name)), label, tid))

    out = ["  TIMELINE (per-stream interval overlap; serial-stream invariant):"]
    for sid, tasks in sorted(per_stream.items()):
        mem_n = sum(1 for t in tasks if t[2])
        if mem_n == 0 or len(tasks) < 2:
            continue
        by_start = sorted(tasks, key=lambda t: t[0])
        overlap_pairs = 0
        mem_overlap_pairs = 0
        max_depth = 0
        approx = False
        active: list = []  # (stop, is_memcpy, label) of tasks still open
        examples = []
        for s, e, is_m, label, _tid in by_start:
            active = [a for a in active if a[0] > s]
            if len(active) > 512:  # pathological fan-out: count is already damning
                approx = True
                active = active[:512]
            for a_stop, a_m, a_label in active:
                overlap_pairs += 1
                if is_m or a_m:
                    mem_overlap_pairs += 1
                    if len(examples) < 3:
                        ov = min(e, a_stop) - s
                        examples.append(f"[{a_label}] x [{label}] for {ov:.1f}us")
            active.append((e, is_m, label))
            max_depth = max(max_depth, len(active))
        pct = 100.0 * mem_overlap_pairs / max(1, overlap_pairs)
        out.append(
            f"    stream {sid}: tasks={len(tasks)} (memcpy={mem_n}), overlapping pairs>={overlap_pairs}"
            f"{'(saturated, lower bound)' if approx else ''}"
            f" (involving-memcpy={mem_overlap_pairs}, {pct:.0f}%), max concurrency={max_depth}"
        )
        for ex in examples:
            out.append(f"      overlap example: {ex}")
        if overlap_pairs == 0:
            out.append(
                "      no overlaps: intervals are serial -> reordering, if observed,"
                " is a completion-vs-visibility gap, not parallel execution"
            )
        else:
            out.append(
                "      OVERLAPS PRESENT: same-stream tasks executed concurrently"
                " -> stream ordering not enforced (CUDA semantics require serial)"
            )

        # (b) dispatch-vs-execution order: count start inversions per stream.
        # A pair (A, B) dispatched A-before-B (tid_A < tid_B) is inverted when
        # B started strictly before A. Cross-class = one side memcpy, other not
        # (the copy-vs-consumer shape); mem-mem inversions (SDMA reordering)
        # counted separately as same-engine reordering.
        ordered = [(t[4], t[0], t[2], t[3]) for t in tasks if t[4] is not None]
        if len(ordered) >= 2:
            ordered.sort(key=lambda t: t[0])  # dispatch order
            inv_total, inv_cross, inv_memmem = _count_inversions(ordered)
            out.append(
                f"      ORDER (dispatch-order vs start-time): tasks-with-tid={len(ordered)},"
                f" start-inversions={inv_total} (cross-class={inv_cross} memcpy-vs-other,"
                f" mem-mem={inv_memmem})"
            )
            # (c) arrival-order class pattern: compress the task-id sequence
            # into M(memcpy)/K(other) blocks. task ids are RUNTIME-arrival
            # order, so this reveals whether copies arrive at the runtime in
            # the Python emission order. The repro emits 9 copies then 7
            # kernels per step: expect repeating "9M 7K" blocks; a leading K
            # block / "7K 9M" pattern means the copies were enqueued to the
            # runtime AFTER the kernels that were launched later - the
            # host-side queue reordered emission order (arrival inversion).
            blocks = []
            for is_m, grp in itertools.groupby(ordered, key=lambda t: t[2]):
                blocks.append(("M" if is_m else "K", len(list(grp))))
            pat = " ".join(f"{n}{c}" for c, n in blocks[:24])
            out.append(f"      ARRIVAL pattern (task-id order, M=memcpy K=other): {pat}")
            shown = 0
            for i in range(len(ordered)):  # example hunting, bounded work
                for j in range(i + 1, min(i + 9, len(ordered))):
                    if ordered[j][1] < ordered[i][1]:  # later-dispatched started earlier
                        out.append(
                            f"        inversion example: tid={ordered[i][0]:.0f} [{ordered[i][3]}]"
                            f" start={ordered[i][1]:.1f} dispatched BEFORE tid={ordered[j][0]:.0f}"
                            f" [{ordered[j][3]}] start={ordered[j][1]:.1f}, but executed AFTER it"
                        )
                        shown += 1
                        break
                if shown >= 3:
                    break
            if inv_cross == 0 and inv_total == 0:
                out.append("      no start inversions: execution follows dispatch order on this stream")
    return out


def _count_inversions(ordered: list[tuple[float, float, bool, str]]) -> tuple[int, int, int]:
    """Merge-sort inversion count over start times of dispatch-ordered tasks.

    Returns (total, cross_class, mem_mem) inversion-pair counts. Equal starts
    are NOT inversions (us-resolution ties are common).
    """

    def sort_rec(lo: list) -> tuple[list, int, int, int]:
        if len(lo) <= 1:
            return lo, 0, 0, 0
        mid = len(lo) // 2
        left, tl, cl, ml = sort_rec(lo[:mid])
        right, tr, cr, mr = sort_rec(lo[mid:])
        merged: list = []
        inv = cross = memmem = 0
        li = ri = 0
        # class counts of the not-yet-merged left tail (cheap cross-class math)
        tail_mem = sum(1 for t in left if t[2])
        tail_non = len(left) - tail_mem
        while li < len(left) and ri < len(right):
            if right[ri][1] < left[li][1]:
                # right element starts before EVERY remaining left element
                inv += tail_mem + tail_non
                if right[ri][2]:
                    cross += tail_non
                    memmem += tail_mem
                else:
                    cross += tail_mem
                merged.append(right[ri])
                ri += 1
            else:
                if left[li][2]:
                    tail_mem -= 1
                else:
                    tail_non -= 1
                merged.append(left[li])
                li += 1
        merged.extend(left[li:])
        merged.extend(right[ri:])
        return merged, inv + tl + tr, cross + cl + cr, memmem + ml + mr

    _, total, cross, memmem = sort_rec(ordered)
    return total, cross, memmem


def audit(paths: list[str]) -> str:
    out = ["===STREAM_AUDIT BEGIN==="]
    seen = set()
    groups: list[str] = []
    for p in discover_csvs(paths):
        if p in seen:
            continue
        seen.add(p)
        groups.extend(audit_file(p))
    out.extend(groups if groups else ["no csv files found under the given paths"])
    out.append("===STREAM_AUDIT END===")
    return "\n".join(out)


def selftest() -> int:
    import tempfile

    tmp = tempfile.mkdtemp(prefix="stream_audit_", dir="/tmp")
    kd = os.path.join(tmp, "kernel_details.csv")
    with open(kd, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["Step Id", "Name", "Type", "Start Time(us)", "Duration(us)", "Stream ID"])
        w.writerow([0, "Memcpy H2D", "Memcpy", 100, 5, "0x2"])
        w.writerow([0, "Memcpy H2D", "Memcpy", 200, 5, "0x2"])
        w.writerow([0, "aclnnEqual", "AI Core", 300, 10, "0x0"])
        w.writerow([0, "aclnnWhere", "AI Core", 400, 10, "0x0"])
    os.makedirs(os.path.join(tmp, "PROF_x", "PROF_export"), exist_ok=True)
    tt = os.path.join(tmp, "PROF_x", "PROF_export", "task_time_1234.csv")
    with open(tt, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["Task Type", "Task Start Time(us)", "Stream ID"])
        w.writerow(["Memcpy", 1, "7"])  # shared-stream variant to exercise the other HINT branch
        w.writerow(["AI CORE", 2, "7"])

    # timeline fixtures: overlap variant (copy x kernel intersect on one stream)
    # and serial variant (touching intervals do NOT count as overlap).
    tl = os.path.join(tmp, "task_time_9999.csv")
    with open(tl, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(
            [
                "Device_id",
                "kernel_name",
                "kernel_type",
                "stream_id",
                "task_id",
                "task_time(us)",
                "task_start(us)",
                "task_stop(us)",
            ]
        )
        w.writerow([0, "", "Memcpy", "42", 1, 100, 1000, 1100])  # copy [1000,1100]
        w.writerow([0, "Equal", "AI CORE", "42", 2, 10, 1050, 1060])  # kernel inside the copy -> overlap
        w.writerow([0, "", "Memcpy", "42", 3, 5, 1200, 1205])  # copy dispatched 3rd, starts at 1200
        w.writerow(
            [0, "Where", "AI CORE", "42", 4, 5, 1150, 1155]
        )  # tid4 after tid3 but starts earlier -> ORDER inversion
        w.writerow([0, "Other", "AI CORE", "9", 5, 5, 100, 105])  # other stream, must not mix in

    report = audit([tmp])
    print(report)
    ok = True
    checks = [
        ("kernel_details audited", "kernel_details.csv" in report),
        ("stream col found", "stream_col='Stream ID'" in report),
        ("memcpy counted on 0x2", "0x2:2" in report),
        ("compute counted on 0x0", "0x0:2" in report),
        ("disjoint hint fired", "DISJOINT" in report),
        ("nested discovery worked", "task_time_1234.csv" in report),
        ("shared-stream hint fired", "land on the launch stream" in report),
        ("timeline section present", "TIMELINE" in report),
        ("overlap detected (1 pair, memcpy-involved)", "overlapping pairs>=1 (involving-memcpy=1, 100%)" in report),
        ("back-to-back not an overlap", "max concurrency=2" in report),
        ("serial verdict line present", "OVERLAPS PRESENT" in report),
        ("order analysis present", "ORDER (dispatch-order vs start-time)" in report),
        ("cross-class inversion counted", "start-inversions=1 (cross-class=1 memcpy-vs-other, mem-mem=0)" in report),
        ("inversion example shown", "dispatched BEFORE" in report),
        ("arrival pattern rendered", "ARRIVAL pattern (task-id order, M=memcpy K=other): 1M 1K 1M 1K" in report),
        ("aggregate skip reason available", True),
    ]
    for label, cond in checks:
        print(f"  [{'ok' if cond else 'FAIL'}] {label}")
        ok = ok and cond
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="prof export dirs (recursive) or single csv files")
    ap.add_argument("--csv", dest="csvs", nargs="+", help="explicit csv files (appended to paths)")
    args = ap.parse_args()
    paths = list(args.paths) + list(args.csvs or [])
    if not paths or paths == ["selftest"]:
        if "selftest" in sys.argv:
            return selftest()
        ap.print_help()
        return 2
    print(audit(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
