#!/usr/bin/env python3
"""Phase 1.5 tax probe: aggregate torch profiler chrome traces into a compact
per-step category TSV (paste-ready for the no-export server protocol).

Runs ON THE SERVER next to the trace files; only its stdout leaves the box
(traces stay where they were written - see AGENTS.md / SUMMARY protocol).

Device-side time is the tax currency. Categories follow the phase1.5 design
(experiments/phase1.5-tax-probe-design.md): attention / gemm / sampling /
bookkeeping / elementwise+norm+rope / memcpy H2D-D2D-D2H / memset / other.
Classification is ordered name-pattern matching: deliberate clashes resolve by
precedence (matmul->gemm before mul->elementwise, argmax->sampling before
max). Unmatched kernels land in other_kernel and are listed in the
top-unclassified block so gaps are visible, never silent. The classifier only
has to be CONSISTENT across runs: the probe reads the ngram-dense DIFFERENTIAL
per category, so absolute mislabels cancel.

Usage:
  profile_step_breakdown.py [--steps N] [--top K] PATH [PATH ...]
  profile_step_breakdown.py [--steps N] --diff label1=PATH label2=PATH [...]
  profile_step_breakdown.py selftest

PATH is a chrome trace (.json / .json.gz) or a directory (auto-discovers
traces, skips files without device events such as the AsyncLLM frontend CPU
trace). --steps divides totals into per-step numbers (runbook passes
PROFILER_STEPS x rounds); without it only totals + shares are meaningful.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
import tempfile

DEVICE_CATS = ("kernel",)
FALLBACK_DEVICE_CATS = ("acl_op",)  # host aclnn API calls, 1:1 with kernels
MEMCPY_CAT = "gpu_memcpy"
MEMSET_CAT = "gpu_memset"
DIR_PATTERNS = (
    ("memcpy_h2d", re.compile(r"htod|h2d|host.*device|pinned.*device", re.I)),
    ("memcpy_d2h", re.compile(r"dtoh|d2h|device.*host|device.*pinned", re.I)),
    ("memcpy_d2d", re.compile(r"dtod|d2d|p2p", re.I)),
)

# Ordered rules, first match wins. Substring risks resolved by precedence:
# matmul(gemm) > mul(elementwise), padding(bookkeeping) > add(elementwise),
# argmax(sampling) > max(elementwise), concat/cat(bookkeeping) > at(elementwise).
RULES = [
    (re.compile(r"flashattention|\bfia|promptattention|pagedattention|attention", re.I), "attention"),
    (re.compile(r"allreduce|all_gather|allgather|reducescatter|all_to_all|broadcast|sendrecv|hccl|hcom", re.I), "comm"),
    (re.compile(r"matmul|gemm|linear|weightquant|conv|mmla", re.I), "gemm"),
    (re.compile(r"softmax|topk|top_k|argmax|argmin|multinomial|gumbel|sampl", re.I), "sampling"),
    (re.compile(r"norm|rope|rotary", re.I), "norm_rope"),
    (
        re.compile(
            r"gather|scatter|index|copyandexpand|slice|concat|\bcat\b|pad|repeat|arange|\brange\b|fill|where|select|split|stack|unsqueeze|squeeze|reshape|flatten|permute|transpose|cast|\bcopy|clone|expand|embedding|onehot|tril|triu|cumsum|masked|shuffle|sort|unique|nonzero|transdata",
            re.I,
        ),
        "bookkeeping",
    ),
    (
        re.compile(
            r"silu|gelu|relu|sigmoid|tanh|softplus|\badd\b|\bsub\b|\bmul\b|\bdiv\b|sqrt|\bpow\b|\babs\b|\bexp\b|\blog\b|clamp|round|floor|ceil|equal|less|greater|\bneg\b|\bmax\b|\bmin\b|\bsum\b|\bmean\b|\band\b|\bor\b|\bnot\b|bitwise|reciprocal",
            re.I,
        ),
        "elementwise",
    ),
]

CAT_ORDER = [
    "attention",
    "comm",
    "gemm",
    "sampling",
    "norm_rope",
    "bookkeeping",
    "elementwise",
    "memcpy_h2d",
    "memcpy_d2d",
    "memcpy_d2h",
    "memcpy_other",
    "memset",
    "other_kernel",
]

OPAQUE_RE = re.compile(r"replay|graph", re.I)


def classify(name: str) -> str:
    for rx, cat in RULES:
        if rx.search(name):
            return cat
    return "other_kernel"


def memcpy_dir(name: str) -> str:
    for cat, rx in DIR_PATTERNS:
        if rx.search(name):
            return cat
    return "memcpy_other"


def load_events(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:  # type: ignore[operator]
        doc = json.load(f)
    if isinstance(doc, dict):
        doc = doc.get("traceEvents", [])
    return [e for e in doc if isinstance(e, dict)]


def discover(paths: list[str]) -> list[str]:
    """Expand dirs into their trace files (newest first per dir)."""
    out: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files = []
            for pat in ("*.json", "*.json.gz"):
                files.extend(glob.glob(os.path.join(p, "**", pat), recursive=True))
            out.extend(sorted(files, key=os.path.getmtime, reverse=True))
        else:
            out.append(p)
    # dirs may share files via recursion duplicates
    seen: set[str] = set()
    return [f for f in out if not (f in seen or seen.add(f))]


def aggregate(paths: list[str], steps: int | None) -> dict:
    agg = {
        "files": [],
        "skipped": [],
        "notes": [],
        "steps": steps,
        "events": 0,
        "span_us": 0.0,
        "cats": {c: {"count": 0, "us": 0.0, "bytes": 0, "bytes_known": False} for c in CAT_ORDER},
        "top_unclassified": {},
        "cat_hist": {},
        "opaque": False,
        "opaque_top": None,
    }
    ts_lo, ts_hi = float("inf"), 0.0
    name_us: dict[str, list[float]] = {}

    for path in paths:
        try:
            events = load_events(path)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            agg["skipped"].append((path, f"unreadable: {e!r}"))
            continue
        kset = [e for e in events if e.get("cat") in DEVICE_CATS]
        if not kset:
            kset = [e for e in events if e.get("cat") in FALLBACK_DEVICE_CATS]
            if kset:
                agg["notes"].append(f"{os.path.basename(path)}: no kernel-cat events, using acl_op fallback")
        dev = kset + [e for e in events if e.get("cat") in (MEMCPY_CAT, MEMSET_CAT)]
        if not dev:
            agg["skipped"].append((path, "no device events"))
            continue
        agg["files"].append(path)
        for e in events:
            agg["cat_hist"][e.get("cat", "?")] = agg["cat_hist"].get(e.get("cat", "?"), 0) + 1
        for e in dev:
            name = str(e.get("name", "?"))
            dur = float(e.get("dur", 0) or 0)
            cat = e.get("cat")
            if cat == MEMCPY_CAT:
                c = memcpy_dir(name)
                b = (e.get("args") or {}).get("bytes")
                if isinstance(b, (int, float)):
                    agg["cats"][c]["bytes"] += int(b)
                    agg["cats"][c]["bytes_known"] = True
            elif cat == MEMSET_CAT:
                c = "memset"
            else:
                c = classify(name)
                if c == "other_kernel":
                    name_us.setdefault(name, [0.0, 0])
                    name_us[name][0] += dur
                    name_us[name][1] += 1
            agg["cats"][c]["count"] += 1
            agg["cats"][c]["us"] += dur
            agg["events"] += 1
            ts = e.get("ts")
            if isinstance(ts, (int, float)):
                ts_lo = min(ts_lo, float(ts))
                ts_hi = max(ts_hi, float(ts) + dur)

    if ts_lo != float("inf"):
        agg["span_us"] = ts_hi - ts_lo
    top = sorted(name_us.items(), key=lambda kv: kv[1][0], reverse=True)[:10]
    agg["top_unclassified"] = top
    total_us = sum(v["us"] for v in agg["cats"].values())
    if total_us > 0 and top:
        share = top[0][1][0] / total_us
        if share > 0.6 and (len(name_us) < 12 or OPAQUE_RE.search(top[0][0])):
            agg["opaque"] = True
            agg["opaque_top"] = (top[0][0], share)
    return agg


def print_table(label: str, agg: dict, top: int = 5) -> None:
    steps = agg["steps"] or 0
    print(
        f"# {label}: files={len(agg['files'])} device_events={agg['events']}"
        f" steps={steps or '-'} span_ms={agg['span_us'] / 1000:.1f}"
        + (f" wall_ms_per_step={agg['span_us'] / 1000 / steps:.1f}" if steps else "")
    )
    hist = " ".join(f"{k}={v}" for k, v in sorted(agg["cat_hist"].items(), key=lambda kv: -kv[1])[:6])
    print(f"# cats: {hist}")
    for note in agg["notes"]:
        print(f"# note: {note}")
    if not agg["files"]:
        for p, why in agg["skipped"]:
            print(f"# skipped: {os.path.basename(p)} ({why})")
        print("# NO DEVICE TRACES - see skipped reasons above")
        return
    print("category\tcnt_per_step\tus_per_step\tshare_pct\tbytes_per_step")
    total_us = sum(v["us"] for v in agg["cats"].values())
    for cat in CAT_ORDER:
        v = agg["cats"][cat]
        if v["count"] == 0:
            continue
        share = 100.0 * v["us"] / total_us if total_us else 0.0
        cnt = v["count"] / steps if steps else v["count"]
        us = v["us"] / steps if steps else v["us"]
        bps = (
            f"{v['bytes'] / steps:.1f}"
            if v["bytes_known"] and steps
            else (str(v["bytes"]) if v["bytes_known"] else "-")
        )
        print(f"{cat}\t{cnt:.2f}\t{us:.1f}\t{share:.1f}\t{bps}")
    for p, why in agg["skipped"]:
        print(f"# skipped: {os.path.basename(p)} ({why})")
    if agg["top_unclassified"]:
        print("# top-unclassified (other_kernel):")
        for name, (us, cnt) in agg["top_unclassified"][:top]:
            per = us / steps if steps else us
            print(f"#   {name[:60]}\tcnt={cnt}\tus_per_step={per:.1f}")
    if agg["opaque"]:
        name, share = agg["opaque_top"]  # type: ignore[misc]
        print(
            f"# OPAQUE-GRAPH WARNING: top event '{name}' = {share:.0%} of device time,"
            " graph replay likely hides per-op kernels;"
            " re-run with EXTRA_SERVE_ARGS=--enforce-eager for op visibility"
        )


def print_diff(labeled: list[tuple[str, dict]]) -> None:
    base_label, base = labeled[0]
    cats = [c for c in CAT_ORDER if any(a["cats"][c]["us"] > 0 for _, a in labeled)]
    hdr = "category\t" + "\t".join(f"{lbl}_us_ps" for lbl, _ in labeled)
    print(hdr)
    steps = base["steps"] or 0
    for cat in cats:
        row = [cat]
        for _lbl, a in labeled:
            us = a["cats"][cat]["us"] / steps if steps else a["cats"][cat]["us"]
            row.append(f"{us:.1f}")
        print("\t".join(row))
    if steps:
        tot = [sum(a["cats"][c]["us"] for c in CAT_ORDER) / steps for _, a in labeled]
        print("TOTAL\t" + "\t".join(f"{t:.1f}" for t in tot))
        print(f"# delta vs {base_label} (us/step):")
        for cat in cats + ["TOTAL"]:
            vals = []
            for lbl, a in labeled:
                if cat == "TOTAL":
                    v = sum(a["cats"][c]["us"] for c in CAT_ORDER) / steps
                else:
                    v = a["cats"][cat]["us"] / steps
                vals.append(v)
            deltas = "  ".join(f"{lbl}:{vals[i] - vals[0]:+.1f}" for i, (lbl, _a) in enumerate(labeled[1:], 1))
            print(f"# {cat}\t{deltas}")


def build_fixture(path: str, steps: int, extra: dict[str, float] | None = None) -> None:
    """Deterministic trace: per step, planted ops with known category budgets."""
    extra = extra or {}
    per_step = [
        ("kernel", "FlashAttentionScoreDev", 5000.0 + extra.get("attention", 0.0), 1),
        ("kernel", "MatMul", 2000.0, 4),
        ("kernel", "Sub&Relu", 150.0, 6),
        ("kernel", "SoftmaxV2", 300.0, 2),
        ("kernel", "GatherV2", 100.0, 3),
        ("kernel", "ZetaUnkOp", 10.0, 1),
        (MEMCPY_CAT, "Memcpy HtoD (Pinned -> Device)", 50.0, 2),
        (MEMSET_CAT, "Memset (Device)", 20.0, 1),
        ("cpu_op", "aten::linear", 1000.0, 100),  # host noise: must be ignored
    ]
    events = []
    ts = 1_000_000.0
    for _s in range(steps):
        for cat, name, dur, n in per_step:
            for _i in range(n):
                args = {"bytes": 512} if cat == MEMCPY_CAT else {}
                events.append({"name": name, "cat": cat, "ts": ts, "dur": dur, "args": args})
                ts += dur
    with open(path, "w") as f:
        json.dump({"traceEvents": events}, f)


def cmd_selftest() -> int:
    with tempfile.TemporaryDirectory() as td:
        f1 = os.path.join(td, "round1.json")
        f2 = os.path.join(td, "round2.json")
        build_fixture(f1, steps=40)
        build_fixture(f2, steps=40, extra={"attention": 1000.0})
        with open(f2, "rb") as fin, gzip.open(f2 + ".gz", "wb") as fout:
            fout.write(fin.read())
        os.remove(f2)

        agg = aggregate(discover([td]), steps=80)
        expect = {
            "attention": (1, 5500.0),  # (cnt/step, us/step): 5000 + 1000 extra on half the steps
            "gemm": (4, 8000.0),
            "elementwise": (6, 900.0),
            "sampling": (2, 600.0),
            "bookkeeping": (3, 300.0),
            "other_kernel": (1, 10.0),
            "memcpy_h2d": (2, 100.0),
            "memset": (1, 20.0),
        }
        ok = True
        for cat, (cnt_ps, us_ps) in expect.items():
            got_us = agg["cats"][cat]["us"] / 80
            got_cnt = agg["cats"][cat]["count"] / 80
            if abs(got_us - us_ps) > 1e-6 or abs(got_cnt - cnt_ps) > 1e-9:
                print(f"FAIL {cat}: us/step {got_us} != {us_ps} or cnt/step {got_cnt} != {cnt_ps}")
                ok = False
        if agg["cats"]["memcpy_h2d"]["bytes"] != 2 * 512 * 80:
            print(f"FAIL memcpy bytes: {agg['cats']['memcpy_h2d']['bytes']}")
            ok = False
        if agg["cats"]["comm"]["us"] or agg["cats"]["memcpy_d2d"]["us"]:
            print("FAIL: phantom comm/d2d events")
            ok = False
        if len(agg["files"]) != 2 or agg["skipped"]:
            print(f"FAIL discovery: files={agg['files']} skipped={agg['skipped']}")
            ok = False

        # opacity detector: a graph-replay-only trace must trip the warning
        f3 = os.path.join(td, "graphy.json")
        with open(f3, "w") as f:
            json.dump(
                {
                    "traceEvents": [
                        {"name": "ACLGraphReplayId", "cat": "kernel", "ts": 1e6 + i * 1000, "dur": 900.0}
                        for i in range(40)
                    ]
                },
                f,
            )
        ag3 = aggregate([f3], steps=40)
        if not ag3["opaque"]:
            print("FAIL: graph-replay trace did not trip OPAQUE warning")
            ok = False

        # differential: ngram(-like) vs dense(-like) delta must be exact
        d_dir = os.path.join(td, "dense")
        n_dir = os.path.join(td, "ngram")
        os.mkdir(d_dir)
        os.mkdir(n_dir)
        build_fixture(os.path.join(d_dir, "a.json"), steps=40)
        build_fixture(os.path.join(n_dir, "b.json"), steps=40, extra={"attention": 1000.0})
        dense = aggregate(discover([d_dir]), steps=40)
        ngram = aggregate(discover([n_dir]), steps=40)
        delta = ngram["cats"]["attention"]["us"] / 40 - dense["cats"]["attention"]["us"] / 40
        if abs(delta - 1000.0) > 1e-6:
            print(f"FAIL diff: attention delta {delta} != 1000.0")
            ok = False

        print("SELFTEST " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="trace files or directories")
    ap.add_argument("--steps", type=int, default=0, help="engine steps covered (per-step divisor)")
    ap.add_argument("--top", type=int, default=5, help="top unclassified names to list")
    ap.add_argument("--diff", action="store_true", help="paths are label=PATH; adds delta table vs first")
    args = ap.parse_args()

    if args.paths == ["selftest"]:
        return cmd_selftest()
    if not args.paths:
        ap.error("need at least one path (or 'selftest')")

    if args.diff:
        labeled = []
        for spec in args.paths:
            label, _, path = spec.partition("=")
            path = path or label
            agg = aggregate(discover([path]), steps=args.steps or None)
            print_table(label, agg, top=args.top)
            print()
            labeled.append((label, agg))
        if len(labeled) >= 2:
            print_diff(labeled)
        return 0

    agg = aggregate(discover(args.paths), steps=args.steps or None)
    print_table("traces", agg, top=args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
