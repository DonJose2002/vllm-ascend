#!/usr/bin/env python3
"""Dual-platform (NPU/GPU) baseline benchmark harness for Phase 0.

Standalone test script; does not modify any vllm/vllm-ascend code.

Measures, against an OpenAI-compatible server (streaming):
  - TTFT percentiles, ITL percentiles (from token arrival timestamps)
  - output throughput per request, aggregate request throughput
  - optional speculative-decoding accept length via the burst-gap method
    (port of bench_sd.py; no /metrics dependency, but counters are snapshotted
    too when the server exposes them)

Matrix: prompt length tiers x concurrency levels. Long prompts are synthesized
by repeating seed paragraphs; actual prompt token counts are read back from
usage and recorded (exact length is best-effort, never silently assumed).

Usage:
  # server: vllm serve <model> --port 8001 ...
  python3 bench_baseline.py run --base-url http://127.0.0.1:8001 \
      --model qwen3-8b-awq --tag gpu-awq-dense --out results_gpu.json \
      [--tiers 4096,16384] [--concs 1,4,16] [--num-prompts 8] \
      [--seed-profile generic|repetitive] [--save-ts]
  python3 bench_baseline.py table results_gpu.json [results_npu.json ...]
  python3 bench_baseline.py summary results.json ...
  # Phase 1 analysis:
  python3 bench_baseline.py kregress k1.json k3.json k5.json k8.json   # E1
  python3 bench_baseline.py diff dense.json planA.json ngram.json ...  # E2 (ngram+planA pair -> derived components)
  python3 bench_baseline.py rt c16run.json                              # E4 (needs --save-ts)

Only stdlib + urllib (no requests/torch), so it runs on server and laptop alike.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import statistics
import sys
import threading
import time
import urllib.request

# ---------------------------------------------------------------------------
# Prompt synthesis
# ---------------------------------------------------------------------------

SEED_PARAGRAPHS = [
    "The history of computing machinery spans mechanical calculators, "
    "electromagnetic relays, vacuum tubes, discrete transistors, and finally "
    "dense integrated circuits carrying billions of switching elements.",
    "A compiler translates source code through lexical analysis, parsing, "
    "semantic checking, intermediate representation, optimization passes, "
    "and target-specific code generation before linking and loading.",
    "Attention mechanisms compute query-key compatibility scores, normalize "
    "them into a distribution, and mix value vectors accordingly, letting "
    "each position gather information from relevant context positions.",
    "Memory hierarchies trade capacity for latency: registers, several cache "
    "levels, main memory, and storage devices each serve a different point "
    "on the cost-performance curve, guided by locality of reference.",
    "Distributed training must contend with gradient staleness, communication "
    "bandwidth, stragglers, and fault tolerance, giving rise to parameter "
    "servers, all-reduce collectives, and sharded data parallelism.",
]


def synthesize_prompt(target_chars: int, question: str) -> str:
    """Repeat seed paragraphs up to target_chars, end with the actual question."""
    body = []
    n = 0
    i = 0
    while n < target_chars:
        p = SEED_PARAGRAPHS[i % len(SEED_PARAGRAPHS)]
        body.append(p)
        n += len(p) + 1
        i += 1
    return " ".join(body) + "\n\n" + question


QUESTIONS = [
    "Summarize the passage above in three sentences.",
    "What is the main idea of the passage above?",
    "List two cause-and-effect relationships mentioned above.",
    "Give a suitable title for the passage above and justify it.",
    "Which claim above would you challenge, and why?",
    "Rewrite the key point above for a general audience.",
    "What terminology above would need definition for a newcomer?",
    "How do the concepts above relate to each other?",
]

# Repetitive profile (Phase 1 E2): questions demand verbatim reproduction of
# the prompt, so prompt-lookup drafters (ngram) find their drafts inside the
# prompt itself. Only the questions swap; seed paragraphs (hence ptok sizing
# and CHARS_PER_TOKEN) stay identical to generic.
REPETITIVE_QUESTIONS = [
    "Repeat the passage above verbatim, word for word.",
    "Quote the passage above exactly as written.",
    "Copy the passage above without changing anything.",
    "Reproduce the passage above exactly, from beginning to end.",
    "Write out the passage above again, character for character.",
    "Echo the passage above back to me precisely.",
]

SEED_PROFILES = {
    "generic": {"questions": QUESTIONS},
    "repetitive": {"questions": REPETITIVE_QUESTIONS},
}

# Rough chars-per-token for English technical text, calibrated for the Qwen3
# BPE on SEED_PARAGRAPHS (measured 3.7 -> actual 63% of target tokens, i.e.
# ~5.87 real chars/token; 5.7 undershoots ~3% so tiers never exceed
# max_model_len). Only used to size the synthesis; actual token counts are
# read back from the server and recorded in ptok.
CHARS_PER_TOKEN = 5.7

# ---------------------------------------------------------------------------
# Streaming request + timestamp capture
# ---------------------------------------------------------------------------

SSE_DATA_RE = re.compile(r"^data:\s*(\{.*\})$", re.MULTILINE)


class ReqResult:
    __slots__ = ("ok", "prompt_tokens", "completion_tokens", "ttft", "token_ts", "err", "saw_done")

    def __init__(self):
        self.ok = False
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.ttft = None  # seconds to first content token
        self.token_ts: list[float] = []
        self.err = None
        self.saw_done = False


def stream_one(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> ReqResult:
    res = ReqResult()
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t_send = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buf = b""
            for chunk in resp:
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    sline = line.decode("utf-8", "replace").strip()
                    if not sline.startswith("data:"):
                        continue
                    data = sline[5:].strip()
                    if data == "[DONE]":
                        res.saw_done = True
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if "usage" in obj and obj.get("usage"):
                        u = obj["usage"]
                        res.prompt_tokens = u.get("prompt_tokens", 0)
                        res.completion_tokens = u.get("completion_tokens", 0)
                    for ch in obj.get("choices", []):
                        delta = ch.get("delta", {}) or {}
                        content = delta.get("content")
                        if content:
                            now = time.monotonic()
                            if res.ttft is None:
                                res.ttft = now - t_send
                            res.token_ts.append(now)
        # A stream is only successful if the server terminated it properly
        # ([DONE] seen). Otherwise the engine may have died mid-stream and
        # we would count a truncated generation as ok (seen for real when
        # the SD engine crashed mid-cell and 5 truncated streams were
        # counted ok=8).
        res.ok = res.saw_done and (bool(res.token_ts) or res.completion_tokens > 0)
        if not res.ok and res.err is None and (res.token_ts or res.completion_tokens):
            res.err = f"stream truncated (saw_done={res.saw_done}, got {max(res.completion_tokens, len(res.token_ts))} tokens)"
    except Exception as e:  # noqa: BLE001
        res.err = repr(e)
    return res


# ---------------------------------------------------------------------------
# SD accept-length estimation (burst-gap; ported from bench_sd.py)
# ---------------------------------------------------------------------------


def estimate_steps(token_ts: list[float], min_gap_s: float = 0.002) -> int:
    if len(token_ts) < 2:
        return max(1, len(token_ts))
    gaps = [b - a for a, b in zip(token_ts, token_ts[1:])]
    median_gap = statistics.median(gaps)
    threshold = max(min_gap_s, 0.3 * median_gap)
    return 1 + sum(g > threshold for g in gaps)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    sv = sorted(values)
    idx = min(len(sv) - 1, max(0, round(q * (len(sv) - 1))))
    return sv[idx] * 1000.0  # report in ms


def http_get(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode()


def snapshot_spec_metrics(base_url: str) -> dict:
    try:
        text = http_get(base_url.rstrip("/") + "/metrics")
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        name, value = parts
        try:
            v = float(value)
        except ValueError:
            continue
        if name.startswith("vllm:spec_decode_num_drafts_total"):
            out["drafts"] = v
        elif name.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted"] = v
    return out


# ---------------------------------------------------------------------------
# Run one (tier, conc) cell
# ---------------------------------------------------------------------------


def run_cell(base_url, model, tier, conc, num_prompts, max_tokens, timeout, tag, results, profile="generic", save_ts=False):
    questions = SEED_PROFILES[profile]["questions"]
    prompts = [
        synthesize_prompt(int(tier * CHARS_PER_TOKEN), questions[i % len(questions)])
        for i in range(num_prompts)
    ]
    # Warmup single short request (compile/cudagraph warm paths), not measured.
    stream_one(base_url, model, "Hello.", 8, timeout=timeout)

    m0 = snapshot_spec_metrics(base_url)
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(stream_one, base_url, model, p, max_tokens, timeout) for p in prompts]
        outs = [f.result() for f in futs]
    wall = time.monotonic() - t0
    m1 = snapshot_spec_metrics(base_url)

    ok = [o for o in outs if o.ok]
    failed = [o for o in outs if not o.ok]
    ttfts = [o.ttft for o in ok if o.ttft is not None]
    itls = []
    for o in ok:
        if len(o.token_ts) >= 2:
            itls.extend(b - a for a, b in zip(o.token_ts, o.token_ts[1:]))
    out_toks = sum(max(o.completion_tokens, len(o.token_ts)) for o in ok)
    prompt_toks = [o.prompt_tokens for o in ok if o.prompt_tokens]

    cell = {
        "tag": tag,
        "tier": tier,
        "conc": conc,
        "num_prompts": num_prompts,
        "max_tokens": max_tokens,
        "seed_profile": profile,
        "ok": len(ok),
        "failed": len(failed),
        "errors": [o.err for o in failed][:3],
        "prompt_tokens_mean": round(statistics.mean(prompt_toks), 1) if prompt_toks else None,
        "ttft_ms_p50": round(pct(ttfts, 0.50), 1) if ttfts else None,
        "ttft_ms_p90": round(pct(ttfts, 0.90), 1) if ttfts else None,
        "itl_ms_p50": round(pct(itls, 0.50), 1) if itls else None,
        "itl_ms_p90": round(pct(itls, 0.90), 1) if itls else None,
        "itl_ms_p99": round(pct(itls, 0.99), 1) if itls else None,
        "out_tok_per_s_per_req_mean": round(
            statistics.mean(
                (max(o.completion_tokens, len(o.token_ts)) / (o.token_ts[-1] - o.token_ts[0]))
                for o in ok
                if len(o.token_ts) >= 2 and o.token_ts[-1] > o.token_ts[0]
            ),
            2,
        )
        if any(len(o.token_ts) >= 2 for o in ok)
        else None,
        "aggregate_out_tok_per_s": round(out_toks / wall, 2),
        "request_per_s": round(len(ok) / wall, 3),
        "wall_s": round(wall, 2),
    }

    # SD accept length: prefer server counters, fall back to burst-gap.
    steps_total = sum(estimate_steps(o.token_ts) for o in ok if len(o.token_ts) >= 1)
    toks_total = sum(max(o.completion_tokens, len(o.token_ts)) for o in ok)
    if m1.get("drafts") and m0.get("drafts") and m1["drafts"] > m0["drafts"]:
        cell["accept_len_counters"] = round(
            1 + (m1["accepted"] - m0["accepted"]) / (m1["drafts"] - m0["drafts"]), 4
        )
    if steps_total > 0:
        cell["accept_len_burst"] = round(toks_total / steps_total, 4)

    if save_ts:
        # Per-request token timestamps (s, relative to cell t0) for offline
        # R(t) occupancy / ITL-vs-R analysis (`rt` subcommand).
        cell["req_token_ts"] = [
            [round(ts - t0, 3) for ts in o.token_ts] for o in ok if len(o.token_ts) >= 1
        ]

    print(
        f"[{tag}] tier={tier:>6} conc={conc:>2}: ok={cell['ok']}/{num_prompts} "
        f"TTFT p50={cell['ttft_ms_p50']}ms p90={cell['ttft_ms_p90']}ms "
        f"ITL p50={cell['itl_ms_p50']}ms out/s={cell['aggregate_out_tok_per_s']}"
        + (
            f" accept(counters)={cell.get('accept_len_counters')}"
            if "accept_len_counters" in cell
            else ""
        )
        + (
            f" accept(burst)={cell.get('accept_len_burst')}"
            if "accept_len_burst" in cell
            else ""
        )
    )
    results.append(cell)
    return cell


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_run(args):
    tiers = [int(x) for x in args.tiers.split(",")]
    concs = [int(x) for x in args.concs.split(",")]
    results: list[dict] = []
    lock = threading.Lock()  # serialize cells; results list append is GIL-safe

    for tier in tiers:
        for conc in concs:
            with lock:
                run_cell(
                    args.base_url,
                    args.model,
                    tier,
                    conc,
                    args.num_prompts,
                    args.max_tokens,
                    args.timeout,
                    args.tag,
                    results,
                    profile=args.seed_profile,
                    save_ts=args.save_ts,
                )

    doc = {
        "harness": "bench_baseline.py v2",
        "base_url": args.base_url,
        "model": args.model,
        "tag": args.tag,
        "seed_profile": args.seed_profile,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": args.note,
        "cells": results,
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out} ({len(results)} cells)")


def cmd_table(args):
    docs = []
    for p in args.paths:
        with open(p) as f:
            docs.append(json.load(f))
    cells = [c for d in docs for c in d["cells"]]
    tags = sorted({c["tag"] for c in cells})
    print(f"{'tag':<22}{'tier':>7}{'conc':>5}{'ok':>5}{'ptok':>8}"
          f"{'TTFT50':>9}{'TTFT90':>9}{'ITL50':>9}{'ITL99':>9}"
          f"{'out/s':>8}{'req/s':>7}{'accL':>7}")
    for tag in tags:
        for c in [c for c in cells if c["tag"] == tag]:
            acc = c.get("accept_len_counters") or c.get("accept_len_burst") or ""
            print(
                f"{tag:<22}{c['tier']:>7}{c['conc']:>5}{c['ok']:>5}"
                f"{c['prompt_tokens_mean'] or 0:>8.0f}"
                f"{c['ttft_ms_p50'] or 0:>9.1f}{c['ttft_ms_p90'] or 0:>9.1f}"
                f"{c['itl_ms_p50'] or 0:>9.1f}{c['itl_ms_p99'] or 0:>9.1f}"
                f"{c['aggregate_out_tok_per_s'] or 0:>8.1f}"
                f"{c['request_per_s'] or 0:>7.2f}{str(acc):>7}"
            )


def cmd_summary(args):
    """Compact paste-ready TSV of result JSONs (for no-export servers)."""
    for p in args.paths:
        with open(p) as f:
            d = json.load(f)
        print(f"# file={p} tag={d['tag']} created={d.get('created','')} profile={d.get('seed_profile','generic')}")
        if d.get("note"):
            print(f"# note={d['note']}")
        cols = (
            "tier conc ok fail ptok ttft50 ttft90 itl50 itl90 itl99 "
            "outs reqs accC accB err"
        ).split(" ")
        print("# " + "\t".join(cols))
        for c in d["cells"]:
            row = [
                str(c.get("tier", "")), str(c.get("conc", "")),
                str(c.get("ok", "")), str(c.get("failed", "")),
                str(c.get("prompt_tokens_mean") or ""),
                str(c.get("ttft_ms_p50") or ""), str(c.get("ttft_ms_p90") or ""),
                str(c.get("itl_ms_p50") or ""), str(c.get("itl_ms_p90") or ""),
                str(c.get("itl_ms_p99") or ""),
                str(c.get("aggregate_out_tok_per_s") or ""),
                str(c.get("request_per_s") or ""),
                str(c.get("accept_len_counters", "")),
                str(c.get("accept_len_burst", "")),
                (c.get("errors") or [""])[0][:60],
            ]
            print("\t".join(row))


# ---------------------------------------------------------------------------
# Phase 1 analysis: K regression / method differential / R(t) occupancy
# ---------------------------------------------------------------------------


def _load_cells(paths):
    """Yield (file, doc, cell) tuples from result JSONs."""
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        for c in d["cells"]:
            yield p, d, c


def _accept_of(cell):
    return cell.get("accept_len_counters") or cell.get("accept_len_burst") or None


def _linreg(xs, ys):
    """Least-squares fit; returns (slope, intercept, r2)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return float("nan"), my, float("nan")
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = (
        1 - sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / ss_tot
        if ss_tot
        else float("nan")
    )
    return slope, intercept, r2


TAG_K_RE = re.compile(r"[-_]k(\d+)")


def cmd_kregress(args):
    """E1: regress itl_ms_p50 ~ K across K-sweep JSONs; K parsed from tags."""
    # group: (tier, conc, profile) -> {K: cell}
    groups: dict[tuple, dict[int, dict]] = {}
    for _p, doc, c in _load_cells(args.paths):
        m = TAG_K_RE.search(c.get("tag", ""))
        if not m or c.get("itl_ms_p50") is None:
            continue
        k = int(m.group(1))
        key = (c["tier"], c["conc"], doc.get("seed_profile", "generic"))
        groups.setdefault(key, {})[k] = c

    print(f"{'tier':>7}{'conc':>5}{'profile':>11}  {'per-K itl50 (ms) / accept':<44}")
    for key in sorted(groups):
        ks = groups[key]
        row = "  ".join(f"K{k}={ks[k]['itl_ms_p50']:.0f}/{_accept_of(ks[k]) or '-'}" for k in sorted(ks))
        print(f"{key[0]:>7}{key[1]:>5}{key[2]:>11}  {row}")

    print()
    print(f"{'tier':>7}{'conc':>5}{'profile':>11}{'nK':>4}"
          f"{'slope ms/K':>12}{'intercept ms':>14}{'R^2':>7}{'netK':>6}{'@itl/acc':>10}")
    for key in sorted(groups):
        ks = groups[key]
        if len(ks) < 3:
            continue  # need >=3 points for a meaningful fit
        k_sorted = sorted(ks)
        xs = [float(k) for k in k_sorted]
        ys = [ks[k]["itl_ms_p50"] for k in k_sorted]
        slope, intercept, r2 = _linreg(xs, ys)
        # net-optimal K: minimize ms per emitted token (itl / accept)
        eff = {
            k: (ks[k]["itl_ms_p50"] / a if a else float("inf"))
            for k in k_sorted
            if (a := _accept_of(ks[k]))
        }
        best_k = min(eff, key=eff.get) if eff else None
        best_v = f"{eff[best_k]:.1f}" if best_k else "-"
        print(f"{key[0]:>7}{key[1]:>5}{key[2]:>11}{len(ks):>4}"
              f"{slope:>12.2f}{intercept:>14.1f}{r2:>7.2f}"
              f"{str(best_k or '-'):>6}{best_v:>10}")
    print("\nreading: slope = marginal cost per draft token (drafter fwd + KV rescan);")
    print("          intercept = per-step fixed cost (graph replay, metadata, sampling glue);")
    print("          netK = argmin itl50/accept (ms per emitted token).")


def cmd_diff(args):
    """E2: dense vs SD methods per cell; derive drafter-chain cost when both
    a planA-tagged and an ngram-tagged run are present.

    drafter chain (planA - ngram) = drafter fwd cost + draft KV tax
    verify+bookkeeping (ngram - dense) = target 6-token verify + SD metadata
    """
    docs = []
    for p in args.paths:
        with open(p) as f:
            docs.append(json.load(f))
    dense = docs[0]
    sds = docs[1:]
    dmap = {(c["tier"], c["conc"]): c for c in dense["cells"] if c.get("itl_ms_p50")}

    print(f"dense = {args.paths[0]} tag={dense['tag']}")
    for d in sds:
        print(f"sd    = tag={d['tag']}")
    print()
    hdr = f"{'tier':>7}{'conc':>5}"
    for d in sds:
        hdr += f"{'| ' + d['tag'][-24:]:>34}"
    print(hdr)
    keys = sorted({(c["tier"], c["conc"]) for d in sds for c in d["cells"] if c.get("itl_ms_p50")})
    for tier, conc in keys:
        base = dmap.get((tier, conc), {}).get("itl_ms_p50")
        row = f"{tier:>7}{conc:>5}"
        for d in sds:
            c = next((x for x in d["cells"] if x["tier"] == tier and x["conc"] == conc), None)
            if not c or c.get("itl_ms_p50") is None:
                row += f"{'-':>34}"
                continue
            itl = c["itl_ms_p50"]
            acc = _accept_of(c)
            spd = f"{itl / base:.2f}x" if base else "-"
            acc_s = f"{acc}" if acc else "-"
            row += f"{f'{itl:.0f}ms {spd} a={acc_s}':>34}"
        print(row)

    # derived components when planA + ngram pair exists (same profile)
    by_tag = {}
    for d in sds:
        by_tag[d["tag"]] = d
    plana = next((d for t, d in by_tag.items() if "planA" in t and "ngram" not in t), None)
    ngram = next((d for t, d in by_tag.items() if "ngram" in t), None)
    if plana and ngram:
        print("\nderived per cell (ms):")
        print(f"{'tier':>7}{'conc':>5}{'dense':>9}{'ngram':>9}{'planA':>9}"
              f"{'drafter-chain':>15}{'verify+bookkeeping':>20}")
        nmap = {(c["tier"], c["conc"]): c for c in ngram["cells"] if c.get("itl_ms_p50")}
        for tier, conc in keys:
            dm = dmap.get((tier, conc), {}).get("itl_ms_p50")
            nm = nmap.get((tier, conc), {}).get("itl_ms_p50")
            pm = next(
                (x.get("itl_ms_p50") for x in plana["cells"] if x["tier"] == tier and x["conc"] == conc),
                None,
            )
            if dm is None or nm is None or pm is None:
                continue
            print(f"{tier:>7}{conc:>5}{dm:>9.1f}{nm:>9.1f}{pm:>9.1f}"
                  f"{pm - nm:>15.1f}{nm - dm:>20.1f}")
        print("\nreading: drafter-chain = planA - ngram (drafter fwd + draft KV tax);")
        print("          verify+bookkeeping = ngram - dense (target multi-token verify + SD glue).")


def cmd_rt(args):
    """E4: R(t) occupancy + ITL-vs-R from cells saved with --save-ts."""
    any_ts = False
    for _p, _doc, c in _load_cells(args.paths):
        ts_lists = c.get("req_token_ts")
        if not ts_lists:
            continue
        any_ts = True
        # occupancy step function from (start, +1) / (end, -1) sweep
        events = []
        for ts in ts_lists:
            if len(ts) >= 1:
                events.append((ts[0], 1))
                events.append((ts[-1], -1))
        events.sort(key=lambda e: (e[0], -e[1]))
        changes = []  # (t, R_after)
        r = 0
        for t, d in events:
            r += d
            changes.append((t, r))
        t_lo, t_hi = changes[0][0], changes[-1][0]
        if t_hi <= t_lo:
            continue

        def r_at(t):
            lo, hi = 0, len(changes) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if changes[mid][0] <= t:
                    lo = mid
                else:
                    hi = mid - 1
            return changes[lo][1]

        occ = [r_at(t_lo + (t_hi - t_lo) * q / 10) for q in range(11)]
        # ITL bucketed by concurrent R at gap midpoint
        buckets: dict[int, list[float]] = {}
        for ts in ts_lists:
            for a, b in zip(ts, ts[1:]):
                mid = (a + b) / 2
                buckets.setdefault(r_at(mid), []).append((b - a) * 1000)
        print(f"[{c['tag']}] tier={c['tier']} conc={c['conc']}:")
        print(f"  R deciles  : {' '.join(f'{x}' for x in occ)}")
        rows = sorted(buckets)
        for rv in rows:
            v = buckets[rv]
            v.sort()
            print(f"  R={rv:<3} n={len(v):<6} ITL p50={v[len(v)//2]:8.1f}ms mean={sum(v)/len(v):8.1f}ms")
    if not any_ts:
        print("no req_token_ts found; re-run bench with --save-ts (cells with conc>1 are the useful ones)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the benchmark matrix")
    r.add_argument("--base-url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--tag", required=True, help="platform/config label, e.g. gpu-awq-dense")
    r.add_argument("--out", required=True)
    r.add_argument("--tiers", default="4096,16384", help="prompt token tiers, comma-sep")
    r.add_argument("--concs", default="1,4,16", help="concurrency levels, comma-sep")
    r.add_argument("--num-prompts", type=int, default=8, help="prompts per cell")
    r.add_argument("--max-tokens", type=int, default=256, help="generation length")
    r.add_argument("--timeout", type=float, default=600.0)
    r.add_argument("--note", default="", help="free-form note stored in JSON")
    r.add_argument("--seed-profile", choices=sorted(SEED_PROFILES), default="generic",
                   help="question set: generic (summarize etc) or repetitive (verbatim recall, ngram-friendly)")
    r.add_argument("--save-ts", action="store_true",
                   help="store per-request token timestamps for R(t) analysis (`rt`)")
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("table", help="print results as a table")
    t.add_argument("paths", nargs="+")
    t.set_defaults(func=cmd_table)

    s = sub.add_parser("summary", help="compact TSV summary for pasting back")
    s.add_argument("paths", nargs="+")
    s.set_defaults(func=cmd_summary)

    kr = sub.add_parser("kregress", help="E1: itl~K regression + net-optimal K (K from tags)")
    kr.add_argument("paths", nargs="+")
    kr.set_defaults(func=cmd_kregress)

    df = sub.add_parser("diff", help="E2: dense.json first, then SD JSONs; derives drafter-chain vs verify cost")
    df.add_argument("paths", nargs="+")
    df.set_defaults(func=cmd_diff)

    rt = sub.add_parser("rt", help="E4: R(t) occupancy + ITL-vs-R (needs --save-ts cells)")
    rt.add_argument("paths", nargs="+")
    rt.set_defaults(func=cmd_rt)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
