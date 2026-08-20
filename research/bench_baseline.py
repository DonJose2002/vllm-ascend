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
      [--tiers 4096,16384] [--concs 1,4,16] [--num-prompts 8] [--sd]
  python3 bench_baseline.py table results_gpu.json [results_npu.json ...]

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


def run_cell(base_url, model, tier, conc, num_prompts, max_tokens, timeout, tag, results):
    prompts = [
        synthesize_prompt(int(tier * CHARS_PER_TOKEN), QUESTIONS[i % len(QUESTIONS)])
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
                )

    doc = {
        "harness": "bench_baseline.py v1",
        "base_url": args.base_url,
        "model": args.model,
        "tag": args.tag,
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
        print(f"# file={p} tag={d['tag']} created={d.get('created','')}")
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
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("table", help="print results as a table")
    t.add_argument("paths", nargs="+")
    t.set_defaults(func=cmd_table)

    s = sub.add_parser("summary", help="compact TSV summary for pasting back")
    s.add_argument("paths", nargs="+")
    s.set_defaults(func=cmd_summary)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
