#!/usr/bin/env python3
"""Benchmark speculative decoding (accept length + latency) for plan A/C.

Standalone test script; does not modify any vllm/vllm-ascend code.

Modes:
  bench   Send a fixed prompt set to an OpenAI-compatible server (streaming),
          snapshot /metrics spec-decode counters around each request, and
          report accept length (incl. bonus), per-position acceptance rates,
          and decode latency (ITL/TPOT percentiles).
  compare Summarize two bench JSON outputs side by side.

Usage (server side, run once per mode):
  # plan C:  VLLM_ASCEND_DRAFT_MODEL_FULL_GRAPH=0 vllm serve ...
  # plan A:  VLLM_ASCEND_DRAFT_MODEL_FULL_GRAPH=1 vllm serve ...
  python3 research/bench_sd.py bench  --base-url http://127.0.0.1:8007 \
      --model /nfs-share/hf_weights/Qwen3-8B --tag A --out bench_A.json
  python3 research/bench_sd.py compare bench_C.json bench_A.json

Metrics source: vllm:spec_decode_num_drafts_total,
vllm:spec_decode_num_accepted_tokens_total and
vllm:spec_decode_num_accepted_tokens_per_pos_total (vllm 0.22.1,
vllm/v1/spec_decode/metrics.py). Mean accept length = 1 + accepted/drafts.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.request

DEFAULT_PROMPTS = [
    "Explain the difference between throughput and latency in LLM serving.",
    "Write a Python function that merges two sorted lists in linear time.",
    "Summarize the causes of the 1929 economic crisis in five bullet points.",
    "Translate into French: The quick brown fox jumps over the lazy dog.",
    "Describe how speculative decoding speeds up LLM inference.",
    "List six practical ways to reduce KV cache memory usage.",
    "Explain why attention scales quadratically with sequence length.",
    "Write a haiku about debugging race conditions at midnight.",
    "Compare INT8 and INT4 quantization trade-offs for edge devices.",
    "What are the main components of the Ascend CANN software stack?",
]


def http_get(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode()


def parse_metrics(text: str) -> dict:
    """Parse the few spec-decode counters from a Prometheus exposition."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("vllm:"):
            continue
        try:
            name_val = line.split()
            name, value = name_val[0], float(name_val[1])
        except (IndexError, ValueError):
            continue
        if name.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            m = re.search(r'position="(\d+)"', name)
            if m:
                out[f"pos_{m.group(1)}"] = value
        elif name.startswith("vllm:spec_decode_num_drafts_total"):
            out["drafts"] = value
        elif name.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted"] = value
        elif name.startswith("vllm:spec_decode_num_draft_tokens_total"):
            out["draft_tokens"] = value
    return out


def snapshot(base_url: str) -> dict:
    return parse_metrics(http_get(base_url.rstrip("/") + "/metrics"))


def estimate_steps_from_timestamps(token_ts: list[float], min_gap_s: float = 0.002) -> int:
    """Estimate engine decode steps from streamed-token arrival timestamps.

    With speculative decoding each engine step releases (1 + accepted) tokens
    back-to-back, so inter-token gaps are bimodal: intra-burst gaps are
    network/serialization noise (sub-ms on localhost) while inter-step gaps
    are one decode step (tens of ms). Count gaps above an adaptive threshold
    (relative to the median gap) to recover the step count. For pooled
    accept length, tokens/steps equals the conventional 1 + accepted/drafts.
    Truncation by max_tokens can clip the final burst (slight underestimate).
    """
    if len(token_ts) < 2:
        return max(1, len(token_ts))
    gaps = [b - a for a, b in zip(token_ts, token_ts[1:])]
    median_gap = statistics.median(gaps)
    threshold = max(min_gap_s, 0.3 * median_gap)
    return 1 + sum(g > threshold for g in gaps)


def run_one(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    poll_timeout: float = 10.0,
) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    before = snapshot(base_url)
    t0 = time.perf_counter()
    first_token_ts = None
    token_ts: list[float] = []
    itls: list[float] = []
    completion_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        prev = t0
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                now = time.perf_counter()
                token_ts.append(now)
                if first_token_ts is None:
                    first_token_ts = now
                else:
                    itls.append(now - prev)
                prev = now
                completion_tokens += 1
    total = time.perf_counter() - t0
    # Spec-decode counters may advance on the engine's periodic stats
    # interval rather than per-request: poll until the drafts counter moves
    # (or timeout) so per-request attribution usually works; global totals
    # computed by the caller remain exact either way.
    after = snapshot(base_url)
    deadline = time.perf_counter() + poll_timeout
    while after.get("drafts", 0.0) <= before.get("drafts", 0.0) and time.perf_counter() < deadline:
        time.sleep(0.5)
        after = snapshot(base_url)
    req_metrics = {
        k: after.get(k, 0.0) - before.get(k, 0.0) for k in set(before) | set(after)
    }
    return {
        "prompt": prompt,
        "completion_tokens": completion_tokens,
        "ttft": (first_token_ts - t0) if first_token_ts else None,
        "total_s": total,
        "itls": itls,
        "est_steps": estimate_steps_from_timestamps(token_ts),
        "metrics_delta": req_metrics,
    }


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def summarize(metrics_delta: dict, results: list[dict]) -> dict:
    drafts = metrics_delta.get("drafts", 0.0)
    accepted = metrics_delta.get("accepted", 0.0)
    pos = {k: v for k, v in metrics_delta.items() if k.startswith("pos_")}
    all_itls = [x for r in results for x in r["itls"]]
    ttfts = [r["ttft"] for r in results if r["ttft"]]
    accept_len = 1 + accepted / drafts if drafts else None
    total_tokens = sum(r["completion_tokens"] for r in results)
    total_steps = sum(r.get("est_steps", 0) for r in results)
    accept_len_est = total_tokens / total_steps if total_steps else None
    return {
        "num_requests": len(results),
        "total_drafts": drafts,
        "total_accepted": accepted,
        "mean_accept_length_incl_bonus": accept_len,
        "mean_accept_length_burst_est": round(accept_len_est, 4) if accept_len_est else None,
        "est_steps_total": total_steps,
        "per_position_accepted": {k: round(v, 1) for k, v in sorted(pos.items())},
        "per_position_rate_vs_drafts": (
            {k: round(v / drafts, 4) for k, v in sorted(pos.items())} if drafts else None
        ),
        "decode_itl_ms": {
            "mean": round(statistics.mean(all_itls) * 1000, 2) if all_itls else None,
            "p50": round(pct(all_itls, 0.50) * 1000, 2) if all_itls else None,
            "p90": round(pct(all_itls, 0.90) * 1000, 2) if all_itls else None,
        },
        "ttft_ms_mean": round(statistics.mean(ttfts) * 1000, 2) if ttfts else None,
        "completion_tokens_total": sum(r["completion_tokens"] for r in results),
    }


def cmd_bench(args: argparse.Namespace) -> None:
    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [ln.strip() for ln in open(args.prompts_file) if ln.strip()]
    prompts = prompts * args.repeat
    start_global = snapshot(args.base_url)
    results = []
    for i, prompt in enumerate(prompts):
        r = run_one(args.base_url, args.model, prompt, args.max_tokens, args.temperature, args.poll_timeout)
        d = r["metrics_delta"]
        al = 1 + d.get("accepted", 0.0) / d["drafts"] if d.get("drafts") else None
        al_m = f"{al:.3f}" if al is not None else "n/a"
        al_e = r["completion_tokens"] / r["est_steps"] if r["est_steps"] else None
        al_est = f"{al_e:.3f}" if al_e is not None else "n/a"
        itl_mean = f"{statistics.mean(r['itls']) * 1000:.2f}" if r["itls"] else "n/a"
        print(
            f"[{i+1}/{len(prompts)}] tokens={r['completion_tokens']} "
            f"steps~={r['est_steps']} accept_len(metrics)={al_m} accept_len(est)={al_est} tpot_ms={itl_mean}",
            flush=True,
        )
        results.append(r)
    end_global = snapshot(args.base_url)
    # Summary from whole-run first/last snapshots: immune to per-request
    # attribution jitter of periodically-updated counters.
    global_delta = {
        k: end_global.get(k, 0.0) - start_global.get(k, 0.0) for k in set(start_global) | set(end_global)
    }
    if not global_delta.get("drafts"):
        print(
            "WARNING: vllm:spec_decode_num_drafts_total did not advance during the run; "
            "accept-length stats unavailable. Check that /metrics exposes spec-decode "
            "counters (no --disable-log-stats / prometheus disabled).",
            file=sys.stderr,
        )
    summary = summarize(global_delta, results)
    out = {"tag": args.tag, "created": time.strftime("%F %T"), "summary": summary, "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    with open(args.a) as f:
        ja = json.load(f)
    with open(args.b) as f:
        jb = json.load(f)
        sa, sb = ja["summary"], jb["summary"]

    def row(name, va, vb, fmt="{:.3f}"):
        try:
            da, db = fmt.format(va), fmt.format(vb)
        except (ValueError, TypeError):
            da, db = str(va), str(vb)
        print(f"{name:<36}{da:>14}{db:>14}")

    print(f"{'':<36}{ja['tag'] + ' (' + ja.get('created','') + ')':>28}"
          f"{jb['tag'] + ' (' + jb.get('created','') + ')':>28}")
    row("requests", sa["num_requests"], jb["summary"]["num_requests"], "{:d}")
    row("draft steps", sa["total_drafts"], sb["total_drafts"], "{:.0f}")
    row("mean accept len (incl bonus)", sa["mean_accept_length_incl_bonus"], sb["mean_accept_length_incl_bonus"])
    row("mean accept len (burst est)", sa.get("mean_accept_length_burst_est"), sb.get("mean_accept_length_burst_est"))
    row("ITL mean (ms)", sa["decode_itl_ms"]["mean"], sb["decode_itl_ms"]["mean"])
    row("ITL p50 (ms)", sa["decode_itl_ms"]["p50"], sb["decode_itl_ms"]["p50"])
    row("ITL p90 (ms)", sa["decode_itl_ms"]["p90"], sb["decode_itl_ms"]["p90"])
    row("TTFT mean (ms)", sa["ttft_ms_mean"], sb["ttft_ms_mean"], "{:.1f}")
    pa, pb = sa.get("per_position_rate_vs_drafts") or {}, sb.get("per_position_rate_vs_drafts") or {}
    for k in sorted(set(pa) | set(pb)):
        row(f"accept rate {k.replace('pos_', 'pos ')}", pa.get(k), pb.get(k), "{:.4f}")
    al_a = sa["mean_accept_length_incl_bonus"]
    al_b = sb["mean_accept_length_incl_bonus"]
    if al_a and al_b:
        print(f"\naccept-len delta (B - A): {al_b - al_a:+.3f}")


def cmd_check(args: argparse.Namespace) -> None:
    """Dump spec-decode related /metrics lines to diagnose missing counters."""
    text = http_get(args.base_url.rstrip("/") + "/metrics")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    spec = [ln for ln in lines if "spec" in ln.lower()]
    print(f"total metric lines: {len(lines)}, containing 'spec': {len(spec)}")
    if spec:
        print("--- spec-decode lines ---")
        for ln in spec:
            print(ln)
    else:
        print("NO spec-decode metrics exposed. Sample of available vllm: lines:")
        vllm_lines = [ln for ln in lines if ln.startswith("vllm:")]
        for ln in vllm_lines[:60]:
            print(ln)
        if not vllm_lines:
            print("(no vllm: prefixed lines at all — /metrics may be disabled or proxied)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--base-url", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--tag", default="run", help="label stored in the output json")
    b.add_argument("--out", default="bench_out.json")
    b.add_argument("--max-tokens", type=int, default=256)
    b.add_argument("--temperature", type=float, default=0.0)
    b.add_argument("--repeat", type=int, default=3, help="repeat the prompt set N times")
    b.add_argument("--prompts-file", default=None, help="one prompt per line; default built-in set")
    b.add_argument(
        "--poll-timeout",
        type=float,
        default=10.0,
        help="seconds to wait per-request for spec-decode counters to advance (they tick on the periodic stats interval)",
    )
    b.set_defaults(func=cmd_bench)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(func=cmd_compare)
    k = sub.add_parser("check", help="inspect /metrics for spec-decode counters")
    k.add_argument("--base-url", required=True)
    k.set_defaults(func=cmd_check)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
