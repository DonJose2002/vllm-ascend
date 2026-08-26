#!/usr/bin/env python3
"""Phase 1.5 T2: bracket a clean decode window with /start_profile // /stop_profile.

Sends one streaming request (same prompt synthesis as bench_baseline so the
context matches the T1 cells), arms /start_profile after the Nth content
token (prefill and first-graph tax excluded), and lets the serve-side
ProfilerConfig.max_iterations auto-stop the recording after a bounded number
of engine steps. Explicit /stop_profile after every round is REQUIRED even
when auto-stop fired: the worker wrapper stays "active" until stopped, and a
second /start_profile would be ignored otherwise.

Prints one line per round; the ITL-after-start is the profiler-taxed step
time (first sanity number, compare against the unprofiled T1 cells). Exits 3
with a PROFILE-FAIL line when /start_profile is not served (404 = profiler
router not mounted: serve must pass --profiler-config '{"profiler":"torch",...}').

Usage:
  profile_window.py --base-url http://127.0.0.1:8001 --model qwen3-8b \
      [--tier 4096] [--max-tokens 256] [--start-after-tokens 24] \
      [--rounds 2] [--timeout 600]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_baseline import CHARS_PER_TOKEN, QUESTIONS, synthesize_prompt  # noqa: E402


def post(url: str, timeout: float = 30.0) -> int:
    req = urllib.request.Request(url, data=b"", method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status


def run_round(
    base_url: str, model: str, tier: int, max_tokens: int, start_after_tokens: int, timeout: float
) -> tuple[bool, str]:
    prompt = synthesize_prompt(int(tier * CHARS_PER_TOKEN), QUESTIONS[0])
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"}
    )
    token_ts: list[float] = []
    start_idx: int | None = None
    saw_done = False
    t0 = time.monotonic()
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
                        saw_done = True
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if any((ch.get("delta") or {}).get("content") for ch in obj.get("choices", [])):
                        token_ts.append(time.monotonic())
                        if start_idx is None and len(token_ts) >= start_after_tokens:
                            try:
                                post(base_url.rstrip("/") + "/start_profile")
                            except urllib.error.HTTPError as e:
                                print(
                                    f"PROFILE-FAIL: /start_profile HTTP {e.code} - "
                                    "profiler router not mounted; serve needs "
                                    "--profiler-config with profiler=torch"
                                )
                                return False, "start_404"
                            except Exception as e:  # noqa: BLE001
                                print(f"PROFILE-FAIL: /start_profile {e!r}")
                                return False, "start_err"
                            start_idx = len(token_ts) - 1
    except Exception as e:  # noqa: BLE001
        return False, f"stream err {e!r} (tokens={len(token_ts)})"

    if start_idx is not None:
        try:
            post(base_url.rstrip("/") + "/stop_profile")
        except Exception as e:  # noqa: BLE001
            print(
                f"# WARN: /stop_profile failed ({e!r}) - wrapper may stay active, next round's start would be ignored"
            )

    if start_idx is None:
        return False, f"stream ended before token {start_after_tokens} (tokens={len(token_ts)})"
    gaps = [b - a for a, b in zip(token_ts[start_idx:], token_ts[start_idx + 1 :])]
    itl = statistics.median(gaps) * 1000 if gaps else float("nan")
    print(
        f"round: tokens={len(token_ts)} start_tok={start_after_tokens} "
        f"itl_after_start_ms_p50={itl:.1f} (n={len(gaps)}) "
        f"round_s={time.monotonic() - t0:.1f} stream_ok={saw_done}"
    )
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tier", type=int, default=4096, help="prompt tier (chars = tier * CHARS_PER_TOKEN)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--start-after-tokens", type=int, default=24, help="content tokens seen before arming /start_profile"
    )
    ap.add_argument("--rounds", type=int, default=2, help="independent windows (one trace file each)")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    ok = 0
    for i in range(1, args.rounds + 1):
        good, _why = run_round(
            args.base_url, args.model, args.tier, args.max_tokens, args.start_after_tokens, args.timeout
        )
        ok += good
        if not good:
            print(f"# round {i}/{args.rounds} failed; continuing")
        time.sleep(2)
    print(f"PROFILE-WINDOW-DONE rounds={args.rounds} ok={ok}")
    return 0 if ok == args.rounds else (3 if ok == 0 else 1)


if __name__ == "__main__":
    sys.exit(main())
