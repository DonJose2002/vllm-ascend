#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""NIAH (needle-in-a-haystack) accuracy harness for Phase 2 KV compression.

The latency bench (bench_baseline.py) cannot see quality: its generic profile
asks soft questions and its repetitive profile demands verbatim echo. This
tool closes that gap with a self-constructed retrieval grid - depth(needle
position) x length(haystack tokens) - where each cell hides one verifiable
fact (a random alphanumeric code in a carrier sentence) and asks for it back.

score = fraction of cells whose response contains the exact code (strict
case-sensitive substring). Running the SAME grid on {dense, topk-2048/4096/
8192} yields the tightness-vs-quality curve - the Phase 4 draft-side
compression selection input.

Why strict substring: the code (e.g. "K7Q-4MX-92Z") is unambiguous, temp=0,
and the prompt says "reply with the code only"; anything else is a miss worth
inspecting via the recorded response snippet.

Qwen3 recipe (e2e-verified): chat_template_kwargs {"enable_thinking": false}
so answers are not buried in reasoning tags.

Usage:
  # against a running server (any mode; same serve as the latency bench):
  python3 research/needle_eval.py run --base-url http://127.0.0.1:8001 \
      --model qwen3-8b --tag npu-bf16-dense \
      --tiers 16384,32768 --depths 0.1,0.25,0.5,0.75,0.9 --samples 2 \
      --out niah.json
  # tightness x quality table across configs (dense json first):
  python3 research/needle_eval.py curve dense.json topk4096.json topk2048.json
  # offline logic check (no server):
  python3 research/needle_eval.py selftest

Only stdlib + urllib (no requests/torch), runs on server and laptop alike.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request

# Rough chars-per-token for the filler below on the Qwen3 BPE (calibrated in
# bench_baseline.py); only sizes the synthesis, actual ptok is read back.
CHARS_PER_TOKEN = 5.7

FILLER_PARAGRAPHS = [
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
    "Operating systems multiplex processors among competing tasks through "
    "scheduling, virtualize physical memory through paging, and shield "
    "applications from device specifics through layered abstractions.",
    "Network protocols stack independent responsibilities: framing and "
    "checksums at the link layer, routing at the network layer, and "
    "reliability, flow, and congestion control at the transport layer.",
    "Numerical linear algebra underpins machine learning: matrix "
    "decompositions, eigenvalue solvers, and iterative methods all exploit "
    "structure such as symmetry, sparsity, and low rank.",
    "Database engines parse declarative queries into algebra trees, choose "
    "join orders and access paths with cost-based optimizers, and guarantee "
    "atomicity, consistency, isolation, and durability through logging.",
    "Cryptography builds primitives from hardness assumptions: one-way "
    "functions give hash chains, trapdoor permutations give public-key "
    "encryption, and both underpin signatures and key exchange.",
]

# (carrier sentence with {code}, matching question asking for it back).
# Carriers are ordinary-looking detail sentences; questions name the exact
# object so retrieval - not guessing - decides the answer.
NEEDLE_TEMPLATES = [
    (
        "One detail worth recording in the margin: the access code for the northern archive is {code}.",
        "What is the access code for the northern archive stated in the text above? Reply with the code only.",
    ),
    (
        "A note pinned to the staff bulletin board reads: the maintenance key for this week is {code}.",
        "According to the note above, what is the maintenance key for this week? Reply with the code only.",
    ),
    (
        "The shipment manifest lists the container seal number as {code}.",
        "What container seal number does the manifest above list? Reply with the code only.",
    ),
    (
        "According to the lab logbook, the calibration passphrase registered on Tuesday was {code}.",
        "What calibration passphrase was registered on Tuesday according to "
        "the logbook above? Reply with the code only.",
    ),
]

# Unambiguous alphabet: no 0/O, 1/I, so a correct recall is never punished by
# transcription confusion.
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def gen_code(rng: random.Random) -> str:
    groups = ["".join(rng.choice(CODE_ALPHABET) for _ in range(3)) for _ in range(3)]
    return "-".join(groups)


def build_haystack(target_chars: int, depth: float, rng: random.Random) -> tuple[str, str, str]:
    """Return (body_with_needle, question, code).

    The body repeats filler paragraphs up to target_chars; the needle carrier
    sentence is inserted at a paragraph boundary closest to depth * body_len
    (depth granularity = one paragraph, ~44 tokens - fine for NIAH).
    """
    body: list[str] = []
    n = 0
    i = 0
    while n < target_chars:
        p = FILLER_PARAGRAPHS[i % len(FILLER_PARAGRAPHS)]
        body.append(p)
        n += len(p) + 1
        i += 1

    carrier_tpl, question = NEEDLE_TEMPLATES[rng.randrange(len(NEEDLE_TEMPLATES))]
    code = gen_code(rng)
    needle = carrier_tpl.format(code=code)

    # Paragraph-boundary index closest to the requested depth (0 < idx < len
    # so the needle is always interior; depth is clamped, never boundary).
    want = depth * len(body)
    idx = max(1, min(len(body) - 1, round(want)))
    body.insert(idx, needle)
    return " ".join(body), question, code


def build_prompt(target_chars: int, depth: float, seed: str) -> tuple[str, str, str, float]:
    """Deterministic: the same (target_chars, depth, seed) -> identical bytes."""
    rng = random.Random(seed)
    body, question, code = build_haystack(target_chars, depth, rng)
    prompt = body + "\n\n" + question
    actual_depth = _needle_depth(prompt, code)
    return prompt, question, code, actual_depth


def _needle_depth(prompt: str, code: str) -> float:
    pos = prompt.find(code)
    if pos < 0:
        return float("nan")
    return pos / len(prompt)


def score_response(text: str, code: str) -> bool:
    """Strict, case-sensitive: the code must appear verbatim."""
    return code in (text or "")


def ask_one(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    enable_thinking: bool = False,
) -> tuple[str, int, str | None]:
    """Non-streaming chat completion. Returns (content, prompt_tokens, err)."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            obj = json.loads(resp.read().decode())
        content = ""
        for ch in obj.get("choices", []):
            content += (ch.get("message") or {}).get("content") or ""
        ptok = (obj.get("usage") or {}).get("prompt_tokens", 0)
        return content, ptok, None
    except Exception as e:  # noqa: BLE001
        return "", 0, repr(e)


# ---------------------------------------------------------------------------
# run / curve / selftest
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    tiers = [int(t) for t in args.tiers.split(",")]
    depths = [float(d) for d in args.depths.split(",")]
    cells: list[dict] = []
    t0 = time.time()
    for tier in tiers:
        target_chars = int(tier * CHARS_PER_TOKEN)
        for depth in depths:
            for s in range(args.samples):
                seed = f"{args.seed}:{tier}:{depth}:{s}"
                prompt, _q, code, want_depth = build_prompt(target_chars, depth, seed)
                assert prompt.count(code) == 1, "needle must appear exactly once"
                content, ptok, err = ask_one(args.base_url, args.model, prompt, args.max_tokens, args.timeout)
                hit = err is None and score_response(content, code)
                ptok_off = ptok > 0 and abs(ptok - tier) / tier > 0.15
                cells.append(
                    {
                        "tier": tier,
                        "depth": depth,
                        "sample": s,
                        "code": code,
                        "hit": hit,
                        "ptok": ptok,
                        "ptok_off": ptok_off,
                        "resp": content[:120],
                        "err": err,
                    }
                )
                mark = "." if hit else ("E" if err else "x")
                print(f"[{time.time() - t0:7.1f}s] tier={tier} depth={depth} s={s} -> {mark}", flush=True)
    out = {
        "tool": "needle_eval",
        "tag": args.tag,
        "params": {
            "tiers": tiers,
            "depths": depths,
            "samples": args.samples,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "chars_per_token": CHARS_PER_TOKEN,
        },
        "cells": cells,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out} ({len(cells)} cells)")
    print_grid(out)
    return 0


def _grid(out: dict) -> tuple[list[float], list[int], dict]:
    """depths x tiers -> {tier: {depth: hits/samples}} plus overall."""
    params = out["params"]
    tiers, depths = params["tiers"], params["depths"]
    g: dict[int, dict[float, list[int]]] = {t: {d: [0, 0] for d in depths} for t in tiers}
    for c in out["cells"]:
        slot = g[c["tier"]][c["depth"]]
        slot[1] += 1
        slot[0] += bool(c["hit"])
    return depths, tiers, g


def print_grid(out: dict) -> None:
    depths, tiers, g = _grid(out)
    hits = sum(1 for c in out["cells"] if c["hit"])
    total = len(out["cells"])
    n_off = sum(1 for c in out["cells"] if c.get("ptok_off"))
    print(
        f"# tag={out['tag']} score={hits}/{total} ({100.0 * hits / max(1, total):.1f}%)"
        + (f" ptok_off={n_off}" if n_off else "")
    )
    header = ["depth"] + [str(t) for t in tiers]
    print("\t".join(header))
    for d in depths:
        row = [f"{d:.2f}"]
        for t in tiers:
            h, n = g[t][d]
            row.append(f"{h}/{n}")
        print("\t".join(row))


def _load(paths: list[str]) -> list[dict]:
    return [json.loads(open(p).read()) for p in paths]  # noqa: SIM115 - short-lived read


def cmd_curve(args: argparse.Namespace) -> int:
    """Tightness x quality: one row per config, one column per tier + overall."""
    runs = _load(args.paths)
    print("config\t" + "\t".join(f"tier{t}" for t in runs[0]["params"]["tiers"]) + "\toverall")
    for run in runs:
        _depths, tiers, g = _grid(run)
        cells = run["cells"]
        overall = sum(1 for c in cells if c["hit"]) / max(1, len(cells))
        per_tier = []
        for t in tiers:
            h = sum(g[t][d][0] for d in _depths)
            n = sum(g[t][d][1] for d in _depths)
            per_tier.append(f"{h / max(1, n):.3f}")
        print(f"{run['tag']}\t" + "\t".join(per_tier) + f"\t{overall:.3f}")


def cmd_selftest(args: argparse.Namespace) -> int:
    del args  # unused
    # 1. determinism: same seed -> byte-identical prompt.
    p1, _, c1, d1 = build_prompt(20000, 0.5, "s:1")
    p2, _, c2, d2 = build_prompt(20000, 0.5, "s:1")
    assert p1 == p2 and c1 == c2, "same seed must rebuild identical prompt+code"
    # different sample -> different code (with overwhelming probability).
    _, _, c3, _ = build_prompt(20000, 0.5, "s:2")
    assert c3 != c1, "different seeds should not collide on the same code"

    # 2. needle uniqueness + depth placement (paragraph granularity).
    for depth in (0.1, 0.25, 0.5, 0.75, 0.9):
        prompt, _q, code, actual = build_prompt(20000, depth, f"t:{depth}")
        assert prompt.count(code) == 1, "code must appear exactly once"
        para = len(FILLER_PARAGRAPHS[0]) / len(prompt)  # one paragraph ~ max drift
        assert abs(actual - depth) <= max(0.05, para), f"depth {depth}: needle landed at {actual:.3f}"

    # 3. scorer.
    code = "K7Q-4MX-92Z"
    assert score_response("The code is K7Q-4MX-92Z.", code)
    assert score_response("K7Q-4MX-92Z", code)
    assert not score_response("k7q-4mx-92z", code), "case-sensitive by design"
    assert not score_response("The code is K7Q-4MX-92.", code), "truncated is a miss"
    assert not score_response("", code)
    assert not score_response("I cannot find the code in the text.", code)

    # 4. grid + curve rendering from a synthetic fixture.
    fixture = {
        "tool": "needle_eval",
        "tag": "fixture-a",
        "params": {"tiers": [16384, 32768], "depths": [0.1, 0.5], "samples": 2, "max_tokens": 48, "seed": "x"},
        "cells": [
            {"tier": t, "depth": d, "sample": s, "hit": (t == 16384), "ptok": t, "ptok_off": False}
            for t in (16384, 32768)
            for d in (0.1, 0.5)
            for s in (0, 1)
        ],
    }
    _depths, _tiers, g = _grid(fixture)
    assert g[16384][0.1] == [2, 2] and g[32768][0.5] == [0, 2], "grid aggregation wrong"
    hits = sum(1 for c in fixture["cells"] if c["hit"])
    assert hits == 4 and len(fixture["cells"]) == 8

    # 5. code charset discipline.
    rng = random.Random("charset")
    for _ in range(50):
        for ch in gen_code(rng).replace("-", ""):
            assert ch in CODE_ALPHABET

    print("PASS: needle_eval selftest (determinism, placement, scorer, grid aggregation)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the NIAH grid against a server")
    r.add_argument("--base-url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--tag", required=True, help="config label, e.g. npu-bf16-dense-hamming-topk4096")
    r.add_argument("--out", required=True)
    r.add_argument("--tiers", default="16384,32768")
    r.add_argument("--depths", default="0.1,0.25,0.5,0.75,0.9")
    r.add_argument("--samples", type=int, default=2)
    r.add_argument("--max-tokens", type=int, default=48)
    r.add_argument("--timeout", type=float, default=600.0)
    r.add_argument("--seed", default="niah-v1")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("curve", help="tightness x quality table across run jsons")
    c.add_argument("paths", nargs="+")
    c.set_defaults(func=cmd_curve)

    st = sub.add_parser("selftest", help="offline logic check (no server)")
    st.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
