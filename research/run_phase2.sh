#!/usr/bin/env bash
# Phase 2 batch driver (2026-09-01). Wraps run_baseline_npu.sh over the C-line
# matrix from experiments/phase2-kv-compression-design.md (notes repo) so one
# command per night runs unattended; collects every SUMMARY into a master log
# and prints a paste-ready digest (per-JSON TSV + NIAH curve + ITL delta table
# + mutual-exclusion 3-way compare) at the end.
#
# Usage:  bash research/run_phase2.sh smoke|cline|b2smoke|digest [start_port]
#   smoke : single 16K/c1 topk4096 eager cell (first-run validation only -
#           already PASSED 2026-09-01 on NPU2; kept for re-runs on new hosts)
#   cline : the full eager matrix, 6 serves (~1 evening):
#             1. dense   eager anchor   16K/32K x 1/16 + NIAH (accuracy ref)
#             2-4. hamming topk {2048,4096,8192}  16K/32K x 1/16 + NIAH
#             5. hammingsd topk4096 (+ngram SD K=5) - mutual-exclusion cell
#             6. ngram   eager anchor K=5 - the exclusion compare target
#   b2smoke: Phase 2 B-line B2 smoke, GRAPH mode (the bet!), 2 serves (~15 min):
#             1. compact 16K/c1 + NIAH(16K/32K on same serve) - verifies:
#                graph-mode serve survives (update_attn_params channel bet),
#                [static-kv-compact] self-evidence lines + KV-usage drop,
#                NIAH quality ~ topk4096 class (conservative selector)
#             2. dense 16K/c1 with --no-enable-prefix-caching - same-caliber
#                latency anchor (prefix state matches run 1; ITL delta ~0
#                expected at this corner - weights dominate, not a failure)
#           If run 1 dies during capture/replay, triage in eager (hint printed
#           at the end); eager green + graph red = bet lost, report the logs.
#   digest: analysis only (no serves, no card) over JSONs under experiments/out/phase2
#
# WHY EAGER EVERYWHERE (cline only): the first graph-mode smoke crashed with an
# aivec (vector core invalid GM) during FULL-graph capture warmup - the kvcomp
# decode path mutates/reassigns tensors per layer/step (hamming_output et al),
# which cannot be safely captured; upstream later removed the whole feature
# (#12049, not in v0.23.0). Eager is the only validated mode for hamming on
# v0.23.0, so ALL cline cells (anchors included) run --enforce-eager for a
# same-mode comparison. Graph-mode numbers from Phase 0/1 are NOT comparable.
# B-line static compaction has NO per-forward mutation (metadata-level view
# only), so b2smoke runs graph mode by design - that difference is the point.
#
# Output isolation: experiments/out/phase2/ - Phase 0/1 JSONs are never
# clobbered (hamming TAGs are unique, but discipline is discipline).
#
# Resource-safety guards copied from run_phase1.sh (2026-08-25 wedged-server
# incident): pin one card per batch, host-RAM floor before each serve, card
# drain after each serve, no core dumps, capped inductor threads.
set -uo pipefail

BATCH="${1:?usage: run_phase2.sh smoke|cline|b2smoke|digest [start_port]}"
PORT="${2:-8130}"
# Output isolation: b2smoke reuses the dense TAG, so it MUST live in its own
# directory (phase2-b2) or it would clobber the cline eager JSONs.
case "$BATCH" in
  b2smoke) OUTROOT_DEFAULT="experiments/out/phase2-b2" ;;
  *)       OUTROOT_DEFAULT="experiments/out/phase2" ;;
esac
OUTROOT="${OUTROOT:-$OUTROOT_DEFAULT}"
mkdir -p "$OUTROOT"
MASTER="$OUTROOT/phase2-${BATCH}-$(date +%Y%m%d-%H%M).log"

# --- resource safety guards (see run_phase1.sh incident note) ---
ulimit -c 0 2>/dev/null || true
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-16}"
FREE_MEM_MIN_MB="${FREE_MEM_MIN_MB:-40000}"
DRAIN_WAIT_MAX="${DRAIN_WAIT_MAX:-900}"
HBM_DRAINED_MB="${HBM_DRAINED_MB:-6144}"

# Guard: never let race-research envs leak into Phase-2 measurements.
unset VLLM_ASCEND_SD_REVIVE_RACE VLLM_ASCEND_SD_COUNTERS VLLM_ASCEND_SD_DEBUG \
      VLLM_ASCEND_SD_STAGED_COPY VLLM_ASCEND_SD_EVENT_COPY || true

npu_hbm_list() {
  python3 - <<'PYEOF'
import re, subprocess
try:
    out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15).stdout
except Exception as e:
    print(f"PARSE-FAIL: npu-smi info failed: {e}")
    raise SystemExit(0)
cur, found = None, False
for line in out.splitlines():
    m = re.match(r"^\|\s*(\d+)\s+\S+\s+\|\s*(OK|Warning|Alarm|Crit\w*|Unknown|Bad)", line)
    if m:
        cur = int(m.group(1))
        continue
    if cur is not None and re.search(r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f]", line):
        pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
        if pairs:
            used, _total = pairs[-1]
            print(f"{cur} {used}")
            found = True
if not found:
    print("PARSE-FAIL: no device blocks parsed")
PYEOF
}

card_hbm() { npu_hbm_list | awk -v id="$1" '$1==id{print $2}'; }

pin_npus() {
  if [ -z "${NPUS:-}" ]; then
    local dev_list pick
    dev_list="$(npu_hbm_list)"
    case "$dev_list" in
      PARSE-FAIL*)
        echo "$dev_list" | tee -a "$MASTER"
        echo "Cannot read NPU occupancy; set NPUS=<id> explicitly." | tee -a "$MASTER"
        exit 1
        ;;
    esac
    pick=$(echo "$dev_list" | awk '{if(!($1 in mx)||$2>mx[$1])mx[$1]=$2} END{best="";for(id in mx){if(best==""||mx[id]<bestv){best=id;bestv=mx[id]}}print best}')
    export NPUS="$pick"
    echo "PINNED: NPUS=$NPUS for the whole batch (auto-picked lowest-HBM card)" | tee -a "$MASTER"
  else
    echo "PINNED: NPUS=$NPUS (user-specified) for the whole batch" | tee -a "$MASTER"
  fi
}

wait_host_ram() {
  while :; do
    local avail
    avail=$(free -m 2>/dev/null | awk '/^Mem:/{print $7}')
    [ -z "$avail" ] && return 0
    [ "$avail" -ge "$FREE_MEM_MIN_MB" ] && return 0
    echo "WAIT: host available RAM ${avail}MB < ${FREE_MEM_MIN_MB}MB $(date +%H:%M:%S)" | tee -a "$MASTER"
    sleep 30
  done
}

drain_card() {
  local deadline=$(( $(date +%s) + DRAIN_WAIT_MAX )) used
  while [ "$(date +%s)" -lt "$deadline" ]; do
    used=$(card_hbm "$NPUS"); used="${used:-0}"
    [ "$used" -le "$HBM_DRAINED_MB" ] && { echo "DRAIN-OK: NPU $NPUS at ${used}MB $(date +%H:%M:%S)" | tee -a "$MASTER"; return 0; }
    sleep 15
  done
  echo "DRAIN-TIMEOUT (${DRAIN_WAIT_MAX}s): NPU $NPUS still at ${used}MB - hard-killing leftovers on port $PORT" | tee -a "$MASTER"
  pkill -KILL -f "vllm serve.*--port $PORT" 2>/dev/null || true
  sleep 30
}

banner() { printf '\n===== %s =====\n' "$*" | tee -a "$MASTER"; }

# run <mode> <tiers> <concs> [env=val ...] - one eager run_baseline_npu.sh call.
# EXTRA_SERVE_ARGS always starts with --enforce-eager (mode-wide decision,
# see header); anything the caller set is appended after it.
run() {
  local mode="$1" tiers="$2" concs="$3"; shift 3
  PORT=$((PORT + 1))
  banner "RUN mode=$mode tiers=$tiers concs=$concs port=$PORT outdir=$OUTROOT eager=1 envs=$* $(date +%H:%M:%S)"
  wait_host_ram
  env -u VLLM_ASCEND_SD_REVIVE_RACE -u VLLM_ASCEND_SD_COUNTERS -u VLLM_ASCEND_SD_DEBUG \
    "$@" TIERS="$tiers" CONCS="$concs" SAVE_TS=1 OUTDIR="$OUTROOT" \
    EXTRA_SERVE_ARGS="--enforce-eager ${EXTRA_SERVE_ARGS:-}" \
    bash research/run_baseline_npu.sh "$mode" "$PORT" 2>&1 | tee -a "$MASTER"
  pkill -TERM -f "vllm serve.*--port $PORT" 2>/dev/null || true
  drain_card
}

# run_graph <mode> <tiers> <concs> [env=val ...] - one GRAPH-mode call (B2 bet:
# no --enforce-eager injected). Extra serve flags come via the EXTRA_SERVE_ARGS
# shell variable read at call time (e.g. the dense anchor's
# --no-enable-prefix-caching); pass nothing for compact - the mode itself adds it.
run_graph() {
  local mode="$1" tiers="$2" concs="$3"; shift 3
  PORT=$((PORT + 1))
  banner "RUN graph mode=$mode tiers=$tiers concs=$concs port=$PORT outdir=$OUTROOT envs=$* $(date +%H:%M:%S)"
  wait_host_ram
  env -u VLLM_ASCEND_SD_REVIVE_RACE -u VLLM_ASCEND_SD_COUNTERS -u VLLM_ASCEND_SD_DEBUG \
    "$@" TIERS="$tiers" CONCS="$concs" SAVE_TS=1 OUTDIR="$OUTROOT" \
    EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}" \
    bash research/run_baseline_npu.sh "$mode" "$PORT" 2>&1 | tee -a "$MASTER"
  pkill -TERM -f "vllm serve.*--port $PORT" 2>/dev/null || true
  drain_card
}

BENCH="$OUTROOT/baseline-npu-qwen3-8b"
NIAHJSON="$OUTROOT/niah"

digest() {
  banner "DIGEST (phase2 $BATCH, all cline cells --enforce-eager) $(date +%H:%M:%S)"
  {
    echo "--- per-file TSV ---"
    ls "$BENCH"-*.json >/dev/null 2>&1 && \
      python3 research/bench_baseline.py summary "$BENCH"-*.json
    echo
    echo "--- NIAH curve (tightness x quality) ---"
    local niah_files=()
    local t
    for t in dense dense-hamming-topk2048 dense-hamming-topk4096 dense-hamming-topk8192; do
      [ -f "$NIAHJSON-npu-bf16-$t.json" ] && niah_files+=("$NIAHJSON-npu-bf16-$t.json")
    done
    if [ "${#niah_files[@]}" -ge 1 ]; then
      python3 research/needle_eval.py curve "${niah_files[@]}" 2>&1
    else
      echo "no NIAH jsons yet"
    fi
    echo
    echo "--- C-line ITL delta vs dense(eager) ---"
    python3 - "$BENCH" <<'PYEOF'
import glob, json, sys, os
bench_prefix = sys.argv[1]
def load(tag):
    p = f"{bench_prefix}-{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None
def cellmap(d):
    return {(c["tier"], c["conc"]): c for c in d["cells"]}
dense = load("npu-bf16-dense")
if not dense:
    print("dense(eager) anchor json missing - delta table skipped"); raise SystemExit
dm = cellmap(dense)
tags = ["npu-bf16-dense-hamming-topk2048", "npu-bf16-dense-hamming-topk4096",
        "npu-bf16-dense-hamming-topk8192"]
loaded = [(t, cellmap(load(t))) for t in tags if load(t)]
if not loaded:
    print("no hamming jsons yet"); raise SystemExit
print("tag\ttier\tconc\titl50\tdense_itl50\tdelta\tratio\touts\tdense_outs")
for t, m in loaded:
    for key in sorted(m):
        c, d = m[key], dm.get(key)
        if not d or not c.get("itl_ms_p50") or not d.get("itl_ms_p50"):
            continue
        di, hi = d["itl_ms_p50"], c["itl_ms_p50"]
        print(f"{t}\t{key[0]}\t{key[1]}\t{hi}\t{di}\t{round(hi-di,1)}\t{round(hi/di,3)}\t"
              f"{c.get('aggregate_out_tok_per_s')}\t{d.get('aggregate_out_tok_per_s')}")
PYEOF
    echo
    echo "--- mutual-exclusion 3-way (expect: hammingsd == ngram, hamming silently OFF) ---"
    python3 - "$BENCH" <<'PYEOF'
import json, sys, os
bench_prefix = sys.argv[1]
def cells(tag):
    p = f"{bench_prefix}-{tag}.json"
    if not os.path.exists(p):
        return None
    return {(c["tier"], c["conc"]): c for c in json.load(open(p))["cells"]}
hsd = cells("npu-bf16-dense-hamming-topk4096-sd-ngram-k5")
ng  = cells("npu-bf16-ngram-k5")
dn  = cells("npu-bf16-dense")
if not (hsd and ng):
    print("hammingsd or ngram json missing - skipped"); raise SystemExit
print("tier\tconc\thsd_itl\tngram_itl\tdense_itl\thsd_accB\tngram_accB")
for key in sorted(hsd):
    h, n = hsd[key], ng.get(key)
    d = dn.get(key) if dn else None
    if not n:
        continue
    print(f"{key[0]}\t{key[1]}\t{h.get('itl_ms_p50')}\t{n.get('itl_ms_p50')}\t"
          f"{d.get('itl_ms_p50') if d else '-'}\t{h.get('accept_len_burst')}\t{n.get('accept_len_burst')}")
PYEOF
  } 2>&1 | tee -a "$MASTER"
}

digest_b2() {
  banner "DIGEST (phase2 b2smoke, graph mode) $(date +%H:%M:%S)"
  {
    echo "--- per-file TSV ---"
    ls "$BENCH"-*.json >/dev/null 2>&1 && \
      python3 research/bench_baseline.py summary "$BENCH"-*.json
    echo
    echo "--- NIAH curve (compact, conservative selector; C-line refs: dense 1.0, topk4096 1.0) ---"
    if [ -f "$NIAHJSON-npu-bf16-compact.json" ]; then
      python3 research/needle_eval.py curve "$NIAHJSON-npu-bf16-compact.json" 2>&1
    else
      echo "no compact NIAH json"
    fi
    echo
    echo "--- B2 compact vs dense (both graph + --no-enable-prefix-caching) ---"
    python3 - "$BENCH" <<'PYEOF'
import json, sys, os
bench_prefix = sys.argv[1]
def cells(tag):
    p = f"{bench_prefix}-{tag}.json"
    if not os.path.exists(p):
        return None
    return {(c["tier"], c["conc"]): c for c in json.load(open(p))["cells"]}
comp, dn = cells("npu-bf16-compact"), cells("npu-bf16-dense")
if not comp:
    print("compact json missing - delta table skipped"); raise SystemExit
print("tag\ttier\tconc\titl50\touts")
for key in sorted(comp):
    c = comp[key]
    print(f"compact\t{key[0]}\t{key[1]}\t{c.get('itl_ms_p50')}\t{c.get('aggregate_out_tok_per_s')}")
    d = dn.get(key) if dn else None
    if d:
        print(f"dense\t{key[0]}\t{key[1]}\t{d.get('itl_ms_p50')}\t{d.get('aggregate_out_tok_per_s')}")
PYEOF
  } 2>&1 | tee -a "$MASTER"
}

case "$BATCH" in
  smoke)
    pin_npus
    run hamming 16384 1 HAMMING_TOPK=4096 NIAH=1
    ;;
  cline)
    pin_npus
    run dense   16384,32768 1,16 NIAH=1
    run hamming 16384,32768 1,16 HAMMING_TOPK=2048 NIAH=1
    run hamming 16384,32768 1,16 HAMMING_TOPK=4096 NIAH=1
    run hamming 16384,32768 1,16 HAMMING_TOPK=8192 NIAH=1
    run hammingsd 16384,32768 1,16 HAMMING_TOPK=4096
    run ngram   16384,32768 1,16
    ;;
  b2smoke)
    pin_npus
    # 1. THE BET: graph mode + static compaction. 16K/c1 bench; the NIAH grid
    #    covers BOTH tiers on the same serve (32K prompts also compact ->
    #    more events + 32K quality). Judgment: serve alive, [static-kv-compact]
    #    lines present with sane numbers, KV usage steps down, NIAH >= 0.9.
    run_graph compact 16384 1 NIAH=1
    # 2. Same-caliber latency anchor: graph dense with prefix caching off
    #    (matches run 1's serve caliber; isolates compaction as the variable).
    #    ITL delta ~0 at this corner is EXPECTED (weights dominate) - B3's
    #    32K/c16 cell is where the bandwidth win should show.
    EXTRA_SERVE_ARGS="--no-enable-prefix-caching"
    run_graph dense 16384 1
    unset EXTRA_SERVE_ARGS
    echo "" | tee -a "$MASTER"
    echo "b2smoke triage: if run 1 died during graph capture/replay, isolate the bet:" | tee -a "$MASTER"
    echo "  EXTRA_SERVE_ARGS='--enforce-eager' NPUS=\$NPUS bash research/run_baseline_npu.sh compact 8141" | tee -a "$MASTER"
    echo "  eager green + graph red = bet lost (paste serve log tail); eager red = deeper bug." | tee -a "$MASTER"
    ;;
  digest)
    ;;
  *) echo "unknown BATCH '$BATCH' (smoke|cline|b2smoke|digest)"; exit 1 ;;
esac

if [ "$BATCH" = "b2smoke" ]; then
  digest_b2
else
  digest
fi
