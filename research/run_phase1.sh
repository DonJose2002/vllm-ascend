#!/usr/bin/env bash
# Phase 1 batch driver (2026-08-25). Wraps run_baseline_npu.sh over the E1/E2/E3
# matrices from experiments/phase1-drafter-cost-design.md so one command per night
# runs unattended; collects every SUMMARY block into a master log and prints a
# paste-ready digest (per-JSON TSV + kregress/diff/KV-tax lines) at the end.
#
# Usage:  bash research/run_phase1.sh e1|e2|e3|p15|all|digest [start_port]
# Night 1: e1 (K sweep, 16 cells, ~1h)      Night 2: e2 (22 cells, ~2h)
# p15: Phase 1.5 tax probe (design: experiments/phase1.5-tax-probe-design.md in
#      the notes repo) - T2 smoke (dense+profiler, biggest unknown FIRST) ->
#      T1 ngram K in {1,3,8} (K=5 reuses E2; the no-profiler differential) ->
#      T2 full (ngram/eagle3). ~10 runs, one night. Runbook: research/p15-runbook.md.
# E3 requires Qwen3-1.7B at $NPU_DRAFT17 (default /nfs-share/hf_weights/Qwen3-1.7B).
# digest: analysis only (summary + kregress + diff + KV tax) over whatever
#         JSONs already sit under experiments/out/phase1 - no serves, no card.
#
# Layout decision: all Phase-1 output goes to experiments/out/phase1/ (E3 under
# e3-1p7b/) so Phase-0 JSONs in experiments/out/ are never clobbered - the E2
# planA-k5 and E3 sda-k5 tags would otherwise reuse Phase-0 filenames.
#
# [incident 2026-08-25] first e1 run wedged the shared server. Root cause
# candidates (unconfirmed, host died): the per-run teardown (kill SERVE_PID +
# sleep 2) underestimates engine death time, so back-to-back serves overlap and
# stack host RAM (weights + pinned pools + compile caches per cold start); a
# crash core-dump of a huge engine could also fill the disk. Guards added
# below - same family of lesson as the D5 `-j=192` incident: batch drivers
# must model resource dimensions, not just wall time.
set -uo pipefail

BATCH="${1:?usage: run_phase1.sh e1|e2|e3|p15|all|digest [start_port]}"
PORT="${2:-8110}"
OUTROOT="${OUTROOT:-experiments/out/phase1}"
NPU_DRAFT17="${NPU_DRAFT17:-/nfs-share/hf_weights/Qwen3-1.7B}"
mkdir -p "$OUTROOT"
MASTER="$OUTROOT/phase1-${BATCH}-$(date +%Y%m%d-%H%M).log"

# --- resource safety guards (see incident note above) ---
ulimit -c 0 2>/dev/null || true   # never core-dump a huge engine onto shared disks
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-16}"
FREE_MEM_MIN_MB="${FREE_MEM_MIN_MB:-40000}"   # host available-RAM floor before each serve
DRAIN_WAIT_MAX="${DRAIN_WAIT_MAX:-900}"       # max seconds to wait for the card to drain
HBM_DRAINED_MB="${HBM_DRAINED_MB:-6144}"      # same threshold the harness preflight uses

# Guard: never let the race-research envs leak into Phase-1 measurements.
unset VLLM_ASCEND_SD_REVIVE_RACE VLLM_ASCEND_SD_COUNTERS VLLM_ASCEND_SD_DEBUG || true

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
  # One card for the WHOLE batch: without pinning, each child invocation
  # re-runs auto-pick and can hop onto a different card when the previous
  # engine is still dying - hiding the overlap instead of preventing it.
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
    [ -z "$avail" ] && return 0  # free unreadable -> skip guard rather than block
    [ "$avail" -ge "$FREE_MEM_MIN_MB" ] && return 0
    echo "WAIT: host available RAM ${avail}MB < ${FREE_MEM_MIN_MB}MB $(date +%H:%M:%S)" | tee -a "$MASTER"
    sleep 30
  done
}

drain_card() {
  # Wait until OUR pinned card actually released its HBM before the next
  # serve claims ~58GB - the sleep-2 of the child teardown is nowhere near
  # enough for an engine to die. Timeout -> hard-kill anything still holding
  # our port (port is unique per run, so the match only ever hits our own
  # processes on this shared server).
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

# run <mode> <tiers> <concs> [env=val ...] - one run_baseline_npu.sh invocation
run() {
  local mode="$1" tiers="$2" concs="$3"; shift 3
  local outdir="$OUTROOT"
  if [ -n "${RUN_SUBDIR:-}" ]; then outdir="$OUTROOT/$RUN_SUBDIR"; mkdir -p "$outdir"; fi
  PORT=$((PORT + 1))
  banner "RUN mode=$mode tiers=$tiers concs=$concs port=$PORT outdir=$outdir envs=$* $(date +%H:%M:%S)"
  wait_host_ram
  env -u VLLM_ASCEND_SD_REVIVE_RACE -u VLLM_ASCEND_SD_COUNTERS -u VLLM_ASCEND_SD_DEBUG \
    "$@" TIERS="$tiers" CONCS="$concs" SAVE_TS=1 OUTDIR="$outdir" \
    bash research/run_baseline_npu.sh "$mode" "$PORT" 2>&1 | tee -a "$MASTER"
  # post-run: catch a still-dying API server by our unique port, then wait
  # for the pinned card to actually drain before the next serve claims it
  pkill -TERM -f "vllm serve.*--port $PORT" 2>/dev/null || true
  drain_card
}

digest() {
  banner "DIGEST (phase1 $BATCH) $(date +%H:%M:%S)"
  local jsons
  jsons=$(ls "$OUTROOT"/*.json "$OUTROOT"/*/*.json 2>/dev/null | sort -u)
  [ -z "$jsons" ] && { echo "no JSONs found under $OUTROOT" | tee -a "$MASTER"; return; }
  # keep only files that exist: a missing member must degrade the digest,
  # never crash it (phase1 incident 2026-08-25: the overwritten ngram file
  # made the hardcoded diff list throw FileNotFoundError and took the whole
  # digest section down with it)
  local k_files=() d_files=() f
  for f in \
    "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-sd-k1-planA.json \
    "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-sd-k3-planA.json \
    "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-sd-k5-planA.json \
    "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-sd-k8-planA.json; do
    [ -f "$f" ] && k_files+=("$f")
  done
  for f in \
    "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-dense.json \
    "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-sd-k5-planA.json \
    "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-ngram-k5.json \
    "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-ngram-k5-repetitive.json \
    "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-eagle3-k5.json \
    "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-dflash-k5.json; do
    [ -f "$f" ] && d_files+=("$f")
  done
  {
    echo "--- per-file TSV ---"
    python3 research/bench_baseline.py summary $jsons
    echo
    if [ "${#k_files[@]}" -ge 2 ]; then
      echo "--- E1 kregress ---"
      python3 research/bench_baseline.py kregress "${k_files[@]}" 2>&1
    fi
    if [ "${#d_files[@]}" -ge 2 ]; then
      echo "--- E2 diff (${#d_files[@]} files) ---"
      python3 research/bench_baseline.py diff "${d_files[@]}" 2>&1
    fi
    if [ "$BATCH" = "p15" ]; then
      # T1: ngram K regression (K=5 from E2 + k1/k3/k8 from this batch)
      local nk=() f
      for f in "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-ngram-k{1,3,5,8}.json; do
        [ -f "$f" ] && nk+=("$f")
      done
      if [ "${#nk[@]}" -ge 2 ]; then
        echo "--- p15 T1 ngram kregress (${#nk[@]} files) ---"
        python3 research/bench_baseline.py kregress "${nk[@]}" 2>&1
      fi
      # T2: cross-config category diff (server-side aggregation only)
      local pd="$OUTROOT/p15-prof" dargs=() m
      for m in dense ngram-k5 eagle3-k5; do
        [ -d "$pd/prof-npu-bf16-$m" ] && dargs+=("$m=$pd/prof-npu-bf16-$m")
      done
      if [ "${#dargs[@]}" -ge 2 ]; then
        echo "--- p15 T2 category diff (${#dargs[@]} configs) ---"
        python3 research/profile_step_breakdown.py \
          --steps $((${PROFILER_STEPS:-40} * ${PROFILER_ROUNDS:-2})) \
          --diff "${dargs[@]}" 2>&1
      fi
    fi
    echo
    echo "--- E4 KV tax (per serve log) ---"
    grep -H "GPU KV cache size" "$OUTROOT"/serve-*.log "$OUTROOT"/*/serve-*.log 2>/dev/null \
      | sed 's/(EngineCore pid=[0-9]*) //' | sort -u
  } 2>&1 | tee -a "$MASTER"
}

{
  banner "PHASE1 BATCH=$BATCH host=$(hostname) $(date)"
  echo "repo: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD) - $(git log -1 --format=%s)"
  echo "model: ${NPU_MODEL:-/nfs-share/hf_weights/Qwen3-8B} draft17: $NPU_DRAFT17"
  echo "guards: FREE_MEM_MIN_MB=$FREE_MEM_MIN_MB DRAIN_WAIT_MAX=${DRAIN_WAIT_MAX}s HBM_DRAINED_MB=$HBM_DRAINED_MB"
} | tee -a "$MASTER"
# digest-only mode runs nothing -> no card pinning needed
[ "$BATCH" != "digest" ] && pin_npus

case "$BATCH" in
  e1|all)
    # E1: K sweep, planA. NUM_PROMPTS stays 8 -> R<=8, the auto-derived
    # (K+1)*1..8 tables cover R*(K+1)/(K+2) for every K in {1,3,5,8}.
    for K in 1 3 5 8; do
      run sda "4096,16384" "1,16" K="$K"
    done
    ;;
esac

case "$BATCH" in
  e2|all)
    # E2: method spectrum at K=5. generic profile for cross-method comparison,
    # repetitive only where it matters (ngram's accept needs it; c1 = cleanest
    # accept measurement, accB distorts at c16).
    run dense      "4096,16384" "1,16"
    run sda        "4096,16384" "1,16" K=5
    run ngram      "4096,16384" "1,16"
    run ngram      "4096,16384" "1"    SEED_PROFILE=repetitive
    run eagle3     "4096,16384" "1,16"
    run dflash     "4096,16384" "1,16"
    ;;
esac

case "$BATCH" in
  e3|all)
    if [ -d "$NPU_DRAFT17" ]; then
      RUN_SUBDIR=e3-1p7b run sda "4096,16384" "1,16" K=5 NPU_DRAFT="$NPU_DRAFT17"
      unset RUN_SUBDIR  # bash keeps prefix-assignments after function calls
    else
      banner "E3 SKIPPED: $NPU_DRAFT17 not present (pull Qwen/Qwen3-1.7B first)"
    fi
    ;;
esac

case "$BATCH" in
  p15)
    # Phase 1.5 (design: experiments/phase1.5-tax-probe-design.md, notes repo).
    # Order per design section 8: T2 SMOKE first - the torch-profiler chain on
    # the pinned vllm v0.23.0 (router gating, TorchNPUProfilerWrapper, trace
    # export) is the biggest unknown; if smoke fails, T1 still runs (profiler
    # not involved) and T2 degrades per the design's fallback. Profiler runs
    # are isolated under p15-prof/ so E2 serve logs (KV-tax evidence) are never
    # clobbered - same layout discipline as e3-1p7b.
    RUN_SUBDIR=p15-prof run dense "4096" "1" PROFILER=1 PROFILE_ONLY=1
    unset RUN_SUBDIR
    # T1: ngram K sweep, K=5 reuses the E2 run. Plain runs (no profiler).
    for K in 1 3 8; do
      run ngram "4096,16384" "1" K="$K"
    done
    # T2 full: ngram + eagle3 under the profiler (dense came from the smoke).
    RUN_SUBDIR=p15-prof run ngram "4096" "1" K=5 PROFILER=1 PROFILE_ONLY=1
    RUN_SUBDIR=p15-prof run eagle3 "4096" "1" K=5 PROFILER=1 PROFILE_ONLY=1
    unset RUN_SUBDIR
    ;;
esac

digest
banner "DONE phase1 $BATCH - master log: $MASTER"
echo ">>> paste back: the DIGEST section + any cell whose SUMMARY shows fail>0 or a fatal block"
