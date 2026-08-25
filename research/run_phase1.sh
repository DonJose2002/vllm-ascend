#!/usr/bin/env bash
# Phase 1 batch driver (2026-08-25). Wraps run_baseline_npu.sh over the E1/E2/E3
# matrices from experiments/phase1-drafter-cost-design.md so one command per night
# runs unattended; collects every SUMMARY block into a master log and prints a
# paste-ready digest (per-JSON TSV + kregress/diff/KV-tax lines) at the end.
#
# Usage:  bash research/run_phase1.sh e1|e2|e3|all [start_port]
# Night 1: e1 (K sweep, 16 cells, ~1h)      Night 2: e2 (22 cells, ~2h)
# E3 requires Qwen3-1.7B at $NPU_DRAFT17 (default /nfs-share/hf_weights/Qwen3-1.7B).
#
# Layout decision: all Phase-1 output goes to experiments/out/phase1/ (E3 under
# e3-1p7b/) so Phase-0 JSONs in experiments/out/ are never clobbered - the E2
# planA-k5 and E3 sda-k5 tags would otherwise reuse Phase-0 filenames.
set -uo pipefail

BATCH="${1:?usage: run_phase1.sh e1|e2|e3|all [start_port]}"
PORT="${2:-8110}"
OUTROOT="${OUTROOT:-experiments/out/phase1}"
NPU_DRAFT17="${NPU_DRAFT17:-/nfs-share/hf_weights/Qwen3-1.7B}"
mkdir -p "$OUTROOT"
MASTER="$OUTROOT/phase1-${BATCH}-$(date +%Y%m%d-%H%M).log"

# Guard: never let the race-research envs leak into Phase-1 measurements.
unset VLLM_ASCEND_SD_REVIVE_RACE VLLM_ASCEND_SD_COUNTERS VLLM_ASCEND_SD_DEBUG || true

banner() { printf '\n===== %s =====\n' "$*" | tee -a "$MASTER"; }

# run <mode> <tiers> <concs> [env=val ...] - one run_baseline_npu.sh invocation
run() {
  local mode="$1" tiers="$2" concs="$3"; shift 3
  local outdir="$OUTROOT"
  if [ -n "${RUN_SUBDIR:-}" ]; then outdir="$OUTROOT/$RUN_SUBDIR"; mkdir -p "$outdir"; fi
  PORT=$((PORT + 1))
  banner "RUN mode=$mode tiers=$tiers concs=$concs port=$PORT outdir=$outdir envs=$* $(date +%H:%M:%S)"
  env -u VLLM_ASCEND_SD_REVIVE_RACE -u VLLM_ASCEND_SD_COUNTERS -u VLLM_ASCEND_SD_DEBUG \
    "$@" TIERS="$tiers" CONCS="$concs" SAVE_TS=1 OUTDIR="$outdir" \
    bash research/run_baseline_npu.sh "$mode" "$PORT" 2>&1 | tee -a "$MASTER"
  sleep 5  # let the port settle before the next serve
}

digest() {
  banner "DIGEST (phase1 $BATCH) $(date +%H:%M:%S)"
  local jsons
  jsons=$(ls "$OUTROOT"/*.json "$OUTROOT"/*/*.json 2>/dev/null | sort -u)
  [ -z "$jsons" ] && { echo "no JSONs found under $OUTROOT" | tee -a "$MASTER"; return; }
  {
    echo "--- per-file TSV ---"
    python3 research/bench_baseline.py summary $jsons
    echo
    case "$BATCH" in
      e1|all)
        echo "--- E1 kregress ---"
        python3 research/bench_baseline.py kregress \
          "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-sd-k1-planA.json \
          "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-sd-k3-planA.json \
          "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-sd-k5-planA.json \
          "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-sd-k8-planA.json 2>&1
        ;;
    esac
    case "$BATCH" in
      e2|all)
        echo "--- E2 diff (dense | planA | ngram | eagle3 | dflash) ---"
        python3 research/bench_baseline.py diff \
          "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-dense.json \
          "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-sd-k5-planA.json \
          "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-ngram-k5.json \
          "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-ngram-k5-repetitive.json \
          "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-eagle3-k5.json \
          "$OUTROOT"/baseline-npu-qwen3-8b-npu-bf16-dflash-k5.json 2>&1
        ;;
    esac
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
} | tee -a "$MASTER"

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

digest
banner "DONE phase1 $BATCH - master log: $MASTER"
echo ">>> paste back: the DIGEST section + any cell whose SUMMARY shows fail>0 or a fatal block"
