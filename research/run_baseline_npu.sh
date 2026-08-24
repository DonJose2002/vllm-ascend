#!/bin/bash
# Phase 0 baseline: NPU (Ascend 910B3) dense + SD benchmark, run inside the
# v0.23.0 Docker on the server. Laptop CANNOT run this (no NPU).
#
# IMPORTANT: the server is strictly no-export (files never leave it, see
# notes AGENTS.md section 3). Therefore this script prints a compact
# ===NPU_BASELINE_BEGIN/END=== summary block at the end. Copy that block
# (everything between the markers) and paste it back into the chat - it
# carries every field needed to backfill the comparison report. Do NOT
# try to copy JSON files or logs out.
#
# Prereq (server, once, inside or outside container):
#   cd /path/to/vllm-ascend && git fetch myfork && git checkout research/main && git pull
#
# Usage (inside container, adjust weight paths via env if needed):
#   bash research/run_baseline_npu.sh dense  8001
#   bash research/run_baseline_npu.sh sd     8002          # plan C, K=5
#   bash research/run_baseline_npu.sh sda    8003          # plan A, K=5
#   K=8 bash research/run_baseline_npu.sh sda 8004         # E1 K-sweep (capture table auto-derived)
#   bash research/run_baseline_npu.sh ngram  8005          # E2 zero-drafter differential
#   bash research/run_baseline_npu.sh eagle3 8006          # E2 (first-ever run; smoke first!)
#   bash research/run_baseline_npu.sh dflash 8007          # E2 (first-ever run; smoke first!)
#   SEED_PROFILE=repetitive bash research/run_baseline_npu.sh ngram 8008   # ngram-friendly workload
#   SAVE_TS=1 ...                                            # store token timestamps for R(t)
#   TIERS=4096 CONCS=1 bash research/run_baseline_npu.sh eagle3 8009       # smoke = 1 cell
#   EXTRA_SERVE_ARGS="--no-enable-prefix-caching" ...        # passthrough extra vllm serve flags (triage)
# Key envs: NPU_MODEL, DRAFT, EAGLE3_MODEL, DFLASH_MODEL, K, NGRAM_MAX/NGRAM_MIN,
#           TIERS, CONCS, NUM_PROMPTS, MAX_TOKENS, SEED_PROFILE, SAVE_TS, NPUS
set -euo pipefail

MODE="${1:?usage: run_baseline_npu.sh dense|sd|sda|ngram|eagle3|dflash PORT}"
PORT="${2:-8001}"
MODEL="${NPU_MODEL:-/nfs-share/hf_weights/Qwen3-8B}"
DRAFT="${NPU_DRAFT:-/nfs-share/hf_weights/Qwen3-0.6B}"
EAGLE3_MODEL="${EAGLE3_MODEL:-/nfs-share/hf_weights/qwen3_8b_eagle3}"
DFLASH_MODEL="${DFLASH_MODEL:-/nfs-share/hf_weights/Qwen3-8B-DFlash-b16}"
K="${K:-5}"
NGRAM_MAX="${NGRAM_MAX:-5}"
NGRAM_MIN="${NGRAM_MIN:-3}"
SEED_PROFILE="${SEED_PROFILE:-generic}"
TIERS="${NPU_TIERS:-4096,16384,32768}"
CONCS="${CONCS:-1,4,16}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
MAX_TOKENS="${MAX_TOKENS:-256}"
OUTDIR="${OUTDIR:-experiments/out}"
mkdir -p "$OUTDIR"

# --- pre-flight: guard against version drift. What matters is the pip-
# installed vllm-ascend distribution (that's what `vllm serve` imports via
# its console script; CWD does NOT affect it). The probe therefore runs
# from a neutral temp dir so the repo checkout under our feet cannot
# shadow the import (a `python3 -` heredoc run from the repo root always
# saw the raw source tree and false-positived).
# Rules: dist version must match EXPECT_VLLM_ASCEND (default 0.23.); if
# the import resolves inside this repo checkout (editable install), the
# served code follows the checked-out branch - fail unless
# ALLOW_REPO_INSTALL=1 (for intentional offline rebuilds from a tag).
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROBE_DIR="${PROBE_DIR:-$(mktemp -d)}"
if ! (cd "$PROBE_DIR" && python3 - "$REPO_DIR" "${EXPECT_VLLM_ASCEND:-0.23.}" <<'PYEOF'
import importlib.metadata, os, sys

repo_dir = os.path.realpath(sys.argv[1])
want_prefix = sys.argv[2]
try:
    dist_ver = importlib.metadata.version("vllm-ascend")
except importlib.metadata.PackageNotFoundError:
    print("PREFLIGHT-FAIL: no vllm-ascend distribution installed (run inside the v0.23.0 docker)")
    sys.exit(1)
try:
    import vllm_ascend
except Exception as e:
    print(f"PREFLIGHT-FAIL: vllm_ascend not importable from neutral cwd: {e!r}")
    sys.exit(1)
path = os.path.realpath(os.path.dirname(vllm_ascend.__file__))
print(f"PREFLIGHT: vllm-ascend dist {dist_ver}, import @ {path}")
ok = True
if not dist_ver.startswith(want_prefix):
    print(f"PREFLIGHT-FAIL: installed vllm-ascend dist version '{dist_ver}' does not start with '{want_prefix}'")
    ok = False
in_repo = path.startswith(repo_dir + os.sep)
if in_repo and os.environ.get("ALLOW_REPO_INSTALL") != "1":
    print("PREFLIGHT-FAIL: import resolves INSIDE this repo checkout (editable install).")
    print("  The served code follows the checked-out branch, not the installed version.")
    print("  Fix A (preferred): pip uninstall -y vllm-ascend  -> falls back to the image's")
    print("    site-packages install if it is still intact (verify with the probe again).")
    print("  Fix B (offline rebuild): git checkout v0.23.0 && rm -rf csrc/build &&")
    print("    pip install -e . --no-build-isolation  (D5-proven, 20-60min), then re-run")
    print("    this script with ALLOW_REPO_INSTALL=1 and keep the tag checked out.")
    ok = False
sys.exit(0 if ok else 1)
PYEOF
)
then
  echo "Pre-flight failed; aborting before serve. (see messages above)"
  exit 1
fi
DIST_VER="$(cd "$PROBE_DIR" && python3 -c 'import importlib.metadata as m; print(m.version("vllm-ascend"))' 2>/dev/null || echo '?')"
rm -rf "$PROBE_DIR"

# --- pre-flight 2: NPU selection (shared server).
# Explicit: NPUS=3 (or NPUS=4,5) -> export ASCEND_RT_VISIBLE_DEVICES.
# Unset: auto-pick the device with the lowest HBM usage from npu-smi info.
# Parser matches the real 25.3.rc1 layout: device header row "| <id> <name> |"
# followed by a Bus-Id row whose LAST "a / b" pair is HBM-Usage(MB). Free-card
# driver noise is ~3.4GB, real jobs show >=28GB, so selected devices must stay
# under HBM_MAX_USED_MB (default 6144) or we refuse to run on someone's card.
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
    m = re.match(r"^\|\s*(\d+)\s+\S+\s+\|\s*(OK|Warning|Alarm|Crit\w*|Unknown|Bad)", line)  # device header (NPU names like 910B3 start with a digit; health cell disambiguates from process rows)
    if m:
        cur = int(m.group(1))
        continue
    if cur is not None and re.search(r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f]", line):
        pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
        if pairs:
            used, _total = pairs[-1]  # last "a / b" on the Bus-Id row = HBM usage
            print(f"{cur} {used}")
            found = True
if not found:
    print("PARSE-FAIL: no device blocks parsed (unexpected npu-smi output format)")
PYEOF
}

DEV_LIST="$(npu_hbm_list)"
case "$DEV_LIST" in
  PARSE-FAIL*)
    echo "$DEV_LIST"
    echo "Cannot read NPU occupancy. Run 'npu-smi info', then re-run with NPUS=<id>."
    exit 1
    ;;
esac
HBM_MAX="${HBM_MAX_USED_MB:-6144}"

if [ -z "${NPUS:-}" ]; then
  PICK=$(echo "$DEV_LIST" | awk '{if(!($1 in mx)||$2>mx[$1])mx[$1]=$2} END{best="";for(id in mx){if(best==""||mx[id]<bestv){best=id;bestv=mx[id]}}print best,bestv}')
  NPUS="${PICK%% *}"; HBM_USED="${PICK##* }"
  echo "PREFLIGHT: auto-picked NPU $NPUS (HBM used ${HBM_USED}/65536 MB; threshold ${HBM_MAX})"
else
  HBM_USED=0
  for d in $(echo "$NPUS" | tr ',' ' '); do
    u=$(echo "$DEV_LIST" | awk -v id="$d" '$1==id{if($2>m)m=$2}END{if(m==""){print "MISSING"}else{print m}}')
    if [ "$u" = "MISSING" ]; then
      echo "PREFLIGHT-FAIL: device $d not found in npu-smi info"
      exit 1
    fi
    [ "$u" -gt "$HBM_USED" ] 2>/dev/null && HBM_USED="$u"
  done
  echo "PREFLIGHT: using NPU(s) $NPUS (max HBM used ${HBM_USED}/65536 MB; threshold ${HBM_MAX})"
fi
if [ "${HBM_USED:-0}" -gt "$HBM_MAX" ]; then
  echo "PREFLIGHT-FAIL: selected NPU(s) already hold ${HBM_USED} MB HBM (another job?)."
  echo "Pick a free one: NPUS=<id> bash research/run_baseline_npu.sh $MODE $PORT"
  exit 1
fi
export ASCEND_RT_VISIBLE_DEVICES="$NPUS"
echo "PREFLIGHT: ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"

SPEC_ARGS=""
TAG="npu-bf16-dense"
NOTE="910B3 x1, v0.23.0 docker, bf16, block_size=128; 32K tier NPU-only (model max_pos=40960)"
BENCH_EXTRA=()

if [ "$SEED_PROFILE" != "generic" ]; then
  TAG="$TAG-$SEED_PROFILE"
fi
if [ "${SAVE_TS:-0}" = "1" ]; then
  BENCH_EXTRA+=(--save-ts)
fi

# Bounded target capture table: (K+1)*i for i=1..8, R<=8 bench. Overridable.
derive_capture_sizes() {
  local step=$((K + 1)) sizes="" i=1 v
  while [ "$i" -le 8 ]; do
    v=$((step * i))
    sizes="${sizes:+$sizes,}$v"
    i=$((i + 1))
  done
  printf '%s' "$sizes"
}

case "$MODE" in
  dense)
    ;;
  sd|sda)
    # K+2 note: draft_model drafter consumes R*(K+2) tokens (extra seed slot);
    # plan C keeps it eager, plan A gives it a derived R*(K+2) table (PR #14510).
    SPEC_ARGS="--speculative-config {\"method\":\"draft_model\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$K}"
    TAG="npu-bf16-sd-k${K}"
    NOTE="$NOTE; plan C (drafter eager, default)"
    if [ "$MODE" = "sda" ]; then
      SPEC_ARGS="$SPEC_ARGS --additional-config {\"draft_model_full_graph\":true}"
      SPEC_ARGS="$SPEC_ARGS --compilation-config {\"cudagraph_capture_sizes\":[$(derive_capture_sizes)]}"
      TAG="npu-bf16-sd-k${K}-planA"
      NOTE="$NOTE -> OVERRIDDEN to plan A (draft_model_full_graph=true, capture table [$(derive_capture_sizes)])"
    fi
    ;;
  ngram)
    # Host-side prompt-lookup proposer: zero drafter forward, zero draft KV.
    # Keys verified against tests/e2e/.../test_ngram.py (prompt_lookup_max/min, no _k suffix).
    SPEC_ARGS="--speculative-config {\"method\":\"ngram\",\"prompt_lookup_max\":$NGRAM_MAX,\"prompt_lookup_min\":$NGRAM_MIN,\"num_speculative_tokens\":$K}"
    TAG="npu-bf16-ngram-k${K}"
    NOTE="$NOTE; ngram host proposer (lookup ${NGRAM_MIN}-${NGRAM_MAX}), default capture table (like dense)"
    ;;
  eagle3|dflash)
    # First-ever runs on this stack (code path exists, never validated here):
    # smoke with TIERS=4096 CONCS=1 before committing a full matrix. Bounded
    # capture table (same derivation as sda) to dodge EE1023 if drafter graphs
    # double like plan A; if capture still dies, retry with GRAPH_MODE
    # (e.g. GRAPH_MODE=FULL_DECODE_ONLY, the mode upstream e2e uses).
    DMODEL="$EAGLE3_MODEL"; [ "$MODE" = "dflash" ] && DMODEL="$DFLASH_MODEL"
    SPEC_ARGS="--speculative-config {\"method\":\"$MODE\",\"model\":\"$DMODEL\",\"num_speculative_tokens\":$K}"
    CCOMP="{\"cudagraph_capture_sizes\":[$(derive_capture_sizes)]"
    if [ -n "${GRAPH_MODE:-}" ]; then
      CCOMP="$CCOMP,\"cudagraph_mode\":\"$GRAPH_MODE\""
    fi
    CCOMP="$CCOMP}"
    SPEC_ARGS="$SPEC_ARGS --compilation-config $CCOMP"
    TAG="npu-bf16-${MODE}-k${K}"
    NOTE="$NOTE; $MODE drafter ($DMODEL), capture [$(derive_capture_sizes)]${GRAPH_MODE:+, mode=$GRAPH_MODE} - FIRST RUN, smoke first"
    ;;
  *)
    echo "unknown MODE '$MODE' (dense|sd|sda|ngram|eagle3|dflash)"; exit 1
    ;;
esac

strip_log() {
  sed -E 's/^\((APIServer|EngineCore) pid=[0-9]+\) //; s/(INFO|ERROR|WARNING) [0-9]{2}-[0-9]{2} [0-9:]{8} \[[^]]*\] //'
}

on_exit() {
  kill "$SERVE_PID" 2>/dev/null || true
  sleep 2
  echo ""
  echo "==================== COPY BELOW (between markers) ===================="
  echo "===NPU_BASELINE_BEGIN==="
  echo "mode=$TAG port=$PORT model=$MODEL"
  echo "visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-unset} hbm_used_mb=${HBM_USED:-?}"
  echo "repo_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "vllm_ascend_dist=$DIST_VER"
  echo "tiers=$TIERS concs=$CONCS nprompts=$NUM_PROMPTS maxtok=$MAX_TOKENS k=$K profile=$SEED_PROFILE"
  SERVE_LOG="$OUTDIR/serve-$TAG.log"
  if [ -s "$SERVE_LOG" ]; then
    grep -E "Available KV cache memory|GPU KV cache size|model weights take|Maximum concurrency|Wrapping draft model|drafter FULL graph enabled|drafter sizes|Capturing CUDA graphs" \
      "$SERVE_LOG" | tail -6 | strip_log | sed 's/^/cfg: /'
  else
    echo "cfg: (serve log empty or missing at $SERVE_LOG)"
  fi
  if [ -f "$OUTDIR/baseline-npu-qwen3-8b-$TAG.json" ]; then
    python3 research/bench_baseline.py summary "$OUTDIR/baseline-npu-qwen3-8b-$TAG.json"
  elif [ -s "$SERVE_LOG" ]; then
    echo "# bench json missing; EngineCore fatal block (if any):"
    grep -B2 -A45 "EngineCore encountered a fatal error" "$SERVE_LOG" \
      | tail -50 | strip_log | sed 's/^/  /'
    echo "# serve log tail (last 10 lines):"
    tail -10 "$SERVE_LOG" | strip_log | sed 's/^/  /'
  else
    echo "# bench json missing AND serve log empty/missing at $SERVE_LOG"
  fi
  echo "===NPU_BASELINE_END==="
  echo "==================== COPY ABOVE ===================="
}
trap on_exit EXIT

echo ">>> serving $TAG on :$PORT (model=$MODEL)"
# NOTE: the server's Qwen3-8B checkpoint has max_position_embeddings=40960
# (no yarn in its config.json), so 40960 is the hard ceiling. Do NOT set
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 to force more - RoPE beyond the derived
# max produces NaNs, and silent corruption is unacceptable for a baseline.
# Long tier is therefore 32K (NPU-only), not 64K.
vllm serve "$MODEL" \
  --served-model-name qwen3-8b \
  --max-model-len 40960 \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --port "$PORT" \
  $SPEC_ARGS \
  ${EXTRA_SERVE_ARGS:-} \
  > "$OUTDIR/serve-$TAG.log" 2>&1 &
SERVE_PID=$!

for i in $(seq 1 120); do
  sleep 10
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then break; fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "server died during startup (summary block below carries the tail)"
    break
  fi
done

if curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
  echo ">>> benching $TAG (tiers=$TIERS concs=$CONCS profile=$SEED_PROFILE)"
  python3 research/bench_baseline.py run \
    --base-url "http://127.0.0.1:$PORT" \
    --model qwen3-8b \
    --tag "$TAG" \
    --tiers "$TIERS" \
    --concs "$CONCS" \
    --num-prompts "$NUM_PROMPTS" \
    --max-tokens "$MAX_TOKENS" \
    --seed-profile "$SEED_PROFILE" \
    "${BENCH_EXTRA[@]:+${BENCH_EXTRA[@]}}" \
    --out "$OUTDIR/baseline-npu-qwen3-8b-$TAG.json" \
    --note "$NOTE" \
    || echo "# bench exited non-zero (partial summary follows)"
else
  echo ">>> server not up; skipping bench"
fi
