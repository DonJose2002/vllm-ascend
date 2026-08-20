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
#   bash research/run_baseline_npu.sh dense 8001
#   bash research/run_baseline_npu.sh sd    8002
set -euo pipefail

MODE="${1:?usage: run_baseline_npu.sh dense|sd PORT}"
PORT="${2:-8001}"
MODEL="${NPU_MODEL:-/nfs-share/hf_weights/Qwen3-8B}"
DRAFT="${NPU_DRAFT:-/nfs-share/hf_weights/Qwen3-0.6B}"
TIERS="${NPU_TIERS:-4096,16384,65536}"
CONCS="${CONCS:-1,4,16}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
MAX_TOKENS="${MAX_TOKENS:-256}"
OUTDIR="${OUTDIR:-experiments/out}"
mkdir -p "$OUTDIR"

# --- pre-flight: guard against version drift (D5 leftover editable installs
# could silently swap vllm_ascend to 0.22.1rc1 source inside this checkout,
# mixing with the image's 0.23.0 vllm). Must be 0.23.x AND resolved outside
# this repo dir; override expected version via EXPECT_VLLM_ASCEND if needed.
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if ! python3 - "$REPO_DIR" "${EXPECT_VLLM_ASCEND:-0.23.}" <<'PYEOF'
import os, sys
repo_dir, want_prefix = sys.argv[1], sys.argv[2]
try:
    import vllm_ascend
except ImportError:
    print(f"PREFLIGHT-FAIL: cannot import vllm_ascend (run inside the v0.23.0 docker)")
    sys.exit(1)
ver = getattr(vllm_ascend, "__version__", "?")
path = os.path.realpath(os.path.dirname(vllm_ascend.__file__))
print(f"PREFLIGHT: vllm_ascend {ver} @ {path}")
ok = True
if not ver.startswith(want_prefix):
    print(f"PREFLIGHT-FAIL: vllm_ascend version {ver} does not start with '{want_prefix}'")
    ok = False
if path.startswith(os.path.realpath(repo_dir) + os.sep):
    print(f"PREFLIGHT-FAIL: vllm_ascend resolves INSIDE this repo checkout (leftover "
          f"'pip install -e .'?). Use a fresh container or a clean checkout at another path.")
    ok = False
sys.exit(0 if ok else 1)
PYEOF
then
  echo "Pre-flight failed; aborting before serve. (see messages above)"
  exit 1
fi

SPEC_ARGS=""
TAG="npu-bf16-dense"
if [ "$MODE" = "sd" ]; then
  SPEC_ARGS="--speculative-config {\"method\":\"draft_model\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":5}"
  TAG="npu-bf16-sd-k5"
fi

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
  echo "repo_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "vllm_ascend_ver=$(python3 -c 'import vllm_ascend,sys; sys.stdout.write(getattr(vllm_ascend,"__version__","?"))' 2>/dev/null || echo '?')"
  echo "tiers=$TIERS concs=$CONCS nprompts=$NUM_PROMPTS maxtok=$MAX_TOKENS"
  grep -E "Available KV cache memory|GPU KV cache size|Maximum concurrency" \
    "$OUTDIR/serve-$TAG.log" 2>/dev/null | tail -4 | strip_log | sed 's/^/cfg: /'
  if [ -f "$OUTDIR/baseline-npu-qwen3-8b-$TAG.json" ]; then
    python3 research/bench_baseline.py summary "$OUTDIR/baseline-npu-qwen3-8b-$TAG.json"
  else
    echo "# bench json missing; serve log tail:"
    tail -15 "$OUTDIR/serve-$TAG.log" 2>/dev/null | strip_log | sed 's/^/  /'
  fi
  echo "===NPU_BASELINE_END==="
  echo "==================== COPY ABOVE ===================="
}
trap on_exit EXIT

echo ">>> serving $TAG on :$PORT (model=$MODEL)"
vllm serve "$MODEL" \
  --served-model-name qwen3-8b \
  --max-model-len 66048 \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --port "$PORT" \
  $SPEC_ARGS \
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
  echo ">>> benching $TAG (tiers=$TIERS concs=$CONCS)"
  python3 research/bench_baseline.py run \
    --base-url "http://127.0.0.1:$PORT" \
    --model qwen3-8b \
    --tag "$TAG" \
    --tiers "$TIERS" \
    --concs "$CONCS" \
    --num-prompts "$NUM_PROMPTS" \
    --max-tokens "$MAX_TOKENS" \
    --out "$OUTDIR/baseline-npu-qwen3-8b-$TAG.json" \
    --note "910B3 x1, v0.23.0 docker, bf16, block_size=128; 64K tier NPU-only" \
    || echo "# bench exited non-zero (partial summary follows)"
else
  echo ">>> server not up; skipping bench"
fi
