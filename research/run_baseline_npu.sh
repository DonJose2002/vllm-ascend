#!/bin/bash
# Phase 0 baseline: NPU (Ascend 910B3) dense + SD benchmark, run inside the
# v0.23.0 Docker on the server. Laptop CANNOT run this (no NPU).
#
# Prereq (server, once):
#   cd /path/to/vllm-ascend && git fetch myfork && git checkout research/main && git pull
#   (bench_baseline.py lives in research/)
#
# Usage (inside container, adjust container name / weight paths if needed):
#   bash research/run_baseline_npu.sh dense  8001
#   bash research/run_baseline_npu.sh sd     8002
# Then copy experiments/out/*.json back to the laptop notes repo experiments/.
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

SPEC_ARGS=""
TAG="npu-bf16-dense"
if [ "$MODE" = "sd" ]; then
  SPEC_ARGS="--speculative-config {\"method\":\"draft_model\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":5}"
  TAG="npu-bf16-sd-k5"
fi

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
    echo "server died, tail of log:"; tail -30 "$OUTDIR/serve-$TAG.log"; exit 1
  fi
done

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
  --note "910B3 x1, vllm-ascend v0.23.0 docker, bf16 weights, block_size=128; 64K tier included (NPU-only)"

kill "$SERVE_PID" 2>/dev/null || true
echo ">>> done: $OUTDIR/baseline-npu-qwen3-8b-$TAG.json"
