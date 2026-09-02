#!/bin/bash
# llm-start.sh - host-side launcher for the company LLM service.
#
# Architecture (user decision 2026-09-02): scripts live on the HOST and
# remote-control the service container via docker exec; nothing is stored
# inside the container (it stays minimal, scripts survive container
# recreation). Card choice = auto-pick the 2 lowest-HBM cards at launch
# time; override with NPUS=4,5.
#
# Flow: container up? -> already serving? -> port free? -> pick 2 cards ->
# docker exec -d (vllm serve, FULL graph mode - see manual §3.5) -> poll
# /v1/models until UP (55GB weight load takes minutes) -> record state.
set -euo pipefail
cd "$(dirname "$0")"
. ./llm-ops.conf
. ./llm-npu-lib.sh

say() { echo "[llm-start] $*"; }

# API key comes from the gitignored llm-ops.secret (public repo - no secrets in git)
[ -f ./llm-ops.secret ] && . ./llm-ops.secret
[ -n "$API_KEY" ] || { say "API_KEY empty - create ./llm-ops.secret next to the scripts (see llm-ops.conf header)"; exit 1; }

if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
  say "container $CONTAINER is not running (start it first, manual §2.5)"
  exit 1
fi

if curl -s -m 5 -o /dev/null "http://127.0.0.1:$PORT/v1/models" -H "Authorization: Bearer $API_KEY"; then
  say "already serving on :$PORT (nothing to do)"
  exit 0
fi

if ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":$PORT$"; then
  say "port $PORT is listening but /v1/models does not answer - another service holds it?"
  exit 1
fi

# --- card selection ---
if [ -n "${NPUS:-}" ]; then
  PICK="$NPUS"
  say "using caller-specified cards: $PICK"
else
  DEV_LIST="$(npu_hbm_list)"
  case "$DEV_LIST" in
    PARSE-FAIL*|"") say "cannot parse npu-smi: ${DEV_LIST:-empty}"; say "fall back: NPUS=4,5 $0"; exit 1 ;;
  esac
  PICK="$(echo "$DEV_LIST" | sort -k2 -n | head -2 | awk '{print $1}' | paste -sd, -)"
  say "auto-picked 2 lowest-HBM cards: $PICK ($(echo "$DEV_LIST" | sort -k2 -n | head -2 | tr '\n' ' '))"
fi
NPICK=$(echo "$PICK" | tr ',' '\n' | grep -c .)
[ "$NPICK" -eq "$TP" ] || { say "need $TP cards, got '$PICK'"; exit 1; }
for d in $(echo "$PICK" | tr ',' ' '); do
  used="$(card_hbm "$d")"
  if [ -n "$used" ] && [ "$used" -gt "$HBM_FREE_MB" ]; then
    say "card $d already holds ${used}MB HBM (> ${HBM_FREE_MB}) - someone's job is there"
    say "either pick others (NPUS=x,y $0) or wait"
    exit 1
  fi
done

# --- launch inside the container (detached; logs to $LOG inside container) ---
docker exec -d \
  -e ASCEND_RT_VISIBLE_DEVICES="$PICK" \
  -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
  "$CONTAINER" \
  bash -c "vllm serve $MODEL_DIR \
      --served-model-name $SERVED_NAME \
      --tensor-parallel-size $TP \
      --max-model-len $MAX_MODEL_LEN \
      --gpu-memory-utilization $GPU_MEM_UTIL \
      --compilation-config '{\"cudagraph_mode\":\"FULL\"}' \
      --api-key '$API_KEY' --port $PORT \
      > $LOG 2>&1"

echo "$(date '+%F %T') cards=$PICK port=$PORT" > llm-ops.state
say "launched in $CONTAINER (cards $PICK), waiting for :$PORT ..."

# --- health wait: weights load + FULL graph capture take minutes ---
deadline=$(( $(date +%s) + 1500 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep 10
  if curl -s -m 5 -o /dev/null "http://127.0.0.1:$PORT/v1/models" -H "Authorization: Bearer $API_KEY"; then
    say "UP on :$PORT (cards $PICK) - company URL: http://<server-ip>:$PORT/v1"
    docker exec "$CONTAINER" grep -m2 -E "Available KV cache memory|GPU KV cache size" "$LOG" 2>/dev/null | sed 's/^/kv: /' || true
    exit 0
  fi
  # died during startup? (docker top reads host-side, no container deps)
  if ! docker top "$CONTAINER" -o pid,cmd 2>/dev/null | grep -q "vllm serve"; then
    say "serve process died during startup - last 25 log lines:"
    docker exec "$CONTAINER" tail -25 "$LOG" 2>/dev/null | sed 's/^/  /'
    exit 1
  fi
done
say "TIMEOUT after 25min - last 25 log lines:"
docker exec "$CONTAINER" tail -25 "$LOG" 2>/dev/null | sed 's/^/  /'
exit 1
