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

if ! $DOCKER inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
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
# ASCEND_RT_VISIBLE_DEVICES REQUIRES ascending device ids (CANN constraint,
# unlike CUDA's order-defines-mapping). 2026-09-02 incident: HBM-sorted pick
# produced "2,1" and every worker died at aclInit 107001 / "Invalid device
# ID" at rtSetDefaultDeviceId(0). Normalize BOTH auto-picked and NPUS input.
PICK="$(echo "$PICK" | tr ',' '\n' | sed 's/ //g' | sort -n | uniq | paste -sd, -)"
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
$DOCKER exec -d \
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

# Failure-path log dump: NPU error dumps are LONG and the interesting
# Python exception sits ABOVE the device-spam tail - so print (a) from the
# LAST "Traceback" line to EOF (bounded), (b) any ERROR/EE/EZ device lines.
dump_fail_log() {
  say "--- last traceback to EOF (max 60 lines) ---"
  $DOCKER exec "$CONTAINER" sh -c \
    "awk '/Traceback/{buf=\"\"} {buf=buf \$0 \"\n\"} END{printf \"%s\", buf}' $LOG | tail -60" \
    2>/dev/null | sed 's/^/  /' || true
  say "--- error lines ---"
  $DOCKER exec "$CONTAINER" sh -c \
    "grep -nE 'ERROR|Traceback|EE[0-9]{4}|EZ[0-9]{4}' $LOG | tail -15" \
    2>/dev/null | sed 's/^/  /' || true
  say "--- raw tail (10) ---"
  $DOCKER exec "$CONTAINER" tail -10 "$LOG" 2>/dev/null | sed 's/^/  /' || true
}

# --- health wait: weights load + FULL graph capture take minutes ---
deadline=$(( $(date +%s) + 1500 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep 10
  if curl -s -m 5 -o /dev/null "http://127.0.0.1:$PORT/v1/models" -H "Authorization: Bearer $API_KEY"; then
    say "UP on :$PORT (cards $PICK) - company URL: http://<server-ip>:$PORT/v1"
    $DOCKER exec "$CONTAINER" grep -m2 -E "Available KV cache memory|GPU KV cache size" "$LOG" 2>/dev/null | sed 's/^/kv: /' || true
    exit 0
  fi
  # died during startup? (docker top reads host-side, no container deps)
  if [ -z "$(serve_pids)" ]; then
    say "serve process died during startup"
    dump_fail_log
    exit 1
  fi
done
say "TIMEOUT after 25min"
dump_fail_log
exit 1
