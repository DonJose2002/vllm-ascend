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
# Pick the TP lowest-HBM cards AMONG FREE ONES (<= HBM_FREE_MB). Returns
# 0 = picked (PICK set), 2 = insufficient free cards, 1 = npu-smi unparseable.
select_cards() {
  local dev_list free_list n_free
  dev_list="$(npu_hbm_list)"
  case "$dev_list" in
    PARSE-FAIL*|"") say "cannot parse npu-smi: ${dev_list:-empty}"; say "fall back: NPUS=4,5 $0"; return 1 ;;
  esac
  free_list="$(echo "$dev_list" | awk -v th="$HBM_FREE_MB" '$2 + 0 <= th' | sort -k2 -n)"
  n_free=$(echo "$free_list" | grep -c . || true)
  if [ "${n_free:-0}" -lt "$TP" ]; then
    say "insufficient free cards: ${n_free:-0}/$TP (threshold ${HBM_FREE_MB}MB) - current occupancy:"
    echo "$dev_list" | sort -k2 -n | sed 's/^/  npu: /'
    return 2
  fi
  PICK="$(echo "$free_list" | head -"$TP" | awk '{print $1}' | paste -sd, -)"
  say "picked $TP lowest-HBM free cards: $PICK ($(echo "$free_list" | head -"$TP" | tr '\n' ' '))"
}

if [ -n "${NPUS:-}" ]; then
  PICK="$NPUS"
  say "using caller-specified cards: $PICK"
else
  rc=0; select_cards || rc=$?
  if [ "$rc" -eq 2 ] && [ -n "${SQUAT_CONTAINER:-}" ]; then
    say "insufficient free cards - restarting the group occupancy container '$SQUAT_CONTAINER' (sanctioned) and retrying"
    if $DOCKER restart "$SQUAT_CONTAINER" >/dev/null 2>&1; then
      # HBM release after the container restart takes a moment to show up
      # in npu-smi; poll for up to ~1 min before declaring truly no cards.
      ok=""
      for _ in 1 2 3 4 5 6; do
        sleep 10
        rc=0; select_cards || rc=$?
        [ "$rc" -eq 0 ] && { ok=1; break; }
        [ "$rc" -eq 1 ] && break   # parser broke, no point retrying
      done
      [ -n "$ok" ] || { say "still insufficient after restarting $SQUAT_CONTAINER - the cards are genuinely busy"; exit 1; }
    else
      say "failed to restart $SQUAT_CONTAINER - continuing with manual selection only"
      exit 1
    fi
  elif [ "$rc" -ne 0 ]; then
    exit 1
  fi
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
TOOL_ARGS=""
if [ "${ENABLE_TOOLS:-1}" = "1" ]; then
  TOOL_ARGS="--enable-auto-tool-choice --tool-call-parser $TOOL_PARSER"
fi
if [ -n "${REASONING_PARSER:-}" ]; then
  TOOL_ARGS="$TOOL_ARGS --reasoning-parser $REASONING_PARSER"
fi
[ -n "$TOOL_ARGS" ] && say "agent tooling flags: $TOOL_ARGS"

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
      $TOOL_ARGS \
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
