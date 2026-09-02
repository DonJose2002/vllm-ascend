#!/bin/bash
# llm-stop.sh - stop the serve processes and give the cards back.
#
# TERM first (graceful), wait for the processes to disappear (engine death
# takes tens of seconds - the research line's drain discipline), hard-kill
# after 300s, then poll npu-smi until BOTH recorded cards release their HBM
# (<= HBM_FREE_MB) so the next user (or the next llm-start) gets clean cards.
set -euo pipefail
cd "$(dirname "$0")"
. ./llm-ops.conf
. ./llm-npu-lib.sh

say() { echo "[llm-stop] $*"; }

# serve_pids / vllm_tree_pids come from llm-npu-lib.sh (via $DOCKER).

PIDS="$(serve_pids || true)"
if [ -z "$PIDS" ]; then
  if [ -n "$(vllm_tree_pids || true)" ]; then
    # No main serve process but workers linger (crashed API server?): kill the tree.
    PIDS="$(vllm_tree_pids)"
    say "main serve gone, orphaned vllm tree found: $(echo "$PIDS" | tr '\n' ' ')"
  else
    say "no serve process found in $CONTAINER (nothing to do)"
    exit 0
  fi
fi
say "TERM: $(echo "$PIDS" | tr '\n' ' ')"

# The serve was started via docker exec, so its processes are children of the
# container's init; killing them by host PID (from docker top) is correct and
# leaves the container itself running.
echo "$PIDS" | xargs -r kill -TERM 2>/dev/null || true

deadline=$(( $(date +%s) + 300 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep 10
  TREE="$(vllm_tree_pids || true)"
  [ -z "$TREE" ] && break
  say "still dying: $(echo "$TREE" | tr '\n' ' ')"
done
if [ -n "${TREE:-}" ]; then
  say "still alive after 300s - hard kill: $(echo "$TREE" | tr '\n' ' ')"
  echo "$TREE" | xargs -r kill -KILL 2>/dev/null || true
  sleep 10
fi

# --- drain check on the cards recorded at start time ---
CARDS="$(state_cards)"
if [ -n "$CARDS" ]; then
  say "waiting for cards $CARDS to drain (<= ${HBM_FREE_MB}MB, max ${DRAIN_WAIT_MAX}s)"
  deadline=$(( $(date +%s) + DRAIN_WAIT_MAX ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    busy=0
    for d in $(echo "$CARDS" | tr ',' ' '); do
      used="$(card_hbm "$d")"
      if [ -n "$used" ] && [ "$used" -gt "$HBM_FREE_MB" ]; then
        busy=1
        say "card $d still at ${used}MB $(date +%H:%M:%S)"
        break
      fi
    done
    [ "$busy" -eq 0 ] && { say "drained - cards $CARDS released"; exit 0; }
    sleep 15
  done
  say "DRAIN-TIMEOUT: cards still busy after ${DRAIN_WAIT_MAX}s (check npu-smi manually)"
  exit 1
fi
say "serve stopped (no card state recorded; skip drain poll)"
