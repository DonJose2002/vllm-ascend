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

PIDS="$(vllm_tree_pids || true)"
if [ -z "$PIDS" ]; then
  say "no live vllm process in $CONTAINER (nothing to do)"
  exit 0
fi
# Stateless service: TERM the WHOLE tree at once (waiting politely on the API
# server alone stalls on keep-alive connections), give it TERM_WAIT seconds,
# then hard kill. Success criterion is the HBM drain below, not process-list
# politeness (zombies hold no resources and are excluded upstream).
# NOTE: the serve processes run as ROOT inside the container (= host uid 0);
# from the ops user's shell the kill MUST go through sudo as well - the
# 2026-09-02 "force kill did nothing" incident was exactly a bare `kill`
# getting EPERM silently swallowed by `2>/dev/null || true`.
say "TERM tree: $(echo "$PIDS" | tr '\n' ' ')"
if ! echo "$PIDS" | xargs -r sudo kill -TERM; then
  say "TERM failed (rc=$?) - sudo asking for a password? Run me from an interactive shell"
  exit 1
fi

deadline=$(( $(date +%s) + TERM_WAIT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep 5
  TREE="$(vllm_tree_pids || true)"
  [ -z "$TREE" ] && { say "tree exited gracefully"; break; }
  say "still dying (${TREE})... $(date +%H:%M:%S)"
done
if [ -n "${TREE:-}" ]; then
  say "graceful window (${TERM_WAIT}s) over - hard kill: $(echo "$TREE" | tr '\n' ' ')"
  if ! echo "$TREE" | xargs -r sudo kill -KILL; then
    say "KILL failed (rc=$?) - check sudo"
    exit 1
  fi
  sleep 5
  LEFT="$(vllm_tree_pids || true)"
  if [ -n "$LEFT" ]; then
    say "processes SURVIVED SIGKILL: $(echo "$LEFT" | tr '\n' ' ')"
    say "likely D-state (uninterruptible NPU driver cleanup) - wait, or last resort:"
    say "  sudo docker restart $CONTAINER"
    exit 1
  fi
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
