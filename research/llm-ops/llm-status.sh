#!/bin/bash
# llm-status.sh - one-glance service report: endpoint, recorded state, the
# two cards' HBM usage, and the tail of the in-container serve log.
set -euo pipefail
cd "$(dirname "$0")"
. ./llm-ops.conf
. ./llm-npu-lib.sh

[ -f ./llm-ops.secret ] && . ./llm-ops.secret
[ -n "$API_KEY" ] || { echo "[llm-status] API_KEY empty - create ./llm-ops.secret (see llm-ops.conf header)"; exit 1; }

if curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" -H "Authorization: Bearer $API_KEY" 2>/dev/null \
   | grep -q "\"$SERVED_NAME\""; then
  echo "service : UP on :$PORT (model $SERVED_NAME)"
else
  echo "service : DOWN (or :$PORT not answering)"
fi

if [ -f llm-ops.state ]; then
  echo "state   : $(cat llm-ops.state)"
  for d in $(state_cards | tr ',' ' '); do
    echo "card $d  : $(card_hbm "$d" || echo '?') MB HBM used"
  done
else
  echo "state   : (never started from this host dir)"
fi

echo "--- serve log tail (inside $CONTAINER:$LOG) ---"
docker exec "$CONTAINER" tail -5 "$LOG" 2>/dev/null | sed 's/^/log: /' || echo "log: (unavailable)"
