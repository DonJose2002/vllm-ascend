#!/bin/bash
# llm-npu-lib.sh - shared helpers for the llm-ops scripts (host-side).
# Card parsing follows the SAME health-cell parser proven against the real
# npu-smi 25.3.rc1 output in the research line (run_baseline_npu.sh):
#   - device header row "| <id> <name> | <health> |" (names like 910B3 start
#     with a DIGIT, so the health cell disambiguates device rows from others)
#   - the Bus-Id row under it carries "a / b" pairs; the LAST pair is HBM MB.
# Free-card driver noise is ~3.4GB; real jobs show >=28GB.

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
    m = re.match(r"^\|\s*(\d+)\s+\S+\s+\|\s*(OK|Warning|Alarm|Crit\w*|Unknown|Bad)", line)
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

card_hbm() { npu_hbm_list | awk -v id="$1" '$1==id{print $2}'; }

# Echo the two lowest-HBM device ids, comma-separated ("4,5").
# Assumes npu_hbm_list already validated (caller must handle PARSE-FAIL).
pick_two_cards() {
  npu_hbm_list | sort -k2 -n | head -2 | awk '{print $1}' | paste -sd, -
}

# Cards recorded by llm-start.sh in llm-ops.state ("... cards=4,5 ...").
state_cards() {
  [ -f "$(dirname "${BASH_SOURCE[0]}")/llm-ops.state" ] || return 0
  awk '{for (i = 1; i <= NF; i++) if ($i ~ /^cards=/) {sub("cards=", "", $i); print $i}}' \
    "$(dirname "${BASH_SOURCE[0]}")/llm-ops.state"
}

# --- serve process management (host-side view via docker top) ---
# Main serve process only (the API server; TERM target for graceful stop).
serve_pids() {
  $DOCKER top "$CONTAINER" -o pid,cmd 2>/dev/null | awk '/vllm serve/ && !/awk/ {print $1}'
}

# Whole vllm process tree: main + EngineCore + Worker_TP* (the "VLLM::" set).
# Liveness checks and the hard-kill phase must use THIS so orphaned workers
# cannot outlive the API server holding the cards.
vllm_tree_pids() {
  $DOCKER top "$CONTAINER" -o pid,cmd 2>/dev/null | awk '/vllm serve|VLLM::/ && !/awk/ {print $1}'
}
