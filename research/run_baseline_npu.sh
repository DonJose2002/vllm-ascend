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
# Either way: refuse to start if picked devices already hold >1GB (someone
# else's job), and record the choice in the summary block.
if [ -z "${NPUS:-}" ]; then
  PICK=$(python3 - <<'PYEOF'
import re, subprocess
try:
    out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15).stdout
except Exception as e:
    print(f"AUTO-PICK-FAIL: npu-smi info failed: {e}")
    raise SystemExit(0)
# Blocks: "NPU 0 | 910B3 ..." then "HBM: 1234 / 65536 (MB)"
best_id, best_used = None, None
cur_id = None
for line in out.splitlines():
    m = re.match(r"\s*NPU\s+(\d+)\s*\|", line)
    if m:
        cur_id = int(m.group(1))
        continue
    h = re.search(r"HBM:\s*(\d+)\s*/\s*(\d+)\s*\(MB\)", line)
    if h and cur_id is not None:
        used = int(h.group(1))
        if best_used is None or used < best_used:
            best_id, best_used = cur_id, used
        cur_id = None
if best_id is None:
    print("AUTO-PICK-FAIL: could not parse any 'NPU n | ... HBM: x / y (MB)' block")
else:
    print(f"{best_id} {best_used}")
PYEOF
)
  case "$PICK" in
    AUTO-PICK-FAIL*)
      echo "$PICK"
      echo "Cannot auto-pick a free NPU. Run 'npu-smi info', then re-run with NPUS=<id>."
      exit 1
      ;;
  esac
  NPUS="${PICK%% *}"
  HBM_USED="${PICK##* }"
  echo "PREFLIGHT: auto-picked NPU $NPUS (HBM used ${HBM_USED}/65536 MB)"
else
  HBM_USED=$(python3 - "$NPUS" <<'PYEOF'
import re, subprocess, sys
wanted = {x.strip() for x in sys.argv[1].split(",") if x.strip()}
try:
    out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15).stdout
except Exception as e:
    print(f"CHECK-FAIL: npu-smi info failed: {e}")
    raise SystemExit(0)
cur_id = None
usages = {}
for line in out.splitlines():
    m = re.match(r"\s*NPU\s+(\d+)\s*\|", line)
    if m:
        cur_id = int(m.group(1))
        continue
    h = re.search(r"HBM:\s*(\d+)\s*/\s*(\d+)\s*\(MB\)", line)
    if h and cur_id is not None:
        usages[cur_id] = int(h.group(1))
        cur_id = None
missing = [d for d in wanted if int(d) not in usages]
if missing:
    print(f"CHECK-FAIL: device(s) {','.join(missing)} not found in npu-smi info")
elif len(usages) < len(wanted):
    print("CHECK-FAIL: could not parse all HBM lines")
else:
    print(max(usages[int(d)] for d in wanted))
PYEOF
)
  case "$HBM_USED" in
    CHECK-FAIL*)
      echo "$HBM_USED"
      exit 1
      ;;
  esac
  echo "PREFLIGHT: using NPU(s) $NPUS (max HBM used ${HBM_USED}/65536 MB)"
fi
if [ "${HBM_USED:-0}" -gt 1024 ]; then
  echo "PREFLIGHT-FAIL: selected NPU(s) already hold ${HBM_USED} MB HBM (another job?)."
  echo "Pick a free one: NPUS=<id> bash research/run_baseline_npu.sh $MODE $PORT"
  exit 1
fi
export ASCEND_RT_VISIBLE_DEVICES="$NPUS"
echo "PREFLIGHT: ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"

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
  echo "visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-unset} hbm_used_mb=${HBM_USED:-?}"
  echo "repo_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "vllm_ascend_dist=$DIST_VER"
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
