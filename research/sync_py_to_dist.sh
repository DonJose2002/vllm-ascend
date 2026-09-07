#!/usr/bin/env bash
# Sync workspace .py files into the pip-installed vllm-ascend dist (research
# iteration shortcut).
#
# WHY: `vllm serve` imports the SITE-PACKAGES copy (sealed non-editable
# install by design since 2026-08-21), so workspace edits need a reinstall.
# But the csrc build CANNOT be incrementally re-run in a dirty temp dir:
# the merge_aic_obj_text step (ld.lld -m aicorelinux in.o -o in.o) rewrites
# .o files IN PLACE - re-running the converter on its own converted output
# fails with "unknown file type" (b2smoke-era lessons 2026-09-04/07, hit
# three times). A .py-only change therefore needs NO wheel rebuild: mirror
# the .py tree straight into the installed dist.
#
# WHEN to use: research loops that touch only *.py / no csrc, no setup.py,
# no new compiled artifacts. For any csrc change - or before recording a
# SUMMARY that cites the dist version - do a CLEAN full install instead:
#   rm -rf build/temp.* && MAX_JOBS=16 pip install . --no-build-isolation
#
# CAVEATS:
# - dist-info metadata stays at the last full install (version/commit suffix
#   goes stale); SUMMARY blocks will cite the stale commit - acceptable for
#   smoke triage, not for the record.
# - additive sync only (deleted/renamed workspace .py files are not pruned
#   from the dist).
set -euo pipefail
cd "$(dirname "$0")/.."

# Resolve the installed dist from a NEUTRAL cwd - running from the repo root
# shadows vllm_ascend with the workspace copy (CWD-shadowing glossary entry).
DIST_DIR="$(cd /tmp && python3 -c 'import vllm_ascend, os; print(os.path.dirname(vllm_ascend.__file__))')" || {
  echo "FAIL: vllm-ascend dist not importable (run inside the v0.23.0 container)"
  exit 1
}
case "$DIST_DIR" in
  "$(pwd)"*) echo "FAIL: dist resolves INSIDE the workspace ($DIST_DIR) - shadowing?"; exit 1 ;;
esac
echo "dist: $DIST_DIR"

(cd vllm_ascend && find . -name '*.py' -print0 | tar --null -cf - -T -) | (cd "$DIST_DIR" && tar -xf -)

# Verification from a neutral cwd: the diagnostic modules must resolve to the
# dist (not the workspace), proving the serve will pick the synced code up.
echo "verify (neutral cwd):"
cd /tmp
python3 -c "import vllm_ascend.patch.platform.patch_static_kv_compact as p; print('  patch  :', p.__file__)"
python3 -c "import vllm_ascend.worker.static_kv_compact as m; print('  module :', m.__file__)"
echo "synced OK - .py-only changes are live; version metadata intentionally stale."
