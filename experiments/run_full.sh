#!/usr/bin/env bash
# Component-ablation sweep from IDEA.md §4: 4 ablations × 4 tasks × 4 budgets × 5 seeds = 320 runs.
# Be aware this is expensive — start by dry-running first to count calls.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$HERE")/.."
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.run \
    --suite full "$@"
echo
echo "[Final-HV reward reference] Aggregating + comparing..."
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.aggregate
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.compare \
    --baseline full --metric aubc
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.compare \
    --baseline full --metric hv_final
