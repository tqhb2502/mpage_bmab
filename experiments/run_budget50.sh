#!/usr/bin/env bash
# Across-tasks comparison at the tight-budget regime (B=50).
# 4 ablations × 4 tasks × {50} × 5 seeds = 80 runs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$HERE")/.."
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.run \
    --suite budget50 "$@"
echo
echo "[budget50] Aggregating + comparing..."
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.aggregate
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.compare \
    --baseline full --metric aubc
