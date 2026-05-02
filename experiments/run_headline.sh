#!/usr/bin/env bash
# The headline AUBC table from IDEA.md §4.
# 4 ablations × bi_tsp × {25,50,100,200} budgets × 3 seeds = 48 runs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$HERE")/.."
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.run \
    --suite headline "$@"
echo
echo "[headline] Aggregating results..."
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.aggregate
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.compare \
    --baseline full --metric aubc
