#!/usr/bin/env bash
# Headline comparison: original MPaGE (budget-capped) vs Final-HV reward.
# 2 methods × bi_tsp × {25,50,100,200} × 3 seeds = 24 runs.
#
# This is the experiment that fills the "MPaGE-orig vs BMAB" row in the
# thesis AUBC table (IDEA.md §4.3).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$HERE")/.."
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.run \
    --suite mpage_compare_full "$@"
echo
echo "[mpage_compare] Aggregating + comparing..."
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.aggregate
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.compare \
    --baseline full --metric aubc
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.compare \
    --baseline full --metric hv_final
