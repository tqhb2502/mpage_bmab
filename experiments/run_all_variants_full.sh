#!/usr/bin/env bash
# Complete sweep for every runnable method in the current harness.
# 10 variants/ablations × 4 tasks × 4 budgets × 5 seeds = 800 runs:
#   Final-HV reward (full), Dense reward (dense_reward),
#   Hybrid reward (hybrid_reward), no_budget_anneal,
#   no_ph, no_diversity, op_only, cluster_only, mpage_budget, mpage_orig
#
# This is expensive. Run with --dry_run first to inspect the planned cells.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$HERE")/.."
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.run \
    --suite all_variants_full "$@"
for arg in "$@"; do
    if [[ "$arg" == "--dry_run" ]]; then
        exit 0
    fi
done
echo
echo "[all_variants_full] Aggregating + comparing all runnable methods..."
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.aggregate
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.compare \
    --baseline full --metric aubc
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.compare \
    --baseline full --metric hv_final
