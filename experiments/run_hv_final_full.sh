#!/usr/bin/env bash
# Full reward-mode comparison for the finalized BMAB implementation.
# 3 variants × 4 tasks × 4 budgets × 5 seeds = 240 runs:
#   Final-HV reward -> full / reward_mode=final_hv
#   Dense reward    -> dense_reward / reward_mode=dense
#   Hybrid reward   -> hybrid_reward / reward_mode=hybrid
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$HERE")/.."
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.run \
    --suite hv_final_full "$@"
echo
echo "[hv_final_full] Aggregating + comparing reward modes..."
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.aggregate
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.compare \
    --baseline full --metric hv_final
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.compare \
    --baseline full --metric aubc
