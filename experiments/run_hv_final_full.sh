#!/usr/bin/env bash
# Internal reward-mode diagnostic suite for the BMAB implementation.
# 3 variants × 4 tasks × 4 budgets × 5 seeds = 240 runs.
# The thesis-facing comparison uses Final-HV and Hybrid; Dense remains here
# only for reproducibility of previous reward-mode diagnostics:
#   Final-HV reward -> internal key full / reward_mode=final_hv
#   Dense reward    -> internal key dense_reward / reward_mode=dense
#   Hybrid reward   -> internal key hybrid_reward / reward_mode=hybrid
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
