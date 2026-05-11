#!/usr/bin/env bash
# Cheapest end-to-end check for the MPaGE-orig vs BMAB-LLM pipeline.
# 2 small runs (~30 LLM calls total) so you can verify the MPaGE-orig runner
# works before paying for the full mpage_compare sweep.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$HERE")/.."
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.run \
    --suite mpage_smoke "$@"
