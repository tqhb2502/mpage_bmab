#!/usr/bin/env bash
# Cheapest end-to-end pipeline test. Runs 2 small experiments (~30 LLM calls
# total) so you can verify everything works before launching a real sweep.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$HERE")/.."  # project root (parent of mpage_bmab/)
"${PYTHON:-mpage_bmab/.venv/bin/python}" -m mpage_bmab.experiments.run \
    --suite smoke "$@"
