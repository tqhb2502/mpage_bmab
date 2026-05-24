#!/usr/bin/env bash
# Archive experiments/results to a tagged folder under experiments/archive/.
# Use this before re-running experiments when you want to keep the old data
# for side-by-side comparison.
#
# Usage
# -----
#
#   mpage_bmab/experiments/archive_results.sh            # tags with current timestamp
#   mpage_bmab/experiments/archive_results.sh my_tag     # tags with a custom string
#
# After archiving, run experiments normally — they will go into a fresh
# experiments/results/. To compare old vs new use:
#
#   mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.compare_versions \\
#       --old mpage_bmab/experiments/archive/<tag> \\
#       --new mpage_bmab/experiments/results
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"  # mpage_bmab/experiments/

TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
ARCHIVE_DIR="archive/${TAG}"

if [[ ! -d results ]]; then
    echo "[archive] No results/ directory to archive — nothing to do."
    exit 0
fi

if [[ -d "$ARCHIVE_DIR" ]]; then
    echo "[archive] $ARCHIVE_DIR already exists. Pick a different tag." >&2
    exit 1
fi

mkdir -p archive
mv results "$ARCHIVE_DIR"
mkdir results
echo "[archive] Moved experiments/results/ -> experiments/${ARCHIVE_DIR}/"
echo "[archive] Fresh experiments/results/ created and ready for new runs."
echo
echo "Next steps:"
echo "  (1) launch experiments as usual, e.g."
echo "      mpage_bmab/experiments/run_mpage_compare.sh"
echo
echo "  (2) compare old vs new with:"
echo "      mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.compare_versions \\"
echo "          --old mpage_bmab/experiments/${ARCHIVE_DIR} \\"
echo "          --new mpage_bmab/experiments/results"
