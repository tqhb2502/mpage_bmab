# Final Consolidated Result Overview

This directory is the canonical thesis-facing location for the final result
tables and figures.

The final comparison includes four distinct setups:

- **Final-HV reward** (`full`)
- **Dense reward** (`dense_reward`)
- **Hybrid reward** (`hybrid_reward`)
- **MPaGE-orig** (`mpage_orig`)

The historical `full_old` result tree is not included here. It is kept only as
archival material because the current `dense_reward` setup is the controlled
current-implementation ablation for immediate-HVI feedback.

The overview is regenerated with:

```bash
MPLCONFIGDIR=/private/tmp/mpl mpage_bmab/.venv/bin/python \
    mpage_bmab/experiments/analyze_all_setups_overview.py

MPLCONFIGDIR=/private/tmp/mpl mpage_bmab/.venv/bin/python \
    mpage_bmab/experiments/analyze_rerun_cell_update.py
```
