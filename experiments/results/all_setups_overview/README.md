# Final Consolidated Result Overview

This directory is the canonical thesis-facing location for the final result
tables and figures.

The final comparison includes four distinct setups:

- **Final-HV reward**
- **Dense reward**
- **Hybrid reward**
- **MPaGE-orig**

The historical `full_old` result tree is not included here. It is kept only as
archival material because Dense reward is the controlled current-implementation
variant for immediate-HVI feedback.

The overview is regenerated with:

```bash
MPLCONFIGDIR=/private/tmp/mpl mpage_bmab/.venv/bin/python \
    mpage_bmab/experiments/analyze_all_setups_overview.py
```
