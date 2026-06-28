# Final Consolidated Result Overview

This directory is the canonical thesis-facing location for the final result
tables and figures used in the main comparison and component ablation study.

Repository: <https://github.com/tqhb2502/mpage_bmab>

The main comparison includes three distinct setups:

- **Final-HV reward**
- **Hybrid reward**
- **MPaGE-orig**

The component ablation study compares **No budget annealing**, **No
Page-Hinkley**, and **No diversity reward** against **Final-HV reward**. Other
completed or historical result trees remain available for reproducibility, but
they are not emphasized in the thesis-facing summary.

The overview is regenerated from the repository root with the overview analysis
script:

```bash
MPLCONFIGDIR=/private/tmp/mpl <python-interpreter> \
    mpage_bmab/experiments/analyze_all_setups_overview.py
```
