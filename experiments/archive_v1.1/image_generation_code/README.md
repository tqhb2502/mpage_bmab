# Image Generation Code

This folder contains the Python script used to generate the analysis figures in `../images`.

From the project root, run:

```bash
MPLCONFIGDIR=/private/tmp/mpl mpage_bmab/.venv/bin/python \
  mpage_bmab/experiments/results/image_generation_code/generate_result_images.py
```

The script reads `../summary.csv` and the per-run JSON artifacts under `../full` and `../mpage_orig`.

It regenerates:

- `aubc_mean_by_budget.png`
- `hv_final_mean_by_budget.png`
- `aubc_delta_percent_heatmap.png`
- `hv_final_delta_percent_heatmap.png`
- `paired_aubc_delta_boxplot.png`
- `budget_curves_B25.png`
- `budget_curves_B50.png`
- `budget_curves_B100.png`
- `budget_curves_B200.png`
- `aubc_to_final_hv_ratio.png`
- `valid_yield_by_budget.png`
- `invalid_null_rate_by_budget.png`
- `valid_yield_delta_percent_heatmap.png`

It also regenerates the derived CSV tables:

- `../derived_cell_stats.csv`
- `../validity_stats.csv`
- `../validity_cell_stats.csv`

Use `--results-root` and `--images-dir` to point the script at another result folder:

```bash
MPLCONFIGDIR=/private/tmp/mpl mpage_bmab/.venv/bin/python \
  mpage_bmab/experiments/results/image_generation_code/generate_result_images.py \
  --results-root mpage_bmab/experiments/results \
  --images-dir mpage_bmab/experiments/results/images
```
