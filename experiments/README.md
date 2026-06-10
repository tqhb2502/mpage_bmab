# BMAB-LLM Experimental Harness

End-to-end pipeline for the experiments described in [`../IDEA.md` §4](../IDEA.md).
Pick a **suite**, run it, and the harness writes a CSV summary plus
Wilcoxon comparisons against the proposed system.

## Layout

```
experiments/
├── configs.py        ← tasks / budgets / seeds / ablations / suite presets
├── run.py            ← single + sweep launcher (sub-process per cell)
├── aggregate.py      ← walk results/, write summary.csv
├── compare.py        ← Wilcoxon signed-rank per (task, budget)
├── run_smoke.sh      ← 2 cheap runs to validate the pipeline
├── run_headline.sh   ← 48 runs (bi_tsp, all budgets, 3 seeds)
├── run_budget50.sh   ← 80 runs (all tasks, B=50, 5 seeds)
├── run_full.sh       ← 320 runs (full sweep)
└── results/          ← per-run output directories + CSVs (gitignored)
```

## Ablations (matches IDEA.md §4.3)

| name | what it changes | flag(s) it sets |
|------|------------------|-----------------|
| `full`         | the proposed BMAB-LLM | (defaults) |
| `dense_reward` | fixed BMAB-LLM with the legacy immediate-HVI reward quality signal | `reward_mode=dense` |
| `hybrid_reward` | fixed BMAB-LLM with half immediate-HVI and half final managed-population HV reward | `reward_mode=hybrid` |
| `no_budget_anneal` | fixed BMAB-LLM without remaining-budget exploration annealing | `disable_budget_annealing=True` |
| `no_ph`        | disable Page-Hinkley drift | `ph_threshold=1e9` |
| `no_diversity` | drop the ΔCDI reward term | `w_diversity=0.0` |
| `op_only`      | uniform random cluster sampling | `disable_cluster_bandit=True` |
| `cluster_only` | round-robin operator selection | `disable_operator_bandit=True` |
| `mpage_budget` | MPaGE-style baseline (no bandits, no diversity) | all three above + `w_rank=0` |
| `mpage_orig` | actual MPaGE wrapper under the same budget accounting | dispatches to `mpage_bmab.mpage_orig` |

In the current code, `full` means the fixed final-HV-oriented method
(`reward_mode=final_hv`). Historical outputs under
`experiments/results/full` were produced before those fixes and should be
reported as the historical/pre-fix `full` result set, not as `dense_reward`.

## Pre-flight

```bash
echo "<openai-api-key>" > secret.txt
echo "<openai-api-key>" > secret_cluster.txt
```

Both files must exist at the path you launch from (the project root that
contains `mpage_bmab/`).

## Suites

| suite | runs | total LLM calls (call-mode) | what it's for |
|-------|-----:|----------------------------:|---------------|
| `smoke` | 2 | ≈ 30 | sanity check the pipeline before paying for the real sweep |
| `headline` | 48 | ≈ 4,650 | the main AUBC table for the thesis chapter |
| `budget50` | 80 | ≈ 4,000 | comparison across all 4 tasks at the tight-budget regime |
| `full` | 320 | ≈ 30,950 | the complete matrix from IDEA.md §4 |
| `hvfix_smoke` | 4 | ≈ 100 | cheap sanity check for the final-HV fixes |
| `hv_final_priority` | 225 | ≈ 13,125 | focused final-HV sweep including reward ablations and `mpage_orig` |
| `hv_final_full` | 240 by default | ≈ 22,500 | full final-HV sweep for `full`, `dense_reward`, and `hybrid_reward`; optional ablations can be added manually |

The "total LLM calls" column is approximate: each cell consumes its
`--budget` plus a small overhead for the cluster-LLM calls (already
counted against the budget).

## Quickstart

```bash
# from the project root that contains mpage_bmab/

# 1. Validate the pipeline end-to-end (~30 cheap calls)
mpage_bmab/experiments/run_smoke.sh

# 2. Run the headline experiment (this is the table for the thesis)
mpage_bmab/experiments/run_headline.sh

# 3. Inspect the results
cat mpage_bmab/experiments/results/summary.csv | column -ts,
```

The headline / full launchers automatically run `aggregate.py` and
`compare.py` after the sweep finishes.

## One-off cells

To re-run a single cell:

```bash
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.run \
    --ablations full \
    --tasks bi_kp \
    --budgets 50 \
    --seeds 2025 \
    --force
```

## Custom slices

```bash
# Compare full vs op_only on bi_tsp + bi_cvrp at B=50,100 across 5 seeds
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.run \
    --ablations full,op_only \
    --tasks bi_tsp,bi_cvrp \
    --budgets 50,100 \
    --seeds 2025,2026,2027,2028,2029
```

Add `--dry_run` first to print the planned cell list without launching
sub-processes — recommended before any run that is going to spend money.

## Resuming / re-running

`run.py` is idempotent: a cell is **skipped** if `aubc.json` already
exists at its destination. Crash-mid-sweep, fix the issue, re-launch the
same command, and only the missing cells run. Pass `--force` to override.

## Aggregating + comparing manually

```bash
# Collect everything under experiments/results/ into a CSV + print a summary
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.aggregate

# Wilcoxon signed-rank: full vs each ablation, paired across seeds
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.compare \
    --baseline full --metric aubc

# Or compare on final hypervolume instead of AUBC
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.compare \
    --baseline full --metric hv_final
```

## Output schema

Per-cell output directory:

```
experiments/results/<ablation>/<task>/B<budget>/seed<seed>/
└── <TIMESTAMP>_<Problem>_BMAB-<ablation>/
    ├── run_log.txt
    ├── samples/samples_*.json     (every heuristic ever evaluated)
    ├── population/pop_*.json      (population at each generation)
    ├── budget_curve.json          ((consumed_budget, hv, pareto_size) triples)
    ├── bandit_log.json            (per-generation bandit statistics)
    ├── budget_history.json        (every LLM-call's budget deduction)
    └── aubc.json                  (the headline AUBC scalar)
```

Aggregated CSV columns:

| column | description |
|--------|-------------|
| `ablation` | method tag such as `full`, `dense_reward`, `hybrid_reward`, `no_budget_anneal`, `no_ph`, `no_diversity`, `op_only`, `cluster_only`, `mpage_budget`, or `mpage_orig` |
| `task` | `bi_tsp` / `tri_tsp` / `bi_cvrp` / `bi_kp` |
| `budget` | total LLM-call budget B |
| `seed` | RNG seed |
| `aubc` | Area-Under-Budget-Curve |
| `hv_final` | hypervolume at the last recorded budget point |
| `pareto_size` | number of non-dominated heuristics on that front |
| `consumed_budget` | calls actually issued (≤ `budget`) |
| `n_calls` | length of `budget_history.json` |
| `run_dir` | absolute path to the timestamped output folder |

## Re-using results across sweeps

All sweeps write into the same `experiments/results/` tree, so you can
mix-and-match: a `headline` run gives you `bi_tsp` data; a later
`budget50` run fills in the other three tasks at B=50. After both
finish, `aggregate.py` produces a single combined `summary.csv`.

## Notes on cost

* All sweeps use `gpt-4o-mini` by default. Override with
  `--llm_model gpt-4o` etc. on `run.py`.
* Token-aware budgeting is supported by `main.py` (`--budget_mode token`)
  but is **not** the default for these suites. Switch the suite's runs
  to token mode by adding `--extra "--budget_mode token"` to `run.py`.
* `run.py` checks for `secret.txt` / `secret_cluster.txt` before the
  first cell and aborts the sweep if either is missing — protects you
  from a silent failure on call #1 of run #1.
