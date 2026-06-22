# Final-HV Improvement Experiments

This file lists the experiments to run after the final-HV fixes.

## Terminology

The method names in this file refer to the **current finalized implementation**:

- `full` (**Final-HV reward**): finalized BMAB method with
  `reward_mode='final_hv'`. The quality part of the bandit reward is the
  normalized HV gain after applying the same managed-population cap used at the
  end of the run.
- `dense_reward` (**Dense reward**): finalized BMAB method with
  `reward_mode='dense'`. The quality
  part of the bandit reward is the legacy immediate-HVI signal.
- `hybrid_reward` (**Hybrid reward**): finalized BMAB method with
  `reward_mode='hybrid'`. The quality part is
  `0.5 * immediate_HVI_norm + 0.5 * final_managed_HV_norm`.
- `no_budget_anneal`: finalized BMAB method with budget-annealed cluster
  exploration disabled. This is optional; skip it if you do not want this
  ablation.
- `mpage_orig`: the original MPaGE wrapper under the same budget accounting.

Important: the historical directory `mpage_bmab/experiments/results/full`
contains runs produced before the final-HV fixes. It is useful as a
historical result set, but it is **not** the same thing as `dense_reward`.
`dense_reward` keeps the new implementation fixes and changes only the reward
quality signal back to immediate HVI.

## What Changed

The `full` method, reported as **Final-HV reward** in the thesis, includes the
final-HV fixes:

- actual generated child score is used for bandit reward;
- pending valid offspring are flushed into the final managed population;
- the last budget unit is not spent on a cluster-only call;
- cluster UCB exploration is annealed by remaining budget;
- cluster priors use front-aware HV contribution, best inner-HV proxy, and runtime;
- parent sampling gives extra weight to PFG-selected elites.

## Smoke Test

Run this first:

```bash
python -m mpage_bmab.experiments.run \
  --suite hvfix_smoke \
  --results_root mpage_bmab/experiments/results_hvfix \
  --dry_run
```

Then run it for real:

```bash
python -m mpage_bmab.experiments.run \
  --suite hvfix_smoke \
  --results_root mpage_bmab/experiments/results_hvfix
```

Aggregate and compare:

```bash
python -m mpage_bmab.experiments.aggregate \
  --results_root mpage_bmab/experiments/results_hvfix \
  --out mpage_bmab/experiments/results_hvfix/summary.csv

python -m mpage_bmab.experiments.compare \
  --summary mpage_bmab/experiments/results_hvfix/summary.csv \
  --metric hv_final
```

Run the archive diagnostic:

```bash
python -m mpage_bmab.experiments.final_hv_diagnostics \
  --summary mpage_bmab/experiments/results_hvfix/summary.csv \
  --ablation full
```

After the flush fix, `managed_minus_recorded` should usually be near zero.

## Priority Final-HV Sweep

This is the recommended next experiment before spending on the full grid:

```bash
python -m mpage_bmab.experiments.run \
  --suite hv_final_priority \
  --results_root mpage_bmab/experiments/results_hvfix
```

This compares:

- `full`: Final-HV reward;
- `dense_reward`: Dense reward with immediate-HVI feedback;
- `hybrid_reward`: Hybrid reward with half immediate-HVI, half final-population HV;
- `no_budget_anneal`: optional Final-HV reward ablation without budget exploration annealing;
- `mpage_orig`: original MPaGE wrapper under the same budget.

If `mpage_orig` has already been run with the same tasks, budgets, seeds,
population sizes, reference points and evaluator code, you do not need to
spend those calls again. Either keep it in the same `results_root`, copy the
existing `mpage_orig` tree into the new result root before aggregation, or
omit `mpage_orig` from the fixed-method sweep and compare against the existing
summary separately.

If you want to skip `no_budget_anneal`, run the same slice manually:

```bash
python -m mpage_bmab.experiments.run \
  --ablations full,dense_reward,hybrid_reward,mpage_orig \
  --tasks bi_tsp,bi_cvrp,bi_kp \
  --budgets 25,50,100 \
  --seeds 2025,2026,2027,2028,2029 \
  --results_root mpage_bmab/experiments/results_hvfix
```

## Full Reward-Mode Sweep

Run this only after the priority sweep looks promising:

```bash
mpage_bmab/experiments/run_hv_final_full.sh \
  --results_root mpage_bmab/experiments/results_hvfix
```

The `hv_final_full` suite compares the three reward modes of the finalized
BMAB implementation across all tasks, budgets, and seeds: `full`
(`reward_mode=final_hv`), `dense_reward` (`reward_mode=dense`), and
`hybrid_reward` (`reward_mode=hybrid`).

## Manual High-Value Slice

If budget is tight, run only the previously weak final-HV cells:

```bash
python -m mpage_bmab.experiments.run \
  --ablations full,dense_reward,hybrid_reward,mpage_orig \
  --tasks bi_tsp,tri_tsp,bi_cvrp,bi_kp \
  --budgets 25,50,100 \
  --seeds 2025,2026,2027 \
  --results_root mpage_bmab/experiments/results_hvfix
```

## Main Metrics to Inspect

Use `hv_final` as the primary metric for these experiments. AUBC should still
be reported, but the purpose of this sweep is to test whether terminal
population quality improves.

Also inspect:

- `valid_yield_per_budget`;
- `managed_minus_recorded` from `final_hv_diagnostics.py`;
- per-task final-HV wins, especially Bi-KP B25/B100 and Bi-TSP B50.
