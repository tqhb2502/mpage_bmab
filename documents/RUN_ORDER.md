# Recommended Experiment Run Order

A practical playbook for executing the thesis experiments in a sensible
sequence. The goal is to spend cheap LLM calls first to catch regressions,
then build up the evidence base from "headline single-task" to "cross-task
generalisation" to "full matrix."

Companion documents:
[experiments/README.md](../experiments/README.md) describes the harness
mechanics; [HYPERPARAMETERS.md](HYPERPARAMETERS.md) records what each cell
will use; [MPAGE_ORIG_COMPARISON.md](MPAGE_ORIG_COMPARISON.md) covers the
upstream-baseline comparison.

---

## Are the experiments still the same?

**Yes.** The suite definitions in [experiments/configs.py](../experiments/configs.py)
did not change — same `(ablation, task, budget, seed)` cells, same output
directories, same idempotency. The only thing that changed is that each
cell now uses the budget-aware `pop_size` from `pop_size_for(task,
budget)`.

* At `B = 50` the new values are **identical** to the old per-task
  defaults (so any earlier `B=50` run would still match).
* At `B ∈ {25, 100, 200}` the old code applied the same per-task default
  (everyone got 6 / 8 / 6 / 6) regardless of budget. The new code applies
  the `pop_size_for` matrix from [HYPERPARAMETERS.md §2](HYPERPARAMETERS.md#2-population-size--per-budget-and-per-task).

If `experiments/results/` is empty, you have not actually launched any
experiment yet and nothing needs re-running. If it contains earlier runs
at `B ≠ 50`, those used a different `pop_size` — delete them (or pass
`--force`) before re-running, otherwise idempotency will skip them.

---

## Recommended order

| # | Step | Suite / launcher | Runs | LLM-call cost ≈ | Why |
|---|------|------------------|-----:|---------------:|-----|
| 1 | **Smoke (BMAB)** | `run_smoke.sh` | 2 | ~30 calls | Confirms `mpage_bmab.main` path works end-to-end before spending real money. |
| 2 | **Smoke (MPaGE-orig)** | `run_mpage_smoke.sh` | 2 | ~30 calls | Confirms the new `mpage_bmab.mpage_orig` runner works the same way. |
| 3 | **Headline AUBC table** | `run_headline.sh` | 48 | ~4,650 calls | The main thesis-chapter table: 4 BMAB ablations × `bi_tsp` × 4 budgets × 3 seeds. Auto-runs `aggregate.py` + `compare.py --baseline full`. |
| 4 | **MPaGE-orig vs BMAB-LLM** | `run_mpage_compare.sh` | 24 | ~2,300 calls | The "is BMAB-LLM actually better than the upstream MPaGE?" comparison. Auto-aggregates and Wilcoxon-tests `full` vs `mpage_orig`. |
| 5 | **Cross-task validation** | `run_budget50.sh` | 80 | ~4,000 calls | "Does the result hold on `tri_tsp`, `bi_cvrp`, `bi_kp`?" — 4 ablations across all 4 tasks at the tight-budget regime `B=50`, all 5 seeds. |
| 6 | **Full sweep** (optional) | `run_full.sh` | 320 | ~30,950 calls | The complete `4 × 4 × 4 × 5` matrix from IDEA.md §4. Only do this if resources allow. |

After step 3 you already have the headline result. Steps 4 and 5 add the
MPaGE-orig baseline and cross-task generalisation — both are necessary
for a defensible thesis chapter. Step 6 is only worthwhile if you have
spare credits and time.

---

## Why this order

1. **Cheapest tests first.** Steps 1–2 catch any pipeline regressions
   (LLM calls, parsing, evaluator, profiler) for ~60 LLM calls total.
   If anything breaks, you find out before paying for the bigger sweeps.
2. **Headline before generalisation.** Step 3 is your strongest
   single-task result. Once you know `full` beats the ablations on
   `bi_tsp`, you have something to defend. Step 5 then tests whether
   the improvement generalises across all four MOCOP benchmarks.
3. **MPaGE-orig comparison goes after BMAB ablations.** Step 4 is the
   "vs upstream" plot; it's most informative once you can describe
   what each BMAB component contributes (from step 3's ablation
   Wilcoxon).
4. **Save the full sweep for last.** Step 6 produces ~320 runs of
   data. Don't launch it until smaller suites have shown the result is
   real — otherwise you spend a lot to discover something step 3 would
   have told you.

---

## Resume-safety

Every launcher uses `experiments/run.py`'s idempotent
skip-if-`aubc.json`-exists logic. So:

* You can stop step 6 halfway and re-run later — it skips finished cells.
* You can re-launch step 3 after step 6 to "fill in" any cells that
  failed during the larger sweep.
* You can add seeds 2030–2034 later via `--seeds` on the CLI and only
  the new cells run.

If you ever need to **force a re-run**, pass `--force` to `run.py` (or
to the wrapper shell script). Use this when you have changed
hyperparameters that affect old cells (e.g. changing `pop_size_for`
would invalidate prior B=25/100/200 runs).

---

## Quick decision tree

```
Did smoke step (1+2) succeed?
├── NO  → debug; fix mpage_orig.py or main.py; do not proceed
└── YES → run step 3 (headline)
         ├── Wilcoxon shows full > op_only or no_ph or no_diversity (p<.05)?
         │   ├── YES → run steps 4 + 5 in parallel (different machines if you have them)
         │   │        ├── full > mpage_orig at B=50 (step 4)?       → thesis claim defensible
         │   │        └── full > ablations on tri_tsp / bi_cvrp / bi_kp?  → result generalises
         │   │        Then step 6 if resources remain.
         │   └── NO  → diagnose. Likely culprits:
         │            – PFG narrowing slab (see OFFSPRING_GENERATION_PATHS §8)
         │            – prior weight `prior_n`
         │            – pop_size too small at the budget you tested
         └── (continue to step 4-6)
```

---

## Concrete commands sequence

```bash
# Pre-flight (cheap, run once)
mpage_bmab/experiments/run_smoke.sh
mpage_bmab/experiments/run_mpage_smoke.sh

# The headline thesis table
mpage_bmab/experiments/run_headline.sh

# vs MPaGE-orig — the baseline comparison
mpage_bmab/experiments/run_mpage_compare.sh

# Cross-task at the tight-budget regime
mpage_bmab/experiments/run_budget50.sh

# Inspect everything aggregated so far
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.aggregate
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.compare --baseline full
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.compare --baseline mpage_orig

# Only if budget allows — the full matrix
mpage_bmab/experiments/run_full.sh
```

---

## Re-running after a code change (keeping the old results)

When you change the code in a way that affects results (e.g. the
Page-Hinkley `+ δ` sign fix, or any reward / bandit tuning) and you want
to re-run while keeping the old data for side-by-side comparison, use
the archive workflow.

### Step 1 — Archive the existing `experiments/results/`

```bash
mpage_bmab/experiments/archive_results.sh v1_pre_ph_fix
```

The script moves the current `experiments/results/` into
`experiments/archive/v1_pre_ph_fix/` (or whatever tag you pass), then
creates a fresh empty `experiments/results/` ready for the new sweep.
Old `summary.csv` and `comparisons_*.csv` are preserved inside the
archive folder.

If you omit the tag a timestamp is used (e.g. `20260516_143055`).

### Step 2 — Re-run experiments as usual

Idempotency now applies to the *fresh* `results/`, so every cell will be
re-executed:

```bash
mpage_bmab/experiments/run_smoke.sh
mpage_bmab/experiments/run_headline.sh
mpage_bmab/experiments/run_mpage_compare.sh
mpage_bmab/experiments/run_budget50.sh
# etc.
```

### Step 3 — Compare old vs new

```bash
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.compare_versions \
    --old mpage_bmab/experiments/archive/v1_pre_ph_fix \
    --new mpage_bmab/experiments/results \
    --out_summary mpage_bmab/experiments/results/version_diff.csv
```

Produces a Markdown table to stdout with one row per
`(ablation, task, budget)` showing `AUBC_old → AUBC_new (ΔAUBC ± std)`
and `HV_old → HV_new (ΔHV)`. Useful for confirming whether a code
change actually moved the metrics in your data.

### Folder layout after archiving

```
experiments/
├── results/                        ← fresh, empty, for new sweep
└── archive/
    └── v1_pre_ph_fix/              ← old results preserved here
        ├── summary.csv
        ├── comparisons_aubc.csv
        ├── comparisons_hv_final.csv
        ├── full/...
        ├── mpage_orig/...
        └── ...
```

You can archive as many times as you want — each call places a new
folder under `experiments/archive/`. Use descriptive tags
(`v1_pre_ph_fix`, `v2_post_ph_fix`, `before_pop_size_change`, etc.) so
each version is easy to identify months later.

---

## What's the minimum viable thesis dataset?

If you are cost-constrained, **steps 1–4** are the minimum viable
thesis dataset (~7,000 LLM calls):

* Steps 1–2 → pipeline correctness.
* Step 3 → ablation table on `bi_tsp` (BMAB-LLM is the proposed
  system, and each ablation's marginal contribution).
* Step 4 → comparison vs the actual upstream MPaGE baseline.

Step 5 adds cross-task generalisation — strongly recommended but
spendable if you must. Step 6 is "comprehensive but expensive."

In thesis terms, the *bare minimum* defendable claim is:

> "On `bi_tsp` under budgets `B ∈ {25, 50, 100, 200}` and 3 seeds per
> cell, BMAB-LLM achieves higher AUBC than (a) the upstream MPaGE
> capped at the same B, and (b) each of three BMAB-LLM ablations
> (Page-Hinkley off, diversity term off, cluster bandit off), with
> Wilcoxon signed-rank `p < .05` per cell."

Steps 1–4 are exactly the data you need to make that claim. Steps 5–6
expand it to four tasks and five seeds.
