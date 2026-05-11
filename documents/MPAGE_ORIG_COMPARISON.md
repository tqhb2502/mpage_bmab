# Running the Original MPaGE Under a Budget — Comparison Setup

This document describes how to run the **original** `MPaGE` class
(vendored at `_llm4ad/method/LLMPFG/eoh.py`) under an LLM-call budget and
compare it against BMAB-LLM on the same task / budget / seed. This is the
experimental protocol for the **MPaGE-budget** / **MPaGE-orig** baseline
referenced in [IDEA.md §4.3](IDEA.md#section-43-baselines).

Before this addition, the project only had a *simulated* MPaGE baseline —
the `mpage_budget` ablation, which runs the BMAB-LLM driver with every
adaptive component disabled. That control is useful for isolating the BMAB
machinery, but it is not the same code path as the upstream `MPaGE` class.
The runner described below uses the actual upstream class.

---

## Table of contents

1. [What's new](#1-whats-new)
2. [How the runner works](#2-how-the-runner-works)
3. [The two simultaneous baselines](#3-the-two-simultaneous-baselines)
4. [Quickstart](#4-quickstart)
5. [Available suites](#5-available-suites)
6. [How budget is counted](#6-how-budget-is-counted)
7. [Comparing AUBC](#7-comparing-aubc)
8. [Caveats and fairness notes](#8-caveats-and-fairness-notes)
9. [External baselines (EoH, MEoH, ReEvo, HSEvo)](#9-external-baselines-eoh-meoh-reevo-hsevo)

---

## 1. What's new

| File | Purpose |
|------|---------|
| [mpage_orig.py](../mpage_orig.py) | New runner module. Defines `MPaGEBudget`, a thin subclass of the vendored `MPaGE` that charges every LLM call (heuristic + cluster) against ``_tot_sample_nums`` and records a budget-vs-HV curve. CLI compatible with the rest of the project. |
| [experiments/configs.py](../experiments/configs.py) | Adds `mpage_orig` to `ALL_ABLATIONS` and lists it in the new `MPAGE_ORIG_ABLATIONS` set. Adds four new suites: `mpage_smoke`, `mpage_compare`, `mpage_compare_full`, `all_methods`. |
| [experiments/run.py](../experiments/run.py) | `_build_cmd` now dispatches `mpage_orig`-family ablations to `mpage_bmab.mpage_orig` instead of `mpage_bmab.main`. Everything else (idempotency, dry-run, aggregation, comparison) works without modification. |
| [experiments/run_mpage_smoke.sh](../experiments/run_mpage_smoke.sh) | 2-run pipeline check for the MPaGE-orig path. |
| [experiments/run_mpage_compare.sh](../experiments/run_mpage_compare.sh) | The 24-run headline comparison: MPaGE-orig vs BMAB-LLM-full on `bi_tsp` × 4 budgets × 3 seeds. Chains run → aggregate → compare. |

No changes to the BMAB-LLM modules themselves. The new module reuses
`BMABProfiler` so the aggregation pipeline (`summary.csv`, Wilcoxon tests)
works seamlessly across both method families.

---

## 2. How the runner works

[mpage_orig.py](../mpage_orig.py) is built around a 100-line subclass:

```python
class MPaGEBudget(MPaGE):
    """MPaGE that counts every LLM call (cluster + heuristic) toward the
    budget and records a budget-vs-HV curve compatible with BMABProfiler."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Patch cluster sampler so its calls also count toward budget
        orig_cluster_thought = self._cluster_sampler.get_thought
        def cluster_thought_counted(prompt):
            if self._tot_sample_nums >= self._max_sample_nums:
                return None
            self._tot_sample_nums += 1
            self._maybe_record_curve()
            return orig_cluster_thought(prompt)
        self._cluster_sampler.get_thought = cluster_thought_counted

    def _sample_evaluate_register(self, prompt):
        # Count EVERY attempt (incl. invalids) — same as BMAB-LLM
        if self._tot_sample_nums >= self._max_sample_nums:
            return
        self._tot_sample_nums += 1
        self._maybe_record_curve()
        # ... rest of the parent logic, with the original _tot_sample_nums++
        # removed (so we don't double-count).

    def _thread_init_population(self):
        # Init loop that also respects the budget
        while (self._population.generation == 0
               and self._tot_sample_nums < self._max_sample_nums):
            ...

    def run(self):
        # After the loop, build a BudgetTracker reflecting consumption and
        # pass it to BMABProfiler.finish(budget=...) so aubc.json is written.
        ...
```

Three design choices worth calling out:

1. **`max_sample_nums = int(budget)`** sets the same hard cap on LLM calls as
   `BudgetTracker.charge` does in BMAB-LLM. The native `MPaGE` loop reads
   this value to decide when to stop, so we don't need a separate exit path.
2. **`max_generations = 10_000_000`** is required because the upstream
   `continue_loop()` returns `False` when *either* generation or
   sample-count condition fails. Setting it to a huge value effectively
   disables the generation cap and lets the budget be the only real
   stopping criterion.
3. **`BMABProfiler` is reused.** The `BMABProfiler` already extends
   `EoHProfiler` with `record_curve_point()` and `aubc()` — exactly the
   methods we need. No new profiler subclass needed.

---

## 3. The two simultaneous baselines

After this addition the project has **two** baselines that approximate the
original MPaGE behaviour. They are not redundant — they probe different
things:

| Name | Dispatch | What it isolates |
|------|----------|------------------|
| `mpage_budget` (BMAB ablation) | `mpage_bmab.main --ablation mpage_budget` | BMAB framework with all adaptive components turned off. Tests whether the framework *itself* introduces overhead. Same loop structure as BMAB but no bandit / no PH / no diversity reward. |
| `mpage_orig`   (true upstream) | `mpage_bmab.mpage_orig`                  | Literally the upstream `MPaGE` class capped at `B` LLM calls. Tests BMAB-LLM against the actual published baseline. |

For the thesis, the **`mpage_orig`** row is the one referenced in
IDEA.md §4.3 as "MPaGE-budget". The `mpage_budget` ablation is a useful
sanity check: if `mpage_orig` and `mpage_budget` produce very different
AUBC, that's a sign the BMAB framework has framework-level overhead
worth investigating.

---

## 4. Quickstart

```bash
# from the project root that contains mpage_bmab/

# 1. Validate the MPaGE-orig pipeline (~30 cheap calls, 2 runs)
mpage_bmab/experiments/run_mpage_smoke.sh

# 2. Run the headline MPaGE-orig vs BMAB-LLM-full comparison (24 runs)
mpage_bmab/experiments/run_mpage_compare.sh

# 3. Inspect the result
cat mpage_bmab/experiments/results/summary.csv | column -ts,
```

The launcher chains the sweep with `aggregate.py` and `compare.py
--baseline full`, so when it finishes you already have a Markdown table of
mean AUBC + Wilcoxon p-values for `mpage_orig` vs `full`.

You can also invoke the runner directly for a single cell:

```bash
mpage_bmab/.venv/bin/python -m mpage_bmab.mpage_orig \
    --task bi_tsp --budget 50 --seed 2025 \
    --log_dir mpage_bmab/experiments/results/mpage_orig/bi_tsp/B50/seed2025
```

---

## 5. Available suites

| Suite | Runs | Cells | What for |
|-------|-----:|-------|----------|
| `mpage_smoke` | 2 | 1 `mpage_orig` + 1 `full` on `bi_tsp` B=15, seed 2025 | Cheapest sanity check |
| `mpage_compare` | 24 | `{mpage_orig, full} × bi_tsp × {25,50,100,200} × 3 seeds` | The headline AUBC comparison |
| `mpage_compare_full` | 160 | `{mpage_orig, full} × 4 tasks × 4 budgets × 5 seeds` | Full matrix; expensive |
| `all_methods` | 60 | `{mpage_orig, full, no_ph, no_diversity, op_only} × bi_tsp × {25,50,100,200} × 3 seeds` | Fully populates an AUBC table that includes both the upstream baseline and all 4 BMAB ablations |

All suites are launched via `mpage_bmab/experiments/run.py --suite <name>`.

---

## 6. How budget is counted

For *fair* comparison with BMAB-LLM, every LLM call costs 1 unit:

| Call site | Original MPaGE | BMAB-LLM | `MPaGEBudget` (new) |
|-----------|----------------|----------|---------------------|
| Heuristic generation (I1/E1/E2/M1/M2) | counts as 1 (only if profiler present and sample is valid) | counts as 1 always | counts as 1 always (incl. invalids) |
| Cluster call | **NOT counted** | counts as 1 | counts as 1 |
| Suggestion (review) call | counts as 1 if used | counts as 1 if used | counts as 1 if used (via the wrapped sampler path) |

The upstream `MPaGE._sample_evaluate_register` only increments
`_tot_sample_nums` (a) when a profiler is attached *and* (b) after a
successful evaluation. The `MPaGEBudget` subclass moves the increment to
the **top** of `_sample_evaluate_register` so every attempt costs budget,
matching `BudgetTracker.charge()` in BMAB-LLM. The cluster sampler is
monkey-patched in `__init__` to do the same.

This is the right accounting for a thesis-grade comparison because both
methods are measured against the *same* B "API requests" — that's the
quantity a thesis reviewer cares about, not internal bookkeeping.

---

## 7. Comparing AUBC

The aggregator and comparator are method-agnostic — they read `aubc.json`
files from `experiments/results/<ablation>/<task>/B<budget>/seed<seed>/`
regardless of whether the ablation is a BMAB-LLM variant or `mpage_orig`.

```bash
# Walk the results tree, write summary.csv with mean±std table to stdout
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.aggregate

# Pair mpage_orig against full per (task, budget) with Wilcoxon signed-rank
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.compare \
    --baseline full --metric aubc

# Use mpage_orig as the baseline if you want to show "how much better
# BMAB-LLM is than the upstream"
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.compare \
    --baseline mpage_orig --metric aubc
```

The output Markdown table contains one row per `(task, budget)` cell with
columns `n_seeds, <baseline>_mean, <other>_mean, Δ (baseline−other),
Wilcoxon W, p`. This is the table that goes into the thesis chapter.

---

## 8. Caveats and fairness notes

* **`pop_size` defaults differ.** The upstream MPaGE auto-adjusts `pop_size`
  based on `max_sample_nums`: 5 for B<200, 10 for 200≤B<1000, etc. (see
  [`eoh.py` lines 117–136](../_llm4ad/method/LLMPFG/eoh.py#L117-L136)). The
  BMAB-LLM default is `pop_size=6` regardless. To make the comparison
  cleaner, the new runner accepts `--pop_size`; the suite presets use 5
  (matching MPaGE-orig at B<200). If you want exact matching, pass
  `--pop_size 6` to `mpage_orig` too.
* **Init phase length.** `MPaGEBudget._thread_init_population` stops on the
  same budget as evolution, so a tiny B<5 may exhaust the budget during
  init before any operator runs. BMAB-LLM has the same property by
  construction — both methods are equally subject to it.
* **Threading.** The upstream class uses a thread/process pool for
  evaluation; BMAB-LLM evaluates inline. This affects wall-clock latency
  but not the LLM-call count. For thesis purposes, AUBC depends only on
  the call count, so this is fine.
* **Cluster call sometimes skipped.** The upstream loop only calls the
  cluster LLM when `len(indivs) >= 3` after PFG selection. With small
  populations this gate may not fire, in which case the budget cost for
  that generation is just 1 (heuristic call). BMAB-LLM calls the cluster
  LLM every generation. This makes BMAB-LLM look "more expensive per
  generation" — but because the comparison is at *fixed total budget*, not
  fixed generation count, this only changes the number of generations
  each method gets to run, not the fairness of the comparison.
* **Seeding.** Both methods seed `random` and `numpy.random` with the
  passed seed. The LLM responses themselves are not deterministic (server-
  side temperature etc.); reproducibility across seeds is statistical,
  hence the `--seeds` parameter and Wilcoxon test.

---

## 9. External baselines (EoH, MEoH, ReEvo, HSEvo)

IDEA.md §4.3 lists external methods (EoH, MEoH, ReEvo, HSEvo) for
"external" comparison. These are **not** implemented in this project
because they live in unrelated repositories with different APIs and
dependencies. The thesis can include their numbers by:

1. Running each external method's official implementation under the same
   `(task, budget, seed)` matrix in their own environment.
2. Hand-copying the resulting AUBC / HV numbers into a CSV with the same
   schema as `summary.csv` (`ablation, task, budget, seed, aubc, hv_final, …`).
3. Concatenating the CSV with the BMAB summary and re-running
   `compare.py --baseline full`.

If you want me to wire one or more external methods into the harness
directly, that's a separate task — it requires their source code as a
dependency in `requirements.txt` and an adapter module similar to
`mpage_orig.py` for each.

---

## Quick reference card

```bash
# Pipeline check (2 runs)
mpage_bmab/experiments/run_mpage_smoke.sh

# Headline MPaGE-orig vs BMAB-LLM-full (24 runs)
mpage_bmab/experiments/run_mpage_compare.sh

# All methods on bi_tsp × all budgets × 3 seeds (60 runs)
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.run --suite all_methods

# Single mpage_orig cell
mpage_bmab/.venv/bin/python -m mpage_bmab.mpage_orig \
    --task bi_tsp --budget 50 --seed 2025

# Statistical comparison
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.compare \
    --baseline full --metric aubc
```
