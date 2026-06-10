# Hyperparameter Settings for MPaGE-orig and BMAB-LLM

This document fixes every hyperparameter for the thesis comparison. Values
are chosen so that

* both methods receive **the same** LLM-call budget, `pop_size`, seeds,
  task, and model, leaving the *scheduler* (round-robin vs two-layer
  bandit) as the only material difference;
* the MPaGE paper's published values are respected where they exist
  (ε for grid-vs-rank, the four-operator alphabet, the grid resolution
  `GK = 4`);
* BMAB-LLM's bandit / Page-Hinkley / reward parameters match
  [IDEA.md §3.2–§3.4](IDEA.md) and the empirical guidance in
  [OFFSPRING_FAQ.md §5](OFFSPRING_FAQ.md).

Where the suite presets in [experiments/configs.py](../experiments/configs.py)
already match these values, the harness applies them automatically. Where
they did not, the section ["Where this is wired"](#where-this-is-wired)
records the code changes made.

---

## Table of contents

1. [Shared experimental matrix](#1-shared-experimental-matrix)
2. [Population size — per budget and per task](#2-population-size--per-budget-and-per-task)
3. [Two-layer bandit (BMAB-LLM only)](#3-two-layer-bandit-bmab-llm-only)
4. [Page-Hinkley drift detector (BMAB-LLM only)](#4-page-hinkley-drift-detector-bmab-llm-only)
5. [Reward function (BMAB-LLM only)](#5-reward-function-bmab-llm-only)
6. [PFG grid (both methods)](#6-pfg-grid-both-methods)
7. [Operator alphabet (both methods)](#7-operator-alphabet-both-methods)
8. [LLM and infrastructure](#8-llm-and-infrastructure)
9. [Ablation overrides](#9-ablation-overrides)
10. [Where this is wired](#10-where-this-is-wired)
11. [One-line summary table](#11-one-line-summary-table)

---

## 1. Shared experimental matrix

These three are set in [experiments/configs.py](../experiments/configs.py)
and apply to every cell in every suite. Both MPaGE-orig and BMAB-LLM use
the same values.

| Parameter | Value | Source |
|-----------|-------|--------|
| LLM-call budget `B` | `{25, 50, 100, 200}` | IDEA §4.4 |
| Seeds | `{2025, 2026, 2027, 2028, 2029}` | IDEA §4.5 (5 seeds) |
| Tasks | `bi_tsp`, `tri_tsp`, `bi_cvrp`, `bi_kp` | IDEA §4.1 |

---

## 2. Population size — per budget and per task

This is the single most consequential hyperparameter for the comparison
because (a) it controls how much in-generation data both bandits get and
(b) it indirectly controls the total number of generations available
under fixed `B`.

### Baseline (per budget)

Derived from OFFSPRING_FAQ.md §5.B's rule `pop_size ≈ 4 · K` with the LLM
clusterer typically producing `K = 2–3` clusters:

| Budget `B` | pop_size | n_gen ≈ `B / (pop + 1)` | obs/arm at K=3 |
|-----------:|---------:|------------------------:|---------------:|
| 25         | 4        | ~5                      | 0.67           |
| 50         | 6        | ~7                      | 1.00           |
| 100        | 8        | ~11                     | 1.33           |
| 200        | 10       | ~18                     | 1.67           |

At small `B` the cluster bandit is largely prior-driven; this is
intentional — the alternative (larger `pop_size`) sacrifices generation
count.

### Per-task adjustment

| Task     | Bonus over baseline | Reason |
|----------|--------------------:|--------|
| `bi_tsp` | 0 | 2-D Pareto front |
| `tri_tsp`| **+2** | 3-D front needs more elites to span |
| `bi_cvrp`| 0 | 2-D |
| `bi_kp`  | 0 | 2-D |

### Final `pop_size` matrix

| Task \ B | 25 | 50 | 100 | 200 |
|----------|---:|---:|----:|----:|
| `bi_tsp` | 4  | 6  | 8   | 10  |
| `tri_tsp`| 6  | 8  | 10  | 12  |
| `bi_cvrp`| 4  | 6  | 8   | 10  |
| `bi_kp`  | 4  | 6  | 8   | 10  |

This is what `pop_size_for(task, budget)` in `configs.py` returns. Both
`mpage_bmab.main` (BMAB-LLM) and `mpage_bmab.mpage_orig` (the
budget-capped MPaGE class) receive this value via `--pop_size` when
launched by [experiments/run.py](../experiments/run.py), overriding the
upstream MPaGE's own auto-default (which would otherwise be 5 at B<200).

### Why override MPaGE's auto-default?

The vendored class auto-adjusts `pop_size` based on `max_sample_nums`
(see [eoh.py:117-136](../_llm4ad/method/LLMPFG/eoh.py#L117-L136)): 5 for
B<200, 10 for 200≤B<1000, etc. Letting it run with its native default
would give MPaGE *different* population sizes than BMAB-LLM, which
would confound the comparison. By passing `--pop_size` explicitly the
two methods are matched cell-by-cell.

---

## 3. Two-layer bandit (BMAB-LLM only)

Defaults from [bmab_llm.py:62-71](../bmab_llm.py#L62-L71) — kept as-is
because they match [IDEA.md §3.2](IDEA.md).

| Parameter | Value | CLI flag | Role |
|-----------|------:|----------|------|
| `c_explore_op`      | `1.0` | `--c_op`           | UCB1 exploration coefficient for the operator bandit (textbook value is `√2 ≈ 1.41`; 1.0 is slightly more exploitative, fine for only 2 arms). |
| `c_explore_cluster` | `1.0` | `--c_cluster`      | UCB1 exploration coefficient for the cluster bandit. Same justification. |
| `gamma_budget`      | `0.5` | `--gamma_budget`   | Exponent for remaining-budget exploration annealing in the cluster bandit. Exploration is scaled by `(remaining/total)^gamma_budget`. |
| `budget_annealing`  | `True` | `--disable_budget_annealing` to turn off | Makes late-budget cluster selection more exploitative for final-HV quality. |
| `prior_n`           | `1.0` | (no CLI flag)      | Virtual count for warm-start. `1` ⇒ first real observation has equal weight. |
| `prior_reward`      | `0.5` | (no CLI flag)      | Neutral mid-range prior reward used when no quality signal is available. |

Operator bandit's softmax temperature is fixed at `1.0`
([bmab_llm.py:244](../bmab_llm.py#L244)). Cluster bandit's tie-breaking
is uniform random ([bandit.py:245-247](../bandit.py#L245-L247)).

---

## 4. Page-Hinkley drift detector (BMAB-LLM only)

From [bandit.py:43-44](../bandit.py#L43-L44):

| Parameter | Value | CLI flag | Role |
|-----------|------:|----------|------|
| `ph_delta`     | `0.005` | `--ph_delta`     | Slack added to each observation; ~0.5 % of the [0, 1] reward range. Lower ⇒ more sensitive, more false alarms. |
| `ph_threshold` | `0.5`   | `--ph_threshold` | Detection threshold on the gap `M_t − m_t`. Lower ⇒ faster detection but more false alarms. |

These were empirically validated to give a **~5 % false-alarm rate**
under stationary noise on rank-normalised rewards in [0, 1] (see
[PAGE_HINKLEY.md §10](PAGE_HINKLEY.md#10-empirical-sanity-check)). They
also detect a 50 % drop in reward within ~10 observations.

Keep at defaults. The `no_ph` ablation sets `ph_threshold = 1e9` which
effectively disables drift detection — that arm is for isolating the
contribution of Page-Hinkley to AUBC.

---

## 5. Reward function (BMAB-LLM only)

From [bmab_llm.py](../bmab_llm.py), [reward.py](../reward.py) and matching
[IDEA.md §3.3](IDEA.md), with the post-analysis final-HV fix:

| Parameter | Value | CLI flag | Role |
|-----------|------:|----------|------|
| `reward_mode`   | `final_hv` | `--reward_mode` | Selects the quality signal: `final_hv`, `dense`, or `hybrid`. |
| `w_quality`     | `1.0` | `--w_quality`   | Weight of the selected quality signal. |
| `w_diversity`   | `0.3` | `--w_diversity` | Weight of ΔCDI diversity gain. Anti-collapse. |
| `w_rank`        | `0.2` | `--w_rank`      | Weight of rank-score (any-objective-better fraction). Dense smoothing. |
| `reward_penalty`| `1.0` | (no CLI flag)   | Penalty subtracted when the heuristic is invalid. Same magnitude as max positive reward. |
| `rank_window`   | `50`  | (no CLI flag)   | Window length for immediate-HVI and final-HV rolling-max normalisation. |

The valid-heuristic reward is:

```text
R = w_quality * quality_signal + w_diversity * max(0, ΔCDI)
    + w_rank * rank_score
```

where `quality_signal` depends on `reward_mode`:

| Mode | Quality signal | Intended comparison |
|------|----------------|---------------------|
| `final_hv` | normalized HV gain after adding the candidate and applying the managed-population cap | Default `full`; aligns the bandit with terminal population HV |
| `dense` | normalized immediate HVI against the current heuristic front | `dense_reward`; closest to the legacy reward before the final-HV fix |
| `hybrid` | `0.5 * dense + 0.5 * final_hv` | `hybrid_reward`; tests whether dense early feedback plus final-HV alignment is better |

Invalid heuristics return only `-reward_penalty`. With the default weights,
positive rewards still stay on a small bounded scale suitable for UCB1. If you
change any `w_*` significantly, also re-tune `ph_delta` and `ph_threshold`
proportionally (PAGE_HINKLEY.md §4).

---

## 6. PFG grid (both methods)

From [population.py:143](../_llm4ad/method/LLMPFG/population.py#L143)
inside `parent_selection`:

| Parameter | Value | Role |
|-----------|------:|------|
| `GK` (cells per axis) | `4` | Grid resolution. With 2-D objectives → 4×4 cell projections, 4-cell pairs available for "adjacent cells" selection. From the MPaGE paper. |
| `sigma` (grid padding) | `0.01` | Small padding so boundary points don't fall outside the grid due to FP rounding. |
| `epsilon` (grid-vs-rank prob) | `0.8` | 80 % of the time use the adjacent-grid-cells rule; 20 % use rank-weighted random over the whole population. From the MPaGE paper. |
| `selection_num` | `2` | Only matters in the 20 % rank-weighted branch — number of rank-weighted parents to pick. |

These are inside `parent_selection` and not exposed as CLI flags. Do
not change them without re-reading the MPaGE paper's grid analysis.

---

## 7. Operator alphabet (both methods)

The five-operator taxonomy `I1, E1, E2, M1, M2` is enabled in full:

| Operator flag | Default | Effect of disabling |
|---------------|---------|---------------------|
| `use_e1_operator` | `True` | Lose the primary crossover variant. |
| `use_e2_operator` | `True` | Lose alternative crossover framing. |
| `use_m1_operator` | `True` | Lose the primary mutation variant. |
| `use_m2_operator` | `True` | Lose alternative mutation framing. |

Both methods use the full alphabet. BMAB-LLM groups `{E1, E2}` under the
operator-bandit arm `crossover` and `{M1, M2}` under `mutate`; within
each group, the choice between the two variants is random
([bmab_llm.py:418-461](../bmab_llm.py#L418-L461)). MPaGE runs all four
in fixed sequence per generation.

---

## 8. LLM and infrastructure

| Parameter | Value | CLI flag | Role |
|-----------|-------|----------|------|
| `llm_model`         | `gpt-4o-mini` | `--llm_model`         | Heuristic-generation model. Cheap and stable. |
| `llm_cluster_model` | `gpt-4o-mini` | `--llm_cluster_model` | Cluster model — same model is fine; cluster prompt is simple structured output. |
| `openai_base_url`   | `https://api.openai.com` | `--openai_base_url` | Endpoint. Override for OpenAI-compatible proxies. |
| `timeout`           | `30` s | (set in `main.py`)   | LLM request timeout. |
| `--secret` / `--secret_cluster` | files | flags | API-key paths. Both methods read from `secret.txt` / `secret_cluster.txt` by default. |
| `multi_thread_or_process_eval` | `thread` (MPaGE only) | `--multi_thread_or_process` | Evaluation pool for MPaGE's subprocess executor. `thread` is safer cross-platform. |
| `budget_mode` (BMAB only) | `call` | `--budget_mode` | Unit cost per LLM request. `token` mode is implemented but not used in the headline sweep. |
| `--review` | `False` (default) | flag | LLM-based suggestion call before crossover (+1 budget unit per call). Kept off in the headline sweep so the budget comparison is clean. |

If your thesis chapter calls for stronger results at high `B`, switch
the heuristic LLM to `gpt-4o` only for the `B=200` cells. Keep
`gpt-4o-mini` everywhere else for cost.

---

## 9. Ablation overrides

The `ABLATIONS` preset table in
[main.py:42-55](../main.py#L42-L55) overrides the above defaults for the
four BMAB-LLM ablations referenced in IDEA.md §4.3:

| Ablation       | Overrides applied | What it isolates |
|----------------|-------------------|------------------|
| `full`         | (none)            | The proposed fixed system: `reward_mode=final_hv`, budget annealing on |
| `dense_reward` | `reward_mode = dense` | Legacy immediate-HVI reward inside the fixed implementation |
| `hybrid_reward` | `reward_mode = hybrid` | Blend of immediate-HVI feedback and final-population HV feedback |
| `no_budget_anneal` | `disable_budget_annealing = True` | Marginal value of remaining-budget exploration annealing |
| `no_ph`        | `ph_threshold = 1e9` | Marginal value of Page-Hinkley |
| `no_diversity` | `w_diversity = 0.0` | Marginal value of ΔCDI in reward |
| `op_only`      | `disable_cluster_bandit = True` (cluster picks uniform random) | Marginal value of the cluster bandit |
| `cluster_only` | `disable_operator_bandit = True` (operators round-robin) | Marginal value of the operator bandit |
| `mpage_budget` | both bandits off + `ph_threshold = 1e9` + `w_diversity = 0` + `w_rank = 0` | A simulated MPaGE baseline within the BMAB framework |
| `mpage_orig`   | (n/a — dispatches to `mpage_bmab.mpage_orig`) | True upstream MPaGE class, budget-capped |

All bandit/PH/reward parameters not in the override dict stay at their
defaults listed in §3–§5.

---

## 10. Where this is wired

Three code changes accompanied the writing of this document:

1. **`pop_size_for(task, budget)`** added to
   [experiments/configs.py](../experiments/configs.py). Returns the
   final `pop_size` from §2 by combining the budget baseline with the
   per-task bonus.
2. **`experiments/run.py`** now calls `pop_size_for(task, budget)`
   instead of the old per-task `POP_SIZES` dict, so every sweep cell
   gets the right `pop_size` automatically.
3. **`POP_SIZES` is preserved** as a backward-compat alias (populated
   from `pop_size_for(task, 50)` at import) so any external script that
   imported it still works.

The other hyperparameters in this document are already the defaults in
[main.py](../main.py) and [bmab_llm.py](../bmab_llm.py), so no other
code changes are needed.

---

## 11. One-line summary table

| Parameter | Value | Source |
|-----------|------:|--------|
| Budgets | `{25, 50, 100, 200}` | IDEA §4.4 |
| Seeds | `{2025–2029}` (5) | IDEA §4.5 |
| Tasks | `bi_tsp, tri_tsp, bi_cvrp, bi_kp` | IDEA §4.1 |
| pop_size at `B=25` | 4 (bi_*) / 6 (tri_tsp) | OFFSPRING_FAQ §5.B |
| pop_size at `B=50` | 6 (bi_*) / 8 (tri_tsp) | OFFSPRING_FAQ §5.B |
| pop_size at `B=100` | 8 (bi_*) / 10 (tri_tsp) | OFFSPRING_FAQ §5.B |
| pop_size at `B=200` | 10 (bi_*) / 12 (tri_tsp) | OFFSPRING_FAQ §5.B |
| selection_num | 2 | MPaGE paper |
| c_op, c_cluster | 1.0 each | IDEA §3.2 |
| gamma_budget | 0.5 | IDEA §3.2 / final-HV fix |
| budget annealing | on | final-HV fix |
| ph_delta | 0.005 | IDEA §3.4 |
| ph_threshold | 0.5 | IDEA §3.4 |
| reward_mode | `final_hv` | final-HV fix |
| w_quality, w_diversity, w_rank | 1.0, 0.3, 0.2 | IDEA §3.3 |
| reward_penalty | 1.0 | IDEA §3.3 |
| rank_window | 50 | empirical |
| GK (grid axis cells) | 4 | MPaGE paper |
| sigma (grid padding) | 0.01 | MPaGE paper |
| epsilon (grid-vs-rank prob) | 0.8 | MPaGE paper |
| All four operators | enabled | MPaGE paper |
| LLM model | gpt-4o-mini | cost / stability |
| LLM timeout | 30 s | safety |
| Budget mode | `call` | IDEA §3.5 |
| LLM review | off | clean comparison |

Anything not in this table is at its code-level default (see
[bmab_llm.py](../bmab_llm.py) and [main.py](../main.py)) and does not
need changing for the thesis sweep.
