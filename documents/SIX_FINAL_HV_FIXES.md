# Six Final-HV Issues and Fixes

This note summarizes the six implementation issues that were fixed to make
BMAB-LLM optimize the final heuristic-population hypervolume (HV) more directly.
Each item points to the actual code that implements the fix.

For experiment naming, the current fixed method is `full`, which uses
`reward_mode='final_hv'`. The ablations `dense_reward` and `hybrid_reward` keep
the other fixes and change only the reward quality signal. This distinction is
documented in `mpage_bmab/documents/HV_FINAL_EXPERIMENTS.md:7-25` and
`mpage_bmab/documents/REWARD.md:99-111`.

## 1. Bandit reward was not aligned with final managed-population HV

### Issue

The old dense reward encouraged immediate hypervolume improvement (HVI) against
the current heuristic front. That can be useful feedback, but it is not exactly
the thesis objective: the final reported score is HV after the managed
population cap is applied. A candidate can have immediate HVI but still fail to
survive the final population-management step.

There is a related reward-path risk: the bandit update must use the score of the
actual generated and evaluated child. If reward is computed from stale
population state, a proxy, or an assumed child, the arm update no longer
reflects what the LLM call really produced.

### Fix

The default `full` method now uses `reward_mode='final_hv'`. The reward computes
the HV before and after inserting the candidate, applies the same population cap
through `managed_scores(...)`, and rewards the normalized managed-population HV
gain.

Code references:

- `mpage_bmab/main.py:99-103` defines `--reward_mode` with default `final_hv`.
- `mpage_bmab/main.py:150-168` passes `reward_mode` into `BMABLLM`.
- `mpage_bmab/bmab_llm.py:126-133` constructs `RewardComputer`.
- `mpage_bmab/reward.py:275-296` computes `managed_hv_delta` and selects
  `dense`, `hybrid`, or `final_hv`.
- `mpage_bmab/bmab_llm.py:320-339` records scores before generation, calls
  `_sample_eval_register(...)`, then computes reward with the returned
  `new_score`.
- `mpage_bmab/bmab_llm.py:370-431` charges budget, samples code, evaluates it,
  stores `func.score = score_list`, registers the function, and returns
  `(True, score_list, func)`.
- `mpage_bmab/bmab_llm.py:341-346` updates the operator and cluster bandits with
  that reward.

The ablation mapping is in `mpage_bmab/main.py:46-50`:

- `full`: fixed method with final-HV reward.
- `dense_reward`: fixed method with legacy immediate-HVI reward.
- `hybrid_reward`: fixed method with `0.5 * dense + 0.5 * final_hv`.

## 2. Pending valid offspring could be lost at the end of a budgeted run

### Issue

The inherited MPaGE population only applies survivor selection after a full
generation of `pop_size` offspring is accumulated. BMAB-LLM can stop when the
LLM-call budget is exhausted in the middle of a generation. Without an explicit
flush, valid pending offspring in `_next_gen_pop` may not enter the final
managed population, so final HV can under-report useful generated heuristics.

### Fix

The population class exposes `flush_pending()` and `pending_population()`.
BMAB-LLM includes pending valid offspring in reward-state calculations and
flushes them into the managed population before finishing the run.

Code references:

- `mpage_bmab/_llm4ad/method/LLMPFG/population.py:235-265` implements
  `flush_pending()` and `pending_population()`.
- `mpage_bmab/bmab_llm.py:512-520` allows `_population_scores(...)` to include
  pending valid offspring.
- `mpage_bmab/bmab_llm.py:320-330` computes reward-state HV/diversity with
  `include_pending=True`.
- `mpage_bmab/bmab_llm.py:522-528` calls `flush_pending()` in `_finish()`.

## 3. The final budget unit could be wasted on clustering only

### Issue

Clustering is an LLM call. Near the end of a run, BMAB-LLM could spend the last
available budget unit on a cluster call, leaving no remaining budget to generate
and evaluate a new heuristic. That cannot improve final HV.

### Fix

Before making a real cluster call, BMAB-LLM checks whether the remaining budget
can afford both clustering and at least one generation call. If not, it skips
the clustering LLM call and uses a fallback singleton partition, preserving the
last call for heuristic generation.

Code references:

- `mpage_bmab/bmab_llm.py:250-262` implements the budget guard and fallback
  partition.
- `mpage_bmab/cluster_manager.py:97-102` exposes `fallback_partition(...)` and
  `cluster_quality(...)` for this no-cluster-call path.

## 4. Cluster priors were weakly connected to final-HV usefulness

### Issue

Using only a simple score proxy for cluster quality can bias the cluster bandit
toward heuristics that look good on one objective but do not contribute much to
the outer Pareto front or to final HV. Runtime also matters because the benchmark
score is multi-objective.

### Fix

Cluster quality is now a front-aware prior. It combines:

- the cluster's HV contribution to the current heuristic front;
- the best inner-HV proxy in the cluster;
- the best runtime proxy in the cluster.

The values are normalized before being used to warm-start the cluster bandit.

Code references:

- `mpage_bmab/cluster_manager.py:106-118` documents the three signals.
- `mpage_bmab/cluster_manager.py:131-158` computes HV contribution, inner-HV
  proxy, and runtime proxy.
- `mpage_bmab/cluster_manager.py:159-163` normalizes the prior.
- `mpage_bmab/bandit.py:183-223` consumes `cluster_quality` when resetting the
  per-generation cluster bandit.

## 5. Late-stage cluster exploration stayed too high

### Issue

Exploration is useful early, but near the end of a strict budget it can hurt
final HV by spending scarce calls on uncertain arms. A previous budget-pressure
term that was added equally to every arm would not change the UCB argmax, so it
did not truly anneal exploration.

### Fix

The cluster bandit now scales the UCB exploration term by the remaining budget
fraction. This gives broad exploration early and stronger exploitation near the
end.

Code references:

- `mpage_bmab/bandit.py:227-259` computes `explore_scale` from
  `budget_fraction ** gamma_budget` and multiplies the UCB exploration term by
  it.
- `mpage_bmab/bmab_llm.py:292-295` passes the live remaining-budget fraction
  into `ClusterBandit.select(...)`.
- `mpage_bmab/main.py:93-107` exposes `--gamma_budget` and
  `--disable_budget_annealing`.
- `mpage_bmab/main.py:46-50` defines the `no_budget_anneal` ablation.

## 6. Parent selection underused PFG-selected elites

### Issue

BMAB-LLM expands the clustering pool to all current population members so the
cluster bandit has a richer search space. However, if parent selection is
uniform within selected clusters, the strongest PFG-selected elites can be
diluted by weaker population members.

### Fix

Parent sampling is weighted. Candidates that also appear in the PFG-selected
elite set receive weight `3.0`; other candidates receive weight `1.0`. Mutation
still selects a parent from the chosen cluster, and crossover still chooses the
second parent from another cluster, preserving the MPaGE intra-cluster mutation
and inter-cluster crossover rule.

Code references:

- `mpage_bmab/bmab_llm.py:237-246` obtains the PFG elites and expands the
  clustering pool to all population members.
- `mpage_bmab/bmab_llm.py:451-461` uses weighted parent choice for mutation.
- `mpage_bmab/bmab_llm.py:463-496` uses weighted parent choice for crossover.
- `mpage_bmab/bmab_llm.py:503-510` implements the `3.0` vs `1.0` weighting.

## Note on the count

The experiment plan lists six final-HV fixes in
`mpage_bmab/documents/HV_FINAL_EXPERIMENTS.md:29-36`. This note keeps the same
six-issue structure, while grouping the reward-mode alignment and actual-child
reward path together because they are implemented in the same reward pipeline.
