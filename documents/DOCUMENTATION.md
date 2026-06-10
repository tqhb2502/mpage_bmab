# BMAB-LLM: Full Project Documentation

> A self-contained extension of [MPaGE](https://arxiv.org/abs/2507.20923) that
> drives LLM-based heuristic design for **Multi-Objective Combinatorial
> Optimisation (MOCOP)** under a strict budget on LLM API calls, using a
> two-layer Budgeted Multi-Objective Multi-Armed Bandit.

This document is the single source of truth for the project. It is written so
that someone with no prior knowledge of MPaGE, the bandit literature, or this
codebase can install, understand, run, and extend it.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [Background you need to know](#2-background-you-need-to-know)
3. [How the system works end-to-end](#3-how-the-system-works-end-to-end)
4. [Repository layout](#4-repository-layout)
5. [File-by-file reference](#5-file-by-file-reference)
6. [The vendored `_llm4ad/` snapshot](#6-the-vendored-_llm4ad-snapshot)
7. [Setup and installation](#7-setup-and-installation)
8. [Running a single experiment](#8-running-a-single-experiment)
9. [The experimental harness](#9-the-experimental-harness)
10. [Output artefacts](#10-output-artefacts)
11. [Configuration reference (every CLI flag)](#11-configuration-reference-every-cli-flag)
12. [Extending the project](#12-extending-the-project)
13. [Troubleshooting and FAQ](#13-troubleshooting-and-faq)

---

## 1. What this project does

### The problem

You have a **multi-objective combinatorial optimisation** problem (e.g. find a
TSP tour that is short *and* fast to compute). You want a *heuristic* — a piece
of Python code that, given an instance of the problem, returns a good
solution.

You ask an LLM to write that heuristic. The LLM is **expensive** (each call
costs money and seconds), so you have a fixed budget `B` of API calls.
**Question: how do you spend `B` calls so that the resulting Pareto front of
heuristics is as good as possible?**

### The original MPaGE answer

MPaGE (Ha et al., 2025) iterates a population of heuristics through generations:

1. Select an elite set with **Pareto-Front-Grid (PFG)** selection.
2. Use a cheap LLM to **cluster** the elites by behavioural similarity.
3. For each generation apply four operators in fixed sequence — `E1`, `E2`
   (cross-cluster crossover), `M1`, `M2` (intra-cluster mutation) — to produce
   offspring.
4. Stop after `max_generations` or `max_sample_nums` calls.

### Three weaknesses BMAB-LLM fixes

| # | Weakness in MPaGE | BMAB-LLM's fix |
|---|--------------------|----------------|
| W1 | Operator probability is **fixed** (round-robin); cannot adapt as the population matures | A persistent **UCB1 bandit over `{Mutate, Crossover}`** carries statistics across generations |
| W2 | Cluster selection is **uniform random**; promising and barren clusters get equal budget | A per-generation **Budgeted UCB1 bandit** over clusters, warm-started with cluster-quality + operator-level priors, with a **Page-Hinkley** drift test that resets dead arms |
| W3 | **No explicit LLM-call budget**; loop only stops on generation count | Every LLM call (init, cluster, mutate, crossover, suggestion) is **debited from a hard budget `B`**; loop terminates exactly when `B` is exhausted |

The current fixed implementation also adds final-HV-oriented corrections:
the default reward quality signal is managed-population HV gain
(`reward_mode=final_hv`), pending valid offspring are flushed into the final
population before reporting, the final usable budget unit is reserved for
candidate generation rather than a cluster-only call, cluster priors use
front-aware HV contribution plus inner-HV/runtime proxies, and parent sampling
is biased toward PFG-selected elites.

### The headline metric: AUBC

To capture not just *final* quality but the *speed* at which budget converts to
quality, this project introduces **AUBC** — Area Under the Budget-vs-HV Curve
(trapezoidal integral of HV(consumed_budget), normalised by `B`). Higher is
better; tells you how well a method does at every intermediate budget.

---

## 2. Background you need to know

### 2.1 Multi-objective optimisation in 30 seconds

A heuristic produces solutions scored on multiple objectives, e.g. `(distance,
time)`. We minimise both. A solution `a` **dominates** `b` if `a` is no worse
than `b` on every objective and strictly better on at least one. The
**Pareto front** is the set of non-dominated solutions.

To compare *fronts* with a single number we use **hypervolume (HV)**: the area
(2-D) or volume (3-D+) of objective space dominated by the front, measured
relative to a fixed reference point. Higher HV ⇒ better front.

### 2.2 Multi-Armed Bandits in 30 seconds

You have *K* slot machines. Each pull returns a stochastic reward. You don't
know the reward distributions. You have *T* pulls. Maximise total reward.

**UCB1** is the standard algorithm. For arm `a` with `n_a` pulls and mean
reward `μ_a`, score it as

```
UCB1(a) = μ_a + c · √( 2·ln(N) / n_a )
```

where `N` is the total number of pulls. Exploit the empirical mean, but boost
under-explored arms. Pick the arm with maximum UCB1 score.

**Budgeted MAB (BwK)**: each pull also has a *cost* `c_a`, and the constraint
is total cost ≤ `B` rather than `T` pulls. Score on **reward per cost**.

**Two-layer bandit (this project)**: separate the *operator* choice (persistent
across generations, only 2 arms) from the *cluster* choice (re-built every
generation). The operator bandit's softmax probability seeds the new cluster
bandit's prior on each reset → no information is lost.

### 2.3 Page-Hinkley test

An online change-point detector. Maintains `m_t = Σ(x_k − μ_k − δ)` and
`M_t = max_{k≤t} m_k`. When `M_t − m_t > λ`, the reward distribution has
**dropped** and we declare drift. Used here to reset a `(cluster, operator)`
arm whose previously good rewards have collapsed — typically because the
population has already absorbed that cluster's style and there is nothing more
to learn from it.

### 2.4 The PFG (Pareto Front Grid)

MPaGE's elite selector. Discretise the objective space into a grid; in each
non-empty cell keep the best (or one) heuristic; output the cells' best as the
elite pool. Encourages even spread along the front rather than concentration on
one tip. Implemented in `_llm4ad/method/LLMPFG/population.py`.

---

## 3. How the system works end-to-end

```
                       ┌──────────────────────────────────────────────┐
                       │           BMABLLM.run() main loop            │
                       └──────────────────────────────────────────────┘
                                          │
                                          ▼
        ┌────────────────────────── _init_population() ──────────────┐
        │  Adaptive warm-up: call LLM with I1 prompt until either    │
        │  pop_size successes OR init_max_calls attempts.            │
        │  Each call charges BudgetTracker.                          │
        └────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                      while not budget.is_exhausted():
                                          │
                                          ▼
        ┌─────────────── _evolve_one_generation() ───────────────────┐
        │  1. PFG select elite set from Population                   │
        │  2. ClusterManager.cluster()                               │
        │       → 1 LLM call (cluster LLM), debits 1 from budget     │
        │       → returns partition + cluster_quality priors         │
        │  3. ClusterBandit.reset(n_clusters, quality, op_priors)    │
        │       op_priors come from OperatorBandit.softmax_probs()   │
        │  4. for ~pop_size offspring while budget remains:          │
        │     a. op = OperatorBandit.select()        # μ or χ        │
        │     b. arm = ClusterBandit.select(op)      # (cluster,op)  │
        │     c. parents = pick from partition[cluster]              │
        │     d. prompt = E1/E2 (χ) or M1/M2 (μ)                     │
        │     e. (sample, evaluate, register) → 1 budget unit        │
        │     f. reward = quality + diversity gain + rank − penalty  │
        │     g. OperatorBandit.update(op, reward)                   │
        │     h. drift = ClusterBandit.update(arm, reward)           │
        │           if drift: arm reset to optimistic prior          │
        │  5. Profiler.record_curve_point(consumed_budget, pop)      │
        │     Profiler.record_bandit_state(gen, op_stats, cl_stats)  │
        └────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                       Profiler.finish() — writes JSON artefacts
                       including aubc.json (the headline metric).
```

Three resources flow through this loop:

* **Budget** flows from `BudgetTracker` into every LLM call. A call that can't
  be paid for is rejected; the loop exits.
* **Heuristics** flow from the LLM through `EoHSampler` → `SecureEvaluator` →
  `Population`. Invalid or timed-out heuristics are penalised in the reward
  but otherwise discarded.
* **Reward signal** flows from `RewardComputer` back into the two bandits and
  the Page-Hinkley drift detector, closing the adaptive loop.

---

## 4. Repository layout

```
mpage_bmab/
├── README.md                ← short intro + quick run
├── IDEA.md                  ← the refined research design (read this if you want the math)
├── DOCUMENTATION.md         ← THIS FILE
├── requirements.txt         ← pip dependencies
├── .gitignore               ← excludes .venv/, logs_bmab/, __pycache__/
├── __init__.py              ← public API of the package
│
├── main.py                  ← CLI entry point (python -m mpage_bmab.main)
├── budget.py                ← BudgetTracker (charges every LLM call)
├── bandit.py                ← OperatorBandit + ClusterBandit + PageHinkleyState
├── reward.py                ← HV utilities + diversity + RewardComputer
├── cluster_manager.py       ← Wraps the cluster LLM call + computes priors
├── profiler.py              ← BMABProfiler (extends EoHProfiler with AUBC)
├── bmab_llm.py              ← BMABLLM main orchestrator (the analogue of MPaGE)
│
├── _llm4ad/                 ← vendored snapshot of upstream MPaGE primitives
│   ├── base/                ← Function/Program, Evaluator, Sampler, ModifyCode
│   ├── tools/               ← LLM API wrappers + ProfilerBase
│   ├── method/LLMPFG/       ← PFG Population, EoHPrompt, EoHSampler, EoHProfiler
│   └── task/optimization/   ← bi_tsp_semo, tri_tsp_semo, bi_cvrp, bi_kp
│
└── experiments/             ← thesis-experiment harness (created in this project)
    ├── README.md            ← harness usage guide
    ├── configs.py           ← named ablations / tasks / budgets / seeds / suites
    ├── run.py               ← sweep launcher
    ├── aggregate.py         ← collects results into summary.csv
    ├── compare.py           ← Wilcoxon signed-rank tests
    ├── run_smoke.sh         ← 2-run pipeline check
    ├── run_headline.sh      ← 48-run headline AUBC table
    ├── run_budget50.sh      ← 80-run B=50 sweep across all 4 tasks
    └── run_full.sh          ← full 320-run matrix
```

---

## 5. File-by-file reference

### 5.1 `__init__.py` — public API

Re-exports the symbols that external code (notebooks, custom scripts) needs:

```python
from .bmab_llm import BMABLLM
from .budget import BudgetTracker
from .bandit import OperatorBandit, ClusterBandit, PageHinkleyState
from .reward import RewardComputer, hypervolume, hvi, shannon_diversity, cumulative_diversity
from .profiler import BMABProfiler
from .cluster_manager import ClusterManager
```

So you can write `from mpage_bmab import BMABLLM, BMABProfiler` without
knowing the internal module layout.

---

### 5.2 `budget.py` — the hard budget cap

Two classes:

#### `BudgetEntry` (dataclass)

A single accounting record. One per LLM call.

```python
@dataclass
class BudgetEntry:
    step: int       # monotonic call index
    label: str      # 'init' | 'cluster' | 'suggestion' | 'mutate#k' | 'crossover#k'
    cost: float     # 1.0 in call mode; n_tokens in token mode
    remaining: float  # budget left *after* this call
```

#### `BudgetTracker`

Thread-safe tracker for the global budget. Two key methods:

```python
def can_afford(self, cost: float = 1.0) -> bool:
    return self._remaining - cost >= -1e-9

def charge(self, cost: float, label: str = 'unknown') -> bool:
    """Atomic check-and-deduct. Returns False without deducting if
    the cost would push remaining below zero."""
    with self._lock:
        if self._remaining - cost < -1e-9:
            return False
        self._remaining -= cost
        self._step += 1
        self._history.append(BudgetEntry(self._step, label, float(cost),
                                         self._remaining))
        return True
```

**Why this matters**: every LLM call in the codebase first calls
`budget.charge(cost, label)` *before* issuing the HTTP request. This is the
single point of enforcement that guarantees the run stops at exactly `B` calls
— there is no other path through which the LLM can be hit.

`fraction_remaining()` returns `remaining / total` and is used by the cluster
bandit's remaining-budget exploration annealing (see §5.4).

---

### 5.3 `bandit.py` — the adaptive selection logic

Three classes, in order of complexity.

#### `PageHinkleyState`

One-sided change-point detector for **decreases** in reward. Maintains a
cumulative sum and its running maximum; when the gap exceeds threshold,
declare drift.

```python
def update(self, value: float) -> bool:
    self.n += 1
    self.mean += (value - self.mean) / self.n           # online mean
    self.sum  += value - self.mean - self.delta         # cumulative deviation
    self.max_sum = max(self.max_sum, self.sum)
    ph = self.max_sum - self.sum
    return ph > self.threshold                          # drift signal
```

The `delta` slack term prevents false alarms from natural noise; `threshold`
controls sensitivity. Defaults `δ=0.005, λ=0.5` give a ≈5 % false-alarm rate
under stationary noise (verified empirically).

#### `OperatorBandit`

Persistent UCB1 bandit over the two operators `{mutate, crossover}`. "Persistent"
means stats are kept for the entire run.

```python
def select(self) -> str:
    self._t += 1
    total_n = sum(s.n for s in self._stats.values())
    scores = {}
    for op, s in self._stats.items():
        exploit = s.reward_per_cost                       # μ_a / c_a
        explore = self._c * sqrt(2 * log(total_n) / s.n)
        scores[op] = exploit + explore
    return max(scores, key=scores.get)
```

Why **reward-per-cost** instead of plain reward? In token-aware mode mutation
and crossover may have different prompt sizes (cost), and we want to maximise
*efficiency*, not absolute reward.

`softmax_probs(temperature)` produces a probability distribution over
operators, used as a warm-start prior when the cluster bandit is reset.

#### `ClusterBandit`

Per-generation Budgeted UCB1 over `(cluster_idx, operator)` arms. Reset every
generation, warm-started from cluster quality and operator priors.

The score combines exploitation and budget-annealed exploration:

```python
exploit       = s.reward_per_cost
explore_scale = (remaining_budget / total_budget) ** gamma_budget
explore       = c · explore_scale · √(2·ln(total_n) / n_a)
score         = exploit + explore
```

The old additive budget-pressure term was uniform across arms and therefore
did not change the argmax. The fixed method scales the exploration term
itself, so cluster selection is broad early in the run and more exploitative
near the end. Pass `--disable_budget_annealing` or use the
`no_budget_anneal` ablation to turn this off.

`update(arm, reward, cost)` does two things:
1. Update arm sufficient statistics.
2. Feed the reward to the arm's `PageHinkleyState`. If drift is detected, reset
   the arm's stats to its optimistic prior and reset the PH state. Returns
   `True` iff drift was triggered.

The warm-start prior is built from:

```python
init_reward = 0.5 * (cluster_quality[k] + operator_prior[op])
```

— a blend of "this cluster looked good last generation" and "this operator
has been good overall".

---

### 5.4 `reward.py` — the bandit reward signal

Five public functions/classes.

#### `hypervolume(points, ref_point)`

Wraps `pymoo.indicators.hv.HV` with a 2-D fallback for environments without
pymoo. Returns 0.0 for empty inputs.

#### `hvi(new_point, existing, ref_point)`

Hypervolume improvement — `HV(existing ∪ {new}) − HV(existing)`. Always ≥ 0.
Returns 0 when `new_point` is dominated by `ref_point` (worse than the
reference) or when `existing ∪ {new}` adds no new dominated volume.

#### `shannon_diversity(cluster_sizes)`

Shannon-Wiener entropy `−Σ p_i·log(p_i)` of a cluster-size histogram. Used as
the SWDI metric.

#### `cumulative_diversity(scores)`

Mean pairwise Euclidean distance in objective space; used as the CDI metric and
in the reward as a diversity-gain bonus.

#### `RewardComputer`

The class that combines everything into the scalar reward fed to the bandits.

```python
def reward(self, new_score, population_scores,
           diversity_before, diversity_after, *,
           invalid=False, managed_pop_size=None):
    if invalid or new_score is None or not _is_valid_score(new_score):
        return -self._pen, {...}                    # penalty for bad heuristic

    h        = hvi(new_score, population_scores, self._ref)
    h_norm   = h / max(recent_hvi_max, ε)           # immediate-HVI signal

    managed_delta = HV(managed_scores(pop + [new], managed_pop_size)) \
                    - HV(managed_scores(pop, managed_pop_size))
    final_norm    = managed_delta / max(recent_managed_delta_max, ε)
    quality       = select_by_reward_mode(h_norm, final_norm)

    rank_score = (# popmembers any-objective-worse than new) / |pop|
    d_gain     = max(0, diversity_after - diversity_before)

    total = w_q · quality  +  w_d · d_gain  +  w_r · rank_score
    return total, breakdown
```

**Why three terms?**
- Pure HVI is sparse (most offspring don't improve the front → reward = 0).
- `rank_score` provides a dense signal (always in [0, 1]).
- `d_gain` discourages collapse onto a single Pareto region.

`reward_mode` controls the quality signal:

| Mode | Meaning |
|------|---------|
| `final_hv` | default `full`; rewards normalized managed-population HV gain |
| `dense` | `dense_reward`; rewards normalized immediate HVI |
| `hybrid` | `hybrid_reward`; averages the two signals |

**Rolling-window normalisation** of HVI keeps the reward bounded in [0, ~1]
even when early-stage hypervolume jumps are huge — this matters because UCB1's
analysis assumes bounded rewards.

---

### 5.5 `cluster_manager.py` — the LLM clustering wrapper

Bridges the cluster-LLM call to the bandit's warm-start.

```python
def cluster(self, elites: List[Function]) -> Tuple[List[List[int]],
                                                    Dict[int, float]]:
    n = len(elites)
    if n == 0:                  return [], {}
    if n < 3:                   return _fallback_partition(elites), {...}

    # 1. Pay for the call BEFORE issuing it
    if not self._budget.charge(self._call_cost, label='cluster'):
        return _fallback_partition(elites), {...}

    # 2. LLM clustering
    try:
        prompt   = EoHPrompt.get_prompt_cluster(self._task, elites, self._template)
        response = self._sampler.get_thought(prompt)
    except Exception:
        response = None

    # 3. Validate or fall back
    partition = (list(response) if _is_valid_partition(response, n)
                 else _fallback_partition(elites))
    return partition, self._cluster_quality(partition, elites)
```

Three behaviours worth highlighting:

1. **Charge before calling.** This is the budget-safety pattern: if the call
   couldn't be paid, no partition. We never spend past `B`.
2. **Singleton fallback.** Any failure (budget rejection, LLM error, malformed
   response, invalid partition) collapses to "every elite is its own cluster"
   — the loop continues degraded but does not crash.
3. **`_cluster_quality()`** maps each cluster to a `[0,1]` quality score
   combining current outer-front HV contribution, the best inner-HV proxy
   (`-score[0]`) and the best runtime proxy. This is the prior fed to
   `ClusterBandit.reset()`.

---

### 5.6 `profiler.py` — logging and the AUBC metric

Extends `EoHProfiler` (vendored from MPaGE) with three concerns specific to a
budgeted run.

```python
class BMABProfiler(EoHProfiler):
    def record_curve_point(self, budget_consumed, population):
        # Compute HV of current Pareto front, append (b, hv, |F|) to curve.

    def record_bandit_state(self, generation, operator_stats, cluster_stats):
        # Snapshot per-arm statistics at end-of-generation.

    def record_budget(self, budget):
        # Capture the full BudgetEntry history.

    def aubc(self, total_budget):
        # Trapezoidal integration of HV(b) over b ∈ [0, B], normalised by B.

    def finish(self, *, budget=None):
        # Write budget_curve.json, bandit_log.json, budget_history.json, aubc.json.
```

The `aubc()` implementation:

```python
pts = sorted(self._curve, key=lambda x: x['budget_consumed'])
xs  = [0.0] + [p['budget_consumed'] for p in pts]
ys  = [0.0] + [p['hv'] for p in pts]
if xs[-1] < total_budget:                  # extend to full budget
    xs.append(total_budget); ys.append(ys[-1])
area = 0.0
for i in range(1, len(xs)):
    area += 0.5 * (ys[i] + ys[i-1]) * (xs[i] - xs[i-1])
return area / max(total_budget, 1e-9)      # mean HV over budget axis
```

Higher AUBC means HV ramps up *faster* as budget is consumed — exactly the
"speed of converting calls into Pareto quality" we care about.

---

### 5.7 `bmab_llm.py` — the orchestrator

The 480-line main class. Owns the budget, both bandits, the population, the
cluster manager, the reward computer, and the evolutionary loop.

#### Public API

```python
class BMABLLM:
    def __init__(self, llm, llm_cluster, evaluation, *,
                 budget, ref_point, pop_size=6,
                 c_explore_op=1.0, c_explore_cluster=1.0, gamma_budget=0.5,
                 ph_delta=0.005, ph_threshold=0.5,
                 budget_annealing=True,
                 disable_cluster_bandit=False, disable_operator_bandit=False,
                 w_quality=1.0, w_diversity=0.3, w_rank=0.2, reward_penalty=1.0,
                 reward_mode='final_hv',
                 random_seed=None, profiler=None, ...)

    def run(self) -> List[Function]      # the main entry; returns final population
```

#### `run()` — the outer loop

```python
def run(self) -> List[Function]:
    try:
        self._init_population()
        self._record_curve_point()
        while not self._budget.is_exhausted():
            if self._max_generations and self._gen >= self._max_generations:
                break
            self._evolve_one_generation()
            self._record_curve_point()
    except KeyboardInterrupt:
        pass
    finally:
        self._finish()
    return list(self._population.population)
```

Two stop conditions: budget exhaustion (canonical) or an optional
`max_generations` cap (sanity). `_finish()` flushes pending valid offspring
into the managed population and records a final curve point before writing
profiler artefacts.

#### `_init_population()` — adaptive warm-up

Replaces MPaGE's fixed `pop_size` initialisation with an adaptive one:

```python
attempts = successes = 0
while (successes < self._init_target_successes
       and attempts  < self._init_max_calls
       and not self._budget.is_exhausted()
       and self._budget.can_afford(self._init_call_cost)):
    ok, _, _ = self._sample_eval_register(
        lambda: EoHPrompt.get_prompt_i1(self._task_description_str,
                                        self._function_to_evolve),
        cost=self._init_call_cost, label='init')
    attempts += 1
    if ok: successes += 1
```

This protects small budgets from being eaten entirely by warm-up. We stop as
soon as we have a working population.

#### `_evolve_one_generation()` — the inner loop

The structural backbone of the project. Breakdown:

```python
# 1. PFG selection
pfg_elites  = self._population.selection(self._selection_num)
full_elites = list(self._population.population)

# 2. Cluster, unless the last affordable call should be saved for generation
partition, cluster_quality = maybe_cluster_or_singleton_fallback(full_elites)

# 3. Reset cluster bandit with warm-start priors
op_priors = self._operator_bandit.softmax_probs(temperature=1.0)
self._cluster_bandit.reset(n_clusters=len(partition),
                           cluster_quality=cluster_quality,
                           operator_priors=op_priors)

# 4. Inner offspring loop
while produced < pop_size and not budget.is_exhausted() \
      and budget.can_afford(self._gen_call_cost):

    # 4a. Pick operator
    if self._disable_operator_bandit:                   # round-robin (ablation)
        op = OPERATORS[self._rr_idx % len(OPERATORS)]
        self._rr_idx += 1
    else:
        op = self._operator_bandit.select()             # UCB1

    # 4b. Pick cluster (or uniform random if disabled)
    if self._disable_cluster_bandit:
        cluster_idx = random.randrange(n_clusters)
        arm = (cluster_idx, op)
    else:
        arm = self._cluster_bandit.select(
            budget_fraction=self._budget.fraction_remaining(),
            restrict_operator=op)

    # 4c. Build prompt + parents
    prompt_builder, parents_used = self._make_prompt(
        op=op, cluster_idx=arm[0],
        partition=partition, full_elites=full_elites,
        pfg_elites=pfg_elites)

    # 4d. Snapshot diversity, then sample/evaluate/register
    score_before = self._population_scores(include_pending=True)
    div_before   = cumulative_diversity(score_before)
    ok, new_score, _ = self._sample_eval_register(prompt_builder,
                                                  cost=cost,
                                                  label=f'{op}#{arm[0]}')
    score_after  = self._population_scores(include_pending=True)
    div_after    = cumulative_diversity(score_after)

    # 4e. Compute reward
    reward, breakdown = self._reward.reward(
        new_score=new_score,
        population_scores=score_before,
        diversity_before=div_before,
        diversity_after=div_after,
        invalid=(not ok),
        managed_pop_size=self._pop_size)

    # 4f. Update bandits
    if not self._disable_operator_bandit:
        self._operator_bandit.update(op, reward, cost=cost)
    if not self._disable_cluster_bandit:
        drift = self._cluster_bandit.update(arm, reward, cost=cost)
        # cluster bandit handles its own PH-driven arm reset
```

**The `disable_*` flags are how the ablations are wired.** Setting both
gives you a "round-robin operator + uniform cluster + no PH + no diversity"
baseline that mirrors MPaGE's behaviour inside this framework.

#### `_make_prompt()` — operator-conditional parent selection

For `mutate`, the parent is drawn from **inside** the selected cluster; for
`crossover`, one parent comes from inside, the other from any **other**
cluster. This matches the original MPaGE prompts. Optional review/suggestion
LLM call (one extra unit of budget) is inserted before crossover when
`--review` is set.

#### `_sample_eval_register()` — the atomic LLM cycle

The single point that **(a) charges the budget**, (b) calls the LLM via the
sampler, (c) parses the response into a `Function`+`Program`, (d) runs the
secure evaluator, and (e) registers a successful function in the population.

Returns `(ok, score, func)`. `ok=False` signals "invalid heuristic" to the
reward computer, which applies the penalty. Returning the actual child score
prevents stale-population scores from being used in the reward.

---

### 5.8 `main.py` — the CLI entry point

Argparse wrapper around `BMABLLM`. The full flag list is in §11. Two
non-obvious bits:

#### Task registry

```python
_TASKS = {
    'bi_tsp':   ('mpage_bmab._llm4ad.task.optimization.bi_tsp_semo',
                 'BITSPEvaluation', (20.0, 60.0)),
    'tri_tsp':  (..., 'TRITSPEvaluation', (20.0, 60.0)),
    'bi_cvrp':  (..., 'BICVRPEvaluation', (40.0, 60.0)),
    'bi_kp':    (..., 'BIKPEvaluation',   (0.0, 60.0)),
}
```

Each entry maps a CLI task name to `(module_path, class_name, default_ref_point)`.
The reference point is the upper bound for HV computation, in the
`(−HV_solutions, runtime)` space.

#### Ablation presets

```python
ABLATIONS = {
    'full':         {},
    'dense_reward': {'reward_mode': 'dense'},
    'hybrid_reward': {'reward_mode': 'hybrid'},
    'no_budget_anneal': {'disable_budget_annealing': True},
    'no_ph':        {'ph_threshold': 1e9},
    'no_diversity': {'w_diversity': 0.0},
    'op_only':      {'disable_cluster_bandit': True},
    'cluster_only': {'disable_operator_bandit': True},
    'mpage_budget': {'disable_cluster_bandit': True,
                     'disable_operator_bandit': True,
                     'ph_threshold': 1e9,
                     'w_diversity': 0.0,
                     'w_rank': 0.0},
}
```

When you pass `--ablation NAME`, the corresponding dict is merged on top of the
parsed CLI args, and the resulting parameters are passed to `BMABLLM`. The
profiler's method tag is set to `BMAB-<ablation>` so that downstream
aggregation can group runs by ablation.

`full` is the current fixed final-HV-oriented method. `dense_reward` and
`hybrid_reward` change only `reward_mode`; they do not revert the other
final-HV fixes.

---

## 6. The vendored `_llm4ad/` snapshot

`_llm4ad/` is a **vendored copy** of the upstream `llm4ad` library with all
absolute `from llm4ad.*` imports rewritten to relative form. It exists so this
project has zero dependency on the parent `MPaGE/` repository — you can copy
`mpage_bmab/` to a new machine and it will run.

| Subtree | What it provides |
|---------|------------------|
| `_llm4ad/base/` | `Function`/`Program` dataclasses, `TextFunctionProgramConverter` (parses LLM output), `LLM` abstract base, `Evaluation` ABC and `SecureEvaluator` (sandboxed subprocess execution), `ModifyCode` (markdown stripping etc.) |
| `_llm4ad/tools/llm/` | `HttpsApiOpenAI` (chat completions wrapper), `HttpsApiOpenAI4Cluster` (structured output via Pydantic for the cluster LLM) |
| `_llm4ad/tools/profiler/` | `ProfilerBase` — handles run directory creation and `run_log.txt` |
| `_llm4ad/method/LLMPFG/` | `Population` (PFG grid + non-dominated sort + crowding distance), `EoHPrompt` (I1/E1/E2/M1/M2/cluster/suggestion prompts), `EoHSampler` (LLM call + thought/code extraction), `EoHProfiler` (population JSON dumps per generation) |
| `_llm4ad/task/optimization/` | Four MOCOP benchmarks: `bi_tsp_semo`, `tri_tsp_semo`, `bi_cvrp`, `bi_kp`. Each provides an `*Evaluation` class, a template heuristic, and a random instance generator |

You can replace `_llm4ad/` with a newer upstream snapshot at any time —
re-copy the directory, rewrite any new absolute imports to relative form, and
re-run.

---

## 7. Setup and installation

### 7.1 Prerequisites

- Python 3.9+ (3.10/3.11 also fine)
- An OpenAI-compatible API key (either OpenAI itself or a compatible proxy)
- ~2 GB free disk for logs

### 7.2 Create the virtual environment

From the directory **containing** `mpage_bmab/`:

```bash
python3 -m venv mpage_bmab/.venv
mpage_bmab/.venv/bin/pip install -r mpage_bmab/requirements.txt
```

### 7.3 Place your API keys

Two files are read by `main.py`:

```bash
echo "<openai-api-key>" > secret.txt
echo "<openai-api-key>" > secret_cluster.txt
```

You can use the same key in both, or use a cheaper model for clustering.

### 7.4 Verify the install

```bash
mpage_bmab/.venv/bin/python -c "from mpage_bmab import BMABLLM; print('OK')"
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.run --suite smoke --dry_run
```

Both commands should succeed without spending API credits.

---

## 8. Running a single experiment

```bash
mpage_bmab/.venv/bin/python -m mpage_bmab.main \
    --task bi_tsp \
    --budget 50 \
    --pop_size 6 \
    --seed 2025 \
    --ablation full \
    --log_dir logs_bmab
```

What happens:
1. Reads `secret.txt` and `secret_cluster.txt`.
2. Builds an `HttpsApiOpenAI` and `HttpsApiOpenAI4Cluster` wrapper (default
   model `gpt-4o-mini`).
3. Loads `BITSPEvaluation`.
4. Constructs a `BMABProfiler` rooted at `logs_bmab/`.
5. Constructs `BMABLLM` with the parameters from CLI + ablation preset.
6. Calls `bmab.run()`.
7. Prints the final AUBC.

The output directory is `logs_bmab/<TIMESTAMP>_BITSPEvaluation_BMAB-full/`
containing all artefacts (see §10).

---

## 9. The experimental harness

The `experiments/` module orchestrates large sweeps for the thesis chapter.

### 9.1 Suites

| Suite | Cells | What for |
|-------|------:|----------|
| `smoke`     | 2   | Cheapest possible end-to-end check (~30 LLM calls) |
| `headline`  | 48  | Main AUBC table: 4 ablations × `bi_tsp` × 4 budgets × 3 seeds |
| `budget50`  | 80  | Tight-budget regime: 4 ablations × 4 tasks × `B=50` × 5 seeds |
| `full`      | 320 | Full sweep from IDEA.md §4: 4×4×4×5 |

### 9.2 Single command, one suite

```bash
mpage_bmab/experiments/run_smoke.sh        # validate pipeline first
mpage_bmab/experiments/run_headline.sh     # produce the headline table
mpage_bmab/experiments/run_full.sh         # full sweep (expensive!)
```

Each shell launcher chains **run → aggregate → compare**, so when it finishes
you already have `summary.csv` and a Markdown table of Wilcoxon p-values.

### 9.3 Custom slices

```bash
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.run \
    --ablations full,op_only \
    --tasks bi_tsp,bi_kp \
    --budgets 50,100 \
    --seeds 2025,2026,2027
```

Always preview with `--dry_run` first to confirm the cell list and count.

### 9.4 Idempotency

`run.py` skips a cell if `aubc.json` already exists at its destination. So:
- a crashed sweep can be **resumed** with the same command;
- you can **add seeds** to a finished suite incrementally;
- pass `--force` to override and re-run.

### 9.5 Aggregating + comparing manually

```bash
# Walk experiments/results/ and write summary.csv (with mean±std table to stdout)
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.aggregate

# Wilcoxon signed-rank test: full vs each other ablation, paired across seeds,
# computed per (task, budget). Markdown report to stdout + comparisons CSV.
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.compare \
    --baseline full --metric aubc

# Or compare on final HV instead of AUBC
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.compare \
    --baseline full --metric hv_final
```

### 9.6 Output directory schema

```
experiments/results/<ablation>/<task>/B<budget>/seed<seed>/
└── <TIMESTAMP>_<Problem>_BMAB-<ablation>/
    ├── run_log.txt
    ├── samples/samples_*.json
    ├── population/pop_*.json
    ├── budget_curve.json
    ├── bandit_log.json
    ├── budget_history.json
    └── aubc.json
```

`aggregate.py` walks this tree, parses one row per `aubc.json`, and writes
`summary.csv` with columns:

`ablation, task, budget, seed, aubc, hv_final, pareto_size, consumed_budget,
total_budget, n_calls, run_dir`

---

## 10. Output artefacts

Every run produces the following files inside its log directory:

| File | What it contains | Useful for |
|------|------------------|-----------|
| `run_log.txt` | Plain-text log compatible with MPaGE's format | Eyeballing a run |
| `samples/samples_*.json` | Every heuristic ever evaluated (code, score, evaluate_time, sample_time, algorithm description) | Forensics; finding the best heuristic |
| `population/pop_*.json` | The population at end of each generation | Visualising population dynamics |
| `budget_curve.json` | List of `{budget_consumed, hv, pareto_size}` triples | Plotting HV-vs-budget; computing AUBC |
| `bandit_log.json` | Per-generation operator and cluster bandit statistics | Diagnosing whether the bandit converged |
| `budget_history.json` | One entry per LLM call: `{step, label, cost, remaining}` | Auditing exactly where budget went |
| `aubc.json` | `{total_budget, consumed_budget, aubc}` | The headline scalar metric |

Plotting tip: `budget_curve.json` is plottable directly with matplotlib —
each row is a point `(x=budget_consumed, y=hv)`.

---

## 11. Configuration reference (every CLI flag)

### Task and budget

| Flag | Default | Meaning |
|------|---------|---------|
| `--task` | `bi_tsp` | One of `bi_tsp`, `tri_tsp`, `bi_cvrp`, `bi_kp` |
| `--budget` | `50.0` | LLM-call budget `B` (ints work too) |
| `--budget_mode` | `call` | `call` = unit cost per LLM call; `token` = token-aware (future) |
| `--pop_size` | `6` | Population size and per-generation offspring target |
| `--max_generations` | `None` | Optional cap (the budget is the canonical stop condition) |

### LLM and API

| Flag | Default | Meaning |
|------|---------|---------|
| `--llm_model` | `gpt-4o-mini` | Heuristic-generation model |
| `--llm_cluster_model` | `gpt-4o-mini` | Cluster model |
| `--openai_base_url` | `https://api.openai.com` | OpenAI-compatible endpoint |
| `--secret` | `secret.txt` | Path to API-key file for the gen LLM |
| `--secret_cluster` | `secret_cluster.txt` | Path to API-key file for the cluster LLM |

### Bandit hyperparameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--c_op` | `1.0` | UCB1 exploration coefficient for the operator bandit |
| `--c_cluster` | `1.0` | UCB1 exploration coefficient for the cluster bandit |
| `--gamma_budget` | `0.5` | Exponent for remaining-budget exploration annealing |
| `--ph_delta` | `0.005` | Page-Hinkley slack |
| `--ph_threshold` | `0.5` | Page-Hinkley drift threshold |

### Reward weights

| Flag | Default | Meaning |
|------|---------|---------|
| `--w_quality` | `1.0` | Weight of the selected quality signal |
| `--w_diversity` | `0.3` | Weight of ΔCDI diversity gain |
| `--w_rank` | `0.2` | Weight of rank score |
| `--reward_mode` | `final_hv` | Quality signal: `final_hv`, `dense`, or `hybrid` |

### Ablation presets and toggles

| Flag | Default | Meaning |
|------|---------|---------|
| `--ablation` | `full` | One of `full`, `dense_reward`, `hybrid_reward`, `no_budget_anneal`, `no_ph`, `no_diversity`, `op_only`, `cluster_only`, `mpage_budget` (overrides the relevant params) |
| `--disable_budget_annealing` | off | Disable remaining-budget exploration annealing |
| `--disable_cluster_bandit` | off | Sample clusters uniformly at random (op_only ablation) |
| `--disable_operator_bandit` | off | Round-robin operators instead of UCB1 |
| `--method_name` | `BMAB-<ablation>` | Override profiler method tag |

### Plumbing

| Flag | Default | Meaning |
|------|---------|---------|
| `--seed` | `2025` | RNG seed |
| `--log_dir` | `logs_bmab` | Output root |
| `--debug` | off | Print per-step bandit decisions to stdout |
| `--review` | off | Enable LLM-based reflection/suggestion call before crossover (extra +1 budget per call) |

---

## 12. Extending the project

### 12.1 Add a new task

1. Drop the task package under `_llm4ad/task/optimization/<your_task>/` with at
   minimum:
   - `__init__.py` re-exporting the `Evaluation` subclass
   - `evaluation.py` defining `class YourTaskEvaluation(Evaluation)` with
     `template_program`, `task_description`, and `evaluate_program(program)
     → score_tuple` methods
   - `template.py` and `get_instance.py` (see existing tasks for the pattern)

2. Patch any absolute `from llm4ad.*` imports in `evaluation.py` to relative
   form (`from ....base import Evaluation`).

3. Register it in `main.py`:

   ```python
   _TASKS['your_task'] = (
       'mpage_bmab._llm4ad.task.optimization.your_task',
       'YourTaskEvaluation',
       (UPPER_BOUND_OBJ_1, UPPER_BOUND_OBJ_2),  # ref point for HV
   )
   ```

4. Add it to `experiments/configs.py:TASKS` if you want it in sweeps.

### 12.2 Add a new ablation

In `main.py`, add an entry to `ABLATIONS`:

```python
'my_ablation': {
    'w_diversity': 0.0,
    'disable_budget_annealing': True,
    'c_explore_cluster': 0.0,
}
```

If your ablation needs a new behaviour (not just a parameter toggle), add a
boolean flag to `BMABLLM.__init__`, wire it into `_evolve_one_generation`,
and expose it via a new `--disable_*` flag in `main.py`. See the
`disable_cluster_bandit` flag as a template.

### 12.3 Add a new reward component

Edit `reward.py:RewardComputer.reward()`. The breakdown dict already has
keys `hvi`, `rank`, `diversity`, `penalty`, `total`; add yours alongside,
weight it with a new `w_*` parameter, and expose the parameter on the CLI in
`main.py`.

### 12.4 Use a different LLM provider

`HttpsApiOpenAI` only requires an OpenAI-compatible chat-completions endpoint.
Point `--openai_base_url` at it (e.g. an Anthropic-via-proxy or local vLLM
server) and put the right key in `secret.txt`. For non-OpenAI-compatible
providers, write a new `LLM` subclass under `_llm4ad/tools/llm/` and import
it in `main.py`.

---

## 13. Troubleshooting and FAQ

### "Missing API-key file"

`run.py` checks `secret.txt` / `secret_cluster.txt` exist before any sub-process.
Run from the project root that contains `mpage_bmab/`, or pass `--secret
/path/to/file` and `--secret_cluster /path/to/file`.

### "AUBC = 0" after a smoke run

Symptoms: the run completed but no Pareto front was built. Causes:
- All sampled heuristics were invalid (LLM returning malformed code). Look at
  `samples/samples_*.json` for the raw responses.
- Reference point in `_TASKS` is too tight — every score lies outside the
  reference box. Pick a larger `(neg_hv_low, time_high)`.

### Run hangs forever

The `SecureEvaluator` runs heuristics in a subprocess with a timeout. If a
heuristic is in an infinite loop the subprocess is killed and the heuristic is
penalised. If the LLM API itself hangs, the `timeout=30` set in `main.py` for
the HTTPS wrapper kicks in.

### "No comparable cells found" from `compare.py`

You ran fewer than 3 seeds for a given `(task, budget)`. Wilcoxon needs at
least three paired non-zero differences. Either run more seeds or accept that
some cells will be blank.

### How do I plot the budget-vs-HV curves?

Each `budget_curve.json` is a list of `{budget_consumed, hv, pareto_size}`
records. A two-line matplotlib plot:

```python
import json, matplotlib.pyplot as plt
data = json.load(open('budget_curve.json'))
plt.plot([d['budget_consumed'] for d in data], [d['hv'] for d in data])
```

For comparison plots across ablations, group by `<ablation>/<task>/B<budget>/`
and overlay one line per ablation.

### Can I resume a crashed sweep?

Yes. `run.py` skips any cell whose `aubc.json` already exists. Re-launching
the same command finishes only the missing cells.

### How do I keep the budget hard-capped under all failure modes?

Every LLM call goes through `BudgetTracker.charge()` *before* the HTTPS
request is issued. If `charge()` returns `False`, the call is never made.
There is no other code path that can hit the LLM.

### What's the minimum sensible budget?

`B ≈ 15` to validate the pipeline (the smoke suite uses this). For
*meaningful* results, `B ≥ 25` so that warm-up + at least one full generation
fits. The thesis's interesting regime is `B = 50`.

---

## Quick reference card

```
# Setup (once)
python3 -m venv mpage_bmab/.venv
mpage_bmab/.venv/bin/pip install -r mpage_bmab/requirements.txt
echo "$OPENAI_KEY" > secret.txt
echo "$OPENAI_KEY" > secret_cluster.txt

# One run
mpage_bmab/.venv/bin/python -m mpage_bmab.main \
    --task bi_tsp --budget 50 --ablation full

# Smoke check the harness
mpage_bmab/experiments/run_smoke.sh

# Headline thesis table (48 runs)
mpage_bmab/experiments/run_headline.sh

# Aggregate + Wilcoxon any time
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.aggregate
mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.compare --baseline full
```
