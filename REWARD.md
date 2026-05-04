# The Reward Function in BMAB-LLM — Deep Dive

This document is a complete reference for the scalar reward signal that drives
every bandit update in BMAB-LLM. It explains *why* the reward is a weighted
sum of four components, *how* each component is computed, where it lives in
the source, how it interacts with the rest of the system, and how to retune
it.

Companion documents:
[BANDITS.md](BANDITS.md) covers the consumers of this reward
(operator and cluster bandits);
[PAGE_HINKLEY.md](PAGE_HINKLEY.md) covers the drift detector that watches it;
[DOCUMENTATION.md](DOCUMENTATION.md) is the project-wide reference.

---

## Table of contents

1. [Why reward design matters here](#1-why-reward-design-matters-here)
2. [The reward formula at a glance](#2-the-reward-formula-at-a-glance)
3. [Component 1 — Normalised Hypervolume Improvement (HVI)](#3-component-1--normalised-hypervolume-improvement-hvi)
4. [Component 2 — Diversity gain ΔCDI](#4-component-2--diversity-gain-cdi)
5. [Component 3 — Rank score](#5-component-3--rank-score)
6. [Component 4 — Invalid penalty](#6-component-4--invalid-penalty)
7. [Putting it together: `RewardComputer.reward()`](#7-putting-it-together-rewardcomputerreward)
8. [Where the reward is consumed](#8-where-the-reward-is-consumed)
9. [Worked numerical example](#9-worked-numerical-example)
10. [Tuning the weights](#10-tuning-the-weights)
11. [Edge cases the code handles](#11-edge-cases-the-code-handles)
12. [The `no_diversity` ablation](#12-the-no_diversity-ablation)
13. [Why these four components and not others](#13-why-these-four-components-and-not-others)
14. [References](#14-references)

---

## 1. Why reward design matters here

Bandits maximise an expected scalar reward. We supply that scalar after every
LLM call — one per generated heuristic. Three constraints make the reward
design non-trivial:

1. **The natural quality signal is sparse.** Hypervolume Improvement (HVI) is
   the textbook quality metric for multi-objective optimisation, but most
   offspring **don't** improve the front (`HVI = 0`). If the reward were
   pure HVI, the bandits would see a long string of zeros and have nothing
   to learn from. UCB1's mean-reward estimates would carry no information
   about which arms are actually better.
2. **Pure quality maximisation collapses diversity.** A bandit rewarded only
   on HVI funnels budget into the cluster currently producing the highest
   HVI improvements. That cluster's representative style takes over the
   population; other promising regions of objective space are abandoned;
   long-term Pareto coverage suffers.
3. **The reward must be bounded.** UCB1's analysis assumes rewards in
   `[0, 1]`. HVI is unbounded (it grows with the size of the dominated
   region), and a single early generation can produce a 100×-larger HVI than
   a late generation. A bandit fed raw HVI would be dominated by the noise
   of one or two outlier rewards.

The reward function is engineered to fix all three problems while preserving
the property that "better heuristic ⇒ higher reward" on average.

---

## 2. The reward formula at a glance

For each newly generated heuristic with objective-space score `s = (s_1,
s_2, …)` (e.g. `(neg_HV_solutions, runtime)`):

```
R(s)  =  w_q · h_norm(s)              ← normalised HVI ∈ [0, 1]
       + w_d · max(0, ΔCDI(s))        ← diversity gain ∈ [0, ∞)
       + w_r · rank_score(s)          ← rank in population ∈ [0, 1]
       − λ_pen · 1[s is invalid]      ← penalty if heuristic is bad
```

Default weights from [bmab_llm.py:67-71](bmab_llm.py#L67-L71):

| Symbol | CLI flag | Default | Role |
|--------|----------|---------|------|
| `w_q`     | `--w_quality`   | `1.0`  | Primary signal, normalised HVI |
| `w_d`     | `--w_diversity` | `0.3`  | Anti-collapse, encourages spreading the population |
| `w_r`     | `--w_rank`      | `0.2`  | Dense smoothing, never zero unless the new heuristic is dominated by everything |
| `λ_pen`   | hard-coded as `reward_penalty` (`1.0`) | `1.0` | Discourages invalid heuristics |

When the heuristic is **invalid** (timeout, parse error, NaN score, no score
returned), the reward short-circuits to **just** `−λ_pen`; the other terms
are zeroed. This is by design — there is no meaningful HVI / rank /
diversity to compute against an absent score.

The whole computation lives in `RewardComputer.reward` at
[reward.py:156-218](reward.py#L156-L218).

---

## 3. Component 1 — Normalised Hypervolume Improvement (HVI)

### 3.1 What hypervolume is

For a minimisation problem with reference point `r ∈ ℝ^M`, the
**hypervolume** of a set of points `P = {p_1, …, p_n}` (each `p_i ∈ ℝ^M`)
is the M-dimensional Lebesgue measure of the dominated region:

```
HV(P, r)  =  λ( ∪_i  [p_i, r] )
```

i.e. the union of "boxes" between each point and the reference. Higher HV
means the front pushes more of the objective space into the dominated
region.

In BMAB-LLM, `M = 2` for `bi_*` tasks and `M = 3` for `tri_tsp`. The
reference point is set in [main.py:30-39](main.py#L30-L39) per task:

```python
_TASKS = {
    'bi_tsp':   ('...', 'BITSPEvaluation',  (20.0, 60.0)),
    'tri_tsp':  ('...', 'TRITSPEvaluation', (20.0, 20.0, 60.0)),
    'bi_cvrp':  ('...', 'BICVRPEvaluation', (40.0, 60.0)),
    'bi_kp':    ('...', 'BIKPEvaluation',   (0.0, 60.0)),
}
```

The numbers are upper bounds — chosen so the entire achievable Pareto front
lies "inside" the box `[origin, ref]`. A point with any objective `≥ ref`
contributes nothing.

### 3.2 The HVI of one new point

Hypervolume *improvement* is the gain in HV when a new point is added to an
existing front:

```
HVI(p_new ; F, r)  =  HV(F ∪ {p_new}, r)  −  HV(F, r)        (≥ 0)
```

Implemented in [reward.py:84-99](reward.py#L84-L99):

```python
def hvi(new_point, existing, ref_point):
    ref = np.asarray(ref_point, dtype=float)
    existing = [np.asarray(p, dtype=float) for p in existing if _is_valid_score(p)]
    new_pt = np.asarray(new_point, dtype=float)
    if not _is_valid_score(new_pt):
        return 0.0
    if np.any(new_pt >= ref):                    # outside the box → contributes 0
        return 0.0
    pts_before = np.array(existing) if existing else np.zeros((0, len(ref)))
    pts_after  = np.vstack([pts_before, new_pt[None, :]])
    hv_before = hypervolume(pts_before, ref) if len(pts_before) else 0.0
    hv_after  = hypervolume(pts_after,  ref)
    return max(0.0, hv_after - hv_before)
```

Three things to note:

1. **`new_pt >= ref` short-circuits to 0.** A point worse than the reference
   in any objective contributes no dominated volume.
2. **`max(0.0, …)`** is paranoia — `pymoo`'s HV indicator is
   monotone-increasing in the number of points, so the difference is
   theoretically non-negative; the floor catches any FP nonsense.
3. **`hypervolume(...)`** delegates to `pymoo.indicators.hv.HV` when
   available, and falls back to a hand-rolled 2-D implementation
   ([reward.py:60-81](reward.py#L60-L81)) when it is not.

### 3.3 Why HVI alone is not enough

HVI is the right signal *in expectation* — averaging across many heuristics,
better operators / clusters produce higher HVI. But on a per-call basis HVI
is **sparse** and **heavy-tailed**:

* **Sparse**: in a typical generation 4 of 6 offspring fail to improve the
  front. They have HVI = 0 even though some of them might be very nearly on
  the front (rank in objective space matters too). The bandit cannot
  distinguish "almost-good" from "useless" — both score 0.
* **Heavy-tailed**: when an offspring *does* improve the front, the gain
  scales with the size of the unfilled region. Early generations have huge
  unfilled regions (HVI may be 30+); late generations have only thin slivers
  left (HVI ≈ 0.1). One huge early HVI can swamp the bandit's reward stream
  for many subsequent updates.

We solve **sparsity** with two extra components (rank score, diversity gain)
that are dense in `[0, 1]`. We solve **heavy-tail** with rolling-window
normalisation, described next.

### 3.4 Rolling-window normalisation

[reward.py:185-191](reward.py#L185-L191):

```python
h = hvi(new_score, population_scores, self._ref)
breakdown['hvi'] = h
self._hvi_history.append(h)
# rolling normalisation: scale so recent max ≈ 1
recent = self._hvi_history[-self._rank_window:]
h_max  = max(recent + [self._hvi_floor + 1e-9])
h_norm = h / h_max if h_max > 0 else 0.0
```

The trailing window is `rank_window = 50` LLM calls (configurable). We
normalise the current HVI by the **max** observed in that window, so:

* In an early-stage burst (HVI = 30), `h_norm = 30 / 30 = 1`.
* When a quiet phase starts (HVI = 1.5 with recent max still 30 in the
  window), `h_norm = 0.05`.
* As the window slides past the burst, recent max drops to ~1.5; now
  HVI = 1.5 normalises to 1 again.

This keeps the bandit's reward roughly bounded in `[0, 1]` regardless of the
absolute HVI scale at any point in the run. The window length 50 is a
practical compromise — long enough to remember bursts, short enough to adapt
to the late-stage regime.

---

## 4. Component 2 — Diversity gain ΔCDI

### 4.1 What CDI measures

The **Cumulative Diversity Index** (CDI) is the mean pairwise Euclidean
distance among current population scores in objective space. Implemented in
[reward.py:117-126](reward.py#L117-L126):

```python
def cumulative_diversity(scores):
    pts = np.array([s for s in scores if _is_valid_score(s)], dtype=float)
    if len(pts) < 2:
        return 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt((diff * diff).sum(-1))
    n = len(pts)
    return float(dists.sum() / (n * (n - 1)))
```

`dists` is the `n × n` pairwise distance matrix. The sum is twice the upper
triangle, divided by `n(n-1)` instead of `n(n-1)/2` — the resulting average
is half the mean pairwise distance, but the absolute scale doesn't matter
because we only use **differences** of CDI, never CDI itself.

CDI is large when the population is spread out across objective space, small
when it has clustered onto one region. It is exactly the diversity metric
that MPaGE uses to *evaluate* runs — by including it in the reward we make
the optimisation target consistent with the evaluation target.

### 4.2 The reward component

[reward.py:209-211](reward.py#L209-L211):

```python
d_gain = max(0.0, diversity_after - diversity_before)
breakdown['diversity'] = d_gain
```

We measure CDI **before** inserting the new heuristic and **after**, and
take the (clipped) increase. A heuristic that opens up a new region of
objective space gets a positive `d_gain`; one that lands inside the existing
cluster gets `d_gain = 0`.

The clip to non-negative is important: a new heuristic that lands at the
*centroid* of the population mathematically reduces CDI, but that's not
something the bandit should be penalised for — the heuristic might still be
useful for HVI. Penalising the diversity drop here would conflict with the
quality signal.

### 4.3 The "before" snapshot

[bmab_llm.py:288-289](bmab_llm.py#L288-L289):

```python
score_before = self._population_scores()
div_before = cumulative_diversity(score_before)
```

The orchestrator captures `population_scores` and `div_before` **before**
calling the LLM. This matters because by the time `reward()` is invoked the
population has already been mutated by `_sample_eval_register` and
`div_before` would be wrong if computed late.

### 4.4 Sibling: Shannon-Wiener Diversity Index

[reward.py:104-114](reward.py#L104-L114) provides `shannon_diversity` over
cluster-size histograms. **It is not used in the reward** — the cluster
sizes change discretely (only when re-clustering happens) so the per-call
ΔSWDI is almost always zero. SWDI is exposed because the IDEA document
mentions it as an evaluation metric; downstream analysis scripts can compute
it from the saved population dumps.

---

## 5. Component 3 — Rank score

### 5.1 The "rank" we use

[reward.py:194-207](reward.py#L194-L207):

```python
rank_score = 0.0
if len(population_scores) > 0:
    n = 0
    better = 0
    for s in population_scores:
        if not _is_valid_score(s):
            continue
        n += 1
        if any(a < b for a, b in zip(new_score, s)):
            better += 1
    rank_score = better / max(n, 1)
breakdown['rank'] = rank_score
```

For each population member, we ask: "is the new heuristic better than this
one on **at least one** objective?" The rank score is the fraction that
answer yes.

### 5.2 Why this specific definition

Several alternatives were considered:

| Candidate | Why we didn't use it |
|-----------|-----------------------|
| Pareto-rank (1 = front, 2 = next layer, …) | Full non-dominated sort is `O(n²M)`; we run reward thousands of times per run. Too expensive per call. |
| `1 − dominance_rank/n` | Same cost as Pareto-rank. |
| Distance to the front | Continuous but unstable; depends on a chosen front-distance metric. |
| Strict domination count (better on **all** objectives) | Almost always 0 because the population is on the front. Same sparsity problem as HVI. |
| **Any-objective-better fraction** ✓ | `O(n·M)` per call, dense in `[0, 1]`, monotone in objective improvement, robust to scale changes |

The *any-objective-better* signal correlates with quality without being as
sparse as strict domination. A heuristic that is much better on objective 1
but worse on objective 2 still gets credit — the bandit then has a chance to
learn that this style is worth pursuing.

### 5.3 Why `0.2` weighting

The default `w_r = 0.2` is small but non-zero. Its job is to ensure that
**the reward is rarely exactly zero**, which is what gives the bandits a
gradient. With `w_r = 0` the bandit reverts to "HVI plus diversity gain"
which can both hit 0 simultaneously and we are back to the sparsity problem.

Empirically `0.2` is enough to keep the reward stream above 0.05 most of the
time without overwhelming the HVI signal.

---

## 6. Component 4 — Invalid penalty

### 6.1 What "invalid" means

A heuristic is *invalid* iff any of the following happen:

| Failure | Where it surfaces |
|---------|-------------------|
| LLM returned no parsable code | `EoHSampler.get_thought_and_function` returns `None` |
| `TextFunctionProgramConverter.function_to_program` raises | Caught in `_sample_eval_register` |
| `SecureEvaluator.evaluate_program_record_time` returns `None`, NaN, or raises | Caught in `_sample_eval_register` |
| Subprocess timeout | The secure evaluator returns sentinel value, treated as invalid |
| Score has wrong dimensionality | `_is_valid_score` returns `False`, treated as invalid |

The orchestrator at [bmab_llm.py:291-310](bmab_llm.py#L291-L310) tracks the
boolean `ok` and forwards it to `RewardComputer.reward(invalid=not ok)`.

### 6.2 The penalty path

[reward.py:177-182](reward.py#L177-L182):

```python
if invalid or new_score is None or not _is_valid_score(new_score):
    breakdown['penalty'] = -self._pen
    breakdown['total']   = -self._pen
    self._reward_history.append(-self._pen)
    self._hvi_history.append(0.0)
    return -self._pen, breakdown
```

Three things happen:

1. The reward is **just** `−λ_pen`, not a sum of components. The other
   terms are skipped because they have nothing meaningful to compute against.
2. A `0.0` is appended to `_hvi_history` — important so the rolling
   normalisation max doesn't get stale (a long invalid streak would
   otherwise leave `h_max` "frozen" at its last good value).
3. The penalty enters `_reward_history` like any other reward, so summary
   stats and downstream analysis see it.

### 6.3 Why `1.0` and not larger

`λ_pen = 1.0` is the same scale as the maximum positive reward
(roughly 1 + 0.3 + 0.2 = 1.5). A larger penalty (e.g. 5.0) would dominate
the bandit's mean-reward estimate; one bad apple in a cluster would be
enough to drive the cluster bandit away from it for the rest of the
generation, even if subsequent samples from that cluster were excellent.
With `λ_pen = 1.0`, two or three good rewards counter one penalty, so the
bandit recovers quickly when the LLM has a transient hiccup.

A larger penalty is justified if the LLM is reliably bad (you want hard
avoidance); a smaller penalty if invalids are rare and you don't want them
to dominate. Both are achievable by overriding the `reward_penalty`
parameter in `BMABLLM.__init__`.

---

## 7. Putting it together: `RewardComputer.reward()`

The full method, with annotations:

```python
def reward(self,
           new_score: Optional[Sequence[float]],
           population_scores: Sequence[Sequence[float]],
           diversity_before: float,
           diversity_after: float,
           *,
           invalid: bool = False) -> Tuple[float, dict]:

    breakdown = {'hvi': 0.0, 'rank': 0.0, 'diversity': 0.0,
                 'penalty': 0.0, 'total': 0.0}

    # ▼ Path A — invalid: short-circuit with penalty only
    if invalid or new_score is None or not _is_valid_score(new_score):
        breakdown['penalty'] = -self._pen
        breakdown['total']   = -self._pen
        self._reward_history.append(-self._pen)
        self._hvi_history.append(0.0)
        return -self._pen, breakdown

    # ▼ Path B — valid: compute three positive components

    # 1. HVI (rolling-window normalised)
    h = hvi(new_score, population_scores, self._ref)
    breakdown['hvi'] = h
    self._hvi_history.append(h)
    recent = self._hvi_history[-self._rank_window:]
    h_max  = max(recent + [self._hvi_floor + 1e-9])
    h_norm = h / h_max if h_max > 0 else 0.0

    # 2. Rank score
    rank_score = 0.0
    if len(population_scores) > 0:
        n = 0
        better = 0
        for s in population_scores:
            if not _is_valid_score(s):
                continue
            n += 1
            if any(a < b for a, b in zip(new_score, s)):
                better += 1
        rank_score = better / max(n, 1)
    breakdown['rank'] = rank_score

    # 3. Diversity gain
    d_gain = max(0.0, diversity_after - diversity_before)
    breakdown['diversity'] = d_gain

    # 4. Combine
    total = (self._w_q * h_norm
             + self._w_d * d_gain
             + self._w_r * rank_score)
    breakdown['total'] = total
    self._reward_history.append(total)
    return total, breakdown
```

The returned `breakdown` dict is consumed by `BMABLLM._evolve_one_generation`
in `--debug` mode for diagnostics — it lets you see, per call, exactly which
component was driving the reward:

```python
if self._debug:
    print(f"[BMAB] gen={self._gen} op={op} cluster={cluster_idx} "
          f"reward={reward:.4f} cost={cost} drift={drift} "
          f"budget={self._budget.remaining:.1f}/"
          f"{self._budget.total:.1f}  breakdown={breakdown}")
```

---

## 8. Where the reward is consumed

The same scalar `reward` is fed to **three** downstream consumers in
[bmab_llm.py:313-320](bmab_llm.py#L313-L320):

```python
if not self._disable_operator_bandit:
    self._operator_bandit.update(op, reward, cost=cost)
if self._disable_cluster_bandit:
    drift = False
else:
    drift = self._cluster_bandit.update(arm, reward, cost=cost)
```

| Consumer | What it does with the reward |
|----------|------------------------------|
| `OperatorBandit.update(op, reward, cost)` | Increments `n_op`, `sum_reward_op`, `sum_cost_op`. UCB1's reward-per-cost score for `op` updates accordingly. |
| `ClusterBandit.update(arm, reward, cost)` | Same but for the per-generation `(cluster, op)` arm. Also feeds the reward into the arm's Page-Hinkley state. |
| `PageHinkleyState.update(reward)` (inside `ClusterBandit.update`) | Updates online mean and cumulative deviation; if the gap to the running max exceeds threshold, `drift = True` and the arm is reset to its prior. |

**A single reward number drives every adaptive component.** This is why the
exact composition of that number matters so much.

---

## 9. Worked numerical example

Setup: bi-TSP, ref point `(20.0, 60.0)`, default weights, `rank_window = 50`,
generation 3 of a `B = 50` run.

State right before the LLM call:

* `population_scores`: 6 points on the current front, e.g.
  `[(7.2, 0.4), (8.1, 0.3), (10.0, 0.2), (12.5, 0.15), (15.3, 0.12),
  (18.0, 0.10)]`.
* `div_before = cumulative_diversity(...)` ≈ `5.83`.
* `_hvi_history`: 18 values from previous calls, e.g. `[15.2, 0.0, 8.4,
  0.0, 0.0, 3.1, …]`. Recent max in last 50: `15.2`.

The LLM produces a new heuristic; the evaluator returns
`new_score = (9.4, 0.18)`.

**Step 1 — HVI.**
The new point is non-dominated (it improves the front between `(8.1, 0.3)`
and `(10.0, 0.2)`). `pymoo` computes `HV(F) = 412.0`,
`HV(F ∪ {new}) = 414.7`. Therefore `h = 2.7`.

`_hvi_history` becomes `[…, 15.2, …, 2.7]`. Recent max in last 50 still
`15.2`. Normalised: `h_norm = 2.7 / 15.2 = 0.178`.

**Step 2 — Rank score.**
For each member of `population_scores`, check if `new_score` beats it on at
least one objective:

* vs `(7.2, 0.4)`: `9.4 < 7.2`? no. `0.18 < 0.4`? **yes**. Counted.
* vs `(8.1, 0.3)`: `0.18 < 0.3`? **yes**. Counted.
* vs `(10.0, 0.2)`: `9.4 < 10.0`? **yes**. Counted.
* vs `(12.5, 0.15)`: `9.4 < 12.5`? **yes**. Counted.
* vs `(15.3, 0.12)`: `9.4 < 15.3`? **yes**. Counted.
* vs `(18.0, 0.10)`: `9.4 < 18.0`? **yes**. Counted.

`better = 6`, `n = 6`, `rank_score = 1.0`.

**Step 3 — Diversity gain.**
After insertion the population has 7 points; the new one adds non-trivial
spread. Suppose `div_after = 5.95`. Then `d_gain = max(0, 5.95 − 5.83) =
0.12`.

**Step 4 — Combine.**
```
total = 1.0 · 0.178   +  0.3 · 0.12  +  0.2 · 1.0
      = 0.178 + 0.036 + 0.200
      = 0.414
```

**Breakdown returned**:
```python
{'hvi': 2.7, 'rank': 1.0, 'diversity': 0.12, 'penalty': 0.0, 'total': 0.414}
```

The bandits both receive `0.414`. UCB1's reward-per-cost for the
selected `(cluster, op)` arm bumps up by roughly that. Page-Hinkley sees a
positive sample, the running mean creeps toward `0.4`, the gap doesn't grow.

For comparison, an offspring that scored `(11.0, 0.5)` (dominated by an
existing point):

* `h = 0` (HVI is 0 — point is dominated, no new region).
* `rank_score`: better-on-1-objective check gives perhaps 4/6 ≈ 0.67.
* `d_gain` ≈ 0.0 (insertion barely moves the centroid).
* `total = 0 + 0 + 0.2·0.67 = 0.134`.

So a dominated heuristic still produces a small positive reward (driven by
the rank score), and a *much* lower one than a Pareto-improving heuristic.
The bandit can distinguish them — that is the entire point of the rank
term.

And an invalid heuristic:

* `total = −1.0` (penalty path, no other components).

The penalty is roughly **3×** the magnitude of an "OK but dominated" reward,
which is the right proportion: invalids are clearly worse than mediocre
heuristics, but not so catastrophically worse that one breaks the bandit.

---

## 10. Tuning the weights

The defaults are calibrated to a `[0, 1]` rolling-normalised HVI. If you
change reward range you must rescale.

| Weight | Default | Effect of increasing | Effect of decreasing |
|--------|---------|----------------------|----------------------|
| `w_quality` | `1.0` | Bandit chases HVI more aggressively; risk of front-collapse | Treats HVI as a tie-breaker; over-emphasises rank/diversity |
| `w_diversity` | `0.3` | Stronger anti-collapse pressure; may slow HVI growth | Front concentrates on whatever region is currently most productive |
| `w_rank` | `0.2` | Reward becomes denser; bandit converges faster but may chase shallow wins | Reward becomes sparser; if too small the bandit sees mostly zeros |
| `reward_penalty` | `1.0` | Hard avoidance of clusters that yielded any invalid; risk of premature exclusion | Invalids barely register; the LLM's noise leaks into the bandit |

Practical rules of thumb:

* If you observe the bandit settling on **one** cluster very early (low
  AUBC despite high final HV), increase `w_diversity` to `0.4–0.5`.
* If runs are **noisy** across seeds, your reward range is likely too wide.
  Print `breakdown` per call with `--debug` and check the maximum total.
  If totals routinely exceed `2.0`, scale the weights down proportionally
  or shorten `rank_window` so the rolling max adapts faster.
* If you switch to **token-aware budgeting** (`--budget_mode token`), the
  costs change scale (e.g. 200–500 tokens per call instead of 1). Reward-
  per-cost in the bandits drops by 200–500×; this doesn't change
  comparative orderings, but it does mean the bandit's exploration term
  dominates more aggressively. Bump `c_op` and `c_cluster` down to compensate
  (e.g. `0.05`).

The `no_diversity` ablation (next section) gives a clean lower-bound for
how much each weight matters.

---

## 11. Edge cases the code handles

| Edge case | Code response |
|-----------|---------------|
| `new_score` is `None` (sampling failed before evaluation) | Treated as `invalid`; penalty applied. |
| `new_score` contains `NaN` or `Inf` | `_is_valid_score` returns `False`; treated as `invalid`. |
| `new_score` is a tuple of wrong length | `_is_valid_score` checks `arr.ndim == 1`; mismatched dimensions ⇒ invalid. |
| Population is empty (very first call after init failure) | `len(population_scores) > 0` guard skips the rank loop; `rank_score = 0.0`. HVI uses an empty front (`HV = 0`). |
| `pymoo` is not installed | `_HAS_PYMOO = False`, `hypervolume()` falls back to a 2-D analytic calculation (see [reward.py:60-81](reward.py#L60-L81)). For `M ≥ 3` the fallback is a degenerate over-estimate; `pymoo` should always be available — `requirements.txt` pins it. |
| All HVI values in the recent window are `0.0` | `h_max` is floored at `1e-9` so `h_norm` is `0` (not `nan`). |
| Pareto-improving point is dominated by reference | `np.any(new_pt >= ref)` triggers, returns 0; the run continues but no quality reward earned. |
| `rank_score` denominator `n = 0` (all population members invalid) | `max(n, 1)` floor; `rank_score = 0`. |
| Diversity drops after insertion (CDI shrunk) | `d_gain = max(0, ...)`; cannot be negative. |

The file is paranoid about NaN propagation because a single NaN in
`_reward_history` would corrupt every UCB score downstream.

---

## 12. The `no_diversity` ablation

The `no_diversity` ablation in [main.py](main.py)'s `ABLATIONS`:

```python
'no_diversity': {'w_diversity': 0.0},
```

sets `w_d = 0`, so `d_gain` no longer contributes. Reward becomes:

```
R(s)  =  w_q · h_norm  +  w_r · rank_score  −  λ_pen · 1[invalid]
```

This isolates the contribution of the diversity term. The thesis Wilcoxon
test [`compare.py --baseline full`](experiments/compare.py) computes the
AUBC gap between `full` and `no_diversity` per `(task, budget)` cell. If
the gap is positive and significant, the diversity term is justified; if it
is zero or negative, the term is overhead and should be removed.

The hypothesis (from IDEA.md §4) is that diversity matters more on:

* **Tri-objective tasks** where there is more "front" to spread along.
* **Larger budgets** where the algorithm has time to explore multiple
  regions; with tiny budgets quality dominates and diversity has little
  room to act.

These hypotheses are testable directly by inspecting `comparisons_aubc.csv`
after the `full` suite runs.

---

## 13. Why these four components and not others

A few design decisions worth defending explicitly:

### Why not raw HVI (without normalisation)?

Two weeks of runs with raw HVI in early prototyping showed UCB1 scores
dominated by 1–2 outlier observations. Normalising in a sliding window keeps
the bandit responsive throughout the run.

### Why not strict-domination rank rather than any-objective-better?

Strict domination is sparse for the same reason as HVI. The "any-objective-
better" version is a *softer* signal that captures the same ordering on
average but is dense in `[0, 1]`. We tested both; the soft version
converges faster.

### Why not include a runtime penalty separately?

Runtime is already an objective in the score tuple `(neg_HV_solutions,
runtime)`, so it appears in HVI and in rank automatically. A separate term
would double-count.

### Why not include the cluster-quality prior in the reward?

The cluster-quality prior already enters the bandit through the warm-start
in `ClusterBandit.reset()`. Including it again in the reward would be
double-counting and make the bandit conservative — it would prefer
high-prior clusters even when their actual rewards are mediocre.

### Why not penalise duplicates?

Duplicate detection would require canonicalising heuristic code (whitespace
normalisation, AST equivalence, ...), which is brittle. Instead, duplicates
naturally contribute `HVI = 0` and `d_gain = 0`, so they get a small reward
of `w_r · rank_score` only. Empirically this is enough to suppress them
without an explicit duplicate term.

### Why is the penalty scalar (one threshold) rather than continuous?

A continuous penalty (e.g. proportional to "how invalid") is hard to define
— most invalidity is binary (the program either runs or it doesn't). A
fixed `−1.0` penalty captures the categorical cost of an invalid heuristic
without inviting the LLM to hover near "almost-valid" garbage.

---

## 14. References

* **Zitzler, E., Thiele, L.** (1999). *Multiobjective evolutionary
  algorithms: A comparative case study and the strength Pareto approach.*
  IEEE Trans. Evolutionary Computation, 3(4), 257–271. — defines the
  hypervolume indicator.
* **Beume, N., Naujoks, B., Emmerich, M.** (2007). *SMS-EMOA: Multiobjective
  selection based on dominated hypervolume.* European Journal of OR, 181(3),
  1653–1669. — establishes HVI as a primary search signal.
* **Fonseca, C. M., Paquete, L., López-Ibáñez, M.** (2006). *An improved
  dimension-sweep algorithm for the hypervolume indicator.* CEC. — the
  algorithm `pymoo` uses internally.
* **Fialho, A., Da Costa, L., Schoenauer, M., Sebag, M.** (2010). *Analyzing
  bandit-based adaptive operator selection mechanisms.* Annals of Math and
  AI, 60. — the rank-normalised reward design that inspired our rolling-
  window normalisation.
* **Drugan, M. M., Nowé, A.** (2013). *Designing multi-objective multi-armed
  bandit algorithms — a study.* IJCNN. — multi-objective MAB; reward
  composition guidelines.
* **Ha, T. M., et al.** (2025). *MPaGE: Pareto-Grid-Guided LLMs for Fast and
  High-Quality Heuristics Design in MOCOP.* arXiv 2507.20923. — the parent
  paper; HV / IGD / SWDI / CDI evaluation conventions.
