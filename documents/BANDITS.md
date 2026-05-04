# The Two Bandit Problems in BMAB-LLM — Deep Dive

This document is a self-contained reference for the bandit machinery that
drives BMAB-LLM. It explains what the multi-armed bandit (MAB) problem is in
the abstract, what the *Budgeted* variant adds, why this project needs **two**
bandits rather than one, how they interact, and exactly where each line of
the implementation lives.

Companion documents: [PAGE_HINKLEY.md](PAGE_HINKLEY.md) covers the drift
detector layered on top of these bandits; [DOCUMENTATION.md](DOCUMENTATION.md)
is the project-wide reference.

---

## Table of contents

1. [The classical MAB problem](#1-the-classical-mab-problem)
2. [UCB1 in detail](#2-ucb1-in-detail)
3. [The Budgeted MAB extension](#3-the-budgeted-mab-extension)
4. [Why one bandit is not enough here](#4-why-one-bandit-is-not-enough-here)
5. [The Operator Bandit (persistent layer)](#5-the-operator-bandit-persistent-layer)
6. [The Cluster Bandit (per-generation layer)](#6-the-cluster-bandit-per-generation-layer)
7. [The two-layer interaction and the warm-start](#7-the-two-layer-interaction-and-the-warm-start)
8. [End-to-end trace of one inner-loop iteration](#8-end-to-end-trace-of-one-inner-loop-iteration)
9. [Theoretical properties](#9-theoretical-properties)
10. [Tuning the bandits](#10-tuning-the-bandits)
11. [Edge cases and how the code handles them](#11-edge-cases-and-how-the-code-handles-them)
12. [Ablations isolating each bandit](#12-ablations-isolating-each-bandit)
13. [References](#13-references)

---

## 1. The classical MAB problem

You face `K` slot machines (called *arms*). At each round `t = 1, 2, …, T`
you pick one arm `a_t`, pull it, and observe a stochastic reward `R_t` drawn
from an unknown distribution `D_a` specific to that arm. Your goal is to
maximise the cumulative reward `Σ R_t`.

The challenge is the **exploration–exploitation trade-off**:

* **Exploit**: pick the arm whose empirical mean reward is highest right now.
  This is greedy and ignores uncertainty — the empirical mean might be high
  by chance after few pulls.
* **Explore**: pick an under-pulled arm to refine its estimate. This costs
  short-term reward but may reveal a globally better arm.

Performance is measured by **regret**: how much you lost compared to an oracle
who always picks the truly best arm `a*`:

```
regret(T)  =  T · μ_{a*}  −  E[ Σ_{t=1..T} R_t ]
```

A "good" bandit algorithm has *sublinear* regret — `regret(T) = o(T)` —
meaning the per-round penalty for not knowing the truth shrinks to zero as
you gather more data.

### Why does this map to LLM-driven heuristic design?

In our project:

| MAB concept | Mapping in BMAB-LLM |
|-------------|---------------------|
| Arm | "Pick operator μ" or "pick cluster `k` with operator μ" |
| Pulling an arm | Issuing one LLM call with the corresponding prompt |
| Reward | A scalar function of HVI + diversity + rank − penalty |
| Round | One iteration of the inner offspring loop |
| Time horizon `T` | The LLM-call budget `B` |

Without a bandit you'd either fix one operator (round-robin MPaGE) or pick
uniformly random; both waste budget on choices that the data already says are
worse.

---

## 2. UCB1 in detail

**Upper Confidence Bound (UCB1)** by Auer, Cesa-Bianchi, Fischer (2002) is the
canonical bandit algorithm and the one used in this project. Per round:

```
UCB1(a, t)  =  μ̂_a   +   c · √( 2 · ln(N_t)  /  n_a )
                ▲              ▲
                │              │
            empirical     uncertainty radius
            mean reward   (Hoeffding bound)
```

where

* `μ̂_a` = empirical mean reward of arm `a` so far,
* `n_a` = number of times `a` has been pulled,
* `N_t = Σ_a n_a` = total pulls so far,
* `c` = exploration coefficient (we use `1.0` by default; the textbook value
  is `√2 ≈ 1.41`, lower values exploit more).

Pick the arm with the largest `UCB1(a, t)`. Why this works:

* If an arm has been **rarely** pulled, `n_a` is small, so `√(ln N / n_a)` is
  large, so even a poor empirical mean cannot keep its UCB low → it *will* be
  picked again soon. **Exploration is forced.**
* If an arm has been **often** pulled, `n_a` is large, so the radius shrinks
  to near zero, so the UCB ≈ the empirical mean. The bandit *trusts* well-
  measured arms.
* The radius is calibrated to the Hoeffding inequality, so with high
  probability the true mean lies inside `[μ̂_a − radius, μ̂_a + radius]`.
  Picking by upper bound = "be optimistic in the face of uncertainty".

### Regret guarantee

UCB1's expected regret is

```
E[regret(T)]  =  O( √(K · T · log T) )
```

i.e. `Õ(√KT)`. This is sublinear and matches the lower bound up to a log
factor.

### Implementation in this project

[bandit.py:114-122](bandit.py#L114-L122):

```python
def select(self) -> str:
    self._t += 1
    scores = {}
    total_n = sum(s.n for s in self._stats.values())
    for op, s in self._stats.items():
        exploit = s.reward_per_cost                                # μ̂_a / c̄_a
        explore = self._c * math.sqrt(2.0 * math.log(max(total_n, 2)) / s.n)
        scores[op] = exploit + explore
    return max(scores, key=scores.get)
```

Note the small but important detail: we use **reward-per-cost**
(`s.sum_reward / s.sum_cost`) rather than plain reward. This is the
signature of a **Budgeted** MAB (next section).

---

## 3. The Budgeted MAB extension

The classical setting limits the number of pulls; the **Budgeted MAB** (or
*Bandits with Knapsacks*, BwK; Tran-Thanh et al. 2010, Badanidiyuru et al.
2013) limits the *cost*. Each arm pull costs `c_a` (which can be different
per arm or per pull), and you stop when the total cost exceeds budget `B`.

The natural generalisation of UCB1 is **Budgeted UCB1** (Ding et al. 2013):
score arms by reward-per-cost rather than reward,

```
BUCB(a, t)  =  S_a^R / S_a^c   +   c · √( 2 · ln(N_t)  /  n_a )
                  ▲
                  │
              reward per
              unit cost
```

where `S_a^R = Σ rewards` and `S_a^c = Σ costs` over all pulls of `a`.

Why this matters: if mutation costs 1 token but crossover costs 5 tokens (long
prompt with two parents), an arm with `μ̂_χ = 0.6, c̄_χ = 5` is worse per
budget unit than `μ̂_μ = 0.2, c̄_μ = 1` (`0.6/5 = 0.12 < 0.2`). Plain UCB1
on raw reward would prefer crossover; Budgeted UCB1 correctly prefers
mutation.

### One more refinement: budget-pressure term

The cluster bandit adds a third term to the score:

```
budget_pressure  =  γ_b · ln( b_t / B )       (≤ 0)
```

where `b_t` is the remaining budget and `B` the total budget. As budget runs
out, `b_t / B → 0` and `ln(b_t/B) → −∞`. This term is the *same* for every
arm in a single decision, so it does not change *which* arm is best — what
it changes is the spread of UCB scores. Effectively it shrinks the
exploration bonus when there is no time left to recover from a bad
exploration decision. It is "explore aggressively early, exploit aggressively
late."

This is documented in IDEA.md §3.2 and implemented in
[bandit.py:236-243](bandit.py#L236-L243):

```python
budget_pressure = self._gamma_b * math.log(max(budget_fraction, 1e-3))
for arm in candidates:
    s = self._stats[arm]
    exploit = s.reward_per_cost
    explore = self._c * math.sqrt(2.0 * math.log(max(total_n, 2)) / s.n)
    scores[arm] = exploit + explore + budget_pressure
```

`max(budget_fraction, 1e-3)` floors the log to avoid `−∞` exactly at
budget exhaustion.

---

## 4. Why one bandit is not enough here

A naive design would have **one bandit** with arms `(cluster, operator)`
covering both decisions. That fails for a structural reason: the two choices
have **different lifetimes**.

* **Operators** (`mutate`, `crossover`) are intrinsic to the algorithm. They
  do not change between generations. Their reward distributions evolve slowly
  as the population matures (e.g. mutation becomes more useful late).
* **Clusters** are constructed by an LLM clustering call at the start of each
  generation. They are *re-built* each time. Cluster `0` in generation `t`
  has nothing to do with cluster `0` in generation `t+1` — the LLM just
  numbered them differently. So **cluster identities are not comparable
  across generations**.

A single flat bandit on `(cluster, operator)` arms would either (a) discard
its statistics every generation (losing all cross-generation information,
including for operators), or (b) keep them and feed garbage to the cluster
dimension because the IDs are no longer meaningful.

The two-layer design solves this:

```
                ┌──────────────────────────────────────────┐
                │  Operator Bandit (persistent, 2 arms)    │
                │  Lives for the entire run.               │
                └──────────────┬───────────────────────────┘
                               │ softmax probabilities
                               │ used as warm-start prior
                               ▼
                ┌──────────────────────────────────────────┐
                │  Cluster Bandit (ephemeral, 2K arms)     │
                │  Reset every generation when clusters    │
                │  are re-built.                           │
                └──────────────────────────────────────────┘
```

**Persistent** stats accumulate over the whole run for the slow-changing
operator decision; **ephemeral** stats are warm-started from the persistent
layer so the new generation does not start blind.

This factorisation also makes ablations cheap: disable the cluster layer to
get "operator UCB1 + uniform cluster" (the `op_only` ablation); disable the
operator layer to get "round-robin operators + cluster UCB1" (the
`cluster_only` ablation).

---

## 5. The Operator Bandit (persistent layer)

### 5.1 Definition

An instance of `OperatorBandit` (defined in [bandit.py:97-143](bandit.py#L97-L143))
is created once at `BMABLLM.__init__` and lives for the entire run.

* Arms: `OPERATORS = ('mutate', 'crossover')` — fixed, 2 of them.
* Statistics per arm: `(n, sum_reward, sum_cost, sum_sq_reward)` in
  [bandit.py:69-74](bandit.py#L69-L74).
* Prior: `n=1, sum_reward=0.5, sum_cost=1` — neutral, mid-range.

### 5.2 Selection

[bandit.py:114-122](bandit.py#L114-L122) — straight UCB1 with reward-per-cost:

```python
def select(self) -> str:
    self._t += 1
    total_n = sum(s.n for s in self._stats.values())
    scores = {}
    for op, s in self._stats.items():
        exploit = s.reward_per_cost                                # μ̂_a / c̄_a
        explore = self._c * math.sqrt(2.0 * math.log(max(total_n, 2)) / s.n)
        scores[op] = exploit + explore
    return max(scores, key=scores.get)
```

### 5.3 Update

```python
def update(self, op: str, reward: float, cost: float = 1.0) -> None:
    self._stats[op].update(reward, cost)
```

Every time an offspring is generated and evaluated, the corresponding
operator's stats are updated with the scalar reward returned by
`RewardComputer`. No drift detection at this level — operator dynamics are
slow, and the offline-style mean is fine.

### 5.4 Softmax probabilities — the cross-layer hook

[bandit.py:134-143](bandit.py#L134-L143):

```python
def softmax_probs(self, temperature: float = 1.0) -> Dict[str, float]:
    scores = []
    keys = list(self._stats.keys())
    for op in keys:
        s = self._stats[op]
        scores.append(s.reward_per_cost)
    m = max(scores)
    exps = [math.exp((sc - m) / max(temperature, 1e-3)) for sc in scores]
    total = sum(exps)
    return {k: e / total for k, e in zip(keys, exps)}
```

Returns a probability distribution `{mutate: p, crossover: 1-p}` from the
current reward-per-cost estimates via temperature-scaled softmax. The
`max` subtraction is the standard numerical-stability trick.

This distribution is **the bridge to the cluster layer**: at the start of
each generation the cluster bandit calls `softmax_probs()` and uses the
result to warm-start its arms (see §7).

### 5.5 What this layer learns

Concretely: across an entire run, which operator should we generally prefer?
For most runs:

* Crossover dominates **early**. The population is sparse; pulling material
  from two clusters generates novel hybrids that often improve HVI.
* Mutation dominates **late**. The population is mature; the elite cluster
  representatives are already good, and small variations on them are more
  productive than big jumps.

The persistent UCB1 captures this *temporal* shift in the reward-per-cost of
the two operators. By generation 5 it might be picking mutation 60% of the
time even though it picked crossover 70% in generation 1. This is the W1 fix
from the IDEA — adaptive `γ` rather than fixed.

---

## 6. The Cluster Bandit (per-generation layer)

### 6.1 Definition

`ClusterBandit` (defined in [bandit.py:148-281](bandit.py#L148-L281)) holds
arms keyed by `(cluster_idx, operator)`. With `K` clusters this is `2K` arms
per generation.

Statistics are reset at every generation boundary by `reset(...)`. Each arm
also has its own `PageHinkleyState` (see [PAGE_HINKLEY.md](PAGE_HINKLEY.md)).

### 6.2 The reset / warm-start

[bandit.py:174-214](bandit.py#L174-L214):

```python
def reset(self, n_clusters: int,
          cluster_quality: Optional[Dict[int, float]] = None,
          operator_priors: Optional[Dict[str, float]] = None) -> None:
    """Reset the bandit at the start of a new generation."""
    self._stats = {}
    self._ph = {}
    self._cluster_priors = {}
    self._n_clusters = n_clusters
    cluster_quality = cluster_quality or {}
    operator_priors = operator_priors or {o: self._prior_reward
                                          for o in OPERATORS}
    # normalise cluster_quality to [0,1]
    if cluster_quality:
        mx = max(cluster_quality.values())
        mn = min(cluster_quality.values())
        rng = mx - mn if mx > mn else 1.0
        cluster_quality = {k: (v - mn) / rng
                           for k, v in cluster_quality.items()}
    for k in range(n_clusters):
        q = cluster_quality.get(k, self._prior_reward)
        self._cluster_priors[k] = q
        for o in OPERATORS:
            op_prior = operator_priors.get(o, self._prior_reward)
            init_reward = 0.5 * (q + op_prior)
            self._stats[(k, o)] = ArmStats(
                n=int(self._prior_n) if self._prior_n >= 1 else 1,
                sum_reward=init_reward * max(self._prior_n, 1.0),
                sum_cost=max(self._prior_n, 1.0),
                sum_sq_reward=0.0,
            )
            self._ph[(k, o)] = PageHinkleyState(
                delta=self._ph_delta, threshold=self._ph_threshold)
    self._t = 0
```

The crucial line is `init_reward = 0.5 * (q + op_prior)`: the warm-start
expected reward of arm `(k, o)` is the **average** of:

* `q` = cluster `k`'s quality (from `ClusterManager._cluster_quality()`,
  derived from the best HV in cluster `k` after min-max normalisation), and
* `op_prior` = persistent operator bandit's softmax probability for `o`.

So an arm gets a high prior if **both** "this cluster has good elites" AND
"this operator has been productive overall." Either one being low pulls the
prior toward neutral; both being low gives a low prior.

The arm's virtual `n_a` and `c̄_a` are set to `prior_n = 1` and `1`
respectively — so the arm acts as if it has been pulled exactly once with a
reward equal to `init_reward`. This is the standard Bayesian-style "pseudo-
count" trick.

### 6.3 Selection

[bandit.py:218-247](bandit.py#L218-L247):

```python
def select(self, budget_fraction: float = 1.0,
           restrict_operator: Optional[str] = None) -> Tuple[int, str]:
    self._t += 1
    candidates = [
        (k, o) for (k, o) in self._stats.keys()
        if (restrict_operator is None or o == restrict_operator)
    ]
    if not candidates:
        raise RuntimeError("ClusterBandit has no arms to select from.")

    total_n = sum(self._stats[a].n for a in candidates)
    budget_pressure = self._gamma_b * math.log(max(budget_fraction, 1e-3))

    scores = {}
    for arm in candidates:
        s = self._stats[arm]
        exploit = s.reward_per_cost
        explore = self._c * math.sqrt(2.0 * math.log(max(total_n, 2)) / s.n)
        scores[arm] = exploit + explore + budget_pressure
    # break ties uniformly
    max_score = max(scores.values())
    best = [a for a, sc in scores.items() if abs(sc - max_score) < 1e-9]
    return random.choice(best)
```

Two things differ from the operator-level `select`:

1. **`restrict_operator` parameter.** When the caller has already chosen an
   operator (the standard flow — operator first, then cluster), only arms
   with that operator are considered. This is what makes the two-layer
   structure correct: the cluster bandit doesn't *re-decide* the operator;
   it picks the best cluster *given* the operator.
2. **Budget pressure term + uniform tie-breaking.** Already discussed in §3.

### 6.4 Update with drift detection

[bandit.py:251-268](bandit.py#L251-L268):

```python
def update(self, arm: Tuple[int, str], reward: float,
           cost: float = 1.0) -> bool:
    if arm not in self._stats:
        return False
    self._stats[arm].update(reward, cost)
    drift = self._ph[arm].update(reward)
    if drift:
        self._ph[arm].reset()
        # Reset arm to optimistic prior, keeping cluster prior
        k, o = arm
        q = self._cluster_priors.get(k, self._prior_reward)
        self._stats[arm] = ArmStats(
            n=int(self._prior_n) if self._prior_n >= 1 else 1,
            sum_reward=q * max(self._prior_n, 1.0),
            sum_cost=max(self._prior_n, 1.0),
        )
    return drift
```

Two effects:

1. Update the arm's UCB1 statistics (just like the operator bandit).
2. Feed the same reward to the arm's Page-Hinkley state. If PH declares
   drift, **reset the arm** to its cluster's optimistic prior. This is the
   second-chance mechanism documented in detail in [PAGE_HINKLEY.md](PAGE_HINKLEY.md).

### 6.5 What this layer learns

Within one generation: **which `(cluster, operator)` combinations produce the
best offspring per unit budget?** The bandit converges quickly because:

* The arm space is small (typically 4–10 clusters × 2 operators = 8–20 arms).
* The budget per generation is `pop_size` calls (default 6) — enough to
  exercise the priors and update them a few times.
* The warm-start prevents wasted exploration of obviously-bad arms.

Across many generations the cluster bandit doesn't really "learn" — its memory
is wiped at every reset. What persists is the *quality of its priors*, fed by
the operator bandit and the cluster manager's quality function.

---

## 7. The two-layer interaction and the warm-start

The bridge between the layers is in [bmab_llm.py:240-249](bmab_llm.py#L240-L249):

```python
# 2. cluster (decrements budget by 1 if a real call is issued)
partition, cluster_quality = self._cluster_mgr.cluster(full_elites)
if not partition:
    return
n_clusters = len(partition)
op_priors = self._operator_bandit.softmax_probs(temperature=1.0)
self._cluster_bandit.reset(
    n_clusters=n_clusters,
    cluster_quality=cluster_quality,
    operator_priors=op_priors,
)
```

Three pieces of information flow into the cluster bandit's reset:

| Source | What it carries | Why it matters |
|--------|-----------------|----------------|
| `cluster_quality` from `ClusterManager` | Per-cluster `[0,1]` HV-derived prior | "Cluster `k` looks promising because its members have high HV" |
| `op_priors` from `OperatorBandit.softmax_probs()` | Per-operator probability `[0,1]` | "Operator `o` has been productive across the run" |
| Hyperparameters `c, γ_b, δ, λ` from the BMAB config | UCB1 / budget-pressure / PH constants | Tuning of exploration strength and drift sensitivity |

The arm prior `init_reward = 0.5 * (q + op_prior)` is the **product channel**
through which knowledge from the long-running operator bandit reaches the
just-instantiated cluster arms. Without it, a freshly reset cluster bandit
would have to discover from scratch every generation that mutation has been
working better than crossover.

### Information flow in one generation

```
                     ┌─────────────────────────┐
                     │  PFG selects elites     │
                     └────────────┬────────────┘
                                  │
                                  ▼
                ┌────────────────────────────────────────┐
                │ ClusterManager.cluster(elites)         │
                │   ↓ debits budget                      │
                │   ↓ LLM clustering call                │
                │   → partition (e.g. [[0,1,3], [2,4]])  │
                │   → cluster_quality (e.g. {0:1.0, 1:0.4})│
                └────────────┬───────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────────────────┐
                │ OperatorBandit.softmax_probs()         │
                │   → {mutate: 0.7, crossover: 0.3}      │
                └────────────┬───────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────────────────┐
                │ ClusterBandit.reset(                   │
                │     n_clusters=2,                       │
                │     cluster_quality={0:1.0, 1:0.4},     │
                │     operator_priors={μ:0.7, χ:0.3})     │
                │                                         │
                │ Builds 4 arms:                          │
                │   (0, μ)  prior = 0.5·(1.0 + 0.7) = 0.85│
                │   (0, χ)  prior = 0.5·(1.0 + 0.3) = 0.65│
                │   (1, μ)  prior = 0.5·(0.4 + 0.7) = 0.55│
                │   (1, χ)  prior = 0.5·(0.4 + 0.3) = 0.35│
                └────────────┬───────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────────────────┐
                │ Inner offspring loop:                   │
                │   for _ in range(pop_size):             │
                │     op  = OperatorBandit.select()       │
                │     arm = ClusterBandit.select(         │
                │              budget_fraction,           │
                │              restrict_operator=op)      │
                │     prompt = E1/E2/M1/M2(arm.cluster)   │
                │     reward = ... (HVI + diversity etc.) │
                │     OperatorBandit.update(op, reward)   │
                │     ClusterBandit.update(arm, reward)   │
                │       ↓ may trigger PH reset            │
                └────────────────────────────────────────┘
```

---

## 8. End-to-end trace of one inner-loop iteration

To make all this concrete, here's a single iteration as the code executes it,
annotated with state changes. Source: [bmab_llm.py:254-322](bmab_llm.py#L254-L322).

```python
while (produced < target_offspring
       and not self._budget.is_exhausted()
       and self._budget.can_afford(self._gen_call_cost)):

    # Step 1 — operator selection (UCB1 over {μ, χ})
    op = self._operator_bandit.select()
    # Suppose: op = 'mutate'  because μ̂_μ/c̄_μ = 0.42 > μ̂_χ/c̄_χ = 0.31
    # plus exploration bonuses balance similarly.

    # Step 2 — cluster selection given op (BUCB1 over (k, μ))
    arm = self._cluster_bandit.select(
        budget_fraction=self._budget.fraction_remaining(),  # e.g. 0.6
        restrict_operator=op,
    )
    # Suppose: arm = (0, 'mutate') with score 0.85 + 0.30 + (-0.26) = 0.89
    # versus (1, 'mutate') with score 0.55 + 0.42 + (-0.26) = 0.71.
    cluster_idx, op_picked = arm

    # Step 3 — prompt construction, parents drawn from cluster 0
    prompt_builder, parents_used = self._make_prompt(
        op=op, cluster_idx=cluster_idx,
        partition=partition, full_elites=full_elites,
    )

    # Step 4 — diversity snapshot (CDI before)
    score_before = self._population_scores()
    div_before   = cumulative_diversity(score_before)   # e.g. 4.2

    # Step 5 — atomic (charge → sample → evaluate → register)
    cost = self._gen_call_cost  # 1.0 in call-mode
    ok = self._sample_eval_register(
        prompt_builder, cost=cost,
        label=f'{op}#{cluster_idx}',     # e.g. "mutate#0"
    )
    # Inside _sample_eval_register: budget -= 1, LLM call issued, code parsed,
    # SecureEvaluator runs the heuristic, score returned.

    # Step 6 — diversity after, reward
    new_score = self._latest_score() if ok else None
    div_after = cumulative_diversity(self._population_scores())
    reward, breakdown = self._reward.reward(
        new_score=new_score,
        population_scores=score_before,
        diversity_before=div_before,
        diversity_after=div_after,
        invalid=(not ok),
    )
    # breakdown: {'hvi': 0.4, 'rank': 0.66, 'diversity': 0.05, 'penalty': 0,
    #             'total': 1.0·0.4 + 0.3·0.05 + 0.2·0.66 = 0.547}

    # Step 7 — bandit updates
    if not self._disable_operator_bandit:
        self._operator_bandit.update(op, reward, cost=cost)
    if self._disable_cluster_bandit:
        drift = False
    else:
        drift = self._cluster_bandit.update(arm, reward, cost=cost)
        # arm (0, 'mutate'): n: 1 → 2,  sum_reward: 0.85 → 1.397
        # PH: sum += 0.547 - μ̂ - δ; max_sum updated; drift returns False.

    produced += 1
```

After this iteration:

* The operator bandit has updated `mutate`'s stats; if mutation kept paying
  off the next `select()` will likely pick mutate again.
* The cluster bandit has nudged arm `(0, mutate)`'s reward-per-cost upward.
  Subsequent selects within this generation will compare it against
  `(1, mutate)` using the new statistics.
* The Page-Hinkley state for `(0, mutate)` has moved one observation
  forward; if the next 5 rewards from this arm collapse, PH will fire and
  the arm will be reset.

---

## 9. Theoretical properties

The IDEA document (§3.7-3.8) sketches the analysis. The relevant statements:

### 9.1 Per-step complexity

| Component | Cost per inner-loop iteration |
|-----------|-------------------------------|
| `OperatorBandit.select`  | `O(2)` = `O(1)` |
| `ClusterBandit.select`   | `O(K)` over candidates |
| `OperatorBandit.update`  | `O(1)` |
| `ClusterBandit.update`   | `O(1)` plus PH update `O(1)` |
| LLM call + evaluation    | dominant cost; LLM-side latency |

Bandit overhead is therefore negligible compared to the resource we are
trying to conserve (LLM calls). This is by design.

### 9.2 Regret bound (sketch)

Under the *stationarity assumption* — fixed but unknown reward distributions
per arm — Budgeted UCB1 has expected regret

```
E[regret(B)]  =  O( √( B · K · log B ) )
```

where `K` is the number of arms (Ding et al. 2013). In our case `K = 2` for
the operator layer (negligible regret) and `K ≤ 2 · max_clusters` per
generation for the cluster layer.

The Page-Hinkley resets buy *non-stationarity* at the cost of a small bounded
number of additional "exploration restarts" per arm — formally analysable as
a piecewise-stationary bandit (Garivier & Moulines 2011).

### 9.3 Bounded reward

UCB1's analysis assumes rewards in `[0, 1]`. We enforce this by:

* HVI rolling-window normalisation in [reward.py:188-191](reward.py#L188-L191).
* `rank_score ∈ [0, 1]` by definition.
* `d_gain` clipped to `[0, ∞)` and small in practice.
* Penalty floored at `−1.0`.

Aggregate reward is therefore in roughly `[−1, 1.5]` — bounded, suitable for
UCB1's Hoeffding-based analysis.

---

## 10. Tuning the bandits

| Hyperparameter | Default | Effect of increasing |
|----------------|---------|----------------------|
| `c_explore_op` (--c_op) | 1.0 | More exploration of the rare operator; slower convergence on the truly best one |
| `c_explore_cluster` (--c_cluster) | 1.0 | More exploration across clusters; mitigates over-confidence in early-pulled arms |
| `gamma_budget` (--gamma_budget) | 0.5 | Stronger budget-pressure damping; explore less when budget low |
| `prior_n` | 1 | Heavier reliance on warm-start; less responsive to first observed rewards |
| `prior_reward` | 0.5 | Higher prior makes new arms look better → more exploration of fresh clusters |
| `ph_delta` (--ph_delta) | 0.005 | Less sensitive PH; ignores small drops; longer detection delay |
| `ph_threshold` (--ph_threshold) | 0.5 | More evidence needed for drift declaration; fewer false alarms |

Practical guidance from running on bi-TSP / bi-CVRP:

* If you see the bandit *never* explore one operator, reduce `c_op` or check
  whether one operator is producing very consistent low-variance rewards
  (ties broken in its favour). Bumping `c_op` to `1.41` (textbook value)
  often helps.
* If runs feel "noisy" (high variance across seeds), check that PH isn't
  firing too often on noise — print drift events with `--debug` and count
  them per generation.
* The `--ablation` flags are the cleanest way to isolate which hyperparameter
  matters. Run `full` vs `no_ph` first; that tells you whether drift
  detection is contributing at all.

---

## 11. Edge cases and how the code handles them

| Edge case | Code response |
|-----------|---------------|
| Cluster bandit selected before reset | `select` raises `RuntimeError` because `self._stats` is empty. The orchestrator catches it as a generic exception and breaks the inner loop. |
| Only one cluster (e.g. fewer than 3 elites) | `ClusterManager._fallback_partition` returns singletons. Cluster bandit operates on `K=1` (so no real choice). UCB1 still runs without numerical issues. |
| `restrict_operator` excludes all arms | Should never happen — both operators always have arms. If it did, `RuntimeError`. |
| Budget exhausts mid-generation | `_evolve_one_generation` checks `self._budget.is_exhausted()` and `can_afford()` at the top of every inner-loop iteration. Cleanly exits. |
| Reward = NaN or `None` | `RewardComputer.reward` short-circuits with `invalid=True`, returns `−penalty`. Bandits update with `−penalty` (legitimate negative reward). |
| Tie in cluster-bandit scores | Broken uniformly at random ([bandit.py:245-247](bandit.py#L245-L247)) so the warm-started priors don't deterministically bias toward the lowest-numbered cluster. |
| Operator bandit invariant | At least one of `n_μ, n_χ ≥ 1` from the priors; `total_n ≥ 2`; `log(total_n)` always defined. |
| Floating-point: `log(0)` from `budget_fraction → 0` | Floored by `max(budget_fraction, 1e-3)` in [bandit.py:236](bandit.py#L236). |

---

## 12. Ablations isolating each bandit

The four ablations defined in [main.py](main.py) `ABLATIONS` slice the
bandit machinery in different ways:

| Ablation | Operator bandit | Cluster bandit | Page-Hinkley | What it isolates |
|----------|-----------------|----------------|---------------|------------------|
| `full` | UCB1 | BUCB1 | active | The proposed system |
| `no_ph` | UCB1 | BUCB1 | disabled (`λ=10⁹`) | Marginal value of drift detection |
| `no_diversity` | UCB1 | BUCB1 | active | Marginal value of `d_gain` reward |
| `op_only` | UCB1 | uniform random | active (cluster-side PH inactive because cluster bandit not used) | Marginal value of cluster-level adaptation |
| `cluster_only` | round-robin | BUCB1 | active | Marginal value of operator-level adaptation |
| `mpage_budget` | round-robin | uniform random | disabled | Pure baseline; should track MPaGE behaviour |

The Wilcoxon comparison framework in
[experiments/compare.py](experiments/compare.py) computes paired-by-seed
significance of `full` against each of these per `(task, budget)`. This is
the experimental protocol that lets you make claims like "PH contributes
+5 % AUBC at B=50, p < 0.05" in the thesis.

To run a single-ablation comparison without launching a sweep:

```bash
mpage_bmab/.venv/bin/python -m mpage_bmab.main --ablation no_ph    --task bi_tsp --budget 50 --seed 2025
mpage_bmab/.venv/bin/python -m mpage_bmab.main --ablation full     --task bi_tsp --budget 50 --seed 2025
```

The two `aubc.json` files can be diffed by eye for a sanity check before
committing to a full sweep.

---

## 13. References

* **Auer, P., Cesa-Bianchi, N., Fischer, P.** (2002). *Finite-time analysis
  of the multiarmed bandit problem.* Machine Learning, 47, 235–256.
  — the UCB1 paper.
* **Tran-Thanh, L., Chapman, A., Munoz de Cote, E., Rogers, A.,
  Jennings, N. R.** (2010). *Epsilon-First Policies for Budget-Limited
  Multi-Armed Bandits.* AAAI. — early Budgeted MAB.
* **Badanidiyuru, A., Kleinberg, R., Slivkins, A.** (2013). *Bandits with
  Knapsacks.* FOCS. — the BwK formulation.
* **Ding, W., Qin, Z., Zhu, W., Yu, Y.** (2013). *Multi-Armed Bandit with
  Budget Constraint and Variable Costs.* AAAI. — Budgeted UCB1, the basis of
  our `ClusterBandit`.
* **Bubeck, S., Cesa-Bianchi, N.** (2012). *Regret Analysis of Stochastic
  and Non-Stochastic Multi-Armed Bandit Problems.* Foundations and Trends in
  Machine Learning, 5(1). — comprehensive survey, regret bounds.
* **Garivier, A., Moulines, E.** (2011). *On Upper-Confidence Bound Policies
  for Switching Bandit Problems.* ALT. — non-stationary regret analysis.
* **Fialho, A., Da Costa, L., Schoenauer, M., Sebag, M.** (2010). *Analyzing
  bandit-based adaptive operator selection mechanisms.* Annals of Math and
  AI, 60. — direct precursor of the operator bandit in this project.
* **Drugan, M. M., Nowé, A.** (2013). *Designing multi-objective multi-armed
  bandit algorithms — a study.* IJCNN. — multi-objective MAB foundations.
