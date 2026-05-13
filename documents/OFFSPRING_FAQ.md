# Offspring Generation — Detailed Q&A

A walkthrough of five common questions about exactly how BMAB-LLM picks
parents, evaluates offspring, updates the population, and learns within
the cluster bandit.

Companion documents:
[OFFSPRING_GENERATION_PATHS.md](OFFSPRING_GENERATION_PATHS.md) covers the
high-level MPaGE-vs-BMAB-LLM pipeline difference;
[BANDITS.md](BANDITS.md) covers the bandit machinery in depth;
[RETAINED_FROM_MPAGE.md](RETAINED_FROM_MPAGE.md) covers what was kept from
the original framework.

---

## Table of contents

1. [Q1: For mutation one cluster, for crossover two clusters — correct? How does the second-cluster pick work?](#1-q1-for-mutation-one-cluster-for-crossover-two-clusters)
2. [Q2: After a cluster is selected, is the heuristic chosen at random?](#2-q2-after-a-cluster-is-selected-is-the-heuristic-chosen-at-random)
3. [Q3: Where does evaluation happen?](#3-q3-where-does-evaluation-happen)
4. [Q4: A concrete worked example of one generation](#4-q4-a-concrete-worked-example-of-one-generation)
5. [Q5: Is one generation enough for the cluster bandit to learn?](#5-q5-is-one-generation-enough-for-the-cluster-bandit-to-learn)
   - [5.A How many clusters K is reasonable?](#5a-how-many-clusters-k-is-reasonable)
   - [5.B What `pop_size` (observations per generation) is right?](#5b-what-pop_size-observations-per-generation-is-right)
   - [5.C Joint rule of thumb](#5c-joint-rule-of-thumb)

---

## 1. Q1: For mutation one cluster, for crossover two clusters?

**Almost — but the second parent in crossover does not come from a *second
bandit-picked cluster*. It comes from the union of *all other clusters*.**

Look at [bmab_llm.py:401-461](../bmab_llm.py#L401-L461), the
`_make_prompt` method:

```python
cluster    = partition[cluster_idx] if cluster_idx < len(partition) else []
in_cluster = [full_elites[i] for i in cluster
              if i < len(full_elites)
              and getattr(full_elites[i], 'algorithm', None)]

if op == 'mutate':
    parent = random.choice(in_cluster)                            # one parent, intra-cluster
    ...
    return (lambda: EoHPrompt.get_prompt_m1/m2(..., parent, ...)), [parent]

# crossover
other_clusters = [i for i in range(len(partition)) if i != cluster_idx]
other_indivs   = []
for j in other_clusters:
    for idx in partition[j]:
        if idx < len(full_elites):
            f = full_elites[idx]
            if getattr(f, 'algorithm', None) is not None:
                other_indivs.append(f)
...
p1 = random.choice(in_cluster)         # one parent from the bandit-picked cluster
p2 = random.choice(other_indivs)       # one parent from the UNION of all other clusters
parents = [p1, p2]
```

So the bandit picks **exactly one cluster — call it `k`**. Then:

* **Mutation (`μ`)**: one parent is sampled uniformly at random from
  cluster `k`. The bandit's cluster decision is the only "where do I draw
  parents from" decision. No second cluster.
* **Crossover (`χ`)**: one parent is sampled from cluster `k`; the **other
  parent is sampled uniformly at random from the union of *every other
  cluster***. There is no second bandit decision — the bandit's cluster
  choice fixes which cluster is the *primary parent's* origin, and the
  second parent is essentially "anyone not in `k`."

### Why this design

The two-layer bandit operates over `(cluster, operator)` arms — there is
no third dimension for "second cluster" because the action space would
balloon to `K²` per generation, and with `pop_size = 6` and one
observation per arm per generation, the bandit would have far too little
data to estimate `K²` arms.

By collapsing "the other parent" to "uniform random from everyone else,"
the bandit credit-assigns the *primary* parent's cluster `k` for the
quality of the resulting offspring, which is a reasonable signal even if
the partner happens to be random.

### Code reference

See [bmab_llm.py:428-444](../bmab_llm.py#L428-L444) for the explicit code
path. Note also that if `other_indivs` ends up empty (e.g. partition
collapsed to one cluster), a final fallback picks from `full_elites`
minus the first member of `in_cluster`, so the crossover prompt is
always able to find two parents.

---

## 2. Q2: After a cluster is selected, is the heuristic chosen at random?

**Yes — uniformly at random within the selected cluster.** Both `p1` (for
mutation or crossover) and `p2` (for crossover only) are picked with
`random.choice`:

```python
parent = random.choice(in_cluster)         # mutation
p1     = random.choice(in_cluster)         # crossover, primary
p2     = random.choice(other_indivs)       # crossover, partner
```

No bandit operates inside the cluster. The bandit's job is to choose
*which cluster* to draw from; once a cluster is chosen, the within-cluster
parent is a flat random draw from the heuristics in that cluster.

### Why uniform within a cluster

Two reasons:

1. **Clusters are constructed precisely to group "behaviourally similar"
   heuristics.** If the cluster did its job, the cluster's members are
   roughly interchangeable as parents — the bandit's decision is about
   *style of heuristic to perturb*, not *which exact heuristic to perturb*.
2. **Per-heuristic statistics would be even more data-starved than
   per-cluster ones.** With `pop_size = 6` and ~3 clusters, each cluster
   has ~2 members. Modelling per-heuristic reward would mean ~12 arms
   instead of ~6, while still getting only ~6 observations per
   generation. Not worth the complexity.

If you want richer behaviour you could bias the within-cluster pick by
crowding distance or by recency. The current implementation deliberately
stays simple.

---

## 3. Q3: Where does evaluation happen?

Evaluation runs inside `_sample_eval_register` in
[bmab_llm.py:338-397](../bmab_llm.py#L338-L397). The relevant lines are:

```python
def _sample_eval_register(self, prompt_builder, *, cost, label):
    if not self._budget.charge(cost, label=label):
        return False                       # ← budget gate (B-aware short circuit)
    prompt = prompt_builder()
    thought, func = self._sampler.get_thought_and_function(prompt)    # ← LLM call
    ...
    program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
    ...
    score, eval_time = self._evaluator.evaluate_program_record_time(program)   # ← evaluation
    ...
    func.score = list(score) if isinstance(score, tuple) else score
    ...
    self._population.register_function(func)        # ← admit into population
```

There are **four objects** that participate in evaluation. Tracing one
heuristic through them:

### 3.1 `_sample_eval_register` — orchestrator

Charges the budget, calls the sampler to get LLM output, parses the
output into a `Function` + `Program`, calls the secure evaluator, copies
the resulting score onto the `Function`, and admits the `Function` to
the population.

### 3.2 `TextFunctionProgramConverter` — parser

Converts the raw LLM text response (which contains code in a markdown
fence + optional thought) into a `Program` object: the heuristic's
Python source plus its import statements, ready to be executed.

### 3.3 `SecureEvaluator` — sandboxed executor

Defined in [_llm4ad/base/evaluate.py](../_llm4ad/base/evaluate.py).
Wraps the task-specific `Evaluation` subclass. Its
`evaluate_program_record_time(program)` method:

1. Spawns a subprocess (timeout-bounded).
2. Inside the subprocess, runs `evaluation.evaluate_program(program)`,
   which delegates to the task's actual scoring logic.
3. Captures the result, kills the subprocess if it exceeds the timeout
   (treating timeouts as invalid heuristics).
4. Returns `(score, eval_time)`.

This sandboxing matters because the LLM might return code with
infinite loops, file-system writes, network calls, or other unsafe
patterns. Running it in a subprocess isolates it from the orchestrator.

### 3.4 The task's `Evaluation.evaluate_program` — actual scoring

Each MOCOP benchmark has its own `Evaluation` subclass:

* `bi_tsp_semo/evaluation.py` — `BITSPEvaluation`
* `tri_tsp_semo/evaluation.py` — `TRITSPEvaluation`
* `bi_cvrp/evaluation.py` — `BICVRPEvaluation`
* `bi_kp/evaluation.py` — `BIKPEvaluation`

For bi-TSP roughly:

```python
class BITSPEvaluation(Evaluation):
    def evaluate_program(self, program: Program) -> Tuple[float, float]:
        instances = self.get_test_instances()       # e.g. 5 random TSPs
        tour_lengths = []
        runtimes = []
        for inst in instances:
            t0 = time.time()
            tour = program.run(inst)                # the LLM heuristic's output
            runtime = time.time() - t0
            tour_lengths.append(compute_two_costs(tour, inst))
            runtimes.append(runtime)
        # Aggregate: a single score tuple per heuristic
        return aggregate(tour_lengths, runtimes)
```

The exact return tuple is `(−HV_on_instances, mean_runtime)` — both
minimised. The first objective is *negative* hypervolume because
hypervolume is naturally maximised and we want a minimisation problem
throughout the framework.

### 3.5 The score tuple's role

After evaluation, `func.score = (s1, s2)` is the heuristic's coordinate
in objective space. This is the **only** value the rest of the pipeline
uses to:

* compute HVI in `RewardComputer.reward()` (input to the bandit)
* check non-dominance in `population_management` (input to the next
  generation's population)
* contribute to `cumulative_diversity()` (input to the diversity-gain
  reward term)
* contribute to `Population.selection` and the PFG grid bookkeeping

If the score is `None` or contains `NaN`, the heuristic is treated as
invalid in `_is_valid_score` ([reward.py:33-40](../reward.py#L33-L40))
and the reward path applies the penalty term.

---

## 4. Q4: A concrete worked example of one generation

Suppose `pop_size = 4`, budget = 20 (LLM calls), and we are running on
`bi_tsp` whose score tuple is `(−HV, runtime)` (both minimised).

### Step 0 — Adaptive warm-up

`_init_population` keeps calling `EoHPrompt.get_prompt_i1` until 4 valid
heuristics have been admitted. Suppose 6 attempts produced 4 successes:

| Heuristic | Score `(−HV, runtime)` | Admitted? |
|-----------|------------------------|-----------|
| `h1` | `(−8.0, 0.5)` | ✓ |
| `h2` | `(−10.0, 0.3)` | ✓ |
| `h3` | `(−7.0, 0.6)` | ✓ |
| (LLM bad output, invalid)   | — | ✗ |
| `h4` | `(−12.0, 0.2)` | ✓ |
| (timeout, invalid) | — | ✗ |

Budget remaining: **20 − 6 = 14**.
Population after warm-up: **{h1, h2, h3, h4}**.

### Step 1 — Cluster the population

`ClusterManager.cluster([h1, h2, h3, h4])` issues one LLM call. Budget
remaining: **14 − 1 = 13**.

Suppose the LLM returns `partition = [[0, 2], [1, 3]]`, meaning:
* **Cluster 0**: `{h1, h3}` — "long-tour-focused" heuristics
* **Cluster 1**: `{h2, h4}` — "fast-runtime-focused" heuristics

Cluster-quality priors (from
[cluster_manager.py:93-114](../cluster_manager.py#L93-L114)): suppose
`{0: 0.6, 1: 1.0}` after min-max normalisation (cluster 1 has the best
HV member, h4).

### Step 2 — Reset the cluster bandit

`OperatorBandit.softmax_probs()` returns the operator-level prior,
e.g. `{mutate: 0.55, crossover: 0.45}` (from earlier warm-up rewards).

`ClusterBandit.reset(n_clusters=2, cluster_quality={0:0.6, 1:1.0},
operator_priors={μ:0.55, χ:0.45})` constructs 4 arms with initial
reward-per-cost values:

| Arm | `0.5 · (cluster_quality + op_prior)` |
|------|--------------------------------------|
| `(0, μ)` | `0.5 · (0.6 + 0.55) = 0.575` |
| `(0, χ)` | `0.5 · (0.6 + 0.45) = 0.525` |
| `(1, μ)` | `0.5 · (1.0 + 0.55) = 0.775` |
| `(1, χ)` | `0.5 · (1.0 + 0.45) = 0.725` |

Each arm starts with virtual `n = 1`. Page-Hinkley states are
initialised to zero.

### Step 3 — Inner offspring loop (`target_offspring = pop_size = 4`)

**Sample 1**:
* `OperatorBandit.select()` → e.g. `mutate` (slightly higher
  reward-per-cost from priors).
* `ClusterBandit.select(restrict_operator=mutate)` → between arms
  `(0, μ)` and `(1, μ)`, picks `(1, μ)` (prior 0.775 > 0.575). Cluster
  1 = `{h2, h4}`.
* `random.choice([h2, h4])` → say `h4`.
* Build prompt `M1(h4)`. Charge budget: **13 − 1 = 12**.
* LLM returns a mutated version → `h5` with score `(−11.0, 0.25)`.
* Reward computation:
  * HVI of `h5` vs current front: small positive (h5 fills a bit
    between h2 and h4); say `HVI = 1.4`, normalised `h_norm = 0.3`.
  * Rank score: h5 beats h1, h3, h4 on objective 1 → some positive.
  * Diversity gain: tiny positive.
  * Total reward, say `R₁ = 0.45`.
* `OperatorBandit.update(mutate, 0.45)`: μ's stats updated.
* `ClusterBandit.update((1, μ), 0.45)`: arm stats updated. PH update:
  reward 0.45 vs mean — no drift yet.
* `h5` added to `next_gen_pop` (population not yet updated).

**Sample 2**:
* `OperatorBandit.select()` → `crossover` (UCB pushed χ above μ now
  that μ has been sampled more).
* `ClusterBandit.select(restrict_operator=crossover)` → between
  `(0, χ)` and `(1, χ)`, picks `(0, χ)` (let's say the budget-pressure
  term tilted things).
* `p1 = random.choice([h1, h3])` → `h1`.
* `p2 = random.choice([h2, h4])` (everyone *not* in cluster 0) → `h2`.
* Build prompt `E1([h1, h2])`. Charge budget: **12 − 1 = 11**.
* LLM returns a hybrid → `h6` with score `(−9.5, 0.45)`.
* Reward, say `R₂ = 0.2` (modest improvement on the front).
* Bandit updates as above.

**Sample 3**: similar; suppose another mutation produces `h7` with
score `(−5.0, 0.8)` (worse, dominated). Reward `R₃ = 0.05` (only rank
score contributes, since HVI = 0).
Budget: **11 − 1 = 10**.

**Sample 4**: suppose `h8` is invalid (LLM produced uncompilable code).
Budget: **10 − 1 = 9**. Reward `R₄ = −1.0` (penalty). Bandit updates;
PH for the chosen arm receives a strongly negative observation —
its `m_t` drops, but PH does not yet fire (one observation is not
enough for the gap to exceed `λ`).

`next_gen_pop` has reached `pop_size = 4`. `register_function` calls
`population_management(population + next_gen_pop, N = 4)`:

### Step 4 — `population_management` rebuilds the population

The union `{h1, h2, h3, h4, h5, h6, h7, h8}` is passed to
[population.py:70-83](../_llm4ad/method/LLMPFG/population.py#L70-L83):

```python
def population_management(population, N):
    fronts = fast_non_dominated_sort(population)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= N:
            selected.extend(front)
        else:
            distances = calculate_crowding_distance(population, front)
            sorted_front = sorted(front, key=lambda i: -distances[i])
            selected.extend(sorted_front[:N - len(selected)])
            break
    return [population[i] for i in selected]
```

Concretely (using NSGA-II's exact recipe):

1. `fast_non_dominated_sort`: classify members into Pareto layers.
   * `h7` is dominated by h2, h4 → goes to a later front.
   * `h8` has no score (invalid) → would have been filtered earlier;
     assume it never made it into `next_gen_pop`. Hmm wait — invalid
     heuristics return `False` from `_sample_eval_register` and are
     never registered, so they are not in the union at all. So the
     real union is 7 members: `{h1, …, h7}`.
   * Front 0 (non-dominated): `{h2, h4, h5, h6}` — these now dominate
     `h1, h3, h7`.
   * Front 1: `{h1, h3}`.
   * Front 2: `{h7}`.

2. Greedy fill up to `N = 4`:
   * Front 0 has 4 members, exactly fills the slot. `selected = {h2,
     h4, h5, h6}`.
   * We don't touch the lower fronts.

### Step 5 — New population, generation += 1

New population: **{h2, h4, h5, h6}**. `next_gen_pop` reset to empty.
`generation` increments from 0 to 1. Budget: **9**.

Three heuristics from the warm-up (h1, h3, h7) have been pruned because
they are dominated. h5 and h6, both bandit-generated this generation,
survived because they joined the Pareto front.

### Step 6 — Next generation

`_evolve_one_generation` runs again with the new population:
* Re-cluster `{h2, h4, h5, h6}` via the cluster LLM. Budget: **9 − 1
  = 8**. Suppose new partition is `[[1, 2, 3], [0]]` = `{h4, h5, h6}`
  and `{h2}`.
* **Cluster bandit resets** with fresh priors (cluster identities are
  not comparable across generations).
* **Operator bandit retains** its statistics — μ now has `n=2` (from
  samples 1 & 3 in the previous generation), χ has `n=2` (samples 2 &
  4), and their reward-per-cost estimates update accordingly.
* Inner loop produces 4 more offspring. Repeat until budget exhausted.

This is the rhythm. The population is replaced in batches of `pop_size`;
PFG (via `population_management`) is what does the replacing; the
bandit drives *which prompts* generate the batch.

---

## 5. Q5: Is one generation enough for the cluster bandit to learn?

**Strictly on its own observations, no — but the warm-start priors do
most of the work.** The two-layer architecture is designed around this
fact.

### The data starvation problem

In one generation the cluster bandit receives `pop_size` observations
(default 6) spread across `K × 2` arms, where `K` is the number of
clusters (typically 2–4). So each arm gets on average **0.75–1.5 raw
observations per generation** — clearly not enough for UCB1's
empirical mean to converge.

If the cluster bandit had to start from zero priors every generation,
its first few decisions would essentially be exploration draws — random
arms tried in turn. By the time the bandit had any usable information
on each arm, the generation would already be over and `reset()` would
wipe the statistics again.

### The warm-start saves it

`ClusterBandit.reset()` in
[bandit.py:174-214](../bandit.py#L174-L214) initialises each arm with
*virtual* observations based on the prior:

```python
op_prior     = operator_priors.get(o, self._prior_reward)   # from persistent OperatorBandit
q            = cluster_quality.get(k, self._prior_reward)   # from cluster manager
init_reward  = 0.5 * (q + op_prior)                          # blend of both signals

self._stats[(k, o)] = ArmStats(
    n          = 1,                       # virtual count of 1
    sum_reward = init_reward * 1,         # virtual reward = blended prior
    sum_cost   = 1,
)
```

So each arm starts as if it had already been pulled once with reward
`0.5 · (cluster_quality + operator_prior)`. Real observations during
the generation update on top of this prior. Effectively the bandit's
"mean reward" is a *prior + 1-or-2 observations* estimate, not "0 + 1
observation".

This is the key design idea: **persistent information lives in two
places** — (a) the operator bandit's cross-generation reward history,
and (b) the cluster manager's per-cluster quality function. Both feed
the cluster bandit's priors at the start of every generation. The
cluster bandit's own statistics, by contrast, only need to capture
*within-generation* refinement on top of those priors.

### What the cluster bandit actually learns within a generation

With ~6 observations per generation, the cluster bandit:

* **Cannot** rank-order all 4–8 arms by mean reward with statistical
  confidence.
* **Can** discriminate strong from weak arms once each has been pulled
  ≥ 1 real time, especially if the rewards are sharply separated (some
  HVI > 0, others HVI = 0).
* **Can** redirect budget within the generation: e.g. if the first
  pull of `(1, μ)` returns a great reward, UCB1 will keep favouring
  it for subsequent pulls of the same generation.
* **Can** trigger Page-Hinkley resets when an arm's reward stream
  drops abruptly — PH only needs a handful of observations to detect a
  drop large enough to matter.

### What the cluster bandit cannot learn within a generation

* Subtle distinctions between two arms whose true reward distributions
  differ by less than the prior noise (`std` ≈ 0.1–0.2).
* The "right" answer for clusters that are never pulled in this
  generation — they keep their warm-start prior as the only signal.

### Why this is fine

The bandit's role is not to converge to the optimal arm within a single
generation. Its role is to **bias the spending of a small, fixed
in-generation budget toward arms that look good on prior + on early
observations**. Across many generations, the operator bandit accumulates
real data; the cluster manager's quality function evolves as the
population changes; both gradually shape the priors. The cluster
bandit's *individual generation* statistics are correspondingly less
load-bearing.

### How you can tune this

If you want richer in-generation data, three knobs are available:

1. **`pop_size`** — larger pop_size means more inner-loop iterations
   per generation, so more observations per arm. Cost: fewer total
   generations under fixed budget.
2. **`gamma_budget`** — lowering the budget-pressure weight makes the
   bandit explore more aggressively in early-budget steps, giving
   under-explored arms a fairer evaluation.
3. **`prior_n`** — raising the virtual count of warm-start observations
   makes the bandit trust the prior more (less responsive to in-
   generation data); lowering it makes the bandit weight in-generation
   data more strongly.

Empirically, default `pop_size = 6` and `gamma_budget = 0.5` gives
acceptable behaviour at `B = 50`. If you push to `B ≤ 25` the cluster
bandit becomes almost pure prior — the in-generation data is just too
thin. Documenting this in the thesis is worth doing.

---

### 5.A How many clusters K is reasonable?

K is decided by the cluster LLM, not by us — the prompt asks it to
group the elites and it returns a partition. In practice, when given
5–8 elites, the LLM typically returns **K = 2–4 clusters**.

The number of bandit arms per generation is **2K** (two operators ×
K clusters). With `pop_size` observations split (roughly evenly) across
operators by the operator bandit, the expected observations per
*(cluster, operator)* arm in one generation is

```
obs_per_arm  ≈  pop_size / (2K)
```

| K | 2K arms | obs/arm at pop_size=6 | obs/arm at pop_size=12 | obs/arm at pop_size=20 |
|---|--------:|----------------------:|-----------------------:|-----------------------:|
| 2 | 4       | 1.50                  | 3.00                   | 5.00                   |
| 3 | 6       | 1.00                  | 2.00                   | 3.33                   |
| 4 | 8       | 0.75                  | 1.50                   | 2.50                   |
| 5 | 10      | 0.60                  | 1.20                   | 2.00                   |
| 6 | 12      | 0.50                  | 1.00                   | 1.67                   |

For UCB1 to update the empirical mean meaningfully away from the
warm-start prior, you want **≥ 2 real observations per arm per
generation**. Lower values mean the bandit is essentially "prior +
noise," not "prior + measurement."

Reading the table: with default `pop_size = 6`, only `K = 2` clusters
gives ≥ 1.5 obs/arm; `K = 3` is borderline; `K ≥ 4` is below 1 and the
bandit can't separate arms statistically.

**Recommended target: `K ≤ pop_size / 4`**, so each arm gets ≥ 2
observations. That means:

* `pop_size = 6`  ⇒ cap K at 2 (rare to exceed in practice anyway)
* `pop_size = 8`  ⇒ cap K at 2
* `pop_size = 12` ⇒ cap K at 3
* `pop_size = 20` ⇒ cap K at 5

Currently K is whatever the LLM returns; if you want to enforce this
cap explicitly, the cleanest place is
[cluster_manager.py:_cluster_quality](../cluster_manager.py#L93-L114)
— add a post-processing step that merges the smallest clusters until
`len(partition) ≤ K_max`. A few lines of code; well worth doing if
your sweeps consistently produce 4+ clusters.

### 5.B What `pop_size` (observations per generation) is right?

`pop_size` controls two competing quantities:

* **Obs per arm per gen** (cluster bandit data richness)
* **Number of generations** at fixed budget `B`
  — roughly `n_gen ≈ B / (pop_size + 1)` (the `+1` accounts for the
  per-generation cluster LLM call)

You want enough generations for (a) the population to evolve through
multiple replacements and (b) the operator bandit to accumulate
enough data — but also enough observations per generation that the
cluster bandit can be meaningfully informative. The two pull in
opposite directions.

Suggested ranges by budget:

| Budget `B` | `pop_size` | Expected `n_gen` | obs/arm at K=3 | Comment |
|-----------:|-----------:|------------------:|----------------:|---------|
| 25  | 4          | ~5  | 0.67 | Bandit is essentially prior-only; PFG and operator bandit still useful |
| 50  | 6 (DEFAULT)| ~7  | 1.00 | Acceptable; warm-start priors carry weight, in-gen data adds a little |
| 100 | 8          | ~11 | 1.33 | Cluster bandit starts to be measurably informative |
| 200 | 12         | ~15 | 2.00 | First regime where cluster bandit hits the "≥2 obs/arm" rule |
| 500 | 16–20      | ~25 | 2.67–3.33 | Both bandits well-fed; PH drift detection plays a real role |

The default `pop_size = 6` is **deliberately under the "≥2 obs/arm"
threshold** because at `B = 50` (the IDEA's headline regime), trading
in-gen depth for more generations gives the operator bandit more data
and gives the population more time to evolve through PFG. Both of
those matter more than cluster-bandit precision at small B.

If your headline thesis result is at `B = 200`+, switch `pop_size`
to 12–16. You can do this per-task in
[experiments/configs.py:POP_SIZES](../experiments/configs.py#L123-L128).

### 5.C Joint rule of thumb

The clean joint guideline is:

```
pop_size  ≈  4 · K        (so each arm gets ≥ 2 in-gen observations)
n_gen     ≈  B / (pop_size + 1)
```

With the LLM tending to produce `K = 2–3`, the natural sweet spot is
`pop_size ∈ [8, 12]`. Below 8 the cluster bandit is prior-driven;
above 16 you start sacrificing generation count.

A useful diagnostic to run on a single completed BMAB-LLM log:

```python
import json
banlog = json.load(open('logs_bmab/.../bandit_log.json'))
for gen in banlog:
    cluster_stats = gen['cluster']                # dict of (k, op) -> stats
    n_per_arm = [v['n'] for v in cluster_stats.values()]
    print(f"gen={gen['generation']:>2}  arms={len(n_per_arm):>2}  "
          f"min_n={min(n_per_arm)}  max_n={max(n_per_arm)}  "
          f"mean_n={sum(n_per_arm)/len(n_per_arm):.2f}")
```

If `mean_n` is hovering near 1 in every generation, the cluster
bandit is prior-driven. If it is hovering near 3+, you're in the
"informative" regime. Use this to decide whether `pop_size` is right
for your budget.

---

## Quick reference card

* Mutation uses **one cluster** (intra-cluster parent). Crossover uses
  **one cluster + everyone else** (no second bandit decision).
* Within a cluster, parent is `random.choice` — flat, uniform.
* Evaluation happens in `SecureEvaluator.evaluate_program_record_time`,
  which spawns a sandboxed subprocess that runs the task's
  `Evaluation.evaluate_program` and returns `(score, eval_time)`.
* Population updates **in batches of `pop_size`**, not after every
  sample. `population_management` (NSGA-II non-dominated sort +
  crowding distance) picks `pop_size` survivors from the union of old
  population + new offspring.
* The cluster bandit cannot fully learn from ~6 observations per
  generation — but warm-start priors from the persistent operator
  bandit and the cluster-quality function carry most of the
  information. Bandit refinement within a generation rides on top of
  those priors.
* Useful sizing rule: **`pop_size ≈ 4 · K`** so each
  `(cluster, operator)` arm receives ≥ 2 in-generation observations.
  With the LLM clusterer typically producing K = 2–3, this lands at
  `pop_size ∈ [8, 12]`. Default `pop_size = 6` is intentionally under
  this threshold so that small-`B` runs still get enough generations
  for PFG and the operator bandit to converge.
