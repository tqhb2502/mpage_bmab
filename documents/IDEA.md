# BMAB-LLM: Budgeted Multi-Objective Multi-Armed Bandit for LLM-Driven Heuristic Design

> **A refined extension of MPaGE for Multi-Objective Combinatorial Optimization (MOCOP)
> under a strict budget on LLM API calls.**

---

## 1. Motivation and Problem Statement

### 1.1 Limitations of MPaGE that Motivate BMAB-LLM

MPaGE (Ha et al., 2025) is the state of the art for LLM-driven heuristic design on
multi-objective combinatorial optimization. It combines (i) the SEMO paradigm,
(ii) Pareto-Front-Grid (PFG) selection of elite heuristics, and (iii) **LLM-based
semantic clustering**, performing intra-cluster *Mutate* and inter-cluster *Crossover*
to generate offspring heuristics.

A close reading of the paper and its source code (`llm4ad/method/LLMPFG/`) reveals
**three structural weaknesses** that severely hurt sample-efficiency under tight
LLM-call budgets:

| # | Weakness in MPaGE | Source-code evidence |
|---|--------------------|------|
| **W1** | Operator probability `γ ∈ {0.3}` between Mutate and Crossover is **fixed for the entire run**, ignoring the fact that Crossover dominates exploration early and Mutate dominates exploitation late. | `prompt.py` & `eoh.py` apply the four operators (`e1`,`e2`,`m1`,`m2`) every generation in equal proportion. |
| **W2** | Cluster selection is **uniform random**: a cluster `C_i` is sampled by `random.choice`, the second parent `h'` is sampled with `U(∪_{k≠i} C_k)`. Promising clusters and "dead" clusters receive identical budget. | `population.py:selection_cluster` line 251–284. |
| **W3** | **No budget on LLM calls**: the loop terminates only when `max_generations` or `max_sample_nums` is reached. Wasteful clusters and redundant calls cannot be aborted early. | `eoh.py:_thread_do_evolutionary_operator` lines 173–262. |

### 1.2 Goal of the Thesis

Given a **fixed budget** `B ∈ ℕ` (max number of LLM calls allowed), design a system that
discovers a Pareto front of heuristics whose hypervolume is **as high as possible**
across all objectives.

Formally, we seek:

```
maximise  HV( PF_heuristic )    subject to  Σ_t cost(call_t) ≤ B
```

where `PF_heuristic` is the Pareto front of generated heuristics in the criteria
space (negative-HV-of-solutions, runtime).

---

## 2. Strengths and Weaknesses of the Original Idea (Vietnamese Draft)

### 2.1 Strengths

1. **Diagnosis is correct**: the three issues identified in the draft (fixed γ,
   uniform cluster sampling, no budget) are real and verifiable directly in the
   source code.
2. **Choice of MAB family is principled**: Budgeted MO-MAB with hypervolume-based
   reward is a well-studied combination (Drugan & Nowé 2013, Yahyaa et al. 2014).
3. **Reuse of LLM-based semantic clustering**: clustering the (potentially infinite)
   heuristic space into a small set of arms is the only realistic way to make MAB
   feasible — the draft is right to stand on top of MPaGE's clustering rather than
   replace it.
4. **Page-Hinkley drift detection**: appropriate because the reward distribution is
   non-stationary (operator effectiveness shifts as the population matures).
5. **Diversity-aware reward** is correctly identified as a guard against
   reward-hacking on a single Pareto region.

### 2.2 Weaknesses Identified, and Refinements

| Weakness | Refinement adopted in BMAB-LLM |
|----------|--------------------------------|
| **Arm definition is loose** — "cluster as arm" but clusters are *re-built every generation* by an LLM, so cluster IDs are not comparable across generations. | We separate the bandit into **two layers**: (a) a *persistent operator-level* bandit (Mutate vs Crossover, 2 arms) carrying statistics across generations; (b) a *per-generation cluster-level* bandit re-initialised from a **warm start** built from the parent cluster's quality and from operator-level priors. This dissolves the cluster-identity problem without losing context. |
| **Reward of zero is too common** — most generations produce no new Pareto-front element, so HVI = 0 dominates the signal. | We use a **rank-based reward with credit smoothing**: reward = max(HVI, ε) + α·rank(score over recent window), and we keep a **fitness-improvement bonus** even for non-Pareto offspring. |
| **No explicit guarantee of diversity** — pure HVI-maximisation collapses onto the densest Pareto region. | Reward augmented with **Δ-Cumulative Diversity Index (ΔCDI)** and **Δ-Shannon Wiener (ΔSWDI)**, the same diversity metrics used for evaluation in MPaGE. This makes the optimisation target consistent with the evaluation target. |
| **Token-aware cost is mentioned but not formalised** — different prompts have wildly different token budgets. | Cost = `len_in_tokens(prompt) + len_in_tokens(response)` retrieved from the LLM API; we cap a single call at `B/(K·5)` tokens so a single arm cannot exhaust the budget. Falls back to unit cost (`c_t=1`) when the API does not return token usage. |
| **MCTS extension is overkill** for a master's thesis given the running cost. | Removed from the core proposal; left as a discussion section for "future work". |
| **Initialisation eats too much budget** — naïve `pop_size` zero-shot calls per run. | Replaced with **adaptive warm-up**: stop initialisation as soon as the population spans `≥ p` distinct grid cells *and* succeeds at least `pop_size//2` times, rather than fixed `pop_size` successes. |

---

## 3. Refined Framework: **BMAB-LLM**

### 3.1 Notation

| Symbol | Meaning |
|--------|---------|
| `B`            | Total budget on LLM calls (or tokens). |
| `b_t`          | Remaining budget at step `t`. |
| `c_t`          | Cost of the `t`-th call. |
| `H(t)`         | Heuristic population at generation `t`. |
| `E(t) ⊆ H(t)`  | Elite set selected by Pareto-Front-Grid (PFG). |
| `{C_1,…,C_K}`  | Semantic clusters returned by LLM clustering on `E(t)`. |
| `O = {μ, χ}`   | Operator alphabet: `μ`=Mutate, `χ`=Crossover. |
| `a = (k, o)`   | An arm: pick cluster `k` and apply operator `o`. |
| `R_t`          | Scalar reward of the action at step `t`. |
| `Δ_HV`         | Hypervolume improvement on the heuristic front. |
| `Δ_SWDI`       | Shannon–Wiener diversity index increment. |
| `Δ_CDI`        | Cumulative Diversity Index increment. |

### 3.2 Two-Layer Bandit Architecture

We use a **hierarchical AOS** that respects the structural difference between
"operator type" (persistent) and "cluster identity" (ephemeral).

```
                ┌──────────────────────────────┐
                │  Operator-level UCB (2 arms) │   <-- carries across generations
                │      μ (Mutate)  /  χ (XO)   │
                └──────────────┬───────────────┘
                               │   p(o)
                               ▼
                ┌──────────────────────────────┐
                │ Cluster-level Budgeted UCB   │
                │       arms = {C_1,…,C_K}     │   <-- reset every generation
                │      reset & warm-started    │       with prior from operator
                └──────────────────────────────┘
```

**Step (i) — Operator selection.** Persistent statistics
`(N_o, S_o^R, S_o^c)` for `o ∈ {μ, χ}` are kept. Probability:

```
score_o = S_o^R / max(S_o^c, 1)  +  c_op · sqrt(2 · ln Σ_o' N_{o'} / N_o)
```

The operator with maximal `score_o` is selected.

**Step (ii) — Cluster selection given `o`.** For each cluster `k`, statistics
`(N_k, S_k^R, S_k^c)` are warm-started from:

* the parent cluster's quality prior (current implementation combines outer
  HV contribution, best inner-HV proxy and runtime proxy);
* an optimistic prior with `N_k = 1`, `S_k^R = R̄ + β·σ_R`.

A **Budgeted UCB1** score is computed:

```
score_k = (S_k^R / max(S_k^c, 1))                       (Exploitation per unit cost)
        + c_cl · (b_t/B)^γ_b · sqrt(2 · ln Σ_k' N_{k'} / N_k)
                                                               (Annealed exploration)
```

The `(b_t/B)^γ_b` factor discourages risky exploration as `b_t → 0`, in line
with finite-horizon MAB intuition. The current code uses `γ_b = 0.5` by
default. Disabling budget annealing recovers ordinary BUCB exploration.

### 3.3 Reward Function (Multi-Objective + Diversity)

For each generated heuristic `h`, evaluated with score `s(h) = (–HV, time)`:

```
R(h) = w_q · quality_signal(s(h) ; F_t)  +  w_d · ΔCDI(h ; H_t)
       + w_r · rank_score(s(h) ; H_t)
       –  λ_pen · 1[h is invalid or times-out]
```

* `quality_signal` is selected by `reward_mode`. The default `final_hv` mode
  uses normalized HV gain after applying the managed-population cap. The
  `dense` mode uses immediate HVI, and `hybrid` averages the two.
* `ΔCDI(h; H_t)` = increase of the cumulative diversity index after inserting
  `h` into the population (uses the same code definition as MPaGE).
* `rank_score(s; H_t)` ∈ [0,1] is the fraction of population members that the
  new score improves on at least one objective, smoothing sparse quality
  feedback into a dense signal.
* Default weights: `w_q = 1.0`, `w_d = 0.3`, `w_r = 0.2`, `λ_pen = 1.0`.
* Immediate HVI and managed-population HV delta are normalized over a rolling
  window before being exposed to UCB, making the reward scale more stable.

### 3.4 Drift Detection (Page–Hinkley)

For each arm we maintain `m_t = m_{t−1} + (R_t − R̄_t + δ)`, `M_t = max_{s ≤ t} m_s`.
If `M_t − m_t > λ_PH`, the arm's statistics are reset to optimistic values
*for that generation only*. This handles the situation where a previously good
cluster becomes barren after the population has adopted its style.

We use `δ = 0.005, λ_PH = 0.5` after rank normalisation, giving an empirical
false-alarm rate of ≈ 5 %.

### 3.5 Budget Accounting

Every callable that hits the API decrements the budget:

| Call type | Default cost |
|-----------|-------------:|
| Initialisation (`i1`) | 1 |
| Mutation (`m1`/`m2`) | 1 |
| Crossover (`e1`/`e2`) | 1 |
| Cluster query | 1 (configurable; can be set to 0 if cheap LLM is used) |
| Suggestion (review) call | 1 (only when `--llm_review` is on) |

For *token-aware* mode we replace the above with the actual number of input + output
tokens reported by the OpenAI API (`response.usage.total_tokens`).

The main loop terminates when `b_t ≤ 0` instead of when generations expire. This
shifts the "stopping criterion" from "fixed-time" to "fixed-cost", which is the
right scientific operationalisation of an LLM-call budget.

### 3.6 Algorithm (Pseudo-code)

```text
Input  : B ∈ ℕ (LLM-call budget), Evaluation E, LLM, LLM_cluster
Output : Pareto front of heuristics

1.  H ← Init(B)                                         # adaptive warm-up
2.  while remaining_budget > 0:
3.      E_t ← PFG_select(H)
4.      groups ← LLM_cluster(E_t)                       # 1 call
5.      bandit.warm_start(groups)
6.      while remaining_budget > 0 AND not generation_done:
7.          o ← OperatorBandit.select()
8.          k ← ClusterBandit.select(o, b_t, B)
9.          parents ← sample_parents(k, o, groups, E_t)
10.         prompt ← Prompt(o, parents, E_t)
11.         (h, cost) ← LLM(prompt)                     # 1 call, –cost
12.         score ← Evaluate(h)
13.         R ← Reward(h, score, H)
14.         OperatorBandit.update(o, R, cost)
15.         ClusterBandit.update(k, o, R, cost)
16.         PageHinkley.check_and_reset((k,o))
17.         H ← H ∪ {h}; H ← non_dominated_filter(H)
18.     end-while
19. end-while
20. return non_dominated(H)
```

### 3.7 Complexity Analysis

| Component | Complexity per step |
|-----------|---------------------:|
| UCB scoring | `O(K)` |
| Page-Hinkley update | `O(1)` per arm |
| HV improvement | `O(|F|·M)` (M = num criteria) |
| LLM call (dominant cost) | API-side |

The bandit overhead is therefore *negligible* compared to the cost of each LLM
call, which is the resource we minimise.

### 3.8 Theoretical Properties (Sketch)

* The cluster-level scheduler is a **Budgeted Knapsack UCB1** (Ding et al. 2013),
  so its expected regret is `O(√(B · K · log B))` *under the stationarity
  assumption*. PH drift-resets buy non-stationarity at the price of a small
  bounded number of additional explorations.
* The reward function is **bounded** (HVI is bounded by the volume of the
  evaluation box; CDI ∈ [0, log K]), so the MAB analysis applies directly.

---

## 4. Experimental Plan

### 4.1 Benchmarks (matching MPaGE)

| Problem | Sizes | Objectives |
|---------|-------|-----------|
| **Bi-TSP** (`bi_tsp_semo`) | 20, 50, 100, 150 | tour length in 2 spaces |
| **Tri-TSP** (`tri_tsp_semo`) | 20, 50, 100 | tour length in 3 spaces |
| **Bi-CVRP** (`bi_cvrp`) | 20, 50, 100 | length, makespan |
| **Bi-KP** (`bi_kp`) | 50, 100, 200 | two profit sums |

### 4.2 Evaluation Metrics

* **HV ↑**, **IGD ↓** — solution quality on the heuristic Pareto front.
* **SWDI ↑**, **CDI ↑** — heuristic-population diversity.
* **AUBC** — *Area-Under-Budget-Curve*: integral of HV(b) over `b ∈ [0, B]`.
  This is **the single most important metric** for the thesis: it summarises
  *how fast* a method converts budget into Pareto-quality. A method that tops the
  HV table after the full budget but is dominated for `b < B/2` should not be
  considered superior.

### 4.3 Baselines

1. **MPaGE-orig**: the unmodified original framework.
2. **MPaGE-budget**: original framework but stopped at the same `B` as ours
   (stops mid-generation).
3. **BMAB-LLM (no PH)**: ablation removing drift detection.
4. **BMAB-LLM (no diversity term)**: ablation with `w_d = 0`.
5. **BMAB-LLM (op-only)**: ablation removing the cluster-level layer.
6. **BMAB-LLM (full)**: the proposed system.
7. (External) **EoH, MEoH, ReEvo, HSEvo** at the same `B`.

### 4.4 Budget Settings

`B ∈ {25, 50, 100, 200}`. The interesting regime is `B = 50`, since
MPaGE-orig cannot even finish its first generation under `B = 50`.

### 4.5 Statistical Protocol

* 5 seeds per setting.
* Wilcoxon signed-rank test on AUBC pairs.
* Report mean ± std for each metric.

---

## 5. Expected Contributions

1. **First budgeted formulation** of LLM-driven heuristic design for MOCOP.
2. **Two-layer hierarchical AOS** (operator + cluster), which solves the
   "cluster-identity-across-generations" problem ignored by previous AOS-on-LLM
   work.
3. **AUBC** as a new evaluation metric that captures the budget-quality
   trade-off, more diagnostic than reporting HV at one fixed `B`.
4. **Open-source reference implementation** released under the same license
   structure as MPaGE.

---

## 6. Why this is "the best version" of the original idea

Compared to the original Vietnamese draft, this refined version:

* **Disambiguates** the meaning of "arm" (operator vs cluster vs hybrid) and
  picks the one that is mathematically tractable AND empirically meaningful.
* **Replaces** the brittle uniform-mixture reward with a rank-normalised
  multi-objective reward whose components are computable from quantities the
  baseline already maintains (no new bookkeeping).
* **Drops** the MCTS extension which would inflate scope without proportional
  benefit; keeps it as future work.
* **Adds AUBC** as the headline metric, because it is the only one that
  *demonstrates* the whole point of the project — that BMAB-LLM produces a
  better Pareto front *for a given budget*.
* **Specifies adaptive warm-up**, fixing the silent bug of fixed-cost
  initialisation that wastes budget when `pop_size` is large.

The result is a research design that is leaner, sharper, and reproducible end-to-end.

---

## 7. References (selected)

* Ha et al., *MPaGE: Pareto-Grid-Guided LLMs for Fast and High-Quality Heuristics
  Design in MOCOP*, arXiv 2507.20923 (2025).
* Fialho L., Costa C.A., Schoenauer M., Sebag M., *Adaptive Operator Selection
  with Dynamic Multi-Armed Bandits*, GECCO 2008.
* Drugan M.M., Nowé A., *Designing Multi-Objective MAB Algorithms — a study*,
  IJCNN 2013.
* Ding Y., Qin Z., Zhu W., Yu Y., *Multi-Armed Bandit with Budget Constraint
  and Variable Costs*, AAAI 2013.
* Yahyaa S.Q., Manuel Drugan M., Manderick B., *Annealing-Pareto Multi-Objective
  Multi-Armed Bandit Algorithm*, ADPRL 2014.
* Bubeck S., Cesa-Bianchi N., *Regret Analysis of Stochastic and Non-Stochastic
  Multi-Armed Bandit Problems*, FnT-ML 2012.
