# What BMAB-LLM Retains From the Original MPaGE Project

The relationship between BMAB-LLM and MPaGE is **"extend, don't replace"** — every primitive that already worked is reused, and only the *adaptive scheduling layer* is new. This document is the inventory, organised by what is retained and what role it plays.

---

## Table of contents

1. [Retained code: the vendored `_llm4ad/` subtree](#1-retained-code-the-vendored-_llm4ad-subtree)
2. [Retained algorithmic ideas](#2-retained-algorithmic-ideas)
3. [Retained operational conventions](#3-retained-operational-conventions)
4. [What's new (for contrast)](#4-whats-new-for-contrast)
5. [One-line summary](#5-one-line-summary)

---

## 1. Retained code: the vendored `_llm4ad/` subtree

The entire upstream `llm4ad` library is included verbatim as `mpage_bmab/_llm4ad/` (relative imports patched, but logic untouched). This is not just convenient — it makes the comparison apples-to-apples, because the BMAB-LLM driver calls the *same* code paths the upstream `MPaGE` driver does for sampling, evaluation, and population maintenance.

### 1.1 `_llm4ad/base/` — data types and evaluation infrastructure

| Symbol | Role kept |
|--------|-----------|
| `Function` | Dataclass representing one heuristic (code + score tuple + algorithm description + timing). Every offspring BMAB-LLM produces is a `Function`. |
| `Program` | A `Function` wrapped with its imports — the unit that the secure evaluator actually executes. |
| `TextFunctionProgramConverter` | Parses LLM-generated text into `Function`/`Program`. Used in [bmab_llm.py:93-97](../bmab_llm.py#L93-L97) for the template, and in [bmab_llm.py:363-364](../bmab_llm.py#L363-L364) for every sampled offspring. Replacing it would mean reimplementing the LLM-text → Python parser. |
| `LLM` | Abstract base for language model interfaces. Both `HttpsApiOpenAI` and `HttpsApiOpenAI4Cluster` inherit from it. |
| `Evaluation` / `SecureEvaluator` | The contract for benchmark scoring + the sandboxed subprocess runner with timeout. BMAB-LLM wraps `SecureEvaluator` for every offspring evaluation ([bmab_llm.py:101-102](../bmab_llm.py#L101-L102)). This is what makes invalid/looping heuristics safe — without it, the LLM could DOS the run. |
| `ModifyCode` | Utilities for stripping markdown fences and extracting function bodies from LLM responses. Called transitively by `EoHSampler`. |

### 1.2 `_llm4ad/tools/llm/` — LLM API wrappers

| Symbol | Role kept |
|--------|-----------|
| `HttpsApiOpenAI` | Chat-completions wrapper. Issues every heuristic-generation request in BMAB-LLM. |
| `HttpsApiOpenAI4Cluster` | Structured-output (Pydantic) wrapper for the clustering LLM — guarantees a typed `{groups: [[…], …]}` response. Without this the clustering step would have to parse free-form JSON, which is brittle. |

### 1.3 `_llm4ad/tools/profiler/`

| Symbol | Role kept |
|--------|-----------|
| `ProfilerBase` | The base profiler that creates the timestamped run directory and writes `run_log.txt`. `BMABProfiler` extends `EoHProfiler` which extends this; the resulting output directory layout is identical to the upstream MPaGE convention so downstream analysis scripts work for either method. |

### 1.4 `_llm4ad/method/LLMPFG/` — the heart of MPaGE

This subtree is the most heavily reused.

| Symbol | Role kept |
|--------|-----------|
| `Population` | **The PFG itself.** Maintains the Pareto-front grid, performs non-dominated sorting and crowding-distance ranking, and exposes `selection(...)` for elite retrieval. BMAB-LLM calls `Population.selection(self._selection_num)` at the start of every generation ([bmab_llm.py:228](../bmab_llm.py#L228)). Replacing it would mean reimplementing the elite-selection algorithm from the MPaGE paper. |
| `EoHPrompt` | **The prompt taxonomy.** Six prompt templates: `get_prompt_i1` (initialisation), `get_prompt_e1`, `get_prompt_e2` (crossover variants), `get_prompt_m1`, `get_prompt_m2` (mutation variants), `get_prompt_cluster` (cluster the elites), `get_prompt_suggestions_only` (review/reflection). BMAB-LLM uses **all six** as-is in [bmab_llm.py:401-461](../bmab_llm.py#L401-L461). The bandit chooses *which* prompt to fire; the prompt content itself is identical to MPaGE. |
| `EoHSampler` | Wraps `LLM` with helpers `get_thought_and_function(prompt)` and `get_thought(prompt)`. Handles thought / code extraction. BMAB-LLM uses the same two methods. |
| `EoHProfiler` | Per-generation population JSON dumps. `BMABProfiler` subclasses it and adds the budget-vs-HV curve recording; everything `EoHProfiler` already wrote (`samples/*.json`, `population/*.json`, `run_log.txt`) is unchanged. |
| `MPaGE` class | The upstream driver. Retained literally — `MPaGEBudget` in [mpage_orig.py](../mpage_orig.py) is a *subclass*, not a rewrite. We override `_sample_evaluate_register` and patch the cluster sampler in `__init__`; the loop structure (`_thread_init_population`, `_thread_do_evolutionary_operator`, `_do_sample`, `run`) is inherited intact. |

### 1.5 `_llm4ad/task/optimization/` — the benchmark suite

| Task | Role kept |
|------|-----------|
| `bi_tsp_semo` | Bi-objective TSP — `BITSPEvaluation` + template heuristic + random instance generator. |
| `tri_tsp_semo` | Tri-objective TSP. |
| `bi_cvrp` | Bi-objective CVRP. |
| `bi_kp` | Bi-objective Knapsack. |

All four are used directly as the experimental matrix in [experiments/configs.py:54](../experiments/configs.py#L54). The only thesis-level reason to swap them out would be to broaden coverage to a fifth task; the four already exercise both 2-D and 3-D Pareto fronts.

---

## 2. Retained algorithmic ideas

Beyond reusing code, BMAB-LLM retains MPaGE's core algorithmic decisions.

### 2.1 The SEMO paradigm

A non-dominated archive of heuristics, updated by inserting one offspring at a time and pruning dominated entries. BMAB-LLM's `Population` *is* the SEMO archive (inherited from MPaGE's `Population` class). The inner loop in [bmab_llm.py:254-322](../bmab_llm.py#L254-L322) follows the SEMO pattern: select parent(s), generate offspring, evaluate, register.

### 2.2 Pareto-Front-Grid (PFG) elite selection

The grid-based elite selector that encourages spread along the objective space rather than concentration on one tip. BMAB-LLM does not touch this — `Population.selection()` is called unchanged. Without PFG, the elite pool would concentrate on whichever Pareto region currently has the highest density and the cluster bandit would have very little to work with.

### 2.3 LLM-based semantic clustering

The single most distinctive idea of MPaGE: rather than syntactic clustering on the code itself, **ask an LLM** to group elites by behavioural similarity. BMAB-LLM uses this exact mechanism via `EoHPrompt.get_prompt_cluster` + `HttpsApiOpenAI4Cluster`. The bandit operates *on top of* the LLM-produced partition — it does not re-cluster. See [cluster_manager.py:76-89](../cluster_manager.py#L76-L89).

### 2.4 The operator taxonomy

MPaGE defines five operator types:

| Operator | Role kept verbatim |
|----------|---------------------|
| `I1` | Initialisation — generate a fresh heuristic from scratch given the task description. Used by BMAB-LLM's adaptive warm-up in [bmab_llm.py:211-212](../bmab_llm.py#L211-L212). |
| `E1` | Crossover by combining ideas from two parents. |
| `E2` | Crossover variant with an alternative prompt framing. |
| `M1` | Mutation that asks the LLM to *modify* a single parent. |
| `M2` | Mutation variant that asks for a *different style* of change. |

BMAB-LLM groups `{E1, E2}` under "crossover" and `{M1, M2}` under "mutation" for the operator bandit's two arms, then randomly selects E1 vs E2 or M1 vs M2 inside `_make_prompt`. The bandit choice is "category level" (μ vs χ); the within-category choice stays uniform random, exactly as in MPaGE.

### 2.5 Intra-cluster mutation, inter-cluster crossover

This dichotomy is the key insight of MPaGE's cluster-aware sampling: a mutation parent should come from *inside* the selected cluster (preserve style), while a crossover should mix parents from *different* clusters (cross-pollinate styles). BMAB-LLM keeps this rule, but parent draws are now weighted so PFG-selected elites are more likely to be chosen:

```python
if op == 'mutate':
    parent = weighted_parent_choice(in_cluster, pfg_elites)   # intra-cluster
    ...
else:  # crossover
    other_clusters = [i for i in range(len(partition)) if i != cluster_idx]
    other_indivs   = [... from those other clusters ...]
    p2 = weighted_parent_choice(other_indivs, pfg_elites)     # inter-cluster
```

### 2.6 Hypervolume as the quality metric

Both methods evaluate the heuristic-Pareto-front via reference-point hypervolume (via `pymoo`). In the current fixed BMAB-LLM implementation, the default reward quality signal is the normalized HV gain after adding a candidate and applying the same managed-population cap used for the final population. The legacy immediate-HVI signal is still available through the `dense_reward` ablation, and a half-immediate/half-final blend is available through `hybrid_reward`.

### 2.7 Multi-objective evaluation conventions

The metric quartet **HV / IGD / SWDI / CDI** from MPaGE is preserved. BMAB-LLM's `RewardComputer` even reuses `cumulative_diversity` / `shannon_diversity` definitions ([reward.py:104-126](../reward.py#L104-L126)) — these are MPaGE's *evaluation* metrics, now also feeding the reward.

### 2.8 Adaptive warm-up (extended, not replaced)

MPaGE's `_thread_init_population` uses `initial_sample_nums_max` as a safety cap. BMAB-LLM keeps the same idea but tightens it: stop when **either** `init_target_successes` valid heuristics OR `init_max_calls` attempts is reached, *and* when the budget runs out. See [bmab_llm.py:200-218](../bmab_llm.py#L200-L218). The underlying I1-prompt-loop is the same.

### 2.9 Optional review/suggestion call

MPaGE's `--review` flag invokes an extra LLM call (`get_prompt_suggestions_only`) before crossover to inject reflection. BMAB-LLM keeps the same flag and the same prompt path in [bmab_llm.py:447-454](../bmab_llm.py#L447-L454); the only difference is that the review call is now charged to the budget like any other LLM call.

---

## 3. Retained operational conventions

These are not algorithmic but they matter for reproducibility and tooling continuity.

| Convention | Role kept |
|------------|-----------|
| API-key files `secret.txt` / `secret_cluster.txt` | Same naming, same content, same location — copy a working MPaGE run's secrets and it just works in BMAB-LLM. |
| Log directory format `<TIMESTAMP>_<Problem>_<method>/` | Produced by `ProfilerBase`; BMAB-LLM inherits this so existing MPaGE analysis scripts that walk this directory pattern work without modification. |
| `run_log.txt` text format | Identical line format — diffable side-by-side with MPaGE runs. |
| `samples/samples_*.json` and `population/pop_*.json` | Same schema (per-heuristic + per-generation dumps). |
| `--debug` interactive prompt inspection | `MPaGE` blocks for `input()` after printing each prompt when `--debug` is set; BMAB-LLM's `--debug` is similar but prints a one-line bandit decision instead of blocking. |
| CLI flag names | `--task`, `--budget` / `--max_sample_nums`, `--pop_size`, `--llm_model`, `--openai_base_url`, `--review`, `--seed` — all match. |

---

## 4. What's new (for contrast)

For completeness, here is the *scheduling layer* added on top of all the above.

| New component | Replaces / augments |
|----------------|-----------------------|
| `BudgetTracker` ([budget.py](../budget.py)) | The hard stop. Replaces MPaGE's "stop at `max_sample_nums`" with a uniform per-call accounting that also covers cluster + review calls. |
| `OperatorBandit` + `ClusterBandit` ([bandit.py](../bandit.py)) | Replaces MPaGE's round-robin operator schedule and uniform random cluster selection. |
| `PageHinkleyState` ([bandit.py:30-63](../bandit.py#L30-L63)) | No analogue in MPaGE — new drift-detection layer for non-stationary cluster rewards. |
| `RewardComputer` ([reward.py](../reward.py)) | No analogue in MPaGE — MPaGE has no reward signal, only an end-of-run HV. Current modes are `final_hv`, `dense`, and `hybrid`. |
| `ClusterManager` ([cluster_manager.py](../cluster_manager.py)) | Thin layer on top of MPaGE's `EoHPrompt.get_prompt_cluster` — adds budget-aware fallback and the warm-start quality priors. |
| `BMABProfiler.record_curve_point` + `aubc()` ([profiler.py](../profiler.py)) | New AUBC metric; MPaGE only reports final HV. |
| Adaptive warm-up with success counting | Extension of MPaGE's init loop, not a replacement. |

---

## 5. One-line summary

> **BMAB-LLM keeps everything in MPaGE that is about *what* heuristic-design primitives exist (population, prompts, operators, clustering, evaluation, benchmarks) and replaces only *how* those primitives are scheduled (adaptive bandits + drift detection + reward signal + hard budget) along with the evaluation metric (AUBC instead of final HV).**

This is also why the `mpage_orig` runner exists ([mpage_orig.py](../mpage_orig.py)): with everything except the scheduler shared, you can run MPaGE-orig and BMAB-LLM in the *same* harness on the *same* tasks with the *same* secrets and the only thing that varies is the scheduler — exactly the comparison [IDEA.md §4.3](IDEA.md) calls for.
