# BMAB-LLM

Budgeted Multi-Objective Multi-Armed Bandit driver for LLM-based heuristic
design on Multi-Objective Combinatorial Optimisation (MOCOP).

This folder is a **standalone project**: it has its own virtualenv, its own
copy of the upstream MPaGE primitives (vendored under `_llm4ad/`) and its own
benchmark tasks. It does **not** depend on the parent project's `llm4ad/` —
you can copy this folder somewhere else, install its requirements, and run.

This is the source-code implementation that accompanies the refined idea
described in [`IDEA.md`](IDEA.md).

## Overview

BMAB-LLM extends MPaGE by:

1. Treating the **operator choice** (`Mutate` vs `Crossover`) as a persistent
   UCB1 bandit that carries statistics across generations.
2. Treating the **cluster choice** (which semantic cluster to draw the parent
   from) as a per-generation Budgeted UCB1 bandit, warm-started from
   front-aware cluster-quality and operator-level priors.
3. Adding a **Page-Hinkley** drift test that resets a `(cluster, operator)`
   arm whenever its reward distribution drops, so a cluster that *was*
   productive but has become barren no longer eats budget.
4. Defining a **bounded multi-objective reward** combining a configurable
   quality signal, a rank-based smoothing term, and a CDI-based diversity
   bonus, with a penalty for invalid heuristics. The default quality signal is
   final managed-population HV improvement; legacy immediate-HVI and hybrid
   modes are available for ablation.
5. Replacing the fixed *generation budget* with an explicit **LLM-call
   budget** `B`. The loop terminates exactly when `B` is exhausted.
6. Recording the full **budget-vs-HV curve** so we can compute the
   **AUBC** metric (Area-Under-Budget-Curve), which captures the speed
   at which budget is converted into Pareto quality.
7. Flushing pending valid offspring into the final managed population and
   avoiding a final cluster-only call when only one generation call remains,
   so terminal HV reflects the useful work paid for by the final budget units.

## Folder layout

```
mpage_bmab/
├── README.md           ← this file
├── IDEA.md             ← refined research design
├── requirements.txt    ← pip requirements for the venv
├── .gitignore
├── __init__.py         ← public exports
├── main.py             ← CLI entry-point
├── budget.py           ← BudgetTracker (call/token-aware)
├── bandit.py           ← OperatorBandit, ClusterBandit, PageHinkleyState
├── reward.py           ← HVI, diversity, RewardComputer
├── cluster_manager.py  ← LLM-clustering wrapper with budget accounting
├── profiler.py         ← BMABProfiler (extends MPaGE EoHProfiler with AUBC)
├── bmab_llm.py         ← BMABLLM orchestrator (the analogue of MPaGE)
└── _llm4ad/            ← vendored MPaGE primitives + benchmarks
    ├── base/           ←   Function/Program, evaluator, sampler, code-modifier
    ├── tools/          ←   ProfilerBase + LLM API wrappers (OpenAI, OpenAI-cluster)
    ├── method/LLMPFG/  ←   PFG Population, EoHPrompt, EoHSampler, EoHProfiler
    └── task/optimization/
        ├── bi_tsp_semo/    ← Bi-objective TSP
        ├── tri_tsp_semo/   ← Tri-objective TSP
        ├── bi_cvrp/        ← Bi-objective CVRP
        └── bi_kp/          ← Bi-objective Knapsack
```

`_llm4ad/` is a snapshot copied from the upstream MPaGE project with all
imports rewritten to be package-relative. You can replace it with a newer
upstream snapshot at any time by re-copying and re-running the relative-import
patches noted in [the project history](IDEA.md).

## Setup

```bash
# from the project root that contains this folder
python3 -m venv mpage_bmab/.venv
mpage_bmab/.venv/bin/pip install -r mpage_bmab/requirements.txt
```

You can also copy `mpage_bmab/` to a new working directory and run the same
two commands — the project is self-contained.

Place your API keys as in MPaGE:

```bash
echo "<openai-api-key>" > secret.txt
echo "<openai-api-key>" > secret_cluster.txt
```

## Running

From the directory **containing** `mpage_bmab/`:

```bash
mpage_bmab/.venv/bin/python -m mpage_bmab.main \
    --task bi_tsp \
    --budget 50 \
    --pop_size 6 \
    --log_dir logs_bmab
```

Use `--help` to see the full set of CLI options. Highlights:

| Option | Default | Effect |
|--------|---------|--------|
| `--task` | `bi_tsp` | One of `bi_tsp`, `tri_tsp`, `bi_cvrp`, `bi_kp`. |
| `--budget` | `50` | LLM-call budget `B`. The loop terminates when `B` is exhausted. |
| `--budget_mode` | `call` | `call` = unit cost per LLM request; `token` = future-work hook for token-aware accounting. |
| `--c_op`, `--c_cluster` | `1.0` | UCB1 exploration coefficients for the operator / cluster bandits. |
| `--gamma_budget` | `0.5` | Exponent used to anneal cluster-bandit exploration as remaining budget shrinks. |
| `--ph_delta`, `--ph_threshold` | `0.005`, `0.5` | Page-Hinkley parameters. |
| `--w_quality`, `--w_diversity`, `--w_rank` | `1.0`, `0.3`, `0.2` | Reward weights. |
| `--reward_mode` | `final_hv` | Quality signal: `final_hv`, `dense`, or `hybrid`. |
| `--disable_budget_annealing` | `False` | Disable remaining-budget exploration annealing in the cluster bandit. |
| `--review` | `False` | Enable LLM-based reflection/suggestion call before crossover (extra +1 budget per call). |

## Output

The profiler writes to `<log_dir>/<TIMESTAMP>_<Problem>_BMAB-LLM/`:

* `run_log.txt` — text log (compatible with MPaGE's format).
* `samples/samples_*.json` — every heuristic ever evaluated.
* `population/pop_*.json` — population at each generation.
* `budget_curve.json` — `(consumed_budget, hv, pareto_size)` triples.
* `bandit_log.json` — per-generation operator and cluster bandit statistics.
* `budget_history.json` — every LLM-call's budget deduction.
* `aubc.json` — the headline **AUBC** metric.

## API usage

```python
from mpage_bmab import BMABLLM, BMABProfiler
from mpage_bmab._llm4ad.tools.llm.llm_api_openai import HttpsApiOpenAI
from mpage_bmab._llm4ad.tools.llm.llm_api_openai_cluster import HttpsApiOpenAI4Cluster
from mpage_bmab._llm4ad.task.optimization.bi_tsp_semo import BITSPEvaluation

llm        = HttpsApiOpenAI(...)
llm_clust  = HttpsApiOpenAI4Cluster(...)
task       = BITSPEvaluation()

profiler = BMABProfiler(log_dir='logs_bmab', ref_point=(20.0, 60.0))

bmab = BMABLLM(
    llm=llm,
    llm_cluster=llm_clust,
    evaluation=task,
    budget=50,                       # 50 LLM calls
    ref_point=(20.0, 60.0),
    pop_size=6,
    profiler=profiler,
)
bmab.run()
print('AUBC =', profiler.aubc(50))
```

## Comparison with the original MPaGE

| Aspect | MPaGE | BMAB-LLM |
|--------|-------|---------|
| Operator selection | Round-robin (E1, E2, M1, M2 every generation) | UCB1 bandit on `{Mutate, Crossover}` (persistent) |
| Cluster selection | Uniform random | Budgeted UCB1 (per-generation, warm-started) |
| Drift handling | None | Page-Hinkley reset per `(cluster, operator)` arm |
| Stopping criterion | Fixed `max_generations` or `max_sample_nums` | Fixed LLM-call budget `B` |
| Headline metric | HV / IGD at fixed quota | **AUBC** + HV / IGD / SWDI / CDI |
| Reward signal | None (population-level non-dominated filter only) | Final managed-population HV improvement by default, plus rank + diversity and invalid-code penalty |

## Ablations supported

The `BMABLLM` constructor accepts flags that disable each upgrade individually:

* `c_explore_cluster=0.0, gamma_budget=0.0` → pure exploitation cluster bandit.
* `ph_threshold=1e9` → effectively disable Page-Hinkley.
* `w_diversity=0.0` → no diversity term.
* `reward_mode='dense'` → legacy immediate-HVI quality signal inside the fixed
  method (`dense_reward` ablation).
* `reward_mode='hybrid'` → half immediate-HVI, half final managed-population HV
  quality signal (`hybrid_reward` ablation).
* `budget_annealing=False` → no remaining-budget exploration annealing
  (`no_budget_anneal` ablation).
* Set both `c_explore_op=0` and only one of `use_m*`/`use_e*` flags → single-operator ablation.

These match the ablation set described in [`IDEA.md`](IDEA.md) §4.3.

For the final-HV-focused follow-up experiments, see
[`experiments/HV_FINAL_EXPERIMENTS.md`](experiments/HV_FINAL_EXPERIMENTS.md).
