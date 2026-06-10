# Offspring-Generation Paths: MPaGE vs BMAB-LLM

A common reading of MPaGE is that it has *two parallel paths* for generating
offspring — one via the **Pareto-Front-Grid (PFG)** and one via **semantic
clustering** — and that BMAB-LLM replaces this duality with a single
bandit-driven pipeline. That framing is close, but it needs one important
refinement before it is accurate.

This document walks through what MPaGE *actually* does per generation, how
PFG and clustering relate, and how BMAB-LLM restructures the same building
blocks.

---

## Table of contents

1. [What MPaGE actually does per generation](#1-what-mpage-actually-does-per-generation)
2. [What BMAB-LLM does per generation](#2-what-bmab-llm-does-per-generation)
3. [Important nuance: PFG is not discarded](#3-important-nuance-pfg-is-not-discarded)
4. [Short version](#4-short-version)
5. [The two distinct roles of PFG](#5-the-two-distinct-roles-of-pfg)
6. [The actual mechanism: ε-based parent selection](#6-the-actual-mechanism--based-parent-selection)
7. [Three implications worth pinning down](#7-three-implications-worth-pinning-down)
8. [PFG-selected elites as a parent-weighting signal](#8-pfg-selected-elites-as-a-parent-weighting-signal)

---

## 1. What MPaGE actually does per generation

Inside the upstream loop ([_llm4ad/method/LLMPFG/eoh.py](../_llm4ad/method/LLMPFG/eoh.py)
function `_thread_do_evolutionary_operator`), **every** generation applies
the four operators in a fixed sequence: `E1 → E2 → M1 → M2`. But the
parent-selection pipeline before each operator differs:

```python
# --- Crossover branch (E1 / E2) -------------------------------------
indivs = self._population.selection(self._selection_num)         # PFG (default selection_num = 2)
if len(indivs) >= 3:
    prompt_cluster = EoHPrompt.get_prompt_cluster(...)
    group = self._cluster_sampler.get_thought(prompt_cluster)     # LLM clusters them
    indivs = self._population.selection_cluster(group, indivs)    # narrow to one cluster
prompt = EoHPrompt.get_prompt_e1(...)                             # or e2
self._sample_evaluate_register(prompt)

# --- Mutation branch (M1 / M2) --------------------------------------
indiv = self._population.selection(1)                             # PFG only, no clustering
prompt = EoHPrompt.get_prompt_m1(...)                             # or m2
self._sample_evaluate_register(prompt)
```

So the picture is **PFG → (optional) clustering → operator**, where:

* **PFG is *always* the first step.** It is what determines which
  heuristics are even *eligible* to be parents. Both crossover and mutation
  start with `Population.selection(...)`.
* **Semantic clustering is a second-stage *filter* on top of PFG, only for
  the crossover branch.** The mutation branch never clusters.
* The clustering step is also gated on `len(indivs) >= 3`. With the default
  `selection_num = 2` you only get clustering when an internal MPaGE call
  returns enough elites; otherwise it is silently skipped.

So the "MPaGE has two ways" intuition *does* map onto a real distinction in
the code — but the two ways are **mutation vs crossover sub-paths**, not
"PFG path vs clustering path." Clustering is always layered *on top of*
PFG, never in place of it.

---

## 2. What BMAB-LLM does per generation

In [bmab_llm.py:222-322](../bmab_llm.py#L222-L322):

```python
def _evolve_one_generation(self) -> None:
    # 1. PFG is still here — maintains the population structure
    elites      = self._population.selection(self._selection_num)
    full_elites = list(self._population.population)         # use the FULL population for clustering

    # 2. Cluster — unconditionally, every generation, on the whole population
    partition, cluster_quality = self._cluster_mgr.cluster(full_elites)
    op_priors = self._operator_bandit.softmax_probs(temperature=1.0)
    self._cluster_bandit.reset(n_clusters=len(partition),
                               cluster_quality=cluster_quality,
                               operator_priors=op_priors)

    # 3. Inner loop — bandit picks (operator, cluster) for every offspring
    while produced < target_offspring and not self._budget.is_exhausted():
        op  = self._operator_bandit.select()                # μ (mutate) or χ (crossover)
        arm = self._cluster_bandit.select(...,              # picks (cluster_idx, op)
                                          restrict_operator=op)
        prompt_builder, parents = self._make_prompt(op=op, cluster_idx=arm[0], ...)
        ok = self._sample_eval_register(prompt_builder, ...)
        # ... update both bandits with reward
```

Three structural differences from MPaGE:

| Aspect | MPaGE | BMAB-LLM |
|--------|-------|----------|
| **PFG** | Run *first*, narrows down elites to `selection_num` of them | Still run (the population is PFG-maintained; admission, non-dominated sorting, crowding distance are all done by `Population`), but the bandit picks parents from the **entire population**, not the PFG-narrowed subset |
| **Clustering** | Conditional — only before crossover, and only when 3+ elites returned | **Unconditional** — every generation, applied to the whole population |
| **Operator + cluster choice** | Round-robin E1 → E2 → M1 → M2; cluster only used for crossover; no choice within a cluster | **Two-layer bandit** picks both the operator μ/χ *and* which cluster to draw the parent from. **Mutation also runs through the cluster bandit** (intra-cluster parent), so mutation now gets the same cluster-aware treatment crossover used to have. |

The single most important conceptual change is that BMAB-LLM **unifies the
two sub-paths**. In MPaGE, mutation skips clustering entirely (a flat
PFG-only draw); in BMAB-LLM, every operator — including mutation — goes
through `cluster_bandit.select(restrict_operator=op)` so the bandit can
also decide *which cluster's style* to mutate. That makes mutation
cluster-aware too, which the original framework never was.

---

## 3. Important nuance: PFG is *not* discarded

A natural follow-up question: "If clustering replaces PFG, can we drop
PFG?" — and the answer is **no**, for two reasons.

1. **PFG still governs the population structure itself.** Every offspring
   is admitted to `self._population` via `register_function(func)`, which
   under the hood runs PFG's grid bookkeeping, non-dominated sort, and
   crowding distance ranking. Without PFG the population would degenerate
   to a flat archive that loses spread.
2. **PFG still controls *what* the clusterer sees.** `full_elites =
   list(self._population.population)` is exactly the PFG-maintained pool.
   The bandits pick from clusters whose members were admitted by PFG; if
   the PFG were uniform-random instead, the cluster bandit's priors would
   be dramatically worse.

So in BMAB-LLM the PFG quietly does background work, and the *visible*
offspring-generation pipeline simplifies to:

```
PFG-maintained population
        →  LLM clusters it
        →  (operator-bandit + cluster-bandit) picks (op, cluster)
        →  parents from that cluster
        →  E1/E2 or M1/M2
        →  offspring
```

vs MPaGE's

```
PFG.selection(2 elites)  →  if crossover and ≥3 elites: cluster + narrow  →  E1/E2 with parents
PFG.selection(1 elite)   →                              (no clustering)   →  M1/M2 with parent
```

---

## 4. Short version

* MPaGE has **two sub-paths per generation** — a crossover sub-path that
  *can* use clustering on top of PFG, and a mutation sub-path that uses
  PFG only. They are *sequential filters*, not parallel alternatives.
* BMAB-LLM **collapses the two sub-paths into one**: cluster the whole
  PFG-maintained population every generation, then let the two-layer
  bandit decide `(operator, cluster)` for every single offspring.
  Mutation is now cluster-aware too.
* PFG is still very much present in BMAB-LLM — it manages the population,
  just doesn't make selection decisions anymore. Those decisions are now
  the bandit's job.

---

## 5. The two distinct roles of PFG

A common confusion is whether BMAB-LLM "uses PFG to select an elite set for
clustering, like MPaGE." To answer that precisely, it helps to separate the
two different jobs the `Population` class actually performs. They are easy
to conflate but operationally distinct.

| PFG role | When it runs | What it does |
|----------|---------------|--------------|
| **Admission** | Every time `register_function(func)` is called on the population | Decides whether a newly evaluated heuristic enters the population, prunes dominated members, maintains the grid + crowding-distance ranking. **The population is *always* a curated elite set because of this rule.** |
| **Selection** | When you explicitly call `Population.selection(N)` | Pulls a narrower subset of elites *out* of the already-curated population — the "narrowing step". |

The first role is **structural**: it determines *what's in* the population.
The second role is **procedural**: it pulls a smaller subset *out of* the
population on demand.

### MPaGE uses both roles

```python
# crossover branch in _thread_do_evolutionary_operator
indivs = self._population.selection(self._selection_num)     # ← SELECTION role
if len(indivs) >= 3:
    prompt_cluster = EoHPrompt.get_prompt_cluster(..., indivs, ...)
    group   = self._cluster_sampler.get_thought(prompt_cluster)
    indivs  = self._population.selection_cluster(group, indivs)
```

MPaGE clusters `indivs` — the *narrowed* subset returned by `selection(...)`
— not the full population. The PFG narrowing is an explicit pre-clustering
step.

### BMAB-LLM uses only the admission role

[bmab_llm.py:227-240](../bmab_llm.py#L227-L240):

```python
try:
    elites = self._population.selection(self._selection_num)  # ← SELECTION role IS called
except Exception:
    return
# The PFG selection above returns ≤ 5 individuals. We expand to a wider
# elite set for clustering: take *all* current population members so we
# can cluster more of them, but bias the bandit's parent draws to the
# PFG selection.
full_elites = list(self._population.population)               # ← but we use the FULL population instead

partition, cluster_quality = self._cluster_mgr.cluster(full_elites)
```

Two things to notice:

1. `Population.selection(self._selection_num)` *is* called — the
   admission-vs-selection split still exists in the code.
2. But the result (`elites`) is **never used downstream**. The clusterer
   gets `full_elites = list(self._population.population)`, which is
   everyone the admission step has let in.

### Why this still counts as "clustering an elite set"

The population that goes into `_cluster_mgr.cluster(full_elites)` *is* an
elite set — because of the admission role. It is small (typically
≤ `pop_size` = 6), non-dominated, grid-spread, and crowding-distance-pruned.
It is exactly the same kind of "PFG-curated" pool that MPaGE narrows from.
The difference is:

```
MPaGE:    Population (PFG-admitted)  →  Population.selection(2)  →  Cluster the narrowed slice
                                          ▲
                                          │
                                     narrowing step

BMAB-LLM: Population (PFG-admitted)  →  Cluster the whole population
                                          ▲
                                          │
                                     no narrowing step
```

Both methods cluster PFG-curated heuristics. MPaGE clusters a narrow
subset; BMAB-LLM clusters all of them.

### Why BMAB-LLM skips the narrowing

1. **The MPaGE narrowing was conservative.** With default `selection_num = 2`,
   the gate `if len(indivs) >= 3` would *also* often skip clustering
   downstream. BMAB-LLM wants real clusters with several members each, so
   it bypasses narrowing and lets the population (size 5–6) be the cluster
   input.
2. **The bandit needs many arms to be useful.** A 2-element pool produces
   1 or 2 clusters; a 6-element pool produces 2–3 clusters; the bandit's
   cluster layer becomes informative around 3+ clusters. Sizing the
   cluster input up matters for the bandit to have something to do.

---

## 6. The actual mechanism: ε-based parent selection

The paper describes the PFG selection step as:

> "With probability **ε**, a group of elitism candidates is selected from a
> set of adjacent grid cells."

This is implemented exactly in
[population.py:143-176](../_llm4ad/method/LLMPFG/population.py#L143-L176):

```python
def parent_selection(pop, m, GK=4, sigma=0.01, epsilon=0.8):
    pop_         = deepcopy(pop)
    knee_point   = cal_knee_point(pop_)
    nadir_point  = cal_nadir_point(pop_)
    # ... normalise scores to [0, 1] in objective space ...
    PFG = Generation_PFG(pop_, GK, knee_point, nadir_point, sigma)   # build the grid

    if random.random() > epsilon:                                    # prob = 1 - ε  (= 0.2 default)
        # Rank-weighted random sampling over the whole population
        funcs   = [f for f in pop if f.score is not None]
        func    = sorted(funcs, key=lambda f: f.score[0])
        p       = [1 / (r + len(func)) for r in range(len(func))]
        p       = np.array(p); p = p / np.sum(p)
        parents = random.choices(pop, k=m, weights=p)
    else:                                                            # prob = ε      (= 0.8 default)
        # Pick a random grid axis and a random cell,
        # return the union of that cell and its neighbour
        i = random.randint(0, len(knee_point) - 1)
        j = random.randint(0, len(PFG[i]) - 2)
        while len(PFG[i][j]) == 0:
            i = random.randint(0, len(knee_point) - 1)
            j = random.randint(0, len(PFG[i]) - 2)
        parents = PFG[i][j] + PFG[i][j + 1]                          # ← adjacent grid cells

    if len(parents) > 5:
        parents = random.sample(parents, 5)
    return parents
```

Mapping the paper text to the code:

| Paper quote | Code line |
|-------------|-----------|
| "With probability **ε**…" | the `else:` branch (taken with probability `epsilon = 0.8`) |
| "…a group of elitism candidates is selected…" | `parents = PFG[i][j] + PFG[i][j + 1]` |
| "…from a set of adjacent grid cells." | the two adjacent cells `PFG[i][j]` and `PFG[i][j+1]` along axis `i` |
| (paper omits, but the code does) | rank-weighted random sampling in the complementary `1 − ε` branch |
| (paper omits, code does) | hard cap at 5 elites via `random.sample(parents, 5)` |

So the **"selection role" / "narrowing step"** of PFG is precisely this
`parent_selection` function, and the 80 %-of-the-time path is the
"adjacent grid cells" mechanism from the paper.

---

## 7. Three implications worth pinning down

### 7.1 `selection_num` is honoured only in the rank-weighted branch

In the rank-weighted branch (`1 − ε` ≈ 20 % of the time), the function
returns exactly `m = selection_num` parents. In the grid-cell branch
(80 %), it returns `len(PFG[i][j]) + len(PFG[i][j+1])` parents, capped at
5 — anywhere from 1 to 5.

So when MPaGE calls `self._population.selection(selection_num=2)`,
**80 % of the time it gets 1–5 elites from adjacent cells, ignoring the `2`**,
and 20 % of the time it gets exactly 2 rank-weighted picks.

### 7.2 This is why MPaGE's `if len(indivs) >= 3` clustering gate actually fires

An earlier reading might suggest "with default `selection_num=2` the gate
should always be False." That is wrong. Because of the 80 % grid-cell
branch returning up to 5 parents, the gate fires whenever (a) the
grid-cell branch is taken **and** (b) the chosen adjacent cells together
contain ≥ 3 elites. So MPaGE's clustering really does run in a
substantial fraction of generations — not "never" — and the paper's
claim that clustering is a core feature is consistent with the code.

### 7.3 What `pfg_elites` holds in BMAB-LLM

The PFG selection line in [bmab_llm.py](../bmab_llm.py):

```python
pfg_elites = self._population.selection(self._selection_num)
```

calls into exactly the `parent_selection` function above. So `pfg_elites`
contains:

* **80 % of the time**: a 1–5 element list drawn from two adjacent grid
  cells along a random objective axis — i.e. a structurally coherent
  slice of the front.
* **20 % of the time**: 2 rank-weighted picks (essentially "the best few
  heuristics overall").

---

## 8. PFG-selected elites as a parent-weighting signal

The comment in [bmab_llm.py:231-234](../bmab_llm.py#L231-L234) says:

> "bias the bandit's parent draws to the PFG selection."

That suggests the original design *intended* to feed `elites` into the
bandit — so the bandit would pick within the adjacent-cell neighbourhood
rather than treating every population member equally. The current fixed
implementation keeps `full_elites = list(self._population.population)` for
clustering, but uses `pfg_elites` inside `_weighted_parent_choice`: candidates
that appear in the PFG-selected slice receive higher sampling weight.

### Why this matters for the thesis

There is still an honest design question lurking here: **should BMAB-LLM
cluster the PFG-narrowing slice itself, rather than clustering the whole
population and only using PFG as a parent-weighting signal?** Arguments either
way:

**For restoring it**
* Matches MPaGE's intent more faithfully (the bandit picks within an
  adjacent-cell neighbourhood).
* The neighbourhood is more semantically homogeneous, so clusters within
  it may be more meaningful.
* Makes the comparison to MPaGE more apples-to-apples — both methods
  would be working off the same kind of slice.

**Against**
* With `pop_size = 6`, the full population is already small and already
  PFG-curated; clustering it gives the bandit more arms, which is what
  the bandit needs to be informative.
* A 2–5 element slice often produces only 1 cluster, which makes the
  cluster bandit degenerate to "uniform random within the only cluster"
  for that generation.

For the thesis comparison this remains a possible **design difference**:
BMAB-LLM gets more clusters to choose from than the equivalent MPaGE slice
would offer, although its final parent draw is now biased back toward PFG
elites. If you want to control for that exactly, add an explicit
`--cluster_pool {full,narrow}` ablation and re-run the headline experiments.

### Practical recommendation

There are three reasonable options:

1. **Leave as-is** and document the difference explicitly: "BMAB-LLM
   clusters the whole PFG-curated population while MPaGE clusters an
   adjacent-cell slice." Defensible because the bandit needs cluster
   counts to be informative.
2. **Add a CLI flag** (e.g. `--cluster_pool {full, narrow}`) and ablate
   both. Cleanest experimentally; adds a small amount of code complexity.
3. **Use the current fixed compromise**: cluster the whole PFG-curated
   population for enough cluster arms, but weight parent draws toward the
   PFG-selected slice.

Option 2 is the cleanest if the thesis defence is going to emphasise
"BMAB-LLM is a drop-in improvement over MPaGE" — the burden of proof for
"drop-in" requires checking whether the cluster input itself matters.
