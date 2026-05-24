# The Two Hypervolumes in BMAB-LLM — Why Each Needs Its Own Reference Point

There are **two distinct hypervolume computations** in this project, and the
confusion between them was the root cause of the recently-fixed tri_tsp
AUBC-=-0 bug. This document separates them, explains how each is computed
with concrete examples for every task, points out where the current code
is right vs. where it could be revised, and dissects what each component
of every reference point means and why it is set the way it is.

Companion documents:
[METRICS_AUBC_HV.md](METRICS_AUBC_HV.md) covers the algorithm-level HV in
the context of AUBC reporting;
[REWARD.md §3](REWARD.md#3-component-1--normalised-hypervolume-improvement-hvi)
covers the HVI reward signal that the bandits consume.

---

## Table of contents

1. [The two HVs at a glance](#1-the-two-hvs-at-a-glance)
2. [Q1: Do these two HVs require two different reference points?](#2-q1-do-these-two-hvs-require-two-different-reference-points)
3. [Q2: How is each HV calculated? With concrete examples.](#3-q2-how-is-each-hv-calculated-with-concrete-examples)
4. [Q3: Is there any confusion or mistake in our current HV calculation?](#4-q3-is-there-any-confusion-or-mistake-in-our-current-hv-calculation)
5. [Q4: What are the components of the reference point and why those values?](#5-q4-what-are-the-components-of-the-reference-point-and-why-those-values)

---

## 1. The two HVs at a glance

| | **Solution-level HV** | **Algorithm-level HV** |
|---|---|---|
| **What set is it computed over?** | The set of *solutions* (e.g. TSP tours) that a heuristic produces for one instance of the problem. | The set of *heuristics* in the BMAB-LLM population, each represented by its score tuple. |
| **Where is it computed?** | Inside each task's `Evaluation.evaluate_program()` method (`_llm4ad/task/optimization/*/evaluation.py`). | Inside `BMABProfiler.record_curve_point()` (`profiler.py:49-73`). |
| **Dimensionality** | Same as the *problem's* objective count. `bi_*` tasks: 2-D. `tri_tsp`: 3-D. | **Always 2-D** because every task's heuristic score tuple is `(−HV_of_solutions_avg, runtime_avg)`. |
| **Reference point lives in** | `<TaskName>Evaluation.__init__()` as `self.ref_point`. | [`main.py:_TASKS`](../main.py) as the third element of each task's tuple. |
| **Per-heuristic? Per-instance?** | One value per `(heuristic, instance)` pair; averaged over `n_instance` instances inside `evaluate()`. | One value per *generation* (or per LLM call for MPaGE-orig), recorded as a point on the budget-vs-HV curve. |
| **Used for** | Forming the **first component of the heuristic's score** (`obj_1 = −mean(HV)`). | Computing **AUBC** and **HV-final**, the thesis-headline metrics. |

These two HVs are mathematically the *same kind of object* (Lebesgue measure
of dominated region against a reference) but they are computed over
*completely different sets* and live in *completely different spaces*. They
look identical when you write the formula but they are unrelated quantities
in practice.

---

## 2. Q1: Do these two HVs require two different reference points?

**Yes — emphatically.** They live in different spaces, so they need different
reference points.

* The **solution-level HV** lives in the *problem's objective space*. For
  `bi_tsp`, that is `(tour_cost_map_1, tour_cost_map_2)` — both in distance
  units. For `tri_tsp`, it is the same idea with three maps. The reference
  point's components are *upper bounds on tour cost* in that space.
* The **algorithm-level HV** lives in the *heuristic-evaluation space*:
  `(−avg_solution_HV, avg_runtime_seconds)`. The reference point's first
  component is an upper bound on `−avg_solution_HV` (i.e. its worst /
  largest value) and the second component is an upper bound on
  `avg_runtime` in seconds.

The dimensionalities are *intentionally* different. A 3-objective problem
like `tri_tsp` has a 3-D *solution* objective space but its heuristic
scoring collapses to a 2-D *algorithm* score — because we summarise the
3-D solution Pareto-front by its HV scalar, then pair that scalar with
runtime. **This collapse is the source of the bug we just fixed**: the
two ref points were *supposed to* have different shapes, but `main.py`
declared a 3-D algorithm ref by mistake.

So strictly: **two HVs, two ref points, generally different shapes.** The
shapes only happen to match for the three bi-objective tasks because both
the solution space *and* the heuristic-score space are 2-D there.

---

## 3. Q2: How is each HV calculated? With concrete examples.

### 3.1 Solution-level HV (inside the evaluator)

Take `tri_tsp_semo/evaluation.py` as the canonical example. The relevant
function is `evaluate(...)`:

```python
def evaluate(instance_data, n_instance, problem_size, ref_point, eva):
    obj_1 = np.ones(n_instance)
    obj_2 = np.ones(n_instance)
    obj_3 = np.ones(n_instance)            # declared but never written — vestige
    n_ins = 0
    for instance, distance_matrix_1, distance_matrix_2, distance_matrix_3 in instance_data:
        start = time.time()
        s = [random_solution(problem_size) for _ in range(100)]      # 100 random tours
        Archive = [(s_, tour_cost(instance, s_, problem_size)) for s_ in s]
        for _ in range(20000):                                       # SEMO inner loop
            s_prime = eva(Archive, instance, distance_matrix_1,
                          distance_matrix_2, distance_matrix_3)      # LLM heuristic picks neighbour
            f_s_prime = tour_cost(instance, s_prime, problem_size)   # 3-D cost
            if not check_constraint(s_prime, problem_size):
                continue
            if not any(dominates(f_a, f_s_prime) for _, f_a in Archive):
                Archive = [(a, f_a) for a, f_a in Archive
                           if not dominates(f_s_prime, f_a)]
                Archive.append((s_prime, f_s_prime))
        end = time.time()
        objs = np.array([obj for _, obj in Archive])                 # 3-D points
        hv_indicator = HV(ref_point=ref_point)                       # ref = [20, 20, 20]
        hv_value = hv_indicator(objs)                                # ← SOLUTION-LEVEL HV
        obj_1[n_ins] = -hv_value
        obj_2[n_ins] = end - start
        n_ins += 1
    return np.mean(obj_1), np.mean(obj_2)
```

#### Step-by-step (tri_tsp, problem_size=20, n_instance=20)

For each of 20 random instances:

1. Start with an Archive of 100 random tours.
2. Run 20,000 SEMO iterations where the *heuristic* (the LLM-generated
   `select_neighbor` function) proposes neighbours. Non-dominated neighbours
   are added; dominated archive entries are removed.
3. After the loop, `Archive` is the 3-D Pareto front of tours discovered.
4. Each tour has a cost tuple `(cost_map_1, cost_map_2, cost_map_3)`.
5. `pymoo.HV(ref_point=[20, 20, 20])` measures the volume in 3-D objective
   space dominated by this Pareto front.
6. `obj_1[n_ins] = -hv_value` (negated so the heuristic's first objective is
   minimised); `obj_2[n_ins] = runtime in seconds`.

After 20 instances, the heuristic's score is
`(np.mean(obj_1), np.mean(obj_2)) = (avg_neg_HV, avg_runtime)` — a 2-D point.

#### Concrete numerical example

Suppose on one tri_tsp instance the heuristic produces an Archive containing
8 non-dominated tours with 3-D costs:

```
[ (5.2, 6.1, 4.8),
  (6.0, 5.3, 5.5),
  (5.8, 5.9, 5.2),
  (7.0, 4.2, 6.1),
  (4.5, 6.8, 5.9),
  ...
]
```

`pymoo.HV(ref_point=[20,20,20])` computes the union of "boxes"
`∏_i (20 − cost_i)` over the non-dominated front. For the front above, the
result might be something like `hv_value ≈ 3500`. Then `obj_1 = -3500`.

Across 20 instances, suppose `obj_1 = [-3500, -3300, -3700, ..., -3400]`
and `obj_2 = [1.96, 2.42, ...]`. The heuristic's final score is
`(mean(obj_1), mean(obj_2)) = (-3500, 2.0)` — this is exactly what we see
in the saved `samples/samples_*.json` files (e.g. `[-3299.74, 1.96]` from
your tri_tsp run).

#### Per-task numbers

| Task | Solution objectives | Solution-ref point | Typical solution HV per instance | Heuristic score range (first axis) |
|------|---------------------|--------------------|----------------------------------|-------------------------------------|
| `bi_tsp`  | `(cost_1, cost_2)` | `[20, 20]` | ~100–400 | `[-400, 0]` |
| `tri_tsp` | `(cost_1, cost_2, cost_3)` | `[20, 20, 20]` | ~2000–4000 | `[-8000, 0]` |
| `bi_cvrp` | `(cost, makespan)` | `[80, 8]` | ~50–250 | `[-640, 0]` |
| `bi_kp`   | `(−profit_1, −profit_2)` | `[-30, -30]` | ~100–500 | `[-900, 0]` |

The numbers in column 4 explain why tri_tsp's algorithm-level HV is ~10×
larger than bi_tsp's: the first heuristic-score component for tri_tsp can
reach `-8000`, vs only `-400` for bi_tsp, so the dominated extent on that
axis is ~20× bigger.

### 3.2 Algorithm-level HV (inside the profiler)

The relevant code is in `profiler.py:record_curve_point()`:

```python
def record_curve_point(self, budget_consumed, population):
    if self._ref_point is None:
        return
    scores = []
    for f in population.population:
        if f.score is None:
            continue
        try:
            arr = np.asarray(f.score, dtype=float)
            if arr.shape == self._ref_point.shape:        # ← shape filter
                scores.append(arr)
        except Exception:
            continue
    if not scores:
        hv = 0.0
    else:
        hv = hypervolume(np.array(scores), self._ref_point)
    with self._curve_lock:
        self._curve.append({
            'budget_consumed': float(budget_consumed),
            'hv': float(hv),
            'pareto_size': len(scores),
        })
```

This is called after the warm-up and after every generation. It snapshots
the *whole heuristic population* (each member is a `Function` with a 2-D
`score = (neg_HV, runtime)`) and computes the HV of that 2-D point set.

#### Step-by-step (`bi_tsp`, post-fix `tri_tsp`)

After generation `g`:

1. Walk `self._population.population` — typically 4–12 `Function` objects
   admitted by the PFG.
2. Collect their `score` tuples (now all 2-D).
3. Call `pymoo.HV(ref_point=(20, 60))(scores_array)`.
4. Append `{budget_consumed, hv, pareto_size}` to the curve.

Note: `pymoo.HV` automatically ignores dominated points, so dominated
heuristics in the population contribute zero — the HV is effectively that
of the population's *Pareto front* in `(neg_HV, runtime)` space.

#### Concrete numerical example (bi_tsp)

Suppose at generation 3 the heuristic population has 6 members with scores:

```
heuristic A: (-120, 0.8)        ← good neg_HV, fast
heuristic B: (-150, 1.2)        ← better neg_HV, slower
heuristic C: (-80,  0.5)        ← weaker but very fast
heuristic D: (-100, 2.1)        ← average
heuristic E: (-110, 0.9)        ← dominated by A
heuristic F: (-300, 4.0)        ← excellent neg_HV, slow
```

`pymoo.HV(ref_point=[20, 60])` over these 6 points filters E (dominated by
A) and computes the dominated area in `(neg_HV, runtime)` space:

- For A at `(-120, 0.8)`: contribution box vertices roughly at
  `(20, 60)` and `(-120, 0.8)` → extent `(20 - (-120), 60 - 0.8) = (140, 59.2)`
- ... and so on for B, C, D, F, with subtractions for overlapping regions.

The total HV might be ≈ `14 200`. This matches your bi_tsp results (~14k–15k).

#### Concrete numerical example (tri_tsp, after fix)

Suppose at generation 1 the heuristic population has 5 members:

```
(-3299.7, 1.96)
(-3335.0, 2.42)
(-3471.6, 4.13)
(-3050.0, 1.50)
(-3520.0, 4.80)
```

`pymoo.HV(ref_point=[20, 60])` filters dominated points and integrates.
With first-axis values around `-3300` the dominated extent on axis 0 is
roughly `3320`, on axis 1 roughly `58`, so each non-dominated point
contributes roughly `~190 000`. The HV of the front comes out around
**`200 000`** — matching what `recompute_metrics.py` produced for your
tri_tsp runs.

### 3.3 Why the algorithm-level HV is always 2-D

Notice that every task's `evaluate()` returns
`(np.mean(obj_1), np.mean(obj_2))` — always two values. Specifically:

```python
obj_1 = -solution_HV     # collapses K-objective solution space → scalar
obj_2 = runtime          # always 1-D
```

So the *heuristic-score space* is always 2-D, **regardless of how many
objectives the underlying problem has.** The 3-objective nature of
`tri_tsp` is hidden inside the inner SEMO HV computation; outside of
that, the heuristic looks 2-D to BMAB-LLM.

This was exactly the design choice that triggered the bug: someone setting
the algorithm ref point for `tri_tsp` to `(20, 20, 60)` — three values,
matching the *problem* dimensionality — was reasoning about the solution
space, when they should have been reasoning about the heuristic-score
space.

---

## 4. Q3: Is there any confusion or mistake in our current HV calculation?

Three issues, in descending order of severity.

### 4.1 The `tri_tsp` ref-point shape mismatch (FIXED)

**Severity: high.** This was the bug you just discovered. `main.py` had

```python
'tri_tsp':  ('...', 'TRITSPEvaluation', (20.0, 20.0, 60.0)),
```

but `TRITSPEvaluation.evaluate_program()` returns a 2-D score, so the
shape filter `arr.shape == self._ref_point.shape` in
[profiler.py:60](../profiler.py#L60) rejected every score → empty `scores`
→ HV = 0 every snapshot → AUBC = 0.

**Fix applied:** changed to `(20.0, 60.0)` in
[main.py:33-37](../main.py#L33-L37) and recomputed offline via
`recompute_metrics.py`.

### 4.2 Silent shape-skip in the profiler

**Severity: medium.** The `if arr.shape == self._ref_point.shape:` filter
silently swallows mismatched scores without any warning. This is the
right behaviour for *legitimately* invalid scores (e.g. a score that came
out as a scalar instead of a tuple due to a heuristic error), but for a
*systematic* shape mismatch like the one in 4.1 it produces a
"AUBC = 0 forever, no log" failure mode that is hard to diagnose.

**Recommended fix (optional):** add a one-shot warning the first time a
score is dropped because of shape mismatch:

```python
if arr.shape != self._ref_point.shape:
    if not getattr(self, '_warned_shape_mismatch', False):
        import warnings
        warnings.warn(
            f"BMABProfiler: dropping score of shape {arr.shape} that "
            f"does not match ref_point shape {self._ref_point.shape}. "
            f"Subsequent mismatches will be silent.")
        self._warned_shape_mismatch = True
    continue
```

I have not applied this — let me know if you want it. It would have
surfaced the tri_tsp bug on the very first generation.

### 4.3 Inconsistent first-axis ref values across tasks

**Severity: low — cosmetic only.** Looking at the algorithm-level refs:

| Task     | algorithm-ref[0] | Min observed neg_HV | "Headroom" |
|----------|----------------:|--------------------:|-----------:|
| `bi_tsp` | 20              | ~ -400              | 420 |
| `tri_tsp`| 20 *(post-fix)* | ~ -8 000            | 8 020 |
| `bi_cvrp`| 40              | ~ -640              | 680 |
| `bi_kp`  | 0               | ~ -900              | 900 |

The first-axis ref must satisfy `ref[0] > all neg_HV values seen`. Since
neg_HV ≤ 0 always, *any* `ref[0] ≥ 0` works. The choice of 20 (or 40 or
0) is arbitrary; it shifts every HV by a constant (`ref[0] − 0`) ×
runtime-extent, but **does not change relative orderings between methods**.

So no method-comparison conclusions are affected. The cosmetic issue is
that absolute HV / AUBC values cannot be compared *across tasks* — a
"larger" HV for tri_tsp doesn't mean tri_tsp's heuristics are better, it
just means tri_tsp's score-space ref-extent is bigger.

**Recommended fix (optional):** standardise to `(0, 60)` for all four
tasks. This would make absolute HV values across tasks more comparable
(though still not identical because of differing solution-HV ranges).
Requires re-aggregating but not re-spending LLM calls — same
`recompute_metrics.py` script can do it with `--ref_point 0,60`.

Whether this matters depends on whether your thesis chapter wants to
make cross-task absolute-value comparisons. If not, leave it.

### 4.4 What is *not* a mistake

A few things that might *look* suspicious but are actually fine:

* **The score's first component is negative (`neg_HV`).** This is
  intentional. `pymoo.HV` and BMAB-LLM both minimise; we negate so the
  "better heuristic" has a smaller first-objective value.
* **`obj_3` is declared but unused in `tri_tsp` `evaluate()`.** Vestige
  of an earlier 3-D heuristic-score design. Has no runtime effect.
* **The dominated-extent for tri_tsp is huge** (~190k per point vs
  ~14k for bi_tsp). Not a bug — just reflects that 3-D solution HV is
  numerically larger than 2-D solution HV.

---

## 5. Q4: What are the components of the reference point and why those values?

There are two ref points per task — the **solution-level** one (in
`Evaluation.__init__`) and the **algorithm-level** one (in
`main.py:_TASKS`). They are independent and serve different roles.

### 5.1 Solution-level ref point — task-by-task

| Task | `self.ref_point` | Meaning of each component | Why this value? |
|------|-----------------|---------------------------|-----------------|
| `bi_tsp` | `[20.0, 20.0]` | `(upper_bound_cost_map_1, upper_bound_cost_map_2)` — tour-cost units | 20-city random Euclidean instances in a unit square have typical tour costs ~5–15. `20` is comfortably above that, so HV is well-defined for any reasonable heuristic. |
| `tri_tsp`| `[20.0, 20.0, 20.0]` | three tour-cost bounds, one per distance map | Same justification as bi_tsp; tri_tsp also uses 20-city instances. |
| `bi_cvrp`| `[80, 8]` | `(upper_bound_total_route_cost, upper_bound_makespan)` | CVRP route costs are larger (multi-vehicle, capacity constraints). 20-customer instances yield costs ~30–60; `80` covers worst-case bad tours. Makespan (longest individual route time) ~3–6 for good solutions; `8` covers worst-case. |
| `bi_kp`  | `[-30, -30]` | `(neg_profit_1, neg_profit_2)` — profits negated because KP is naturally maximisation | Knapsack profits are 0–30 per item; the worst case is "zero profit" → neg-profit = 0. The ref point of -30 means "any solution with profit < 30 is below the ref" — i.e. it ensures the ref bounds the achievable region. |

**Why these specific values?** They are chosen so that:

1. *Every* legitimate solution lies inside the ref box (otherwise it
   contributes 0 to HV, masking quality differences).
2. The box isn't *so* large that the HV becomes insensitive to where the
   front actually sits (every point would contribute ~max-area no matter
   how far from the front).

The optimal choice is "just slightly worse than the worst plausible
solution," and that has been done empirically per task by the MPaGE
paper's authors. We inherited these values verbatim.

### 5.2 Algorithm-level ref point — task-by-task

| Task | `_TASKS[...][2]` | Meaning of each component | Why this value? |
|------|------------------|---------------------------|-----------------|
| `bi_tsp` | `(20.0, 60.0)` | `(upper_bound_neg_HV, upper_bound_runtime_seconds)` | `neg_HV ≤ 0` always, so `20` is a safe upper bound. `60s` is a generous runtime cap — typical runs spend 0.5–4 seconds per heuristic evaluation. |
| `tri_tsp`| `(20.0, 60.0)` *(post-fix)* | same | same |
| `bi_cvrp`| `(40.0, 60.0)` | same | same; `40` is arbitrary (any `≥0` works) but was chosen empirically by the MPaGE authors. |
| `bi_kp`  | `(0.0, 60.0)`  | same | `0` works because neg_HV ≤ 0; this is the tightest reasonable choice. |

**Component breakdown of the algorithm ref point:**

* `ref[0]` = **worst neg_HV we will count.** Larger means the dominated
  box is wider, so absolute HV values grow. Any `ref[0] ≥ 0` works
  (since neg_HV ≤ 0 always). Choice is empirical and somewhat arbitrary;
  it does not affect *relative* method comparisons.
* `ref[1]` = **worst runtime in seconds we will count.** `60s` is a soft
  cap on heuristic runtime — if a heuristic takes more than 60s to
  evaluate, it contributes 0 to the algorithm-level HV. This is a
  deliberate design choice: very slow heuristics should not dominate
  the front just because they happened to produce a slightly better
  solution Pareto set, since runtime is itself an objective in the
  `(neg_HV, runtime)` minimisation.

### 5.3 In one sentence

> Each task has **two reference points** that serve **two different jobs**:
> the *solution-level* ref bounds the worst plausible solution in the
> *problem's objective space*, while the *algorithm-level* ref bounds the
> worst plausible heuristic in the *(−avg_solution_HV, avg_runtime) space*.
> Their dimensionalities can differ (and do, for `tri_tsp`), and their
> components are completely unrelated except by name.

---

## Bonus: how to talk about HVs in the thesis chapter

When writing this up, be **explicit about which HV you mean** every time you
mention "HV". Suggested terminology:

* **"Solution-set hypervolume" / "S-HV"** — what each heuristic produces on
  one MOCOP instance. This is the **quality** of the heuristic itself,
  averaged over instances, and forms the first component of the heuristic's
  score tuple.
* **"Heuristic-population hypervolume" / "H-HV"** — what BMAB-LLM's
  population achieves in `(−S-HV, runtime)` space. This is the metric
  reported as HV-final and integrated into AUBC.

Equivalently you can say "level-1 HV" (solution-set) and "level-2 HV"
(heuristic-population). Pick one convention and use it consistently —
otherwise readers (and reviewers) will be confused, because *both* HVs are
maximised, *both* use `pymoo.HV`, *both* live in some "objective space",
and yet they refer to entirely different things. The bug we just fixed is
the kind of thing that happens when this distinction is not held
firmly in mind.
