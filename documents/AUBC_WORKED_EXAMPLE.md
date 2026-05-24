# AUBC — Step-by-Step Worked Example

This document explains the **Area-Under-Budget-Curve (AUBC)** metric in the
exact way it is computed inside this project, then walks through a complete
numerical example end-to-end, then contrasts two hypothetical runs with
identical HV-final but different AUBC to show *why* the metric exists.

If you've already read [METRICS_AUBC_HV.md](METRICS_AUBC_HV.md), this
document is the **slower, more pedagogical version** with all arithmetic
expanded.

---

## Table of contents

1. [What AUBC actually measures, in one paragraph](#1-what-aubc-actually-measures-in-one-paragraph)
2. [The formula, twice](#2-the-formula-twice)
3. [The two structural choices: the (0, 0) anchor and the B-extension](#3-the-two-structural-choices-the-0-0-anchor-and-the-b-extension)
4. [Worked example #1: a single run, all arithmetic shown](#4-worked-example-1-a-single-run-all-arithmetic-shown)
5. [Worked example #2: same HV-final, different AUBC](#5-worked-example-2-same-hv-final-different-aubc)
6. [Plugging in your real numbers](#6-plugging-in-your-real-numbers)
7. [How to sanity-check AUBC by eye](#7-how-to-sanity-check-aubc-by-eye)
8. [Common pitfalls](#8-common-pitfalls)

---

## 1. What AUBC actually measures, in one paragraph

Every run produces a sequence of "snapshots" — at each generation we record
*(budget consumed so far, hypervolume of the current population)*. Plotted,
this is a step-wise rising curve from `(0, 0)` toward `(B, HV-final)`. **AUBC
is the area underneath that curve, divided by `B`.** Because it's an
average of HV across the entire budget axis, it answers "if you stopped me
at a random point during the run, what HV would you typically see?" — a
quantity HV-final alone cannot tell you, because HV-final reports only the
endpoint of the trajectory.

A method that finds high HV *early* in the budget has a high AUBC. A method
that catches up to the same HV-final but only at the very end has a low
AUBC. Two methods with identical HV-final and identical AUBC are
indistinguishable on this metric pair; two methods with identical HV-final
but different AUBC differ in **how quickly they got there**.

---

## 2. The formula, twice

### 2.1 Continuous form (the definition)

```
              1   ⌠ B
   AUBC  =   ─── │   HV(b) db
              B  ⌡ 0
```

where `HV(b)` is the (unknown but real-valued) HV-of-population at the
moment when the cumulative LLM-call budget consumed equals `b`. This is the
**mean HV over the budget axis** in the closed interval `[0, B]`.

### 2.2 Discrete form (what the code computes)

In practice we have HV measured only at the generation boundaries — say `n`
data points `(b_1, h_1), (b_2, h_2), …, (b_n, h_n)`, sorted with
`0 < b_1 < b_2 < … < b_n ≤ B`. We approximate the integral by the
**trapezoidal rule**:

```
              1   ┌                                                       ┐
   AUBC  ≈   ─── │ Σ  ½·(h_i + h_{i-1}) · (b_i − b_{i-1})                 │
              B  └  i                                                     ┘
```

with two extra "padding" points:

* **A leading anchor** `(0, 0)` (no budget spent → no dominated volume).
* **A trailing extension** `(B, h_n)` (if the last recorded snapshot is
  before `B`, treat HV as flat from there to `B`).

In code, [profiler.py:96-112](../profiler.py#L96-L112):

```python
def aubc(self, total_budget: float) -> float:
    if not self._curve:
        return 0.0
    pts = sorted(self._curve, key=lambda x: x['budget_consumed'])
    xs = [0.0] + [p['budget_consumed'] for p in pts]
    ys = [0.0] + [p['hv']             for p in pts]
    if xs[-1] < total_budget:
        xs.append(total_budget)
        ys.append(ys[-1])
    area = 0.0
    for i in range(1, len(xs)):
        area += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    return area / max(total_budget, 1e-9)
```

That's the entire definition.

---

## 3. The two structural choices: the (0, 0) anchor and the B-extension

### 3.1 Why `(0, 0)`?

Before any LLM call has been made, no heuristic exists, no population
exists, no Pareto front exists, so **the HV is 0**. Anchoring at `(0, 0)`
makes the trapezoidal area on the first segment a triangle, not a rectangle
— which correctly represents "we built up from zero." Skipping this anchor
would give a constant offset that favours methods that record their first
curve point early.

### 3.2 Why extend flat to `B`?

If the run completes its last generation at `b = 17` of a `B = 20` budget,
nothing has changed by the time `b` actually reaches 20 (the population is
fixed, no more LLM calls). So `HV` stays constant. The extension says: from
the last snapshot to the budget limit, treat HV as flat.

This is also a **defensive bias**: if a run crashes at `b = 12` and never
reaches `B = 20`, we still extend the last `HV` flat to `B`. This may
slightly **over**estimate AUBC for crashed runs (it pretends "nothing more
was learned but no time was wasted either"), which is fine because both
methods being compared are treated identically by this convention.

---

## 4. Worked example #1: a single run, all arithmetic shown

### 4.1 The setup

Imagine running BMAB-LLM on `bi_tsp` with:

* total budget `B = 20` (so 20 LLM calls allowed),
* `pop_size = 4`,
* after warm-up the population has 4 valid heuristics,
* the run completes two evolution generations before budget runs out.

Three `record_curve_point()` calls fire:

| Generation | After... | `budget_consumed` | `hv` |
|------------|----------|------------------:|------:|
| 0          | warm-up  | 4                 | 10 000 |
| 1          | gen 1    | 10                | 14 000 |
| 2          | gen 2    | 17                | 15 000 |

The last snapshot is at `b = 17`, **not** at `b = 20`. (The run stopped
because `_budget.is_exhausted()` became true before another generation
could complete, or because gen 2 finished and the next iteration's
`can_afford` check failed.)

### 4.2 Apply the padding

Take the three recorded points and add the (0, 0) anchor and the
flat-extension to `B = 20`:

```
xs = [0,      4,      10,     17,     20]
ys = [0,  10000,  14000,  15000,  15000]
```

Five anchor points → **four trapezoidal segments** to integrate over.

### 4.3 Compute each segment's area

Apply `0.5 · (h_left + h_right) · (b_right − b_left)`:

**Segment 1**: from `(0, 0)` to `(4, 10 000)`
```
0.5 × (0 + 10 000) × (4 − 0)
= 0.5 × 10 000 × 4
= 20 000
```
This is geometrically a **triangle** (left edge has height 0). It
represents the work done during the warm-up.

**Segment 2**: from `(4, 10 000)` to `(10, 14 000)`
```
0.5 × (10 000 + 14 000) × (10 − 4)
= 0.5 × 24 000 × 6
= 72 000
```
A **trapezoid**: 4 budget units wide of warm-up at HV=10000 has expanded
into 6 budget units wide of generation 1, with HV rising from 10 000 to
14 000.

**Segment 3**: from `(10, 14 000)` to `(17, 15 000)`
```
0.5 × (14 000 + 15 000) × (17 − 10)
= 0.5 × 29 000 × 7
= 101 500
```
Another trapezoid, this time covering generation 2.

**Segment 4**: from `(17, 15 000)` to `(20, 15 000)`
```
0.5 × (15 000 + 15 000) × (20 − 17)
= 0.5 × 30 000 × 3
= 45 000
```
A **rectangle** (constant height). This is the flat extension — we ran out
of budget but the curve is treated as flat for the remaining 3 budget units.

### 4.4 Sum the segments and divide by B

```
   total area  =  20 000 + 72 000 + 101 500 + 45 000  =  238 500

   AUBC        =  238 500 / 20                         =  11 925
```

So for this run: **AUBC = 11 925**, HV-final = 15 000.

### 4.5 Visual representation

```
   HV
   ▲
   │
   │
15000 ────────────●─────────────●
   │           ░░░│░░░░░░░░░░░░░│       ← Segment 4: rectangle, area 45 000
   │         ░░░░░│░░░░░░░░░░░░░│
14000 ────●░░░░░░░│░░░░░░░░░░░░░│       ← Segment 3: trapezoid, area 101 500
   │    ░░│░░░░░░░│░░░░░░░░░░░░░│
   │   ░░░│░░░░░░░│░░░░░░░░░░░░░│       ← Segment 2: trapezoid, area 72 000
   │  ░░░░│░░░░░░░│░░░░░░░░░░░░░│
10000 ●░░░░│░░░░░░░│░░░░░░░░░░░░░│
   │░░░░░░│░░░░░░░│░░░░░░░░░░░░░│      ← Segment 1: triangle, area 20 000
   │░░░░░░│░░░░░░░│░░░░░░░░░░░░░│
   │░░░░░░│░░░░░░░│░░░░░░░░░░░░░│
   0──────────────────────────────────► budget
   0      4      10              17    20
```

The total shaded area is **238 500**. Divided by the budget axis length 20,
we get the **mean HV = 11 925**, which is the AUBC.

### 4.6 Interpretation

The mean HV across the budget axis was about 12 000 of the 15 000 we
ended with — i.e. on average across the run the population was about 80%
of the way to its final quality. The ratio `AUBC / HV-final = 11 925 / 15 000
= 0.795` is a useful "trajectory efficiency" number; closer to 1 means the
HV ramped up fast and stayed high.

---

## 5. Worked example #2: same HV-final, different AUBC

To see why AUBC is informative beyond HV-final, here are two runs that
*both* reach `HV-final = 15 000` at `B = 20` but follow very different
trajectories.

### 5.1 Method A — early-ramp ("fast learner")

Curve points:

| `budget_consumed` | `hv` |
|------------------:|-----:|
|  4                | 14 000 |
| 10                | 15 000 |
| 17                | 15 000 |

After padding:
```
xs = [0,     4,     10,     17,     20]
ys = [0, 14000, 15000,  15000,  15000]
```

Segments:
* `(0, 0) → (4, 14000)`: `0.5 × 14 000 × 4 = 28 000`
* `(4, 14000) → (10, 15000)`: `0.5 × 29 000 × 6 = 87 000`
* `(10, 15000) → (17, 15000)`: `0.5 × 30 000 × 7 = 105 000`
* `(17, 15000) → (20, 15000)`: `0.5 × 30 000 × 3 = 45 000`

Total = `28 000 + 87 000 + 105 000 + 45 000 = 265 000`.
**AUBC_A = 265 000 / 20 = 13 250**.

### 5.2 Method B — late-ramp ("slow learner")

Curve points:

| `budget_consumed` | `hv` |
|------------------:|-----:|
|  4                |  5 000 |
| 10                | 10 000 |
| 17                | 15 000 |

After padding:
```
xs = [0,    4,     10,    17,     20]
ys = [0, 5000, 10000, 15000,  15000]
```

Segments:
* `(0, 0) → (4, 5000)`: `0.5 × 5 000 × 4 = 10 000`
* `(4, 5000) → (10, 10000)`: `0.5 × 15 000 × 6 = 45 000`
* `(10, 10000) → (17, 15000)`: `0.5 × 25 000 × 7 = 87 500`
* `(17, 15000) → (20, 15000)`: `0.5 × 30 000 × 3 = 45 000`

Total = `10 000 + 45 000 + 87 500 + 45 000 = 187 500`.
**AUBC_B = 187 500 / 20 = 9 375**.

### 5.3 Compare

| | Method A | Method B |
|--|--------:|--------:|
| **HV-final** | 15 000 | 15 000 |
| **AUBC**     | **13 250** | **9 375** |
| **AUBC / HV-final** | 0.883 | 0.625 |

Same endpoint, very different journey. AUBC sees Method A as **41 % better**
than Method B (`13 250 / 9 375 = 1.41`) even though they share HV-final.
That is the entire point of the metric — to detect *trajectory* quality,
not just *endpoint* quality.

### 5.4 Visual side-by-side

```
   HV
   ▲                          Method A  (early ramp)
   │
15000 ─────────────────● ● ● ● ● ● ● ● ● ●
   │              ░░░░░│
14000 ────●░░░░░░░░░░░░│
   │   ░░░░░░░░░░░░░░░░│
   │  ░░░░░░░░░░░░░░░░░│
10000 ░░░░░░░░░░░░░░░░░│
   │ ░░░░░░░░░░░░░░░░░░│
 5000 │░░░░░░░░░░░░░░░░░│
   │░░░░░░░░░░░░░░░░░░░│
   0──────────────────────────────────► budget
   0   4              10           17  20

   HV
   ▲                          Method B  (late ramp)
   │
15000 ─────────────────────────────────● ● ●
   │                            ░░░░░░░│
14000 │                       ░░░░░░░░░│
   │                       ░░░░░░░░░░░│
10000 │                ●░░░░░░░░░░░░░░│
   │                 ░░░░░░░░░░░░░░░░│
 5000 │      ●░░░░░░░░░░░░░░░░░░░░░░░░│
   │    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
   0──────────────────────────────────► budget
   0   4              10           17  20
```

Visually, the dotted area of Method A is much fatter — that's the AUBC
difference, made concrete.

---

## 6. Plugging in your real numbers

From your `comparisons_aubc.csv` for **bi_tsp at B = 100**:

| Method | AUBC | HV-final | AUBC / HV-final |
|--------|-----:|---------:|----------------:|
| `full`        | 13 961 | 14 956 | 0.933 |
| `mpage_orig`  | 13 170 | 15 024 | 0.876 |

A few things to notice from this table alone:

* **HV-final is essentially tied** (full: 14 956 vs mpage_orig: 15 024 — the
  upstream is even slightly higher). If you reported only HV-final you'd say
  the methods are equivalent or that `mpage_orig` is marginally better.
* **AUBC is decisively higher for `full`** (+791, or about +6 %). The
  trajectory was better even though the endpoint was a wash.
* **AUBC / HV-final is higher for `full`** (0.933 vs 0.876). Across the
  budget axis, `full` averaged ≈ 93 % of its final HV vs `mpage_orig`'s
  ≈ 88 %. `full` got close to its peak earlier.

For tri_tsp at B = 25 (post-fix):

| Method | AUBC | HV-final | AUBC / HV-final |
|--------|-----:|---------:|----------------:|
| `full`        | 191 983 | 206 935 | 0.928 |
| `mpage_orig`  | 179 375 | 206 411 | 0.869 |

Same pattern: nearly identical HV-finals (206k vs 206k), but `full` reaches
its final HV faster (AUBC ratio 0.93 vs 0.87, gap of +6 percentage points).

This pattern — *"HV-final tied, AUBC distinctly higher"* — is exactly what
the thesis claims BMAB-LLM should do, and AUBC is the metric that makes the
claim visible.

---

## 7. How to sanity-check AUBC by eye

Two quick checks before reporting an AUBC number:

### 7.1 The bracketing check

```
   0  ≤  AUBC  ≤  HV-final
```

* `AUBC ≥ 0` because every term in the trapezoidal sum is ≥ 0 (HV is
  non-negative and budget intervals are positive).
* `AUBC ≤ HV-final` because the curve is monotonically non-decreasing
  (the population only gets better) and the mean of a monotone-increasing
  bounded function is at most its supremum.

If you see `AUBC > HV-final`, something is wrong — likely a corrupt
`budget_curve.json`, a wrong sort order, or an error in the data.

### 7.2 The "ramp speed" check

A method that does nothing for the first half of the budget and then
suddenly jumps to its final HV will have `AUBC ≈ HV-final / 2`.

A method that hits its final HV almost immediately and stays there will
have `AUBC ≈ HV-final`.

So `AUBC / HV-final` is a "fraction of final value typically available
during the run." Numbers under 0.5 indicate a very late-ramping method;
numbers above 0.9 indicate an early-ramping one. Most well-behaved BMAB
or MPaGE runs land in the 0.8–0.95 range.

---

## 8. Common pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **All HV values are 0** (the tri_tsp bug) | `AUBC = 0` for every seed | Ref-point shape mismatch. See `HV_TWO_LEVELS.md §4.1`. |
| **Curve has only 1 point** | AUBC equals roughly `0.5 · h_1 · b_1 / B + h_1 · (B − b_1) / B` | This is the correct integral over the available data; just note it's a coarse approximation. |
| **Curve is unsorted** | AUBC can come out negative | The implementation sorts before integrating; if you've hand-edited `budget_curve.json`, make sure it's still increasing in `budget_consumed`. |
| **Run crashed mid-generation** | Last `budget_consumed` < `B` | The flat-extension to `B` assumes nothing more would have happened. This is a defensive bias, not a bug. Document it if it matters. |
| **HV-final taken from an unsorted curve** | `hv_final` is not actually the last value chronologically | `aggregate.py` reads `curve[-1]`, which is the last item in file order — for normal runs this is also the chronologically last. If you've manipulated the file, re-sort. |
| **AUBC > HV-final** | Should be impossible | Likely a corrupt curve or wrong B. Inspect `budget_curve.json` by hand. |

---

## TL;DR

* **AUBC = (1/B) · ∫ HV(b) db**, computed by trapezoidal integration
  over the saved `(budget_consumed, hv)` records, anchored at `(0, 0)`
  and extended flat to `B`.
* Same units as HV — it is literally the **mean HV across the budget axis**.
* **AUBC ≤ HV-final** always; the ratio `AUBC / HV-final` is the
  "trajectory efficiency."
* Two methods with the same HV-final can have very different AUBC, and
  that difference is the headline result this metric exists to expose.
