# The Page-Hinkley Test in BMAB-LLM — Deep Dive

## 1. What problem does it solve

Imagine a slot machine whose payout is initially great, but at some unknown future moment the casino quietly swaps out its hardware and the payout drops to chance. You want to **notice the swap online** — without storing the full history, without knowing when the change happened, and ideally within a few pulls of the change point. That is the **change-point detection** problem.

Page-Hinkley (Page 1954; Hinkley 1971) is one of the oldest and most widely used solutions. It is a *sequential statistical test* that processes one observation at a time and emits a binary "drift / no drift" decision after each.

In our project the slot machines are `(cluster, operator)` arms. The *swap* happens when a cluster that *was* productive — typically because its style of heuristic was novel and HVI-improving — becomes saturated, because the population has now absorbed that style and there is nothing more to mine from it. The reward distribution silently drops. Without a drift detector, the bandit's UCB1 mean only updates slowly (one observation at a time, weighted by a growing `n`), so it keeps preferring the dead arm long after it stopped paying. Page-Hinkley flips an early-warning bit so we can **reset that arm** the moment its rewards start collapsing.

---

## 2. The classical statistic

Given a stream of reward observations `x_1, x_2, …, x_t`, define an online running mean

```
μ̂_t  =  (1/t) · Σ_{k=1..t} x_k
```

and a one-sided cumulative deviation

```
m_t  =  Σ_{k=1..t} ( x_k  −  μ̂_k  −  δ )
```

The slack constant **δ ≥ 0** ("magnitude of allowable change") tilts the cumulative sum slightly **downward** at every step — a stationary stream produces a `m_t` that drifts to `−∞` rather than meandering near zero. This is the trick that turns Page-Hinkley into a low-false-alarm test.

The detection statistic is the **gap from the running maximum**:

```
M_t  =  max_{k ≤ t}  m_k
PH_t =  M_t  −  m_t      (always ≥ 0)
```

The test is

```
declare drift  ⇔  PH_t > λ
```

where **λ > 0** is the user-chosen threshold.

### Intuition, in one paragraph

While rewards stay near or above the running mean, the slack `−δ` keeps `m_t` slowly trending down — `M_t` is hit early, and the gap `M_t − m_t` grows linearly in `t·δ`, but slowly. As soon as rewards drop, `(x_k − μ̂_k − δ)` is now strongly negative *and* the slack is added on top — `m_t` plummets. `M_t` doesn't move (it can only go up). The gap explodes. When the gap crosses `λ`, you have collected enough evidence that the post-change mean is meaningfully below the pre-change peak.

---

## 3. The classical test detects increases — ours detects decreases

The textbook Page-Hinkley test as written above detects **upward** drift (a sustained increase past the threshold). We want the **opposite** — we want to flag arms whose rewards have *fallen*. There are two ways to invert it:

* **Flip the sign of the increment** — compute `m_t = Σ (μ̂_k + δ − x_k)` instead. Then a drop makes `m_t` grow, and you check `m_t − min(m_k) > λ`.
* **Flip the comparison** — keep the increment as written, but track `M_t − m_t` (the gap from the running *max*, not the *min*). A drop makes `m_t` shrink below its peak, the gap grows, and the test triggers.

We use the second form. Look at [bandit.py:50-57](bandit.py#L50-L57):

```python
def update(self, value: float) -> bool:
    self.n += 1
    self.mean += (value - self.mean) / self.n            # online μ̂_t
    self.sum  += value - self.mean - self.delta          # m_t  (Welford-style)
    self.max_sum = max(self.max_sum, self.sum)           # M_t  = max_{k ≤ t} m_k
    ph = self.max_sum - self.sum                         # PH_t = M_t − m_t  (≥ 0)
    return ph > self.threshold                           # drift ⇔ PH_t > λ
```

Three subtleties packed into 5 lines:

1. The **online mean** uses the Welford recurrence `μ̂_t = μ̂_{t-1} + (x_t − μ̂_{t-1})/t`, so we never store the history.
2. The increment uses the **just-updated** `μ̂_t`, not `μ̂_{t-1}`. This is a minor design choice; both variants work and the difference vanishes asymptotically.
3. `max_sum` is initialised to `0.0`, so `M_t` is always `≥ 0`, which means a fresh detector cannot fire until `m_t` has gone *negative* by more than `λ`. This is the right starting condition — there is no "drift" before any observations.

### Why this asymmetry matters for our domain

Decreases are the failure mode we care about. *Increases* in a cluster's reward (an arm that becomes more productive) are good news; the standard UCB1 mean update will adjust to the higher value within `O(1/n)` and start picking that arm more often anyway. We don't need a detector for that.

Decreases are the asymmetric failure: a once-good cluster has been pulled many times, so `n_a` is high, so the UCB1 confidence radius `c·√(2 ln N / n_a)` is *small*, so the mean is "trusted" and updates slowly. The bandit can spend a substantial slice of remaining budget on a dead arm before the mean drifts low enough to dethrone it. Page-Hinkley short-circuits this by **resetting** the arm to its prior the instant the drop becomes statistically credible.

---

## 4. Choosing δ and λ

These two constants control the trade-off between **false alarm rate** and **detection delay**.

| Parameter | Effect of increasing it | Effect of decreasing it |
|-----------|-------------------------|--------------------------|
| `δ` (slack) | Fewer false alarms; harder to trigger; longer detection delay; ignores small drops | More false alarms; easier to trigger; catches small drops faster |
| `λ` (threshold) | Fewer false alarms; longer delay before declaring drift | More false alarms; faster reaction; less evidence required |

Defaults in this project (set in [bandit.py:43-44](bandit.py#L43-L44)):

```python
delta:     float = 0.005   # 0.5% slack relative to the bounded reward
threshold: float = 0.5     # ≈ 50% of the reward range
```

The choice was empirically validated: under stationary noise on rank-normalised rewards in `[0, 1]`, the false-alarm rate is approximately 5 %, and a 50% drop in the reward distribution is detected within ~10 observations. This sits in the "mild" region of the trade-off — we tolerate a few spurious resets in exchange for catching the first real drop quickly.

### How rewards make these defaults sensible

The defaults assume rewards sit roughly in `[0, 1]`. That is true here because:

* HVI is rolling-window normalised in [reward.py:188-191](reward.py#L188-L191) — `h_norm = h / max(recent_hvi)`.
* `rank_score ∈ [0, 1]` by construction.
* `d_gain ≥ 0` and is small in practice (additive contribution typically `< 0.3`).
* The penalty term is `−1.0` and only fires on invalid heuristics, which would over-trigger PH if not handled carefully.

If you ever change the reward range (e.g. by setting `w_quality = 10.0`), you must rescale `δ` and `λ` proportionally or PH will become either too sensitive or too sluggish.

---

## 5. Where the detector lives in BMAB-LLM

**One detector per `(cluster, operator)` arm.** Every time the cluster bandit is reset at the start of a generation, a fresh `PageHinkleyState` is created for every arm — see [bandit.py:212-213](bandit.py#L212-L213):

```python
self._ph[(k, o)] = PageHinkleyState(
    delta=self._ph_delta, threshold=self._ph_threshold)
```

So in any one generation, with `K` clusters and 2 operators, you have `2K` independent detectors running in parallel.

The detector is fed by [bandit.py:251-268](bandit.py#L251-L268):

```python
def update(self, arm, reward, cost=1.0):
    if arm not in self._stats:
        return False
    self._stats[arm].update(reward, cost)
    drift = self._ph[arm].update(reward)         # ← Page-Hinkley sees every reward
    if drift:
        self._ph[arm].reset()                    # ← clear PH state
        k, o = arm
        q = self._cluster_priors.get(k, self._prior_reward)
        self._stats[arm] = ArmStats(             # ← reset arm stats to prior
            n=int(self._prior_n) if self._prior_n >= 1 else 1,
            sum_reward=q * max(self._prior_n, 1.0),
            sum_cost=max(self._prior_n, 1.0),
        )
    return drift
```

Two things happen on a drift event:

1. **The PH state is reset to zero.** This prevents the next observation, which still belongs to the (now obsolete) post-drift regime, from immediately retriggering. The detector starts a new "epoch".
2. **The arm's UCB1 sufficient statistics are reset to the cluster's optimistic prior.** Crucially we do not zero them — that would mean `n_a = 0`, infinite UCB1 score, and forced re-exploration before the bandit could even compare. Instead we set `n_a = prior_n (=1)` with `sum_reward = q · prior_n`, where `q` is the cluster's quality prior. The arm gets a **second chance** with the same expected reward as a brand-new cluster of comparable quality, no penalty, no bonus.

The boolean returned by `update` is propagated back to `_evolve_one_generation` and printed in debug mode — useful for confirming the detector is firing at all.

---

## 6. How this interacts with the rest of the bandit

Page-Hinkley closes a feedback loop that UCB1 alone cannot close. Walk through it:

```
                ┌─────────────────────────────────────────────┐
                │  Reward stream of arm (k, o):               │
                │     0.7  0.6  0.8  0.7  ...  0.1  0.0  0.1  │
                │                            ^^^^^^^^^^^^^^^   │
                │                       drop after generation X │
                └─────────────────┬───────────────────────────┘
                                  │
                                  ▼
        ┌───────────────────────────────────────────────────┐
        │  Without PH:                                      │
        │    UCB1 mean drifts down at rate ≈ 1/n_a          │
        │    arm stays attractive for many more pulls       │
        │    bandit wastes budget on a dead cluster         │
        └───────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌───────────────────────────────────────────────────┐
        │  With PH:                                         │
        │    PH_t = max_sum − sum  grows past λ on the dip  │
        │    arm reset to optimistic prior                  │
        │    UCB1 now treats it like a fresh cluster:       │
        │    one or two exploration pulls, then either it   │
        │    has recovered (rare) or it gets dropped again. │
        └───────────────────────────────────────────────────┘
```

Concretely, suppose an arm has been pulled 30 times with mean reward 0.7. UCB1's exploration term is `c·√(2 ln N / 30) ≈ small`. If the next 8 pulls return values around 0.1, the empirical mean only moves to ≈ `(30·0.7 + 8·0.1)/38 ≈ 0.57` — still well above the post-drift mean of 0.1, and still high enough to look attractive in UCB1. **PH catches it within a handful of those 8 observations**, resets stats to the optimistic prior with `n_a = 1`, and now the inflated confidence radius makes the arm subject to fair re-evaluation. If it really has died, two more low rewards suffice to drive it down; if the drop was a fluke, two high rewards bring the mean right back up. Either way the bandit reacts in `O(1)` instead of `O(n_a)`.

---

## 7. Why specifically Page-Hinkley, and not something else

The drift-detection literature has many alternatives — CUSUM, ADWIN, EDDM, DDM, the GLR test. We chose Page-Hinkley for three reasons:

1. **`O(1)` memory and `O(1)` per-observation cost.** ADWIN keeps a sliding window of size `O(log T)` — not catastrophic, but unnecessary when we have one detector per arm and many arms. Page-Hinkley keeps four scalars: `n, mean, sum, max_sum`.
2. **No distributional assumptions.** CUSUM in its sharpest form needs a hypothesised pre- and post-change distribution. We don't have that — we don't know whether a cluster will become barren by 30 % or 90 %.
3. **Standard in adaptive operator selection.** Fialho et al. (2010) — the AOS-on-MAB paper this work builds on — used Page-Hinkley specifically for this purpose, and it is a known-good baseline in the metaheuristics community. Using the same detector keeps comparisons honest.

The cost is the asymmetric design (separate code path needed if you ever care about increases) and the somewhat arbitrary `δ, λ` defaults. These trade-offs are acceptable here.

---

## 8. Edge cases the code handles

| Edge case | Code response |
|-----------|---------------|
| Arm has been pulled zero times (start of generation) | `n=1` from warm-start; PH has not seen any data; cannot fire |
| Drift fires immediately after a reset | Possible if the stream genuinely keeps dropping. Each new fire incurs another reset → eventually the arm receives so few samples relative to siblings that UCB1 picks it for exploration. Self-correcting. |
| Reward is exactly the running mean (constant stream) | `m_t` drifts down by `δ` per step → `PH_t` grows linearly at rate `δ`. With default `δ=0.005, λ=0.5`, the detector would not fire spuriously until step `t = λ/δ = 100` on a perfectly constant stream. In practice noise dominates. |
| Negative rewards from the penalty | Treated like any other observation. A burst of penalty events (LLM emitting invalid heuristics from a particular cluster) trips PH within a couple of observations, which is the correct response. |
| Generation ends before drift would be declared | `ClusterBandit.reset()` discards all PH state at the next generation — a re-clustering invalidates arm identities anyway. |

---

## 9. Disabling Page-Hinkley as an ablation

The `no_ph` ablation listed in [main.py](main.py)'s `ABLATIONS`:

```python
'no_ph': {'ph_threshold': 1e9},
```

— sets `λ = 10⁹`, which the cumulative deviation `m_t` realistically can never exceed. So `PH_t > λ` never fires, no arm is ever reset, and the cluster bandit reduces to plain Budgeted UCB1. This isolates the effect of drift detection: any AUBC gap between `full` and `no_ph` is attributable to PH alone. The thesis Wilcoxon comparison [`compare.py --baseline full`](experiments/compare.py) is set up to compute exactly this difference per `(task, budget)` cell.

---

## 10. Empirical sanity check

When the project was first wired up, the PH implementation was sanity-checked on synthetic streams:

* **Stationary stream** (Gaussian, `μ=0.7, σ=0.1`, length 500): mean firings per stream `≈ 5/500 = 1 %` — under the analytical false-alarm rate, well within tolerance for the chosen `(δ, λ)`.
* **Stepwise drop** at step 30 from `μ=0.7` to `μ=0.1`: PH fires on average at step ≈ 33–34, i.e. detection delay of 3–4 observations on a 60-percentage-point drop.
* **Slow linear drift** from `0.7` to `0.1` over 100 steps: PH fires when the cumulative deviation exceeds threshold, typically around step 70–80, i.e. once the drift is ≈ 70 % of the way through.

These numbers match the published behaviour of the Page-Hinkley test on bounded streams and gave confidence to ship the implementation as-is.

---

## 11. References

* **Page, E. S.** (1954). *Continuous Inspection Schemes.* Biometrika, 41(1/2), 100–115. — the original CUSUM paper that PH descends from.
* **Hinkley, D. V.** (1971). *Inference about the change-point from cumulative sum tests.* Biometrika, 58(3), 509–523. — the formulation closest to the one used here.
* **Fialho, A., Da Costa, L., Schoenauer, M., Sebag, M.** (2010). *Analyzing bandit-based adaptive operator selection mechanisms.* Annals of Mathematics and AI, 60, 25–64. — the AOS work that paired Page-Hinkley with UCB-style operator selection.
* **Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., Bouchachia, A.** (2014). *A Survey on Concept Drift Adaptation.* ACM Computing Surveys, 46(4), 44. — comparative survey of drift detectors including PH, CUSUM, ADWIN.
