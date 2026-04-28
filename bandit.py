"""Two-layer Budgeted Multi-Armed Bandit for BMAB-LLM.

Layers
------
1. **OperatorBandit**: persistent across generations. Two arms ``μ`` (Mutate)
   and ``χ`` (Crossover). UCB1.
2. **ClusterBandit**: re-initialised every generation when the elite set is
   re-clustered by the LLM. Per arm ``(cluster_idx, operator)`` we maintain
   the budget-aware reward / cost ratio plus a Page-Hinkley test so that an
   arm that suddenly stops producing good heuristics is reset to the
   optimistic prior.

A *warm-start* strategy initialises new clusters with optimistic prior values
plus a draw from the operator-level statistics, so we never start from zero
samples.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


OPERATORS = ('mutate', 'crossover')


# --------------------------------------------------------------------- Page-Hinkley

@dataclass
class PageHinkleyState:
    """Online change-point detector for *decreases* in reward.

    Standard one-sided Page-Hinkley test:
        m_t   = Σ_{k=1..t} (x_k − mean_k − δ)
        M_t   = max_{k ≤ t} m_k
        PH_t  = M_t − m_t      (≥ 0)
        drift = (PH_t > λ)

    A sustained drop in `x_t` makes `m_t` decrease away from its peak `M_t`,
    so `PH_t` grows past the threshold.
    """
    delta: float = 0.005
    threshold: float = 0.5
    sum: float = 0.0
    max_sum: float = 0.0
    n: int = 0
    mean: float = 0.0

    def update(self, value: float) -> bool:
        """Update with a new reward observation. Returns True iff drift detected."""
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.sum += value - self.mean - self.delta
        self.max_sum = max(self.max_sum, self.sum)
        ph = self.max_sum - self.sum
        return ph > self.threshold

    def reset(self) -> None:
        self.sum = 0.0
        self.max_sum = 0.0
        self.n = 0
        self.mean = 0.0


# --------------------------------------------------------------------- ArmStats

@dataclass
class ArmStats:
    """Sufficient statistics for one bandit arm."""
    n: int = 0
    sum_reward: float = 0.0
    sum_cost: float = 0.0
    sum_sq_reward: float = 0.0  # for variance / Bernstein-style bounds (unused)

    def update(self, reward: float, cost: float = 1.0) -> None:
        self.n += 1
        self.sum_reward += reward
        self.sum_cost += max(cost, 1e-6)
        self.sum_sq_reward += reward * reward

    @property
    def mean_reward(self) -> float:
        return self.sum_reward / self.n if self.n > 0 else 0.0

    @property
    def mean_cost(self) -> float:
        return self.sum_cost / self.n if self.n > 0 else 1.0

    @property
    def reward_per_cost(self) -> float:
        return self.sum_reward / max(self.sum_cost, 1e-6)


# --------------------------------------------------------------------- Operator-level

class OperatorBandit:
    """Persistent UCB1 bandit over the operator alphabet {μ, χ}."""

    def __init__(self, c_explore: float = 1.0, prior_n: int = 1,
                 prior_reward: float = 0.5):
        self._c = c_explore
        self._stats: Dict[str, ArmStats] = {
            o: ArmStats(n=prior_n,
                        sum_reward=prior_reward * prior_n,
                        sum_cost=prior_n)
            for o in OPERATORS
        }
        self._t = 0

    def total_pulls(self) -> int:
        return self._t

    def select(self) -> str:
        self._t += 1
        scores = {}
        total_n = sum(s.n for s in self._stats.values())
        for op, s in self._stats.items():
            exploit = s.reward_per_cost
            explore = self._c * math.sqrt(2.0 * math.log(max(total_n, 2)) / s.n)
            scores[op] = exploit + explore
        return max(scores, key=scores.get)

    def update(self, op: str, reward: float, cost: float = 1.0) -> None:
        self._stats[op].update(reward, cost)

    def stats(self) -> Dict[str, dict]:
        return {op: {'n': s.n,
                     'mean_reward': s.mean_reward,
                     'mean_cost': s.mean_cost,
                     'reward_per_cost': s.reward_per_cost}
                for op, s in self._stats.items()}

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


# --------------------------------------------------------------------- Cluster-level

class ClusterBandit:
    """Budgeted UCB1 over per-generation arms ``(cluster_idx, operator)``.

    Statistics are reset every generation but warm-started with a prior derived
    from the cluster's representative quality and from the persistent operator
    statistics.
    """

    def __init__(self, c_explore: float = 1.0, gamma_budget: float = 0.5,
                 prior_n: float = 1.0, prior_reward: float = 0.5,
                 ph_delta: float = 0.005, ph_threshold: float = 0.5):
        self._c = c_explore
        self._gamma_b = gamma_budget
        self._prior_n = prior_n
        self._prior_reward = prior_reward
        self._ph_delta = ph_delta
        self._ph_threshold = ph_threshold

        self._stats: Dict[Tuple[int, str], ArmStats] = {}
        self._ph: Dict[Tuple[int, str], PageHinkleyState] = {}
        self._cluster_priors: Dict[int, float] = {}
        self._t = 0
        self._n_clusters = 0

    # -------------------------------------------------------- (re-)initialisation

    def reset(self, n_clusters: int,
              cluster_quality: Optional[Dict[int, float]] = None,
              operator_priors: Optional[Dict[str, float]] = None) -> None:
        """Reset the bandit at the start of a new generation.

        Args:
            n_clusters: number of clusters in the new generation.
            cluster_quality: map cluster_idx -> non-negative quality (e.g.
                normalised inverse score of the best heuristic in that cluster).
            operator_priors: map operator -> normalised prior from the
                persistent OperatorBandit.
        """
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

    # -------------------------------------------------------- selection

    def select(self, budget_fraction: float = 1.0,
               restrict_operator: Optional[str] = None) -> Tuple[int, str]:
        """Choose an arm.

        Args:
            budget_fraction: ``b_t / B`` ∈ (0, 1]. Used in the budget-pressure
                term ``γ_b · ln(budget_fraction)`` (≤ 0).
            restrict_operator: if set, only arms with this operator are considered.
        """
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

    # -------------------------------------------------------- update

    def update(self, arm: Tuple[int, str], reward: float,
               cost: float = 1.0) -> bool:
        """Update statistics for ``arm``. Returns True iff PH detected drift."""
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

    # -------------------------------------------------------- introspection

    def stats(self) -> Dict[Tuple[int, str], dict]:
        return {arm: {'n': s.n,
                      'mean_reward': s.mean_reward,
                      'mean_cost': s.mean_cost,
                      'reward_per_cost': s.reward_per_cost}
                for arm, s in self._stats.items()}

    @property
    def n_clusters(self) -> int:
        return self._n_clusters
