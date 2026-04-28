"""Reward computation for BMAB-LLM.

Combines three signals:

1. **HVI** (Hypervolume Improvement) — primary multi-objective quality signal,
   computed on the *heuristic* Pareto front (criteria = (-HV, runtime)).
2. **Diversity gain** — change in Cumulative Diversity Index (CDI) and/or
   Shannon-Wiener Diversity Index (SWDI).
3. **Rank score** — a dense, smoothed signal that ranks the new heuristic
   against recent population members (so we never observe pure-zero rewards).

A penalty term is subtracted when the heuristic is invalid (None, NaN or a
TIMEOUT during evaluation).

All intermediate signals are bounded so the standard MAB regret analysis applies.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from pymoo.indicators.hv import HV  # type: ignore
    _HAS_PYMOO = True
except Exception:  # pragma: no cover - pymoo is in requirements but be defensive
    _HAS_PYMOO = False


# --------------------------------------------------------------------------- HV

def _is_valid_score(score) -> bool:
    if score is None:
        return False
    try:
        arr = np.asarray(score, dtype=float)
    except Exception:
        return False
    return arr.ndim == 1 and np.all(np.isfinite(arr))


def _pareto_front(points: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset of ``points`` (minimisation)."""
    if len(points) == 0:
        return points
    keep = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not keep[i]:
            continue
        for j in range(len(points)):
            if i == j or not keep[j]:
                continue
            # i dominates j ?
            if np.all(points[i] <= points[j]) and np.any(points[i] < points[j]):
                keep[j] = False
    return points[keep]


def hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Hypervolume of ``points`` with respect to ``ref_point`` (minimisation)."""
    if len(points) == 0:
        return 0.0
    if _HAS_PYMOO:
        ind = HV(ref_point=np.asarray(ref_point, dtype=float))
        return float(ind(np.asarray(points, dtype=float)))
    # crude 2D fallback
    pts = np.asarray(points, dtype=float)
    pts = _pareto_front(pts)
    if pts.shape[1] != 2:
        # Degenerate fallback: sum of (ref - point) volumes (overestimate)
        return float(np.maximum(0.0, ref_point - pts).prod(axis=1).sum())
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_y = ref_point[1]
    for x, y in pts:
        if x >= ref_point[0] or y >= ref_point[1]:
            continue
        hv += (ref_point[0] - x) * (prev_y - y)
        prev_y = y
    return float(max(0.0, hv))


def hvi(new_point: Sequence[float],
        existing: Iterable[Sequence[float]],
        ref_point: Sequence[float]) -> float:
    """Hypervolume improvement of ``new_point`` over ``existing``."""
    ref = np.asarray(ref_point, dtype=float)
    existing = [np.asarray(p, dtype=float) for p in existing if _is_valid_score(p)]
    new_pt = np.asarray(new_point, dtype=float)
    if not _is_valid_score(new_pt):
        return 0.0
    if np.any(new_pt >= ref):  # strictly worse than reference, no contribution
        return 0.0
    pts_before = np.array(existing) if existing else np.zeros((0, len(ref)))
    pts_after = np.vstack([pts_before, new_pt[None, :]])
    hv_before = hypervolume(pts_before, ref) if len(pts_before) else 0.0
    hv_after = hypervolume(pts_after, ref)
    return max(0.0, hv_after - hv_before)


# ---------------------------------------------------------------------- Diversity

def shannon_diversity(cluster_sizes: Sequence[int]) -> float:
    """Shannon-Wiener diversity over the cluster size histogram."""
    sizes = [s for s in cluster_sizes if s > 0]
    total = sum(sizes)
    if total == 0:
        return 0.0
    entropy = 0.0
    for s in sizes:
        p = s / total
        entropy -= p * math.log(p)
    return entropy


def cumulative_diversity(scores: Sequence[Sequence[float]]) -> float:
    """Cumulative Diversity Index. Mean pairwise Euclidean distance in
    objective space, normalised by population size. Bounded in [0, ∞)."""
    pts = np.array([s for s in scores if _is_valid_score(s)], dtype=float)
    if len(pts) < 2:
        return 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt((diff * diff).sum(-1))
    n = len(pts)
    return float(dists.sum() / (n * (n - 1)))


# ------------------------------------------------------------------------- Reward

class RewardComputer:
    """Computes the scalar reward used by the bandit and tracks rolling stats
    so we can normalise on-the-fly."""

    def __init__(self,
                 ref_point: Sequence[float],
                 w_quality: float = 1.0,
                 w_diversity: float = 0.3,
                 w_rank: float = 0.2,
                 penalty: float = 1.0,
                 rank_window: int = 50,
                 hvi_floor: float = 0.0):
        self._ref = np.asarray(ref_point, dtype=float)
        self._w_q = w_quality
        self._w_d = w_diversity
        self._w_r = w_rank
        self._pen = penalty
        self._rank_window = rank_window
        self._hvi_floor = hvi_floor

        self._reward_history: List[float] = []
        self._hvi_history: List[float] = []

    # ----------------------------------------------------------------- compute

    def reward(self,
               new_score: Optional[Sequence[float]],
               population_scores: Sequence[Sequence[float]],
               diversity_before: float,
               diversity_after: float,
               *,
               invalid: bool = False) -> Tuple[float, dict]:
        """Compute a scalar reward and return it together with a breakdown.

        Args:
            new_score: tuple/list of objective values for the new heuristic, or
                None / NaN if the heuristic was invalid.
            population_scores: iterable of objective tuples for the current
                heuristic Pareto front (used as HVI baseline).
            diversity_before / diversity_after: diversity index values before /
                after inserting the new heuristic.
            invalid: whether the heuristic is invalid (penalty applied).
        """
        breakdown = {'hvi': 0.0, 'rank': 0.0, 'diversity': 0.0,
                     'penalty': 0.0, 'total': 0.0}

        if invalid or new_score is None or not _is_valid_score(new_score):
            breakdown['penalty'] = -self._pen
            breakdown['total'] = -self._pen
            self._reward_history.append(-self._pen)
            self._hvi_history.append(0.0)
            return -self._pen, breakdown

        # 1. HVI
        h = hvi(new_score, population_scores, self._ref)
        breakdown['hvi'] = h
        self._hvi_history.append(h)
        # rolling normalisation: scale so recent max ≈ 1
        recent = self._hvi_history[-self._rank_window:]
        h_max = max(recent + [self._hvi_floor + 1e-9])
        h_norm = h / h_max if h_max > 0 else 0.0

        # 2. Rank score within window
        rank_score = 0.0
        if len(population_scores) > 0:
            # Simple metric: how many population members have any objective worse
            # than new_score? Rescaled to [0, 1].
            n = 0
            better = 0
            for s in population_scores:
                if not _is_valid_score(s):
                    continue
                n += 1
                if any(a < b for a, b in zip(new_score, s)):
                    better += 1
            rank_score = better / max(n, 1)
        breakdown['rank'] = rank_score

        # 3. Diversity gain
        d_gain = max(0.0, diversity_after - diversity_before)
        breakdown['diversity'] = d_gain

        total = (self._w_q * h_norm
                 + self._w_d * d_gain
                 + self._w_r * rank_score)
        breakdown['total'] = total
        self._reward_history.append(total)
        return total, breakdown

    # ----------------------------------------------------------------- stats

    @property
    def history(self) -> List[float]:
        return list(self._reward_history)

    def stats(self) -> dict:
        if not self._reward_history:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}
        a = np.asarray(self._reward_history)
        return {'mean': float(a.mean()), 'std': float(a.std()),
                'max': float(a.max()), 'min': float(a.min())}
