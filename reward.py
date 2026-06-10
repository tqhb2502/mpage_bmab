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


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def managed_scores(scores: Sequence[Sequence[float]], max_size: int) -> np.ndarray:
    """Score-only equivalent of MPaGE's population survivor selection.

    This mirrors ``population_management``: keep non-dominated fronts first and
    use crowding distance to truncate the last accepted front. It is used by the
    final-HV reward so the bandit learns the value of a candidate after the same
    population cap used at the end of the run.
    """
    pts = np.asarray([s for s in scores if _is_valid_score(s)], dtype=float)
    if len(pts) == 0 or max_size <= 0:
        return np.zeros((0, 0), dtype=float)
    if len(pts) <= max_size:
        return pts

    n = len(pts)
    dominates_list = [[] for _ in range(n)]
    dominated_count = [0 for _ in range(n)]
    fronts: List[List[int]] = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(pts[i], pts[j]):
                dominates_list[i].append(j)
            elif _dominates(pts[j], pts[i]):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)

    front_idx = 0
    while front_idx < len(fronts):
        nxt: List[int] = []
        for i in fronts[front_idx]:
            for j in dominates_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    nxt.append(j)
        if nxt:
            fronts.append(nxt)
        front_idx += 1

    selected: List[int] = []
    for front in fronts:
        if len(selected) + len(front) <= max_size:
            selected.extend(front)
            continue
        remaining = max_size - len(selected)
        distances = {idx: 0.0 for idx in front}
        if len(front) <= 2:
            for idx in front:
                distances[idx] = float('inf')
        else:
            for obj in range(pts.shape[1]):
                ordered = sorted(front, key=lambda idx: pts[idx, obj])
                distances[ordered[0]] = float('inf')
                distances[ordered[-1]] = float('inf')
                lo = pts[ordered[0], obj]
                hi = pts[ordered[-1], obj]
                if hi == lo:
                    continue
                for pos in range(1, len(ordered) - 1):
                    distances[ordered[pos]] += (
                        pts[ordered[pos + 1], obj]
                        - pts[ordered[pos - 1], obj]
                    ) / (hi - lo)
        selected.extend(sorted(front, key=lambda idx: -distances[idx])[:remaining])
        break
    return pts[selected]


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
                 hvi_floor: float = 0.0,
                 reward_mode: str = 'final_hv'):
        self._ref = np.asarray(ref_point, dtype=float)
        self._w_q = w_quality
        self._w_d = w_diversity
        self._w_r = w_rank
        self._pen = penalty
        self._rank_window = rank_window
        self._hvi_floor = hvi_floor
        if reward_mode not in {'dense', 'final_hv', 'hybrid'}:
            raise ValueError("reward_mode must be one of: dense, final_hv, hybrid")
        self._reward_mode = reward_mode

        self._reward_history: List[float] = []
        self._hvi_history: List[float] = []
        self._quality_history: List[float] = []

    # ----------------------------------------------------------------- compute

    def reward(self,
               new_score: Optional[Sequence[float]],
               population_scores: Sequence[Sequence[float]],
               diversity_before: float,
               diversity_after: float,
               *,
               invalid: bool = False,
               managed_pop_size: Optional[int] = None) -> Tuple[float, dict]:
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
        breakdown = {'hvi': 0.0, 'managed_hv_delta': 0.0,
                     'quality': 0.0, 'rank': 0.0, 'diversity': 0.0,
                     'penalty': 0.0, 'total': 0.0}

        if invalid or new_score is None or not _is_valid_score(new_score):
            breakdown['penalty'] = -self._pen
            breakdown['total'] = -self._pen
            self._reward_history.append(-self._pen)
            self._hvi_history.append(0.0)
            self._quality_history.append(0.0)
            return -self._pen, breakdown

        # 1. HVI
        h = hvi(new_score, population_scores, self._ref)
        breakdown['hvi'] = h
        self._hvi_history.append(h)
        # rolling normalisation: scale so recent max ≈ 1
        recent = self._hvi_history[-self._rank_window:]
        h_max = max(recent + [self._hvi_floor + 1e-9])
        h_norm = h / h_max if h_max > 0 else 0.0

        # 1b. Final-population HV delta after applying the population cap.
        managed_delta = h
        if managed_pop_size is not None and managed_pop_size > 0:
            before = managed_scores(population_scores, managed_pop_size)
            after_scores = list(population_scores) + [new_score]
            after = managed_scores(after_scores, managed_pop_size)
            hv_before = hypervolume(before, self._ref) if len(before) else 0.0
            hv_after = hypervolume(after, self._ref) if len(after) else 0.0
            managed_delta = max(0.0, hv_after - hv_before)
        breakdown['managed_hv_delta'] = managed_delta

        recent_quality = self._quality_history[-self._rank_window:]
        q_max = max(recent_quality + [managed_delta, self._hvi_floor + 1e-9])
        final_norm = managed_delta / q_max if q_max > 0 else 0.0
        self._quality_history.append(managed_delta)

        if self._reward_mode == 'dense':
            quality_signal = h_norm
        elif self._reward_mode == 'hybrid':
            quality_signal = 0.5 * h_norm + 0.5 * final_norm
        else:
            quality_signal = final_norm
        breakdown['quality'] = quality_signal

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

        total = (self._w_q * quality_signal
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
