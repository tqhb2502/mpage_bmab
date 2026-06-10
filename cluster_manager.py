"""Cluster manager for BMAB-LLM.

Wraps the LLM-based semantic clustering call from MPaGE so that:

* the budget tracker is decremented per call;
* malformed responses are recovered with a fallback partition;
* cluster representatives (an heuristic per cluster) and per-cluster quality
  scores are extracted for the bandit's warm start.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ._llm4ad.base import Function
from ._llm4ad.method.LLMPFG.prompt import EoHPrompt
from ._llm4ad.method.LLMPFG.sampler import EoHSampler

from .budget import BudgetTracker
from .reward import _is_valid_score, hypervolume


def _flat_indices(group: Sequence[Sequence[int]]) -> List[int]:
    return [idx for sub in group for idx in sub]


def _is_valid_partition(group, n) -> bool:
    if not group:
        return False
    flat = _flat_indices(group)
    return sorted(flat) == list(range(n)) and len(set(flat)) == n


def _fallback_partition(elites: List[Function]) -> List[List[int]]:
    """Singleton clusters: every elite gets its own cluster."""
    return [[i] for i in range(len(elites))]


class ClusterManager:
    """Wraps the cluster-LLM call and computes warm-start priors."""

    def __init__(self, cluster_sampler: EoHSampler, task_description: str,
                 template_function: Function, budget: BudgetTracker,
                 cluster_call_cost: float = 1.0,
                 score_invert_cap: float = 100.0,
                 ref_point: Sequence[float] | None = None):
        self._sampler = cluster_sampler
        self._task = task_description
        self._template = template_function
        self._budget = budget
        self._call_cost = cluster_call_cost
        self._score_invert_cap = score_invert_cap
        self._ref_point = (np.asarray(ref_point, dtype=float)
                           if ref_point is not None else None)

    # -------------------------------------------------------- API

    def cluster(self, elites: List[Function]) -> Tuple[List[List[int]],
                                                       Dict[int, float]]:
        """Cluster the given elite heuristics into groups.

        Returns:
            (group_indices, cluster_quality) where group_indices is a list of
            sublists (each sublist is a list of indices into ``elites``) and
            cluster_quality is a dict mapping cluster_idx -> a non-negative
            quality score used as a warm-start prior for the bandit.
        """
        n = len(elites)
        if n == 0:
            return [], {}
        if n < 3:
            partition = _fallback_partition(elites)
            return partition, self._cluster_quality(partition, elites)

        # 1. Charge budget BEFORE issuing call (avoid spending past budget)
        if not self._budget.charge(self._call_cost, label='cluster'):
            partition = _fallback_partition(elites)
            return partition, self._cluster_quality(partition, elites)

        # 2. Issue the LLM clustering call
        try:
            prompt = EoHPrompt.get_prompt_cluster(
                self._task, elites, self._template)
            response = self._sampler.get_thought(prompt)
        except Exception:
            response = None

        # 3. Parse / fall back
        if response is None or not _is_valid_partition(response, n):
            partition = _fallback_partition(elites)
        else:
            partition = list(response)

        return partition, self._cluster_quality(partition, elites)

    def fallback_partition(self, elites: List[Function]) -> List[List[int]]:
        return _fallback_partition(elites)

    def cluster_quality(self, partition: List[List[int]],
                        elites: List[Function]) -> Dict[int, float]:
        return self._cluster_quality(partition, elites)

    # -------------------------------------------------------- priors

    def _cluster_quality(self, partition: List[List[int]],
                         elites: List[Function]) -> Dict[int, float]:
        """Quality prior per cluster.

        Combines three final-HV-relevant signals:

        * cluster hypervolume contribution to the current heuristic front;
        * best inner-HV proxy in the cluster (``-score[0]``);
        * best runtime proxy in the cluster.

        The previous prior only used ``-min(score[0])``, which ignored runtime
        and whether the cluster actually contributed to the outer Pareto front.
        """
        quality: Dict[int, float] = {}
        all_scores = []
        score_by_index: Dict[int, np.ndarray] = {}
        for i, f in enumerate(elites):
            if f.score is None or not _is_valid_score(f.score):
                continue
            arr = np.asarray(f.score, dtype=float)
            if self._ref_point is not None and arr.shape != self._ref_point.shape:
                continue
            score_by_index[i] = arr
            all_scores.append(arr)

        hv_total = 0.0
        if self._ref_point is not None and all_scores:
            hv_total = hypervolume(np.asarray(all_scores), self._ref_point)

        for c, sub in enumerate(partition):
            scores = [score_by_index[i] for i in sub if i in score_by_index]
            if not scores:
                quality[c] = 0.0
                continue

            hv_contribution = 0.0
            if self._ref_point is not None and hv_total > 0.0:
                without = [s for i, s in score_by_index.items() if i not in sub]
                hv_without = (hypervolume(np.asarray(without), self._ref_point)
                              if without else 0.0)
                hv_contribution = max(0.0, hv_total - hv_without)

            first_objectives = [float(s[0]) for s in scores]
            best_hv_proxy = max(0.0, -min(first_objectives))
            if self._score_invert_cap > 0:
                best_hv_proxy = min(best_hv_proxy, self._score_invert_cap)

            runtimes = [float(s[1]) for s in scores if len(s) > 1 and s[1] >= 0]
            runtime_proxy = 0.0
            if runtimes:
                runtime_proxy = 1.0 / (1.0 + min(runtimes))

            quality[c] = hv_contribution + 0.25 * best_hv_proxy + runtime_proxy
        if quality:
            mx = max(quality.values())
            if mx > 0:
                quality = {k: min(v / mx, 1.0) for k, v in quality.items()}
        return quality
