"""Cluster manager for BMAB-LLM.

Wraps the LLM-based semantic clustering call from MPaGE so that:

* the budget tracker is decremented per call;
* malformed responses are recovered with a fallback partition;
* cluster representatives (an heuristic per cluster) and per-cluster quality
  scores are extracted for the bandit's warm start.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ._llm4ad.base import Function
from ._llm4ad.method.LLMPFG.prompt import EoHPrompt
from ._llm4ad.method.LLMPFG.sampler import EoHSampler

from .budget import BudgetTracker


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
                 score_invert_cap: float = 100.0):
        self._sampler = cluster_sampler
        self._task = task_description
        self._template = template_function
        self._budget = budget
        self._call_cost = cluster_call_cost
        self._score_invert_cap = score_invert_cap

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

    # -------------------------------------------------------- priors

    def _cluster_quality(self, partition: List[List[int]],
                         elites: List[Function]) -> Dict[int, float]:
        """Quality prior per cluster.

        Uses ``-min(score[0])`` (since score[0] = -HV; lower is better).
        Capped to a reasonable range and shifted to non-negative.
        """
        quality: Dict[int, float] = {}
        for c, sub in enumerate(partition):
            scores = [elites[i].score[0]
                      for i in sub
                      if elites[i].score is not None]
            if not scores:
                quality[c] = 0.0
                continue
            best = min(scores)            # most negative => best HV
            quality[c] = max(0.0, -best)  # convert back to positive HV proxy
        if quality:
            mx = max(quality.values())
            if mx > 0:
                quality = {k: min(v / mx, 1.0) for k, v in quality.items()}
        return quality
