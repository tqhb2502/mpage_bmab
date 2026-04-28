"""Profiler for BMAB-LLM.

Extends MPaGE's :class:`EoHProfiler` with:

* a budget-vs-HV curve (the data needed to compute the AUBC metric);
* per-arm bandit statistics, dumped at the end of every generation;
* a record of the budget consumed by each LLM call.
"""
from __future__ import annotations

import json
import os
from threading import Lock
from typing import List

import numpy as np

from ._llm4ad.method.LLMPFG.profiler import EoHProfiler
from ._llm4ad.method.LLMPFG.population import Population

from .budget import BudgetTracker
from .reward import hypervolume


class BMABProfiler(EoHProfiler):
    """Profiler with budget tracking + AUBC computation."""

    def __init__(self,
                 log_dir: str | None = None,
                 evaluation_name: str = 'Problem',
                 method_name: str = 'BMAB-LLM',
                 *,
                 ref_point=None,
                 initial_num_samples: int = 0,
                 log_style: str = 'complex',
                 **kwargs):
        super().__init__(log_dir=log_dir, evaluation_name=evaluation_name,
                         method_name=method_name,
                         initial_num_samples=initial_num_samples,
                         log_style=log_style, **kwargs)
        self._ref_point = np.asarray(ref_point) if ref_point is not None else None
        self._curve: List[dict] = []      # budget-vs-HV samples
        self._curve_lock = Lock()
        self._bandit_log: List[dict] = []
        self._budget_history: List[dict] = []

    # -------------------------------------------------------- AUBC tracking

    def record_curve_point(self, budget_consumed: float,
                           population: Population) -> None:
        """Append a (consumed_budget, HV) record to the curve."""
        if self._ref_point is None:
            return
        scores = []
        for f in population.population:
            if f.score is None:
                continue
            try:
                arr = np.asarray(f.score, dtype=float)
                if arr.shape == self._ref_point.shape:
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

    def record_bandit_state(self, generation: int,
                            operator_stats: dict,
                            cluster_stats: dict) -> None:
        with self._curve_lock:
            self._bandit_log.append({
                'generation': generation,
                'operator': operator_stats,
                'cluster': {f'{k}_{o}': v
                            for (k, o), v in cluster_stats.items()},
            })

    def record_budget(self, budget: BudgetTracker) -> None:
        with self._curve_lock:
            self._budget_history = [
                {'step': e.step, 'label': e.label,
                 'cost': e.cost, 'remaining': e.remaining}
                for e in budget.history
            ]

    # -------------------------------------------------------- summary metrics

    def aubc(self, total_budget: float) -> float:
        """Area-Under-Budget-Curve. Uses trapezoidal integration on the
        (consumed_budget, hv) samples recorded so far. Higher is better."""
        if not self._curve:
            return 0.0
        # Sort curve by consumed budget
        pts = sorted(self._curve, key=lambda x: x['budget_consumed'])
        xs = [0.0] + [p['budget_consumed'] for p in pts]
        ys = [0.0] + [p['hv'] for p in pts]
        # extend to total_budget
        if xs[-1] < total_budget:
            xs.append(total_budget)
            ys.append(ys[-1])
        area = 0.0
        for i in range(1, len(xs)):
            area += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
        return area / max(total_budget, 1e-9)  # mean HV over budget axis

    # -------------------------------------------------------- finishing

    def finish(self, *, budget: BudgetTracker | None = None) -> None:
        super().finish()
        if not self._log_dir:
            return
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            with open(os.path.join(self._log_dir, 'budget_curve.json'),
                      'w') as f:
                json.dump(self._curve, f, indent=2)
            with open(os.path.join(self._log_dir, 'bandit_log.json'),
                      'w') as f:
                json.dump(self._bandit_log, f, indent=2)
            with open(os.path.join(self._log_dir, 'budget_history.json'),
                      'w') as f:
                json.dump(self._budget_history, f, indent=2)
            if budget is not None:
                with open(os.path.join(self._log_dir, 'aubc.json'), 'w') as f:
                    json.dump({
                        'total_budget': budget.total,
                        'consumed_budget': budget.consumed,
                        'aubc': self.aubc(budget.total),
                    }, f, indent=2)
        except Exception:
            pass
