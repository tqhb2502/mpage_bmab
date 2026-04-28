"""Budget tracker for BMAB-LLM.

Maintains the global remaining budget across all LLM calls (initialisation,
clustering, suggestion, mutation and crossover). Supports both unit-cost mode
(``call_count``) and token-aware mode (``token_count``).

The budget tracker is thread-safe so the same instance can be passed to every
sampler / cluster invocation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import List, Tuple


@dataclass
class BudgetEntry:
    """One LLM call's accounting record."""
    step: int
    label: str          # 'init', 'cluster', 'suggestion', 'mutate', 'crossover'
    cost: float
    remaining: float


class BudgetTracker:
    """Token / call budget tracker."""

    def __init__(self, total_budget: float, mode: str = 'call'):
        assert mode in ('call', 'token'), "mode must be 'call' or 'token'"
        self._total = float(total_budget)
        self._remaining = float(total_budget)
        self._mode = mode
        self._lock = Lock()
        self._history: List[BudgetEntry] = []
        self._step = 0

    # ------------------------------------------------------------------ accessors
    @property
    def total(self) -> float:
        return self._total

    @property
    def remaining(self) -> float:
        return self._remaining

    @property
    def consumed(self) -> float:
        return self._total - self._remaining

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def history(self) -> List[BudgetEntry]:
        return list(self._history)

    def is_exhausted(self) -> bool:
        return self._remaining <= 0

    # ------------------------------------------------------------------ deduction
    def can_afford(self, cost: float = 1.0) -> bool:
        return self._remaining - cost >= -1e-9

    def charge(self, cost: float, label: str = 'unknown') -> bool:
        """Charge the budget. Returns True if successful, False if insufficient."""
        with self._lock:
            if self._remaining - cost < -1e-9:
                return False
            self._remaining -= cost
            self._step += 1
            self._history.append(BudgetEntry(self._step, label, float(cost),
                                             self._remaining))
            return True

    def force_charge(self, cost: float, label: str = 'unknown') -> None:
        """Charge the budget regardless of sign. Used when an LLM call has been
        issued *before* the budget check (e.g. cost-aware token mode where the
        cost is only known after the call returns)."""
        with self._lock:
            self._remaining -= cost
            self._step += 1
            self._history.append(BudgetEntry(self._step, label, float(cost),
                                             self._remaining))

    # ------------------------------------------------------------------ helpers
    def fraction_remaining(self) -> float:
        if self._total <= 0:
            return 0.0
        return max(0.0, self._remaining / self._total)

    def snapshot(self) -> Tuple[float, float, int]:
        return self._remaining, self._total, self._step

    def __repr__(self) -> str:
        return (f"BudgetTracker(mode={self._mode}, "
                f"remaining={self._remaining:.2f}/{self._total:.2f}, "
                f"steps={self._step})")
