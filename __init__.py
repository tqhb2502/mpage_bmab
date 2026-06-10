"""BMAB-LLM: Budgeted Multi-Objective Multi-Armed Bandit driver for
LLM-based heuristic design on Multi-Objective Combinatorial Optimisation.

A refined extension of MPaGE that explicitly optimises a fixed budget ``B`` of
LLM calls. See ``IDEA.md`` for the full description.
"""

__all__ = [
    'BMABLLM',
    'BudgetTracker',
    'OperatorBandit',
    'ClusterBandit',
    'PageHinkleyState',
    'RewardComputer',
    'BMABProfiler',
    'ClusterManager',
    'hypervolume',
    'hvi',
    'shannon_diversity',
    'cumulative_diversity',
]


def __getattr__(name):
    if name == 'BMABLLM':
        from .bmab_llm import BMABLLM
        return BMABLLM
    if name == 'BMABProfiler':
        from .profiler import BMABProfiler
        return BMABProfiler
    if name == 'ClusterManager':
        from .cluster_manager import ClusterManager
        return ClusterManager
    if name == 'BudgetTracker':
        from .budget import BudgetTracker
        return BudgetTracker
    if name in {'OperatorBandit', 'ClusterBandit', 'PageHinkleyState'}:
        from .bandit import OperatorBandit, ClusterBandit, PageHinkleyState
        return {
            'OperatorBandit': OperatorBandit,
            'ClusterBandit': ClusterBandit,
            'PageHinkleyState': PageHinkleyState,
        }[name]
    if name in {'RewardComputer', 'hypervolume', 'hvi',
                'shannon_diversity', 'cumulative_diversity'}:
        from .reward import (RewardComputer, hypervolume, hvi,
                             shannon_diversity, cumulative_diversity)
        return {
            'RewardComputer': RewardComputer,
            'hypervolume': hypervolume,
            'hvi': hvi,
            'shannon_diversity': shannon_diversity,
            'cumulative_diversity': cumulative_diversity,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
