"""BMAB-LLM: Budgeted Multi-Objective Multi-Armed Bandit driver for
LLM-based heuristic design on Multi-Objective Combinatorial Optimisation.

A refined extension of MPaGE that explicitly optimises a fixed budget ``B`` of
LLM calls. See ``IDEA.md`` for the full description.
"""
from .bmab_llm import BMABLLM
from .budget import BudgetTracker
from .bandit import OperatorBandit, ClusterBandit, PageHinkleyState
from .reward import RewardComputer, hypervolume, hvi, shannon_diversity, cumulative_diversity
from .profiler import BMABProfiler
from .cluster_manager import ClusterManager

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
