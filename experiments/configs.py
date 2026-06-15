"""Experimental matrix from IDEA.md §4.

A *config* in this harness is a 4-tuple ``(ablation, task, budget, seed)``.
The runner enumerates the cartesian product and dispatches each cell to
``mpage_bmab.main``.

Suite presets
-------------

Suite presets are provided for common workflows:

* **smoke** — 2 ablations × 1 task × 1 budget × 1 seed = 2 runs.
  Tiny budget; used to confirm the pipeline end-to-end with the cheapest
  possible spend before kicking off an expensive sweep.
* **headline** — 4 ablations × 1 task × 4 budgets × 3 seeds = 48 runs.
  This is the experiment that produces the AUBC table in the thesis chapter.
* **full** — 4 ablations × 4 tasks × 4 budgets × 5 seeds = 320 runs.
  Full coverage; only run after you've validated the pipeline.
* **hvfix_smoke / hv_final_priority / hv_final_full** — follow-up suites for
  the final-HV fixes and reward-mode ablations.

You can compose your own suite with command-line flags to ``run.py``.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple


# --------------------------------------------------------------------------- #
# Method ablations (must match the keys of ABLATIONS in mpage_bmab/main.py)
# --------------------------------------------------------------------------- #

# The four BMAB-LLM ablations referenced in IDEA.md §4.3 plus a degenerate
# control (mpage_budget) that disables both bandits and the diversity term.
# Useful when you want a self-contained "MPaGE-budget" baseline without
# leaving the standalone project to run the parent llm4ad/eoh.py.
ABLATIONS: List[str] = [
    'full',          # the proposed system (operator + cluster bandit + PH + diversity)
    'no_ph',         # ablate Page-Hinkley drift detection
    'no_diversity',  # ablate ΔCDI term in the reward
    'op_only',       # ablate the cluster-level bandit
    # 'cluster_only' and 'mpage_budget' are available but not in the headline
    # comparison.  Add them explicitly via --ablations on the CLI if you want.
]

ALL_ABLATIONS: List[str] = [
    'full', 'dense_reward', 'hybrid_reward', 'no_budget_anneal',
    'no_ph', 'no_diversity', 'op_only',
    'cluster_only', 'mpage_budget',
    # `mpage_orig` dispatches to mpage_bmab.mpage_orig (the actual MPaGE class
    # capped at B sample calls) instead of mpage_bmab.main. The runner picks
    # the right entry-point automatically.
    'mpage_orig',
]

# Ablations that run through ``mpage_bmab.mpage_orig`` instead of the standard
# ``mpage_bmab.main`` entry point. These exist so we can compare BMAB-LLM
# against the actual upstream MPaGE class under the same budget, matching the
# "MPaGE-budget" / "MPaGE-orig" baselines in IDEA.md §4.3.
MPAGE_ORIG_ABLATIONS: List[str] = ['mpage_orig']


# --------------------------------------------------------------------------- #
# Tasks, budgets, seeds — straight from IDEA.md §4.1, §4.4, §4.5.
# --------------------------------------------------------------------------- #

TASKS: List[str] = ['bi_tsp', 'tri_tsp', 'bi_cvrp', 'bi_kp']

BUDGETS: List[int] = [25, 50, 100, 200]

# Five seeds — fixed so reruns are reproducible and aggregation merges cleanly.
SEEDS: List[int] = [2025, 2026, 2027, 2028, 2029]


# --------------------------------------------------------------------------- #
# Suite presets
# --------------------------------------------------------------------------- #

class Suite(NamedTuple):
    name: str
    ablations: List[str]
    tasks: List[str]
    budgets: List[int]
    seeds: List[int]
    description: str

    @property
    def n_runs(self) -> int:
        return (len(self.ablations) * len(self.tasks)
                * len(self.budgets) * len(self.seeds))


SUITES: Dict[str, Suite] = {
    'smoke': Suite(
        name='smoke',
        ablations=['full', 'mpage_budget'],
        tasks=['bi_tsp'],
        budgets=[15],
        seeds=[2025],
        description='2 cheap runs; confirms the pipeline end-to-end.',
    ),
    'headline': Suite(
        name='headline',
        ablations=['full', 'no_ph', 'no_diversity', 'op_only'],
        tasks=['bi_tsp'],
        budgets=[25, 50, 100, 200],
        seeds=[2025, 2026, 2027],
        description='4 ablations × bi_tsp × 4 budgets × 3 seeds.',
    ),
    'budget50': Suite(
        name='budget50',
        ablations=['full', 'no_ph', 'no_diversity', 'op_only'],
        tasks=TASKS,
        budgets=[50],
        seeds=SEEDS,
        description='B=50 (tightest interesting regime) across all 4 tasks, all seeds.',
    ),
    'full': Suite(
        name='full',
        ablations=['full', 'no_ph', 'no_diversity', 'op_only'],
        tasks=TASKS,
        budgets=BUDGETS,
        seeds=SEEDS,
        description='4 ablations × 4 tasks × 4 budgets × 5 seeds (320 runs).',
    ),

    # --- MPaGE-comparison suites (use the actual upstream MPaGE class) ---
    'mpage_smoke': Suite(
        name='mpage_smoke',
        ablations=['mpage_orig', 'full'],
        tasks=['bi_tsp'],
        budgets=[15],
        seeds=[2025],
        description='Cheapest end-to-end sanity check for the MPaGE-orig vs '
                    'BMAB-LLM comparison pipeline.',
    ),
    'mpage_compare': Suite(
        name='mpage_compare',
        ablations=['mpage_orig', 'full'],
        tasks=['bi_tsp'],
        budgets=[25, 50, 100, 200],
        seeds=[2025, 2026, 2027],
        description='Headline MPaGE-orig vs BMAB-LLM-full on bi_tsp, '
                    'all budgets, 3 seeds (24 runs). Drop-in equivalent to '
                    'the IDEA.md §4.3 "MPaGE-budget" comparison.',
    ),
    'mpage_compare_full': Suite(
        name='mpage_compare_full',
        ablations=['full', 'mpage_orig'],
        tasks=TASKS,
        budgets=BUDGETS,
        seeds=SEEDS,
        description='Full MPaGE-orig vs BMAB-LLM-full matrix '
                    '(2 × 4 × 4 × 5 = 160 runs). Expensive.',
    ),
    'hvfix_smoke': Suite(
        name='hvfix_smoke',
        ablations=['full', 'dense_reward', 'no_budget_anneal', 'mpage_orig'],
        tasks=['bi_kp'],
        budgets=[25],
        seeds=[2025],
        description='Cheap sanity check for the final-HV fixes on the weakest '
                    'previous cell: Bi-KP, B=25, one seed.',
    ),
    'hv_final_priority': Suite(
        name='hv_final_priority',
        ablations=['full', 'dense_reward', 'hybrid_reward',
                   'no_budget_anneal', 'mpage_orig'],
        tasks=['bi_tsp', 'bi_cvrp', 'bi_kp'],
        budgets=[25, 50, 100],
        seeds=SEEDS,
        description='Focused final-HV sweep over cells where terminal HV was '
                    'mixed or weak. 5 × 3 × 3 × 5 = 225 runs.',
    ),
    'hv_final_full': Suite(
        name='hv_final_full',
        ablations=[
            'full',
            'dense_reward',
            'hybrid_reward',
            #'no_budget_anneal',
            #'mpage_orig'
        ],
        tasks=TASKS,
        budgets=BUDGETS,
        seeds=SEEDS,
        description='Full reward-mode comparison for the finalized BMAB '
                    'implementation: full/final_hv, dense_reward/dense, and '
                    'hybrid_reward/hybrid across all tasks, budgets, and seeds '
                    '(3 × 4 × 4 × 5 = 240 runs). Add no_budget_anneal or '
                    'mpage_orig manually if needed.',
    ),
    'all_methods': Suite(
        name='all_methods',
        ablations=['mpage_orig', 'full', 'no_ph', 'no_diversity', 'op_only'],
        tasks=['bi_tsp'],
        budgets=[25, 50, 100, 200],
        seeds=[2025, 2026, 2027],
        description='MPaGE-orig + all 4 BMAB ablations on bi_tsp × all budgets × '
                    '3 seeds (60 runs). Fully populates an AUBC table.',
    ),
}


# --------------------------------------------------------------------------- #
# Per-task / per-budget pop_size.
# --------------------------------------------------------------------------- #

# Budget-driven baseline. Derived from OFFSPRING_FAQ.md §5.B's
# `pop_size ≈ 4 · K` rule with the LLM clusterer typically producing
# K = 2–3.  At small B the rule is violated (cluster bandit ends up
# prior-driven) but trying to enforce it would leave too few generations
# for PFG and the operator bandit to make progress.
BUDGET_POP_SIZE = [
    # (max_budget_excl, pop_size)
    (50,    4),    # B < 50 → 4
    (100,   6),    # 50  ≤ B < 100 → 6
    (200,   8),    # 100 ≤ B < 200 → 8
    (float('inf'), 10),    # 200 ≤ B → 10
]

# Per-task adjustment on top of the budget-driven baseline.
TASK_POP_BONUS: Dict[str, int] = {
    'bi_tsp':  0,
    'tri_tsp': 2,    # 3-D Pareto front needs more diversity
    'bi_cvrp': 0,
    'bi_kp':   0,
}


def pop_size_for(task: str, budget: int) -> int:
    """Recommended pop_size for a (task, budget) cell.

    See `documents/HYPERPARAMETERS.md` for the rationale. The baseline
    follows the `pop_size ≈ 4·K` rule from OFFSPRING_FAQ.md §5.B; tri-TSP
    gets a +2 bonus because its 3-D Pareto front is wider.
    """
    base = next(ps for max_b, ps in BUDGET_POP_SIZE if budget < max_b)
    return base + TASK_POP_BONUS.get(task, 0)


# Kept for backward compatibility. New code should call pop_size_for().
POP_SIZES: Dict[str, int] = {t: pop_size_for(t, 50) for t in TASKS}


def run_id(ablation: str, task: str, budget: int, seed: int) -> str:
    """Stable identifier used both as a directory name and as a sweep key."""
    return f"{ablation}__{task}__B{int(budget)}__seed{int(seed)}"
