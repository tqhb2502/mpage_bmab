"""Experimental matrix from IDEA.md §4.

A *config* in this harness is a 4-tuple ``(ablation, task, budget, seed)``.
The runner enumerates the cartesian product and dispatches each cell to
``mpage_bmab.main``.

Suite presets
-------------

Three ready-made suites are provided for common workflows:

* **smoke** — 2 ablations × 1 task × 1 budget × 1 seed = 2 runs.
  Tiny budget; used to confirm the pipeline end-to-end with the cheapest
  possible spend before kicking off an expensive sweep.
* **headline** — 4 ablations × 1 task × 4 budgets × 3 seeds = 48 runs.
  This is the experiment that produces the AUBC table in the thesis chapter.
* **full** — 4 ablations × 4 tasks × 4 budgets × 5 seeds = 320 runs.
  Full coverage; only run after you've validated the pipeline.

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
    'full', 'no_ph', 'no_diversity', 'op_only',
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
        ablations=['mpage_orig', 'full'],
        tasks=TASKS,
        budgets=BUDGETS,
        seeds=SEEDS,
        description='Full MPaGE-orig vs BMAB-LLM-full matrix '
                    '(2 × 4 × 4 × 5 = 160 runs). Expensive.',
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
# Per-task pop_size / max_generations overrides.
# --------------------------------------------------------------------------- #

# A small population keeps init from eating the whole budget at small B.
# Pop_size 6 matches the README default; we raise it for tri-objective TSP
# because the elite Pareto-front is intrinsically wider in 3-D objective space.
POP_SIZES: Dict[str, int] = {
    'bi_tsp':  6,
    'tri_tsp': 8,
    'bi_cvrp': 6,
    'bi_kp':   6,
}


def run_id(ablation: str, task: str, budget: int, seed: int) -> str:
    """Stable identifier used both as a directory name and as a sweep key."""
    return f"{ablation}__{task}__B{int(budget)}__seed{int(seed)}"
