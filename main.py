"""Entry-point for running BMAB-LLM on a chosen MOCOP benchmark.

Usage
-----

From the project root::

    python -m mpage_bmab.main --task bi_tsp --budget 50

API keys are read from ``secret.txt`` (heuristic-generation LLM) and
``secret_cluster.txt`` (cluster LLM), exactly the same convention as MPaGE.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make project root importable when run via `python mpage_bmab/main.py`
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mpage_bmab import BMABLLM, BMABProfiler
from mpage_bmab._llm4ad.tools.llm.llm_api_openai import HttpsApiOpenAI
from mpage_bmab._llm4ad.tools.llm.llm_api_openai_cluster import HttpsApiOpenAI4Cluster


_TASKS = {
    'bi_tsp':   ('mpage_bmab._llm4ad.task.optimization.bi_tsp_semo',
                 'BITSPEvaluation',  (20.0, 60.0)),
    'tri_tsp':  ('mpage_bmab._llm4ad.task.optimization.tri_tsp_semo',
                 # 2-D ref because TRITSPEvaluation.evaluate() returns
                 # (−HV_of_3D_solutions_avg, runtime_avg) — the 3-D structure
                 # is collapsed into a scalar HV at the inner SEMO level.
                 'TRITSPEvaluation', (20.0, 60.0)),
    'bi_cvrp':  ('mpage_bmab._llm4ad.task.optimization.bi_cvrp',
                 'BICVRPEvaluation', (40.0, 60.0)),
    'bi_kp':    ('mpage_bmab._llm4ad.task.optimization.bi_kp',
                 'BIKPEvaluation',   (0.0, 60.0)),
}

# Named ablation presets matching IDEA.md §4.3.
# Each value is a dict of overrides applied on top of the parsed CLI args.
ABLATIONS = {
    'full':         {},
    'no_ph':        {'ph_threshold': 1e9},
    'no_diversity': {'w_diversity': 0.0},
    'op_only':      {'disable_cluster_bandit': True},
    'cluster_only': {'disable_operator_bandit': True},
    'mpage_budget': {'disable_cluster_bandit': True,
                     'disable_operator_bandit': True,
                     'ph_threshold': 1e9,
                     'w_diversity': 0.0,
                     'w_rank': 0.0},
}


def _load_evaluation(task_name: str):
    if task_name not in _TASKS:
        raise ValueError(f"Unknown task '{task_name}'. "
                         f"Available: {list(_TASKS.keys())}")
    module_name, class_name, default_ref = _TASKS[task_name]
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(), default_ref


def _read_key(path: str) -> str:
    with open(path, 'r') as f:
        return f.readline().strip()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run BMAB-LLM on a MOCOP benchmark")
    p.add_argument('--task', default='bi_tsp', choices=list(_TASKS.keys()))
    p.add_argument('--budget', type=float, default=50.0,
                   help="Total LLM-call budget B (e.g. 25, 50, 100, 200).")
    p.add_argument('--budget_mode', default='call', choices=['call', 'token'])
    p.add_argument('--pop_size', type=int, default=6)
    p.add_argument('--max_generations', type=int, default=None,
                   help="Optional cap on number of generations.")
    p.add_argument('--llm_model', default='gpt-4o-mini')
    p.add_argument('--llm_cluster_model', default='gpt-4o-mini')
    p.add_argument('--openai_base_url', default='https://api.openai.com')
    p.add_argument('--secret', default='secret.txt')
    p.add_argument('--secret_cluster', default='secret_cluster.txt')
    p.add_argument('--log_dir', default='logs_bmab')
    p.add_argument('--c_op', type=float, default=1.0)
    p.add_argument('--c_cluster', type=float, default=1.0)
    p.add_argument('--gamma_budget', type=float, default=0.5)
    p.add_argument('--w_quality', type=float, default=1.0)
    p.add_argument('--w_diversity', type=float, default=0.3)
    p.add_argument('--w_rank', type=float, default=0.2)
    p.add_argument('--ph_delta', type=float, default=0.005)
    p.add_argument('--ph_threshold', type=float, default=0.5)
    p.add_argument('--disable_cluster_bandit', action='store_true',
                   help="Sample clusters uniformly at random (op_only ablation).")
    p.add_argument('--disable_operator_bandit', action='store_true',
                   help="Round-robin operators instead of UCB1 (ablation).")
    p.add_argument('--ablation', default='full', choices=list(ABLATIONS.keys()),
                   help="Named ablation preset (see IDEA.md §4.3).")
    p.add_argument('--method_name', default=None,
                   help="Override profiler method tag. Defaults to BMAB-<ABLATION>.")
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--review', action='store_true',
                   help="Enable LLM-based reflection / suggestion call.")
    args = p.parse_args(argv)

    # Apply ablation preset on top of parsed args. Explicit CLI flags always
    # win — applying the preset only sets fields that the preset declares.
    preset = ABLATIONS[args.ablation]
    for key, value in preset.items():
        # only apply when user did not override on the CLI
        # (argparse stores None defaults for unset args; but most have
        # numeric defaults, so we apply unconditionally for preset keys)
        setattr(args, key, value)

    api_key = _read_key(args.secret)
    cluster_key = _read_key(args.secret_cluster)

    llm = HttpsApiOpenAI(base_url=args.openai_base_url, api_key=api_key,
                         model=args.llm_model, timeout=30)
    llm_cluster = HttpsApiOpenAI4Cluster(base_url=args.openai_base_url,
                                         api_key=cluster_key,
                                         model=args.llm_cluster_model,
                                         timeout=30)

    task, default_ref = _load_evaluation(args.task)

    method_name = args.method_name or f"BMAB-{args.ablation}"
    profiler = BMABProfiler(log_dir=args.log_dir,
                            evaluation_name=task.__class__.__name__,
                            method_name=method_name,
                            ref_point=default_ref,
                            log_style='complex')

    bmab = BMABLLM(
        llm=llm,
        llm_cluster=llm_cluster,
        evaluation=task,
        budget=args.budget,
        budget_mode=args.budget_mode,
        ref_point=default_ref,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        c_explore_op=args.c_op,
        c_explore_cluster=args.c_cluster,
        gamma_budget=args.gamma_budget,
        w_quality=args.w_quality,
        w_diversity=args.w_diversity,
        w_rank=args.w_rank,
        ph_delta=args.ph_delta,
        ph_threshold=args.ph_threshold,
        disable_cluster_bandit=args.disable_cluster_bandit,
        disable_operator_bandit=args.disable_operator_bandit,
        random_seed=args.seed,
        debug_mode=args.debug,
        llm_review=args.review,
        profiler=profiler,
    )
    bmab.run()
    print(f"\n[BMAB] Done. ablation={args.ablation} task={args.task} "
          f"budget={args.budget} seed={args.seed} "
          f"AUBC = {profiler.aubc(args.budget):.6f}")


if __name__ == '__main__':
    main()
