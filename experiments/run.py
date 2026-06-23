"""Run BMAB-LLM experiments — single, sweep, or named suite.

Examples
--------

    # Cheapest possible end-to-end check (2 runs, B=15)
    python -m mpage_bmab.experiments.run --suite smoke

    # The headline AUBC table (48 runs, ~all on bi_tsp)
    python -m mpage_bmab.experiments.run --suite headline

    # Custom slice
    python -m mpage_bmab.experiments.run \\
        --ablations full,op_only \\
        --tasks bi_tsp,bi_kp \\
        --budgets 50,100 \\
        --seeds 2025,2026

    # Just dry-run to count and list the experiments
    python -m mpage_bmab.experiments.run --suite full --dry_run

Each cell of the sweep becomes a subprocess invocation of
``mpage_bmab.main`` with the ablation preset applied. Output for run
``(ablation, task, budget, seed)`` lives at::

    <results_root>/<ablation>/<task>/B<budget>/seed<seed>/

The runner is **idempotent**: a cell is skipped when an ``aubc.json`` file
already exists at its destination (override with ``--force``).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from glob import glob
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_PKG_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mpage_bmab.experiments.configs import (
    ALL_ABLATIONS, BUDGETS, MPAGE_ORIG_ABLATIONS, POP_SIZES, SEEDS, SUITES,
    TASKS, pop_size_for, run_id,
)


def _csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(',') if x.strip()]


def _csv_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def _output_dir(results_root: str, ablation: str, task: str,
                budget: int, seed: int) -> str:
    return os.path.join(results_root, ablation, task,
                        f"B{int(budget)}", f"seed{int(seed)}")


def _already_done(out_dir: str) -> bool:
    """A run is considered done when at least one timestamped subdir contains
    an ``aubc.json`` artefact."""
    matches = glob(os.path.join(out_dir, '*', 'aubc.json'))
    return bool(matches)


def _build_cmd(*, python: str, ablation: str, task: str, budget: int,
               seed: int, log_dir: str, secret: str, secret_cluster: str,
               llm_model: str, llm_cluster_model: str,
               openai_base_url: str, pop_size: int,
               extra: Optional[List[str]] = None) -> List[str]:
    """Build the sub-process command for one cell.

    `mpage_orig`-family ablations dispatch to ``mpage_bmab.mpage_orig`` (the
    actual upstream MPaGE class capped at B sample calls). All other
    ablations dispatch to ``mpage_bmab.main`` and use its ``--ablation`` preset.
    """
    if ablation in MPAGE_ORIG_ABLATIONS:
        method_name = 'MPaGE-orig' if ablation == 'mpage_orig' else f"MPaGE-{ablation}"
        cmd = [
            python, '-m', 'mpage_bmab.mpage_orig',
            '--task', task,
            '--budget', str(budget),
            '--seed', str(seed),
            '--pop_size', str(pop_size),
            '--log_dir', log_dir,
            '--secret', secret,
            '--secret_cluster', secret_cluster,
            '--llm_model', llm_model,
            '--llm_cluster_model', llm_cluster_model,
            '--openai_base_url', openai_base_url,
            '--method_name', method_name,
        ]
    else:
        cmd = [
            python, '-m', 'mpage_bmab.main',
            '--ablation', ablation,
            '--task', task,
            '--budget', str(budget),
            '--seed', str(seed),
            '--pop_size', str(pop_size),
            '--log_dir', log_dir,
            '--secret', secret,
            '--secret_cluster', secret_cluster,
            '--llm_model', llm_model,
            '--llm_cluster_model', llm_cluster_model,
            '--openai_base_url', openai_base_url,
        ]
    if extra:
        cmd.extend(extra)
    return cmd


def _resolve_suite(args) -> tuple:
    if args.suite:
        s = SUITES[args.suite]
        return s.ablations, s.tasks, s.budgets, s.seeds
    return (
        _csv_list(args.ablations) if args.ablations else ['full'],
        _csv_list(args.tasks)     if args.tasks     else TASKS,
        _csv_int_list(args.budgets) if args.budgets else BUDGETS,
        _csv_int_list(args.seeds)   if args.seeds   else SEEDS,
    )


def _print_plan(cells, results_root, force):
    print(f"\nPlanned runs: {len(cells)}")
    print(f"Results root : {results_root}")
    print(f"Force-rerun  : {force}\n")
    for i, (abl, task, b, seed) in enumerate(cells, 1):
        print(f"  [{i:>3}/{len(cells)}] {run_id(abl, task, b, seed)}")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="BMAB-LLM experiment sweep")
    p.add_argument('--suite', default=None, choices=list(SUITES.keys()),
                   help="Named suite (overrides --ablations/--tasks/...).")
    p.add_argument('--ablations', default=None,
                   help=f"CSV subset of internal ablation keys {ALL_ABLATIONS}. "
                        "Default: Final-HV reward.")
    p.add_argument('--tasks', default=None,
                   help=f"CSV subset of {TASKS}. Default: all four.")
    p.add_argument('--budgets', default=None,
                   help=f"CSV ints. Default: {BUDGETS}")
    p.add_argument('--seeds', default=None,
                   help=f"CSV ints. Default: {SEEDS}")

    p.add_argument('--results_root',
                   default=os.path.join(_PKG_ROOT, 'experiments', 'results'),
                   help="Where per-run outputs are written.")
    p.add_argument('--secret', default='secret.txt')
    p.add_argument('--secret_cluster', default='secret_cluster.txt')
    p.add_argument('--llm_model', default='gpt-4o-mini')
    p.add_argument('--llm_cluster_model', default='gpt-4o-mini')
    p.add_argument('--openai_base_url', default='https://api.openai.com')
    p.add_argument('--python', default=sys.executable,
                   help="Python interpreter to use for sub-processes.")
    p.add_argument('--force', action='store_true',
                   help="Re-run even if aubc.json already exists.")
    p.add_argument('--dry_run', action='store_true',
                   help="Print the plan, do not launch any sub-processes.")
    p.add_argument('--extra', default='',
                   help="Extra args appended verbatim to every sub-call "
                        "(e.g. '--debug --review').")
    args = p.parse_args(argv)

    ablations, tasks, budgets, seeds = _resolve_suite(args)

    # Validate
    bad = [a for a in ablations if a not in ALL_ABLATIONS]
    if bad:
        print(f"[run] Unknown ablation(s): {bad}. Choose from {ALL_ABLATIONS}",
              file=sys.stderr)
        return 2

    cells = [(a, t, b, s) for a in ablations for t in tasks
             for b in budgets for s in seeds]

    _print_plan(cells, args.results_root, args.force)
    if args.dry_run:
        return 0

    # Sanity-check the secrets up front so the user does not start a 3-hour
    # sweep only to fail on the first call.
    for path in (args.secret, args.secret_cluster):
        if not os.path.isfile(path):
            print(f"[run] Missing API-key file: {path}", file=sys.stderr)
            return 2

    extra_args = args.extra.split() if args.extra else []
    failures: List[str] = []
    skipped = 0
    t0 = time.time()
    for i, (abl, task, budget, seed) in enumerate(cells, 1):
        out_dir = _output_dir(args.results_root, abl, task, budget, seed)
        rid = run_id(abl, task, budget, seed)
        if not args.force and _already_done(out_dir):
            print(f"[{i:>3}/{len(cells)}] SKIP  {rid} (aubc.json present)")
            skipped += 1
            continue
        os.makedirs(out_dir, exist_ok=True)
        cmd = _build_cmd(
            python=args.python,
            ablation=abl, task=task, budget=budget, seed=seed,
            log_dir=out_dir,
            secret=args.secret, secret_cluster=args.secret_cluster,
            llm_model=args.llm_model,
            llm_cluster_model=args.llm_cluster_model,
            openai_base_url=args.openai_base_url,
            pop_size=pop_size_for(task, budget),
            extra=extra_args,
        )
        print(f"[{i:>3}/{len(cells)}] RUN   {rid}")
        rc = subprocess.call(cmd, cwd=_PROJECT_ROOT)
        if rc != 0:
            print(f"           ! exit-code {rc}", file=sys.stderr)
            failures.append(rid)

    elapsed = time.time() - t0
    print(f"\n[run] Done. {len(cells) - skipped - len(failures)} ok, "
          f"{skipped} skipped, {len(failures)} failed in {elapsed:.1f}s.")
    if failures:
        print("Failed cells:")
        for r in failures:
            print(f"  {r}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
