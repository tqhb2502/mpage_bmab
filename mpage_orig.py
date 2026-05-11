"""Runner for the **original** MPaGE under an LLM-call budget.

This module wraps the vendored ``llm4ad.method.LLMPFG.eoh.MPaGE`` class with a
thin subclass that

1. counts **every** LLM call (heuristic generation + clustering) toward
   ``_tot_sample_nums``, matching BMAB-LLM's accounting scheme, and
2. records a ``(consumed_budget, hv, pareto_size)`` curve point after every
   call so we can compute AUBC the same way as BMAB-LLM.

The result is an apples-to-apples baseline for IDEA.md §4.3's
"MPaGE-budget" comparison: same total LLM-call budget B, same task, same
seed, no BMAB adaptive components.

Usage
-----

From the project root that contains ``mpage_bmab/``::

    mpage_bmab/.venv/bin/python -m mpage_bmab.mpage_orig \\
        --task bi_tsp --budget 50 --seed 2025

Output goes to ``--log_dir`` (default ``logs_mpage_orig``); the artefacts
``budget_curve.json``, ``aubc.json``, ``budget_history.json`` are written so
that the existing ``experiments/aggregate.py`` picks the runs up alongside
BMAB-LLM runs without modification.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import traceback

import numpy as np

# Make project root importable when run via `python mpage_bmab/mpage_orig.py`
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mpage_bmab._llm4ad.base import TextFunctionProgramConverter
from mpage_bmab._llm4ad.method.LLMPFG.eoh import MPaGE
from mpage_bmab._llm4ad.method.LLMPFG.profiler import EoHProfiler
from mpage_bmab._llm4ad.method.LLMPFG.prompt import EoHPrompt
from mpage_bmab._llm4ad.tools.llm.llm_api_openai import HttpsApiOpenAI
from mpage_bmab._llm4ad.tools.llm.llm_api_openai_cluster import HttpsApiOpenAI4Cluster

from mpage_bmab.budget import BudgetTracker
from mpage_bmab.main import _TASKS, _load_evaluation, _read_key
from mpage_bmab.profiler import BMABProfiler


# A very large generations cap so the budget is the real stop criterion.
# (See `MPaGE._thread_do_evolutionary_operator.continue_loop` — it requires
# BOTH max_generations and max_sample_nums to be set; if either is None the
# loop exits immediately because the corresponding continue flag is False.)
_HUGE_GENERATIONS = 10_000_000


class MPaGEBudget(MPaGE):
    """`MPaGE` that counts every LLM call (cluster + heuristic) toward the
    budget and records a budget-vs-HV curve compatible with `BMABProfiler`.

    Three differences from the parent class:

    * `_sample_evaluate_register` increments ``_tot_sample_nums`` **before** the
      LLM call so even invalid heuristics consume budget — same as BMAB-LLM.
    * The `_cluster_sampler.get_thought` method is monkey-patched in `__init__`
      to increment the same counter when a cluster call is issued.
    * `run()` constructs a `BudgetTracker` mirroring the final consumption and
      passes it to `BMABProfiler.finish(budget=...)` so `aubc.json` is written.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Patch the cluster sampler so its calls also count toward
        # _tot_sample_nums, matching BMAB-LLM's accounting.
        orig_cluster_thought = self._cluster_sampler.get_thought

        def cluster_thought_counted(prompt):
            if self._tot_sample_nums >= self._max_sample_nums:
                return None
            self._tot_sample_nums += 1
            self._maybe_record_curve()
            try:
                return orig_cluster_thought(prompt)
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                return None

        self._cluster_sampler.get_thought = cluster_thought_counted

    # --------------------------------------------------------------- helpers

    def _maybe_record_curve(self) -> None:
        prof = self._profiler
        if prof is not None and hasattr(prof, 'record_curve_point'):
            try:
                prof.record_curve_point(self._tot_sample_nums, self._population)
            except Exception:
                pass

    # ----------------------------------------------------- core override

    def _sample_evaluate_register(self, prompt):
        """Same as parent, but counts EVERY attempt against the budget (incl.
        invalid samples) and records a curve point before and after."""
        if self._tot_sample_nums >= self._max_sample_nums:
            return

        # Charge the attempt up front.
        self._tot_sample_nums += 1
        self._maybe_record_curve()

        sample_start = time.time()
        try:
            thought, func = self._sampler.get_thought_and_function(prompt)
        except Exception:
            if self._debug_mode:
                traceback.print_exc()
            return
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            return

        try:
            program = TextFunctionProgramConverter.function_to_program(
                func, self._template_program)
        except Exception:
            return
        if program is None:
            return

        # Evaluate (subprocess pool from the parent).
        try:
            score, eval_time = self._evaluation_executor.submit(
                self._evaluator.evaluate_program_record_time,
                program,
            ).result()
        except Exception:
            score, eval_time = None, 0.0

        func.score = score
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time

        if self._profiler is not None:
            try:
                self._profiler.register_function(func)
                if isinstance(self._profiler, EoHProfiler):
                    self._profiler.register_population(self._population)
            except Exception:
                pass

        try:
            self._population.register_function(func)
        except Exception:
            pass

        self._maybe_record_curve()

    def _thread_init_population(self):
        """Init loop that also respects the budget — the parent's version only
        stops at ``_initial_sample_nums_max``, which can exceed B for tiny
        budgets."""
        while (self._population.generation == 0
               and self._tot_sample_nums < self._max_sample_nums):
            try:
                prompt = EoHPrompt.get_prompt_i1(self._task_description_str,
                                                 self._function_to_evolve)
                self._sample_evaluate_register(prompt)
                if self._tot_sample_nums > self._initial_sample_nums_max:
                    print(f"[mpage_orig] init not accomplished in "
                          f"{self._initial_sample_nums_max} samples")
                    break
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                continue

    # ---------------------------------------------------------- finalisation

    def run(self):
        try:
            if not self._resume_mode:
                self._init_population()
            self._do_sample()
        except KeyboardInterrupt:
            pass
        finally:
            self._finalise_profiler()

    def _finalise_profiler(self) -> None:
        prof = self._profiler
        if prof is None:
            return
        # Build a BudgetTracker reflecting the consumption so far so that
        # BMABProfiler.finish() can write aubc.json with the right numbers.
        total = float(self._max_sample_nums)
        consumed = float(min(self._tot_sample_nums, self._max_sample_nums))
        bt = BudgetTracker(total_budget=total, mode='call')
        if consumed > 0:
            bt.force_charge(consumed, label='mpage_orig_total')
        try:
            if hasattr(prof, 'record_budget'):
                prof.record_budget(bt)
            if hasattr(prof, 'finish'):
                try:
                    prof.finish(budget=bt)
                except TypeError:
                    prof.finish()
        except Exception:
            if self._debug_mode:
                traceback.print_exc()


# --------------------------------------------------------------------------- CLI


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Run the original MPaGE under an LLM-call budget.")
    p.add_argument('--task', default='bi_tsp', choices=list(_TASKS.keys()))
    p.add_argument('--budget', type=float, default=50.0,
                   help="LLM-call budget B (counts BOTH heuristic-gen and "
                        "cluster calls, matching BMAB-LLM accounting).")
    p.add_argument('--pop_size', type=int, default=5)
    p.add_argument('--llm_model', default='gpt-4o-mini')
    p.add_argument('--llm_cluster_model', default='gpt-4o-mini')
    p.add_argument('--openai_base_url', default='https://api.openai.com')
    p.add_argument('--secret', default='secret.txt')
    p.add_argument('--secret_cluster', default='secret_cluster.txt')
    p.add_argument('--log_dir', default='logs_mpage_orig')
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--review', action='store_true')
    p.add_argument('--method_name', default='MPaGE-orig',
                   help="Profiler method tag (use 'MPaGE-budget' if you want "
                        "the baseline labelled that way in summary.csv).")
    p.add_argument('--multi_thread_or_process', default='thread',
                   choices=['thread', 'process'])
    args = p.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    api_key = _read_key(args.secret)
    cluster_key = _read_key(args.secret_cluster)

    llm = HttpsApiOpenAI(base_url=args.openai_base_url, api_key=api_key,
                         model=args.llm_model, timeout=30)
    llm_cluster = HttpsApiOpenAI4Cluster(base_url=args.openai_base_url,
                                         api_key=cluster_key,
                                         model=args.llm_cluster_model,
                                         timeout=30)

    task, default_ref = _load_evaluation(args.task)

    profiler = BMABProfiler(log_dir=args.log_dir,
                            evaluation_name=task.__class__.__name__,
                            method_name=args.method_name,
                            ref_point=default_ref,
                            log_style='complex')

    mpage = MPaGEBudget(
        llm=llm,
        llm_cluster=llm_cluster,
        evaluation=task,
        profiler=profiler,
        max_generations=_HUGE_GENERATIONS,
        max_sample_nums=int(args.budget),
        pop_size=args.pop_size,
        debug_mode=args.debug,
        llm_review=args.review,
        multi_thread_or_process_eval=args.multi_thread_or_process,
    )
    mpage.run()

    used = mpage._tot_sample_nums
    print(f"\n[MPaGE-orig] Done. task={args.task} budget={args.budget} "
          f"seed={args.seed} samples_used={used} "
          f"AUBC = {profiler.aubc(args.budget):.6f}")


if __name__ == '__main__':
    main()
