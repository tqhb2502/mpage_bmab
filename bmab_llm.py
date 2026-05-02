"""BMAB-LLM: Budgeted Multi-Objective Multi-Armed Bandit driver for
LLM-based heuristic design.

This module is the ``MPaGE`` analogue: it owns the population, the budget,
the operator and cluster bandits, and the main evolutionary loop. Where MPaGE
runs four operators in fixed order and consumes generations until quota,
BMAB-LLM:

* picks the operator (Mutate/Crossover) via a persistent UCB1 bandit, and
* picks the parent cluster via a per-generation budgeted UCB1 bandit
  (re-initialised each time the elite set is re-clustered),
* monitors each (cluster, operator) arm with a Page-Hinkley test for drift,
* terminates when the LLM-call budget ``B`` is exhausted, regardless of
  generation count.
"""
from __future__ import annotations

import random
import time
import traceback
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._llm4ad.base import (
    Evaluation, LLM, Function, Program,
    SecureEvaluator, TextFunctionProgramConverter,
)
from ._llm4ad.method.LLMPFG.population import Population
from ._llm4ad.method.LLMPFG.prompt import EoHPrompt
from ._llm4ad.method.LLMPFG.sampler import EoHSampler

from .bandit import OperatorBandit, ClusterBandit, OPERATORS
from .budget import BudgetTracker
from .cluster_manager import ClusterManager
from .profiler import BMABProfiler
from .reward import (RewardComputer, cumulative_diversity, shannon_diversity,
                     _is_valid_score)


class BMABLLM:
    """Budgeted MAB-driven LLM heuristic-design framework."""

    def __init__(self,
                 llm: LLM,
                 llm_cluster: LLM,
                 evaluation: Evaluation,
                 *,
                 budget: float,
                 ref_point: Tuple[float, float] = (20.0, 60.0),
                 budget_mode: str = 'call',
                 pop_size: int = 6,
                 selection_num: int = 2,
                 max_generations: Optional[int] = None,
                 # operator alphabet usage (kept for back-compat with ablations)
                 use_e1_operator: bool = True,
                 use_e2_operator: bool = True,
                 use_m1_operator: bool = True,
                 use_m2_operator: bool = True,
                 # bandit hyperparams
                 c_explore_op: float = 1.0,
                 c_explore_cluster: float = 1.0,
                 gamma_budget: float = 0.5,
                 ph_delta: float = 0.005,
                 ph_threshold: float = 0.5,
                 disable_cluster_bandit: bool = False,
                 disable_operator_bandit: bool = False,
                 # reward weights
                 w_quality: float = 1.0,
                 w_diversity: float = 0.3,
                 w_rank: float = 0.2,
                 reward_penalty: float = 1.0,
                 # call costs
                 cluster_call_cost: float = 1.0,
                 generation_call_cost: float = 1.0,
                 init_call_cost: float = 1.0,
                 review_call_cost: float = 1.0,
                 llm_review: bool = False,
                 # warm-up
                 init_target_successes: Optional[int] = None,
                 init_max_calls: Optional[int] = None,
                 # plumbing
                 profiler: Optional[BMABProfiler] = None,
                 debug_mode: bool = False,
                 random_seed: Optional[int] = None,
                 **kwargs):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._function_to_evolve: Function = (
            TextFunctionProgramConverter.text_to_function(
                self._template_program_str))
        self._template_program: Program = (
            TextFunctionProgramConverter.text_to_program(
                self._template_program_str))

        # core components
        self._budget = BudgetTracker(total_budget=budget, mode=budget_mode)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode,
                                          **kwargs)
        llm.debug_mode = debug_mode
        self._sampler = EoHSampler(llm, self._template_program_str)
        self._cluster_sampler = EoHSampler(llm_cluster,
                                           self._template_program_str)
        self._cluster_mgr = ClusterManager(
            cluster_sampler=self._cluster_sampler,
            task_description=self._task_description_str,
            template_function=self._function_to_evolve,
            budget=self._budget,
            cluster_call_cost=cluster_call_cost,
        )

        self._operator_bandit = OperatorBandit(c_explore=c_explore_op)
        self._cluster_bandit = ClusterBandit(c_explore=c_explore_cluster,
                                             gamma_budget=gamma_budget,
                                             ph_delta=ph_delta,
                                             ph_threshold=ph_threshold)
        self._reward = RewardComputer(
            ref_point=ref_point,
            w_quality=w_quality,
            w_diversity=w_diversity,
            w_rank=w_rank,
            penalty=reward_penalty,
        )

        # population
        self._population = Population(pop_size=pop_size)

        # config
        self._pop_size = pop_size
        self._selection_num = selection_num
        self._max_generations = max_generations
        self._use_e1 = use_e1_operator
        self._use_e2 = use_e2_operator
        self._use_m1 = use_m1_operator
        self._use_m2 = use_m2_operator
        self._review = llm_review
        self._init_target_successes = init_target_successes or pop_size
        self._init_max_calls = init_max_calls or max(pop_size * 5, 25)
        self._gen_call_cost = generation_call_cost
        self._init_call_cost = init_call_cost
        self._review_call_cost = review_call_cost
        self._ref_point = np.asarray(ref_point, dtype=float)
        self._debug = debug_mode
        self._disable_cluster_bandit = disable_cluster_bandit
        self._disable_operator_bandit = disable_operator_bandit
        self._rr_idx = 0  # round-robin index used when operator bandit is off

        # profiler
        self._profiler = profiler
        if profiler is not None:
            try:
                profiler.record_parameters(llm, evaluation, self)
            except Exception:
                pass
            if hasattr(profiler, '_ref_point'):
                profiler._ref_point = self._ref_point

        self._tot_sample_nums = 0
        self._gen = 0

    # ============================================================ PUBLIC API

    def run(self) -> List[Function]:
        """Run the BMAB-LLM evolutionary loop until the budget is exhausted."""
        try:
            self._init_population()
            self._record_curve_point()
            while not self._budget.is_exhausted():
                if self._max_generations is not None and \
                        self._gen >= self._max_generations:
                    break
                self._evolve_one_generation()
                self._record_curve_point()
        except KeyboardInterrupt:
            pass
        finally:
            self._finish()
        return list(self._population.population)

    @property
    def population(self) -> Population:
        return self._population

    @property
    def budget(self) -> BudgetTracker:
        return self._budget

    # ============================================================ INTERNAL

    def _record_curve_point(self) -> None:
        if self._profiler is not None:
            try:
                self._profiler.record_curve_point(
                    self._budget.consumed, self._population)
            except Exception:
                pass

    # -------------------------------------------------------- initialisation

    def _init_population(self) -> None:
        """Adaptive warm-up: keep sampling I1 prompts until either (i) we have
        ``init_target_successes`` valid heuristics, or (ii) we issued
        ``init_max_calls`` calls."""
        attempts = 0
        successes = 0
        while (successes < self._init_target_successes
               and attempts < self._init_max_calls
               and not self._budget.is_exhausted()
               and self._budget.can_afford(self._init_call_cost)):
            ok = self._sample_eval_register(
                lambda: EoHPrompt.get_prompt_i1(
                    self._task_description_str, self._function_to_evolve),
                cost=self._init_call_cost,
                label='init',
            )
            attempts += 1
            if ok:
                successes += 1

    # -------------------------------------------------------- one generation

    def _evolve_one_generation(self) -> None:
        if len(self._population.population) == 0:
            return

        # 1. select an elite pool from PFG (re-using MPaGE's parent_selection)
        try:
            elites = self._population.selection(self._selection_num)
        except Exception:
            return
        # The PFG selection above returns ≤ 5 individuals. We expand to a wider
        # elite set for clustering: take *all* current population members so we
        # can cluster more of them, but bias the bandit's parent draws to the
        # PFG selection.
        full_elites = list(self._population.population)
        if len(full_elites) < 3:
            full_elites = full_elites or []

        # 2. cluster (decrements budget by 1 if a real call is issued)
        partition, cluster_quality = self._cluster_mgr.cluster(full_elites)
        if not partition:
            return
        n_clusters = len(partition)
        op_priors = self._operator_bandit.softmax_probs(temperature=1.0)
        self._cluster_bandit.reset(
            n_clusters=n_clusters,
            cluster_quality=cluster_quality,
            operator_priors=op_priors,
        )

        # 3. inner loop: run a budget of ~pop_size offspring per generation
        target_offspring = max(1, self._pop_size)
        produced = 0
        while (produced < target_offspring
               and not self._budget.is_exhausted()
               and self._budget.can_afford(self._gen_call_cost)):

            # Decide operator first (persistent bandit), then cluster.
            if self._disable_operator_bandit:
                op = OPERATORS[self._rr_idx % len(OPERATORS)]
                self._rr_idx += 1
            else:
                op = self._operator_bandit.select()

            if self._disable_cluster_bandit:
                cluster_idx = random.randrange(n_clusters)
                arm = (cluster_idx, op)
            else:
                try:
                    arm = self._cluster_bandit.select(
                        budget_fraction=self._budget.fraction_remaining(),
                        restrict_operator=op,
                    )
                except Exception:
                    break
                cluster_idx, op_picked = arm

            # Build prompt + parents
            prompt_builder, parents_used = self._make_prompt(
                op=op, cluster_idx=cluster_idx,
                partition=partition, full_elites=full_elites,
            )
            if prompt_builder is None:
                # fallback -- consume one budget unit on a vanilla i1
                prompt_builder = (lambda: EoHPrompt.get_prompt_i1(
                    self._task_description_str, self._function_to_evolve))
                parents_used = []

            # Charge budget and call LLM
            cost = self._gen_call_cost
            if self._review and parents_used:
                # extra suggestion call
                if self._budget.can_afford(self._review_call_cost):
                    self._budget.charge(self._review_call_cost,
                                        label='suggestion')

            score_before = self._population_scores()
            div_before = cumulative_diversity(score_before)

            ok = self._sample_eval_register(
                prompt_builder, cost=cost,
                label=f'{op}#{cluster_idx}',
            )

            # Compute reward + update bandit
            new_score = None
            if ok and len(self._population.population) > 0:
                # The newly registered function is the most recent one in
                # _next_gen_pop or the population.
                new_score = self._latest_score()
            score_after = self._population_scores()
            div_after = cumulative_diversity(score_after)

            reward, breakdown = self._reward.reward(
                new_score=new_score,
                population_scores=score_before,
                diversity_before=div_before,
                diversity_after=div_after,
                invalid=(not ok),
            )

            if not self._disable_operator_bandit:
                self._operator_bandit.update(op, reward, cost=cost)
            if self._disable_cluster_bandit:
                drift = False
            else:
                drift = self._cluster_bandit.update(arm, reward, cost=cost)

            if self._debug:
                print(f"[BMAB] gen={self._gen} op={op} cluster={cluster_idx} "
                      f"reward={reward:.4f} cost={cost} drift={drift} "
                      f"budget={self._budget.remaining:.1f}/"
                      f"{self._budget.total:.1f}  breakdown={breakdown}")

            produced += 1

        # bandit / curve dumps for this generation
        if self._profiler is not None:
            try:
                self._profiler.record_bandit_state(
                    generation=self._gen,
                    operator_stats=self._operator_bandit.stats(),
                    cluster_stats=self._cluster_bandit.stats(),
                )
            except Exception:
                pass
        self._gen += 1

    # -------------------------------------------------------- sampling

    def _sample_eval_register(self, prompt_builder, *,
                              cost: float, label: str) -> bool:
        """Run one (sample → evaluate → register) cycle. Decrements the budget.
        Returns True iff a valid heuristic was registered."""
        if not self._budget.charge(cost, label=label):
            return False
        try:
            prompt = prompt_builder()
        except Exception:
            return False
        if prompt is None:
            return False

        sample_start = time.time()
        try:
            thought, func = self._sampler.get_thought_and_function(prompt)
        except Exception:
            if self._debug:
                traceback.print_exc()
            return False
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            return False

        try:
            program = TextFunctionProgramConverter.function_to_program(
                func, self._template_program)
        except Exception:
            return False
        if program is None:
            return False

        # evaluate (this may take a while). We do it inline (single thread).
        try:
            score, eval_time = self._evaluator.evaluate_program_record_time(
                program)
        except Exception:
            score, eval_time = None, 0.0

        if not _is_valid_score(score):
            return False

        func.score = list(score) if isinstance(score, tuple) else score
        func.evaluate_time = eval_time
        func.sample_time = sample_time
        func.algorithm = thought

        if self._profiler is not None:
            try:
                self._profiler.register_function(func)
                self._profiler.register_population(self._population)
            except Exception:
                pass

        self._tot_sample_nums += 1
        try:
            self._population.register_function(func)
        except Exception:
            pass
        return True

    # -------------------------------------------------------- prompt selection

    def _make_prompt(self, *, op: str, cluster_idx: int,
                     partition: List[List[int]],
                     full_elites: List[Function]):
        """Return a *callable* that produces a fresh prompt, plus the parent
        functions actually used (so we can also issue an optional review call).
        """
        if not full_elites:
            return None, []
        cluster = partition[cluster_idx] if cluster_idx < len(partition) else []
        in_cluster = [full_elites[i] for i in cluster
                      if i < len(full_elites)
                      and getattr(full_elites[i], 'algorithm', None)]
        if not in_cluster:
            return None, []

        if op == 'mutate':
            parent = random.choice(in_cluster)
            use_m2 = self._use_m2 and (
                random.random() < 0.5 or not self._use_m1)
            if use_m2:
                return (lambda: EoHPrompt.get_prompt_m2(
                    self._task_description_str, parent,
                    self._function_to_evolve)), [parent]
            return (lambda: EoHPrompt.get_prompt_m1(
                self._task_description_str, parent,
                self._function_to_evolve)), [parent]

        # crossover
        other_clusters = [i for i in range(len(partition)) if i != cluster_idx]
        other_indivs: List[Function] = []
        for j in other_clusters:
            for idx in partition[j]:
                if idx < len(full_elites):
                    f = full_elites[idx]
                    if getattr(f, 'algorithm', None) is not None:
                        other_indivs.append(f)
        if not other_indivs:
            other_indivs = [f for f in full_elites
                            if f is not in_cluster[0]
                            and getattr(f, 'algorithm', None) is not None]
        if not other_indivs:
            return None, []
        p1 = random.choice(in_cluster)
        p2 = random.choice(other_indivs)
        parents = [p1, p2]
        suggestions = None
        if self._review:
            try:
                suggestion_prompt = EoHPrompt.get_prompt_suggestions_only(
                    self._task_description_str, parents,
                    self._function_to_evolve)
                suggestions = self._sampler.get_thought(suggestion_prompt)
            except Exception:
                suggestions = None
        if self._use_e2 and (random.random() < 0.5 or not self._use_e1):
            return (lambda s=suggestions, par=parents:
                    EoHPrompt.get_prompt_e2(self._task_description_str, par,
                                            self._function_to_evolve, s)), parents
        return (lambda s=suggestions, par=parents:
                EoHPrompt.get_prompt_e1(self._task_description_str, par,
                                        self._function_to_evolve, s)), parents

    # -------------------------------------------------------- helpers

    def _population_scores(self):
        out = []
        for f in self._population.population:
            if f.score is not None and _is_valid_score(f.score):
                out.append(list(f.score))
        return out

    def _latest_score(self):
        if not self._population.population:
            return None
        f = self._population.population[-1]
        return list(f.score) if f.score is not None else None

    def _finish(self):
        if self._profiler is not None:
            try:
                self._profiler.record_budget(self._budget)
                self._profiler.finish(budget=self._budget)
            except TypeError:
                self._profiler.finish()
