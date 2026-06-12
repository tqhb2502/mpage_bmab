# Evaluation Process Clarification

This note clarifies how generated heuristics are evaluated in the project. It is intended as a standalone explanation for understanding the experiments, not as thesis text.

## Are There Multiple Instances or Multiple Runs?

Both ideas are partly correct, but the evaluation is closer to the following:

For each problem type, there are multiple fixed problem instances. A generated heuristic is evaluated on all of them, and the final score is averaged across instances.

From the current evaluators:

- `Bi-TSP`: `n_instance = 4`
- `Tri-TSP`: `n_instance = 20`
- `Bi-CVRP`: `n_instance = 8`
- `Bi-KP`: `n_instance = 8`

The instances are generated with `np.random.seed(2025)`, so the benchmark instances are intended to be fixed and reproducible for a given problem type.

However, a heuristic does not simply produce one final solution per instance.

Instead, the heuristic is a `select_neighbor(...)` function. For each instance, the evaluator:

1. Creates an initial random archive of solutions.
2. Repeatedly calls the heuristic many times on the same instance.
3. Each call proposes a new candidate solution.
4. Feasible and non-dominated candidates are inserted into the archive.
5. After the loop finishes, HV is computed from the final archive.
6. The HV values are averaged across all instances.

For example:

- `Bi-TSP`: 100 random initial solutions, then 2,000 heuristic calls per instance.
- `Tri-TSP`: 100 random initial solutions, then 20,000 heuristic calls per instance.
- `Bi-CVRP`: 10 random initial solutions, then 6,000 heuristic calls per instance.
- `Bi-KP`: 20 random initial solutions, then 8,000 heuristic calls per instance.

So the heuristic is not evaluated as "one heuristic gives one solution for one instance." It is evaluated as "one heuristic defines a search operator that is repeatedly applied to build a Pareto archive for each instance."

There is also stochasticity. Even though the problem instances are fixed, the initial archives use Python `random`, and many generated heuristics may themselves use `random` or `np.random`. Therefore, applying the same heuristic again to the same instance can produce different archives and different HV values unless all random sources are explicitly seeded.

The clean summary is:

> Each problem type contains multiple fixed benchmark instances. A heuristic is evaluated across all instances, but on each instance it is repeatedly applied as a stochastic or deterministic neighborhood operator to build a Pareto archive. The final score is the mean final archive HV, plus runtime, over all instances.

## What Does "Repeatedly Applied as a Neighborhood Operator" Mean?

Think of the generated heuristic as a move rule, not as a complete solver.

For example, in Bi-TSP, a solution is one tour:

```text
[0, 5, 2, 8, ..., 1]
```

The heuristic function does not receive an empty problem and return the final best tour. Instead, it receives the current archive of existing tours and proposes one new candidate tour.

A simplified version of the evaluator is:

```text
archive = 100 random tours

repeat 2000 times:
    new_tour = heuristic(archive, instance, distance_matrices)

    if new_tour is valid:
        compute its objective values

        if it is not dominated by the archive:
            remove archive solutions dominated by new_tour
            add new_tour to archive

final_HV = hypervolume(archive)
```

So "repeatedly applied" means:

> The same heuristic function is called many times on the same problem instance. Each call proposes one new candidate solution. Over many calls, these candidates gradually build or improve the Pareto archive.

The Pareto archive is the set of currently useful trade-off solutions.

For Bi-TSP, each tour has two objective values, for example:

```text
tour A = (distance_objective_1 = 5.2, distance_objective_2 = 7.8)
tour B = (distance_objective_1 = 6.1, distance_objective_2 = 6.4)
tour C = (distance_objective_1 = 8.0, distance_objective_2 = 9.0)
```

Because this is minimization, smaller is better. If one tour is no worse in all objectives and better in at least one, it dominates another tour. Dominated tours are removed from the archive.

Therefore, the archive is not one best solution. It is a set of non-dominated solutions representing different trade-offs.

"Stochastic or deterministic neighborhood operator" means:

- Neighborhood operator: the heuristic usually modifies an existing solution from the archive to create a nearby new solution.
- Deterministic: given the same archive and instance, it always returns the same candidate.
- Stochastic: it uses randomness, so it may return different candidates even with the same archive and instance.

For example, a stochastic TSP heuristic might say:

```text
Choose a random tour from the archive.
Randomly swap two cities.
Return the modified tour.
```

A deterministic one might say:

```text
Choose the tour with the best first objective.
Swap the pair of cities that most reduces objective 1.
Return the modified tour.
```

In this project, the LLM generates this heuristic or move rule. The evaluator then tests how good that rule is by running it thousands of times and measuring the final archive HV.

The important distinction is:

> The heuristic is not judged by one solution it produces. It is judged by the quality of the Pareto archive produced after repeatedly using it as a search operator.
