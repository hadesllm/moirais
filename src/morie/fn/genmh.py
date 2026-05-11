# morie.fn — function file (hadesllm/morie)
"""
Genetic algorithm for global optimization.

Population-based evolutionary approach using selection, crossover, and mutation.
"""

import numpy as np

__all__ = ['genmh']


def genmh(f, bounds, pop_size=50, generations=100, pc=0.7, pm=0.1, full_output=False, seed=None):
    """
    Genetic algorithm for global optimization.

    Evolves a population of candidate solutions via selection, crossover, mutation.

    Parameters
    ----------
    f : callable
        Objective function f(x) to minimize.
    bounds : list of tuple
        [(x_min, x_max), ...] bounds for each dimension.
    pop_size : int, optional
        Population size (default 50).
    generations : int, optional
        Number of generations (default 100).
    pc : float, optional
        Crossover probability (default 0.7).
    pm : float, optional
        Mutation probability per gene (default 0.1).
    full_output : bool, optional
        If True, return (x_min, info_dict).
    seed : int, optional
        Random seed.

    Returns
    -------
    x_min : ndarray
        Estimated minimizer.
    info_dict : dict, optional
        Dictionary with keys: 'generations', 'converged', 'final_value'.

    References
    ----------
    Holland, J. H. (1975). Adaptation in Natural and Artificial Systems.
    University of Michigan Press.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import genmh
    >>> f = lambda x: (x[0] - 2)**2 + (x[1] - 3)**2
    >>> bounds = [(0, 5), (0, 5)]
    >>> x_min = genmh(f, bounds, pop_size=30, generations=50, seed=42)
    >>> np.allclose(x_min, [2, 3], atol=1.0)
    True
    """
    if seed is not None:
        np.random.seed(seed)

    n_vars = len(bounds)
    bounds = np.array(bounds)

    # Initialize population
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, n_vars))
    fitness = np.array([f(x) for x in pop])

    for gen in range(generations):
        # Selection: tournament
        selected = []
        for _ in range(pop_size):
            idx = np.random.choice(pop_size, 2, replace=False)
            if fitness[idx[0]] < fitness[idx[1]]:
                selected.append(pop[idx[0]].copy())
            else:
                selected.append(pop[idx[1]].copy())
        selected = np.array(selected)

        # Crossover
        offspring = []
        for i in range(0, pop_size - 1, 2):
            if np.random.rand() < pc:
                alpha = np.random.rand()
                child1 = alpha * selected[i] + (1 - alpha) * selected[i + 1]
                child2 = alpha * selected[i + 1] + (1 - alpha) * selected[i]
                offspring.extend([child1, child2])
            else:
                offspring.extend([selected[i].copy(), selected[i + 1].copy()])
        offspring = np.array(offspring[:pop_size])

        # Mutation
        for i in range(pop_size):
            mask = np.random.rand(n_vars) < pm
            offspring[i, mask] += np.random.normal(0, 0.1, np.sum(mask))
            offspring[i] = np.clip(offspring[i], bounds[:, 0], bounds[:, 1])

        # Elitism: keep best
        best_idx = np.argmin(fitness)
        offspring[0] = pop[best_idx].copy()

        pop = offspring
        fitness = np.array([f(x) for x in pop])

    best_idx = np.argmin(fitness)
    if full_output:
        return pop[best_idx], {
            'generations': generations,
            'converged': False,
            'final_value': fitness[best_idx]
        }
    return pop[best_idx]
