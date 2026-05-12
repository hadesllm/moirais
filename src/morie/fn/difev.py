# morie.fn -- function file (hadesllm/morie)
"""
Differential evolution for global optimization.

Mutation strategy based on differences between population members.
"""

import numpy as np

__all__ = ['difev']


def difev(f, bounds, pop_size=50, generations=100, F=0.8, Cr=0.7, full_output=False, seed=None):
    """
    Differential evolution (DE) for global optimization.

    Mutation via weighted difference vectors: v = x_r1 + F * (x_r2 - x_r3).

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
    F : float, optional
        Differential weight (default 0.8).
    Cr : float, optional
        Crossover probability (default 0.7).
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
    Storn, R., & Price, K. (1997). Differential evolution - a simple and
    efficient heuristic for global optimization over continuous spaces.
    Journal of Global Optimization, 11(4), 341-359.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import difev
    >>> f = lambda x: (x[0] - 2)**2 + (x[1] - 3)**2
    >>> bounds = [(0, 5), (0, 5)]
    >>> x_min = difev(f, bounds, pop_size=30, generations=50, seed=42)
    >>> np.allclose(x_min, [2, 3], atol=0.5)
    True
    """
    if seed is not None:
        np.random.seed(seed)

    n_vars = len(bounds)
    bounds = np.array(bounds)

    # Initialize
    x = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, n_vars))
    fx = np.array([f(xi) for xi in x])

    for gen in range(generations):
        for i in range(pop_size):
            # Select three distinct random indices
            idxs = np.random.choice(pop_size, 3, replace=False)
            r1, r2, r3 = idxs

            # Mutation
            v = x[r1] + F * (x[r2] - x[r3])
            v = np.clip(v, bounds[:, 0], bounds[:, 1])

            # Crossover
            j_rand = np.random.randint(n_vars)
            u = x[i].copy()
            for j in range(n_vars):
                if np.random.rand() < Cr or j == j_rand:
                    u[j] = v[j]

            fu = f(u)
            if fu < fx[i]:
                x[i] = u
                fx[i] = fu

    best_idx = np.argmin(fx)
    if full_output:
        return x[best_idx], {
            'generations': generations,
            'converged': False,
            'final_value': fx[best_idx]
        }
    return x[best_idx]
