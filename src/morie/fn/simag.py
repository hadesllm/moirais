"""
Simulated annealing for global optimization.

Probabilistically accepts uphill moves to escape local minima.
"""

import numpy as np

__all__ = ['simag']


def simag(f, x0, bounds, T_init=1.0, cooling_rate=0.95, max_iter=10000, full_output=False, seed=None):
    """
    Simulated annealing for global optimization.

    Mimics physical annealing process: accept moves with probability
    p = exp(-delta_f / T). Temperature decreases gradually.

    Parameters
    ----------
    f : callable
        Objective function f(x) to minimize.
    x0 : ndarray
        Initial point.
    bounds : list of tuple
        [(x_min, x_max), ...] bounds for each dimension.
    T_init : float, optional
        Initial temperature (default 1.0).
    cooling_rate : float, optional
        Temperature decay rate (default 0.95).
    max_iter : int, optional
        Maximum iterations (default 10000).
    full_output : bool, optional
        If True, return (x_min, info_dict).
    seed : int, optional
        Random seed.

    Returns
    -------
    x_min : ndarray
        Estimated minimizer.
    info_dict : dict, optional
        Dictionary with keys: 'iterations', 'converged', 'final_value'.

    References
    ----------
    Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by
    simulated annealing. Science, 220(4598), 671-680.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import simag
    >>> f = lambda x: (x[0] - 2)**2 + (x[1] - 3)**2
    >>> bounds = [(0, 5), (0, 5)]
    >>> x_min = simag(f, np.array([0.0, 0.0]), bounds, seed=42)
    >>> np.allclose(x_min, [2, 3], atol=0.5)
    True
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.atleast_1d(x0).astype(float)
    f_x = f(x)
    x_best = x.copy()
    f_best = f_x
    T = T_init

    for iteration in range(max_iter):
        # Random neighbor (Gaussian perturbation)
        x_new = x + np.random.normal(0, 1, len(x))

        # Apply bounds
        for i, (lo, hi) in enumerate(bounds):
            x_new[i] = np.clip(x_new[i], lo, hi)

        f_new = f(x_new)
        delta_f = f_new - f_x

        # Metropolis criterion
        if delta_f < 0 or np.random.rand() < np.exp(-delta_f / (T + 1e-14)):
            x = x_new
            f_x = f_new

            if f_x < f_best:
                x_best = x.copy()
                f_best = f_x

        T *= cooling_rate

    if full_output:
        return x_best, {
            'iterations': max_iter,
            'converged': False,
            'final_value': f_best
        }
    return x_best
