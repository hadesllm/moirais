# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""
Bayesian optimization using acquisition functions.

Uses Gaussian process surrogate with upper confidence bound or expected improvement.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

__all__ = ['bysop']


def bysop(f, bounds, n_init=10, n_iter=20, acq='ucb', kappa=2.576, full_output=False, seed=None, cdf=None):
    """
    Bayesian optimization for global optimization.

    Maintains Gaussian process surrogate; selects next point via acquisition function
    (upper confidence bound or expected improvement).

    Parameters
    ----------
    f : callable
        Objective function f(x) to minimize.
    bounds : list of tuple
        [(x_min, x_max), ...] bounds for each dimension.
    n_init : int, optional
        Number of initial design points (default 10).
    n_iter : int, optional
        Number of BO iterations (default 20).
    acq : str, optional
        Acquisition function: 'ucb' or 'ei' (default 'ucb').
    kappa : float, optional
        Exploration-exploitation trade-off for UCB (default 2.576).
    full_output : bool, optional
        If True, return (x_min, info_dict).
    seed : int, optional
        Random seed.

    Returns
    -------
    x_min : ndarray
        Estimated minimizer.
    info_dict : dict, optional
        Dictionary with keys: 'n_evals', 'converged', 'final_value'.

    References
    ----------
    Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian
    optimization of machine learning algorithms. In NIPS (pp. 2960-2968).

    Examples
    --------
    >>> import numpy as np
    >>> from moirais.fn import bysop
    >>> f = lambda x: (x[0] - 2)**2 + (x[1] - 3)**2
    >>> bounds = [(0, 5), (0, 5)]
    >>> x_min = bysop(f, bounds, n_init=5, n_iter=10, seed=42)
    >>> np.allclose(x_min, [2, 3], atol=1.0)
    True
    """
    if seed is not None:
        np.random.seed(seed)

    n_vars = len(bounds)
    bounds = np.array(bounds)

    # Initial design (Latin hypercube)
    X_init = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_init, n_vars))
    y_init = np.array([f(x) for x in X_init])

    X = X_init
    y = y_init

    for it in range(n_iter):
        # Simple GP: use kernel trick with RBF
        from scipy.spatial.distance import cdist
        K = np.exp(-cdist(X, X, 'sqeuclidean') / (2 * 1.0))
        K += np.eye(len(X)) * 1e-6

        # Solve for GP weights
        try:
            alpha = np.linalg.solve(K, y)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(K, y, rcond=None)[0]

        # Acquisition function
        def acq_func(x_):
            x_ = np.atleast_1d(x_)
            k = np.exp(-cdist(x_.reshape(1, -1), X, 'sqeuclidean') / (2 * 1.0)).ravel()
            mu = np.dot(k, alpha)
            v = np.dot(k, np.linalg.solve(K, k))
            sigma = np.sqrt(np.abs(v))

            if acq == 'ucb':
                return -(mu - kappa * sigma)
            else:  # EI
                y_min = np.min(y)
                Z = (y_min - mu) / (sigma + 1e-10)
                ei = (y_min - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
                return -ei

        # Optimize acquisition
        res = minimize(acq_func, x0=np.mean(bounds, axis=0), bounds=bounds,
                      method='L-BFGS-B')
        x_next = res.x
        y_next = f(x_next)

        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

    best_idx = np.argmin(y)
    if full_output:
        return X[best_idx], {
            'n_evals': len(y),
            'converged': False,
            'final_value': y[best_idx]
        }
    return X[best_idx]
