# morie.fn -- function file (hadesllm/morie)
"""
MM algorithm (minorization-maximization) for optimization.

Constructs majorizing function; easier to optimize than original objective.
"""

import numpy as np

__all__ = ['mhfit']


def mhfit(f, majorizer, x0, tol=1e-6, max_iter=100, full_output=False):
    """
    MM (minorization-maximization) algorithm.

    Minimizes f(x) by iteratively minimizing majorizing function g(x|x_k)
    such that g(x_k|x_k) = f(x_k) and g(x|x_k) >= f(x).

    Parameters
    ----------
    f : callable
        Objective function f(x).
    majorizer : callable
        Function majorizer(x, x_k) -> majorizing function value.
        Should satisfy: majorizer(x, x_k) >= f(x) for all x, with
        majorizer(x_k, x_k) = f(x_k).
    x0 : ndarray
        Initial point.
    tol : float, optional
        Convergence tolerance (default 1e-6).
    max_iter : int, optional
        Maximum iterations (default 100).
    full_output : bool, optional
        If True, return (x_min, info_dict).

    Returns
    -------
    x_min : ndarray
        Estimated minimizer.
    info_dict : dict, optional
        Dictionary with keys: 'iterations', 'converged', 'final_value'.

    References
    ----------
    Hunter, D. R., & Lange, K. (2004). A tutorial on MM algorithms.
    The American Statistician, 58(1), 30-37.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import mhfit
    >>> f = lambda x: x**2  # True minimizer at x=0
    >>> # Majorizer: quadratic upper bound
    >>> majorizer = lambda x, xk: (x - xk)**2 + 0.5*(x + xk)**2
    >>> x_min = mhfit(f, majorizer, 2.0)
    >>> np.isclose(x_min, 0.0, atol=1e-4)
    True
    """
    x = np.atleast_1d(x0).astype(float)

    for iteration in range(max_iter):
        f_x = f(x)

        # Minimize majorizer via simple grid/gradient search
        def maj_obj(x_new):
            return majorizer(x_new, x)

        from scipy.optimize import minimize as scipy_minimize
        result = scipy_minimize(maj_obj, x, method='BFGS', options={'maxiter': 50})
        x_new = result.x

        residual = np.linalg.norm(x_new - x)
        if residual < tol:
            if full_output:
                return x_new, {
                    'iterations': iteration + 1,
                    'converged': True,
                    'final_value': f(x_new)
                }
            return x_new

        x = x_new

    if full_output:
        return x, {
            'iterations': max_iter,
            'converged': False,
            'final_value': f(x)
        }
    return x
