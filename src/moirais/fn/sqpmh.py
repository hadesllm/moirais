"""
Sequential quadratic programming (SQP) for constrained optimization.

Solves equality and inequality constrained problems via iterative quadratic programs.
"""

import numpy as np
from scipy.optimize import minimize

__all__ = ['sqpmh']


def sqpmh(f, grad_f, constraints, x0, tol=1e-6, max_iter=100, full_output=False):
    """
    Sequential quadratic programming (SQP) for constrained optimization.

    At each iteration, solves a quadratic program based on local quadratic
    approximation of Lagrangian.

    Parameters
    ----------
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient of f(x).
    constraints : list of dict
        Each dict has 'type': 'eq'/'ineq', 'fun': c(x), 'jac': grad c(x).
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
    Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.).
    Springer.

    Examples
    --------
    >>> import numpy as np
    >>> from moirais.fn import sqpmh
    >>> f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
    >>> gf = lambda x: np.array([2*(x[0]-1), 2*(x[1]-2)])
    >>> c = {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 2}
    >>> x0 = np.array([0.5, 1.5])
    >>> x_min = sqpmh(f, gf, [c], x0)
    >>> np.isclose(x_min[0] + x_min[1], 2.0, atol=1e-4)
    True
    """
    x = np.atleast_1d(x0).astype(float)

    # Use scipy's SLSQP (which implements SQP internally)
    result = minimize(
        f,
        x,
        method='SLSQP',
        jac=grad_f,
        constraints=constraints,
        options={'ftol': tol, 'maxiter': max_iter}
    )

    if full_output:
        return result.x, {
            'iterations': result.nit,
            'converged': result.success,
            'final_value': result.fun
        }
    return result.x
