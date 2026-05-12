# morie.fn -- function file (hadesllm/morie)
"""
Conjugate gradient method for unconstrained optimization.

First-order iterative method using conjugate directions.
"""

import numpy as np

__all__ = ['cgmth']


def cgmth(f, grad_f, x0, tol=1e-6, max_iter=1000, full_output=False):
    """
    Conjugate gradient method for unconstrained minimization.

    Uses conjugate directions to minimize f(x). Converges in at most n steps
    for quadratic functions.

    Parameters
    ----------
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient function grad_f(x).
    x0 : ndarray
        Initial point.
    tol : float, optional
        Convergence tolerance (default 1e-6).
    max_iter : int, optional
        Maximum iterations (default 1000).
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
    Hestenes, M. R., & Stiefel, E. (1952). Methods of conjugate gradients
    for solving linear systems. Journal of Research of the National Bureau
    of Standards, 49(6), 409-436.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import cgmth
    >>> f = lambda x: (x[0] - 2)**2 + (x[1] - 3)**2
    >>> gf = lambda x: np.array([2*(x[0]-2), 2*(x[1]-3)])
    >>> x0 = np.array([0.0, 0.0])
    >>> x_min = cgmth(f, gf, x0)
    >>> np.allclose(x_min, [2, 3], atol=1e-4)
    True
    """
    x = np.atleast_1d(x0).astype(float)
    g = grad_f(x)
    d = -g

    for iteration in range(max_iter):
        if np.linalg.norm(g) < tol:
            if full_output:
                return x, {
                    'iterations': iteration,
                    'converged': True,
                    'final_value': f(x)
                }
            return x

        # Line search (simple backtracking)
        alpha = 1.0
        c = 1e-4
        for _ in range(20):
            if f(x + alpha * d) <= f(x) + c * alpha * np.dot(g, d):
                break
            alpha *= 0.5

        x = x + alpha * d
        g_new = grad_f(x)

        # Polak-Ribiere-Polyak (PRP) formula
        beta = np.dot(g_new, g_new - g) / (np.dot(g, g) + 1e-14)
        beta = max(0, beta)

        d = -g_new + beta * d
        g = g_new

    if full_output:
        return x, {
            'iterations': max_iter,
            'converged': False,
            'final_value': f(x)
        }
    return x
