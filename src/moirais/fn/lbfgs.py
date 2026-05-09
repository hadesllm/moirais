# moirais.fn — function file (hadesllm/moirais)
"""
L-BFGS (Limited-memory BFGS) quasi-Newton optimization.

Memory-efficient variant of BFGS for large-scale optimization.
"""

import numpy as np

__all__ = ['lbfgs']


def lbfgs(f, grad_f, x0, m=10, tol=1e-6, max_iter=1000, full_output=False):
    """
    Limited-memory BFGS (L-BFGS) quasi-Newton method.

    Approximates Hessian using only m most recent gradient pairs.
    Efficient for high-dimensional problems.

    Parameters
    ----------
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient function grad_f(x).
    x0 : ndarray
        Initial point.
    m : int, optional
        Number of BFGS updates to store (default 10).
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
    Nocedal, J. (1980). Updating quasi-Newton matrices with limited storage.
    Mathematics of Computation, 35(151), 773-782.

    Examples
    --------
    >>> import numpy as np
    >>> from moirais.fn import lbfgs
    >>> f = lambda x: (x[0] - 1)**2 + 100*(x[1] - x[0]**2)**2
    >>> gf = lambda x: np.array([2*(x[0]-1) - 400*x[0]*(x[1]-x[0]**2),
    ...                            200*(x[1]-x[0]**2)])
    >>> x0 = np.array([0.0, 0.0])
    >>> x_min = lbfgs(f, gf, x0)
    >>> np.allclose(x_min, [1, 1], atol=0.01)
    True
    """
    x = np.atleast_1d(x0).astype(float)
    g = grad_f(x)
    d = -g

    s_history = []
    y_history = []

    for iteration in range(max_iter):
        if np.linalg.norm(g) < tol:
            if full_output:
                return x, {
                    'iterations': iteration,
                    'converged': True,
                    'final_value': f(x)
                }
            return x

        # Line search (backtracking)
        alpha = 1.0
        c = 1e-4
        for _ in range(20):
            if f(x + alpha * d) <= f(x) + c * alpha * np.dot(g, d):
                break
            alpha *= 0.5

        s = alpha * d
        x_new = x + s
        g_new = grad_f(x_new)
        y = g_new - g

        # Store s and y
        s_history.append(s)
        y_history.append(y)
        if len(s_history) > m:
            s_history.pop(0)
            y_history.pop(0)

        # L-BFGS 2-loop recursion
        q = g_new.copy()
        alpha_vals = []
        for i in range(len(s_history) - 1, -1, -1):
            rho = 1.0 / (np.dot(y_history[i], s_history[i]) + 1e-14)
            alpha_i = rho * np.dot(s_history[i], q)
            alpha_vals.append(alpha_i)
            q = q - alpha_i * y_history[i]

        # Initial Hessian approximation: I (scaled)
        if len(s_history) > 0:
            gamma = np.dot(s_history[-1], y_history[-1]) / (np.dot(y_history[-1], y_history[-1]) + 1e-14)
        else:
            gamma = 1.0
        d = gamma * q

        for i in range(len(s_history)):
            rho = 1.0 / (np.dot(y_history[i], s_history[i]) + 1e-14)
            beta = rho * np.dot(y_history[i], d)
            d = d + (alpha_vals[len(s_history) - 1 - i] - beta) * s_history[i]

        d = -d
        x = x_new
        g = g_new

    if full_output:
        return x, {
            'iterations': max_iter,
            'converged': False,
            'final_value': f(x)
        }
    return x
