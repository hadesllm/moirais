# morie.fn -- function file (hadesllm/morie)
"""
Coordinate descent optimization.

Updates one coordinate at a time, holding others fixed.
"""

import numpy as np

__all__ = ['cdesc']


def cdesc(f, grad_f, x0, tol=1e-6, max_iter=1000, cyclic=True, full_output=False):
    """
    Coordinate descent for unconstrained optimization.

    Minimizes each coordinate separately in sequence (cyclic) or
    greedily (steepest descent coordinate).

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
    cyclic : bool, optional
        If True, cycle through coordinates. If False, pick steepest (default True).
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
    Wright, S. J. (2015). Coordinate descent algorithms. Mathematical
    Programming, 151(1), 3-34.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import cdesc
    >>> f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
    >>> gf = lambda x: np.array([2*(x[0]-1), 2*(x[1]-2)])
    >>> x0 = np.array([0.0, 0.0])
    >>> x_min = cdesc(f, gf, x0)
    >>> np.allclose(x_min, [1, 2], atol=1e-3)
    True
    """
    x = np.atleast_1d(x0).astype(float)
    n = len(x)

    for iteration in range(max_iter):
        x_old = x.copy()
        g = grad_f(x)

        if cyclic:
            coords = range(n)
        else:
            # Steepest descent coordinate
            coords = [np.argmax(np.abs(g))]

        for j in coords:
            # Line search in direction j
            def f_j(alpha):
                x_test = x.copy()
                x_test[j] = alpha
                return f(x_test)

            # Simple backtracking
            alpha = x[j]
            step = -g[j] / (np.abs(g[j]) + 1e-10)
            for _ in range(20):
                if f_j(alpha + 0.1 * step) < f_j(alpha):
                    alpha += 0.1 * step
                else:
                    break

            x[j] = alpha

        residual = np.linalg.norm(x - x_old)
        if residual < tol:
            if full_output:
                return x, {
                    'iterations': iteration + 1,
                    'converged': True,
                    'final_value': f(x)
                }
            return x

    if full_output:
        return x, {
            'iterations': max_iter,
            'converged': False,
            'final_value': f(x)
        }
    return x
