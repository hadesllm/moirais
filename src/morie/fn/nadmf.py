# morie.fn -- function file (hadesllm/morie)
"""
Nesterov accelerated gradient (NAG) for faster convergence.

Momentum method with lookahead for improved convergence rates.
"""

import numpy as np

__all__ = ['nadmf']


def nadmf(f, grad_f, x0, learning_rate=0.01, momentum=0.9, tol=1e-6, max_iter=1000,
          full_output=False, seed=None):
    """
    Nesterov accelerated gradient (NAG) for unconstrained minimization.

    Accelerated first-order method with momentum. Looks ahead before gradient.

    Parameters
    ----------
    f : callable
        Objective function f(x).
    grad_f : callable
        Gradient function grad_f(x).
    x0 : ndarray
        Initial point.
    learning_rate : float, optional
        Learning rate / step size (default 0.01).
    momentum : float, optional
        Momentum coefficient (default 0.9).
    tol : float, optional
        Convergence tolerance (default 1e-6).
    max_iter : int, optional
        Maximum iterations (default 1000).
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
    Nesterov, Y. (1983). A method of solving a convex programming problem
    with convergence rate O(1/k^2). Soviet Mathematics Doklady, 27(2), 372-376.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import nadmf
    >>> f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
    >>> gf = lambda x: np.array([2*(x[0]-1), 2*(x[1]-2)])
    >>> x0 = np.array([0.0, 0.0])
    >>> x_min = nadmf(f, gf, x0)
    >>> np.allclose(x_min, [1, 2], atol=1e-3)
    True
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.atleast_1d(x0).astype(float)
    v = np.zeros_like(x)
    best_x = x.copy()
    best_f = float(f(x))

    for iteration in range(max_iter):
        x_lookahead = x + momentum * v
        g = np.asarray(grad_f(x_lookahead), dtype=float)

        g_norm = np.linalg.norm(g)
        if not np.all(np.isfinite(g)) or g_norm > 1e8:
            x = best_x.copy()
            v = np.zeros_like(x)
            learning_rate *= 0.5
            continue
        if g_norm > 1e3:
            g = g * (1e3 / g_norm)

        v_new = momentum * v - learning_rate * g
        x_new = x + v_new

        if np.all(np.isfinite(x_new)):
            f_new = float(f(x_new))
            if np.isfinite(f_new) and f_new < best_f:
                best_x = x_new.copy()
                best_f = f_new

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
        v = v_new

    if full_output:
        return best_x, {
            'iterations': max_iter,
            'converged': False,
            'final_value': best_f,
        }
    return best_x
