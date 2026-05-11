# morie.fn — function file (hadesllm/morie)
"""
FISTA (fast iterative shrinkage-thresholding algorithm).

Accelerated proximal gradient method with momentum.
"""

import numpy as np

__all__ = ['fista']


def fista(f, grad_f, prox_g, x0, step_size=0.01, tol=1e-6, max_iter=1000, full_output=False):
    """
    FISTA for composite optimization.

    Accelerated proximal gradient with momentum: y_k = x_k + t_{k-1}/t_k (x_k - x_{k-1}).

    Parameters
    ----------
    f : callable
        Smooth convex function f(x).
    grad_f : callable
        Gradient of f(x).
    prox_g : callable
        Proximal operator of g(x, step_size).
    x0 : ndarray
        Initial point.
    step_size : float, optional
        Step size (default 0.01).
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
    Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems. SIAM Journal on Imaging Sciences,
    2(1), 183-202.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import fista
    >>> f = lambda x: np.sum(x**2)
    >>> gf = lambda x: 2*x
    >>> prox_g = lambda x, ss: np.sign(x) * np.maximum(np.abs(x) - ss, 0)
    >>> x0 = np.array([1.0, 2.0])
    >>> x_min = fista(f, gf, prox_g, x0)
    >>> np.allclose(x_min, [0, 0], atol=1e-4)
    True
    """
    x = np.atleast_1d(x0).astype(float)
    y = x.copy()
    t = 1.0

    for iteration in range(max_iter):
        g = grad_f(y)
        x_new = prox_g(y - step_size * g, step_size)

        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        y_new = x_new + ((t - 1) / t_new) * (x_new - x)

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
        y = y_new
        t = t_new

    if full_output:
        return x, {
            'iterations': max_iter,
            'converged': False,
            'final_value': f(x)
        }
    return x
