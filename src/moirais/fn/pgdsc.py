# moirais.fn — function file (hadesllm/moirais)
"""
Proximal gradient descent for composite optimization.

Minimizes f(x) + g(x) where f is smooth and g is convex (possibly non-smooth).
"""

import numpy as np

__all__ = ['pgdsc']


def pgdsc(f, grad_f, prox_g, x0, step_size=0.01, tol=1e-6, max_iter=1000, full_output=False):
    """
    Proximal gradient descent for composite optimization.

    Iteration: x_{k+1} = prox_{step_size * g}(x_k - step_size * grad_f(x_k)).

    Parameters
    ----------
    f : callable
        Smooth convex function f(x).
    grad_f : callable
        Gradient of f(x).
    prox_g : callable
        Proximal operator of g: prox_g(x, step_size) -> argmin_u [g(u) + 0.5*||u-x||^2/step_size].
    x0 : ndarray
        Initial point.
    step_size : float, optional
        Step size / learning rate (default 0.01).
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
    Nesterov, Y. (2013). Gradient methods for minimizing composite functions.
    Mathematical Programming, 140(1), 125-161.

    Examples
    --------
    >>> import numpy as np
    >>> from moirais.fn import pgdsc
    >>> f = lambda x: np.sum(x**2)
    >>> gf = lambda x: 2*x
    >>> prox_g = lambda x, ss: np.sign(x) * np.maximum(np.abs(x) - ss, 0)
    >>> x0 = np.array([1.0, 2.0])
    >>> x_min = pgdsc(f, gf, prox_g, x0)
    >>> np.allclose(x_min, [0, 0], atol=1e-4)
    True
    """
    x = np.atleast_1d(x0).astype(float)

    for iteration in range(max_iter):
        g = grad_f(x)
        x_new = prox_g(x - step_size * g, step_size)

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
