# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Bisection method for root finding.

Iteratively narrows an interval [a, b] where f changes sign.
"""

import numpy as np

__all__ = ['bsctn']


def bsctn(f, a, b, tol=1e-6, max_iter=100, full_output=False):
    """
    Bisection method for root finding.

    Assumes f(a) and f(b) have opposite signs. Iteratively bisects [a, b]
    until a root is bracketed within tolerance.

    Parameters
    ----------
    f : callable
        Function f(x).
    a : float
        Left endpoint (must have f(a) * f(b) < 0).
    b : float
        Right endpoint.
    tol : float, optional
        Convergence tolerance (default 1e-6).
    max_iter : int, optional
        Maximum iterations (default 100).
    full_output : bool, optional
        If True, return (root, info_dict).

    Returns
    -------
    root : float
        Estimated root.
    info_dict : dict, optional
        Dictionary with keys: 'iterations', 'converged', 'final_residual'.

    References
    ----------
    Burden, R. L., & Faires, J. D. (2010). Numerical Analysis (9th ed.).
    Cengage Learning.

    Examples
    --------
    >>> from morie.fn import bsctn
    >>> f = lambda x: x**2 - 2
    >>> root = bsctn(f, 1.0, 2.0)
    >>> import numpy as np
    >>> np.isclose(root, np.sqrt(2), atol=1e-6)
    True
    """
    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    for iteration in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)

        if np.abs(fc) < tol or (b - a) / 2.0 < tol:
            if full_output:
                return c, {
                    'iterations': iteration + 1,
                    'converged': True,
                    'final_residual': np.abs(fc)
                }
            return c

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    if full_output:
        c = (a + b) / 2.0
        return c, {
            'iterations': max_iter,
            'converged': False,
            'final_residual': np.abs(f(c))
        }
    return (a + b) / 2.0
