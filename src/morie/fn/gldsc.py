# morie.fn -- function file (hadesllm/morie)
"""
Golden section search for univariate minimization.

Iteratively brackets the minimum of a unimodal function.
"""

import numpy as np

__all__ = ['gldsc']


def gldsc(f, a, b, tol=1e-6, max_iter=100, full_output=False):
    """
    Golden section search for univariate minimization.

    Brackets the minimum of a unimodal function f on [a, b] using the golden
    ratio φ = (1 + √5) / 2 ≈ 1.618.

    Parameters
    ----------
    f : callable
        Unimodal function f(x) to minimize.
    a : float
        Left boundary.
    b : float
        Right boundary.
    tol : float, optional
        Convergence tolerance (default 1e-6).
    max_iter : int, optional
        Maximum iterations (default 100).
    full_output : bool, optional
        If True, return (x_min, info_dict).

    Returns
    -------
    x_min : float
        Estimated minimizer.
    info_dict : dict, optional
        Dictionary with keys: 'iterations', 'converged', 'final_value'.

    References
    ----------
    Kiefer, J. (1953). Sequential minimax search for a maximum. Proceedings
    of the American Mathematical Society, 4(3), 502-506.

    Examples
    --------
    >>> from morie.fn import gldsc
    >>> f = lambda x: (x - 3)**2
    >>> x_min = gldsc(f, 0.0, 5.0)
    >>> import numpy as np
    >>> np.isclose(x_min, 3.0, atol=1e-4)
    True
    """
    golden_ratio = (3 - np.sqrt(5)) / 2  # ≈ 0.381966

    x1 = a + golden_ratio * (b - a)
    x2 = b - golden_ratio * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    for iteration in range(max_iter):
        if (b - a) < tol:
            if full_output:
                x_min = (a + b) / 2
                return x_min, {
                    'iterations': iteration + 1,
                    'converged': True,
                    'final_value': f(x_min)
                }
            return (a + b) / 2

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + golden_ratio * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - golden_ratio * (b - a)
            f2 = f(x2)

    x_min = (a + b) / 2
    if full_output:
        return x_min, {
            'iterations': max_iter,
            'converged': False,
            'final_value': f(x_min)
        }
    return x_min
