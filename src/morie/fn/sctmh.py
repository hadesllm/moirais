# morie.fn -- function file (hadesllm/morie)
"""
Secant method for root finding.

Approximates derivative using finite differences without requiring fprime.
"""

import numpy as np

__all__ = ['sctmh']


def sctmh(f, x0, x1, tol=1e-6, max_iter=100, full_output=False):
    """
    Secant method for root finding.

    Similar to Newton's method but approximates f'(x) using finite differences:
    f'(x) ≈ (f(x_{n}) - f(x_{n-1})) / (x_{n} - x_{n-1}).

    Parameters
    ----------
    f : callable
        Function f(x).
    x0 : float
        First initial guess.
    x1 : float
        Second initial guess.
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
    >>> from morie.fn import sctmh
    >>> f = lambda x: x**2 - 2
    >>> root = sctmh(f, 1.0, 2.0)
    >>> import numpy as np
    >>> np.isclose(root, np.sqrt(2), atol=1e-5)
    True
    """
    x_prev = float(x0)
    x_curr = float(x1)
    f_prev = f(x_prev)
    f_curr = f(x_curr)

    for iteration in range(max_iter):
        # Secant approximation
        denom = f_curr - f_prev
        if np.abs(denom) < 1e-14:
            if full_output:
                return x_curr, {
                    'iterations': iteration,
                    'converged': False,
                    'final_residual': np.abs(f_curr)
                }
            return x_curr

        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        residual = np.abs(x_next - x_curr)

        if residual < tol:
            if full_output:
                return x_next, {
                    'iterations': iteration + 1,
                    'converged': True,
                    'final_residual': np.abs(f(x_next))
                }
            return x_next

        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_next
        f_curr = f(x_curr)

    if full_output:
        return x_curr, {
            'iterations': max_iter,
            'converged': False,
            'final_residual': np.abs(f_curr)
        }
    return x_curr
