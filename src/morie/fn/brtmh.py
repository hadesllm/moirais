# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""
Brent's method for root finding.

Combines bisection, secant, and inverse quadratic interpolation.
"""

import numpy as np

__all__ = ['brtmh']


def brtmh(f, a, b, tol=1e-6, max_iter=100, full_output=False):
    """
    Brent's method for root finding.

    Robust hybrid method combining bisection, secant, and inverse quadratic
    interpolation. Faster than bisection, more reliable than Newton's method.

    Parameters
    ----------
    f : callable
        Function f(x).
    a : float
        Left bracket (f(a) * f(b) < 0 assumed).
    b : float
        Right bracket.
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
    Brent, R. P. (1973). Algorithms for Minimization Without Derivatives.
    Prentice Hall.

    Examples
    --------
    >>> from morie.fn import brtmh
    >>> f = lambda x: x**3 - 2
    >>> root = brtmh(f, 1.0, 2.0)
    >>> import numpy as np
    >>> np.isclose(root, 2**(1/3), atol=1e-6)
    True
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    if np.abs(fa) < np.abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = c
    mflag = True

    for iteration in range(max_iter):
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            L0 = a * fb * fc / ((fa - fb) * (fa - fc))
            L1 = b * fa * fc / ((fb - fa) * (fb - fc))
            L2 = c * fa * fb / ((fc - fa) * (fc - fb))
            s = L0 + L1 + L2
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        # Bisection fallback conditions
        if ((s - b) * (3 * a - b) > 0 or
            (mflag and np.abs(s - b) >= np.abs(b - c) / 2) or
            (not mflag and np.abs(s - b) >= np.abs(c - d) / 2) or
            (mflag and np.abs(b - c) < tol) or
            (not mflag and np.abs(c - d) < tol)):
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d = c
        c = b

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if np.abs(fa) < np.abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        if np.abs(fb) < tol or (b - a) < tol:
            if full_output:
                return b, {
                    'iterations': iteration + 1,
                    'converged': True,
                    'final_residual': np.abs(fb)
                }
            return b

    if full_output:
        return b, {
            'iterations': max_iter,
            'converged': False,
            'final_residual': np.abs(fb)
        }
    return b
