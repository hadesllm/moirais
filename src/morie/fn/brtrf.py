# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Brent's method for root finding."""

from __future__ import annotations

from collections.abc import Callable

from ._containers import DescriptiveResult


def brent_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-12,
    maxiter: int = 100,
) -> DescriptiveResult:
    """Brent's method for finding a root of f in [a, b].

    Combines bisection, secant, and inverse quadratic interpolation for
    guaranteed convergence with superlinear speed.

    Parameters
    ----------
    f : callable
        Continuous scalar function.
    a : float
        Left bracket (f(a) and f(b) must have opposite signs).
    b : float
        Right bracket.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    DescriptiveResult
        ``value`` is the root; ``extra`` has iterations and bracket width.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c, fc = a, fa
    mflag = True
    d = 0.0
    for it in range(1, maxiter + 1):
        if abs(fb) < tol or abs(b - a) < tol:
            return DescriptiveResult(
                name="Brent's Method",
                value=float(b),
                extra={"iterations": it, "bracket_width": abs(b - a)},
            )
        if abs(fa - fc) > 1e-15 and abs(fb - fc) > 1e-15:
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            s = b - fb * (b - a) / (fb - fa + 1e-30)
        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = (not mflag) and abs(c - d) < tol
        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False
        fs = f(s)
        d = c
        c, fc = b, fb
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
    return DescriptiveResult(
        name="Brent's Method",
        value=float(b),
        extra={"iterations": maxiter, "bracket_width": abs(b - a)},
    )


brtrf = brent_root
