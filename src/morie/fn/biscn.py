# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bisection method for root finding."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "So this is how liberty dies. With thunderous applause. -- Padme"


def bisection_root(f, a: float, b: float, tol: float = 1e-10, max_iter: int = 100, **kwargs) -> DescriptiveResult:
    r"""
    Find a root of f(x) = 0 using the bisection method.

    Requires :math:`f(a) \\cdot f(b) < 0` (sign change). Converges
    linearly with guaranteed error bound :math:`|x^* - x_n| < (b-a)/2^n`.

    :param f: Callable f(x) whose root is sought.
    :param a: Left endpoint of bracket.
    :param b: Right endpoint of bracket.
    :param tol: Convergence tolerance on interval width. Default 1e-10.
    :param max_iter: Maximum iterations. Default 100.
    :return: DescriptiveResult with root estimate.
    :raises ValueError: If f(a) and f(b) have the same sign.

    References
    ----------
    Burden, R. L. & Faires, J. D. (2011). *Numerical Analysis* (9th ed.).
    Brooks/Cole.
    """
    a, b = float(a), float(b)
    fa, fb = float(f(a)), float(f(b))

    if fa * fb > 0:
        raise ValueError(f"f(a)={fa:.6g} and f(b)={fb:.6g} must have opposite signs.")

    history = []
    for i in range(max_iter):
        c = (a + b) / 2.0
        fc = float(f(c))
        history.append(c)

        if abs(fc) < 1e-300 or (b - a) / 2.0 < tol:
            return DescriptiveResult(
                name="bisection_root",
                value=c,
                extra={
                    "root": c,
                    "f_at_root": fc,
                    "iterations": i + 1,
                    "converged": True,
                    "bracket_width": b - a,
                    "history": np.array(history),
                },
            )

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    c = (a + b) / 2.0
    return DescriptiveResult(
        name="bisection_root",
        value=c,
        extra={
            "root": c,
            "f_at_root": float(f(c)),
            "iterations": max_iter,
            "converged": False,
            "bracket_width": b - a,
            "history": np.array(history),
        },
    )


biscn = bisection_root


def cheatsheet() -> str:
    return "bisection_root({}) -> Bisection method for root finding."
