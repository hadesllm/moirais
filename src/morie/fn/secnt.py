# morie.fn — function file (hadesllm/morie)
"""Secant method for root finding."""

from __future__ import annotations

from collections.abc import Callable

from ._containers import DescriptiveResult


def secant_method(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    *,
    tol: float = 1e-10,
    maxiter: int = 100,
) -> DescriptiveResult:
    """Secant method for finding roots of f(x) = 0.

    Does not require a derivative; uses two prior evaluations to
    approximate the slope.

    Parameters
    ----------
    f : callable
        Scalar function.
    x0 : float
        First initial guess.
    x1 : float
        Second initial guess.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    DescriptiveResult
        ``value`` is the root; ``extra`` has iterations and converged flag.
    """
    f0 = f(x0)
    f1 = f(x1)
    converged = False
    for it in range(1, maxiter + 1):
        if abs(f1 - f0) < 1e-30:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < tol:
            converged = True
            x1 = x2
            break
        x0, f0 = x1, f1
        x1 = x2
        f1 = f(x1)
    return DescriptiveResult(
        name="Secant Method",
        value=float(x1),
        extra={"iterations": it, "converged": converged, "f_root": float(f(x1))},
    )


secnt = secant_method
