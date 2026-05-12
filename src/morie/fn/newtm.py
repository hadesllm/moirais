# morie.fn -- function file (hadesllm/morie)
"""Newton's method for root finding (1D and multivariate)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._containers import DescriptiveResult


def newton_method(
    f: Callable,
    x0: float | np.ndarray,
    *,
    fprime: Callable | None = None,
    tol: float = 1e-10,
    maxiter: int = 100,
    h: float = 1e-7,
) -> DescriptiveResult:
    """Newton's method for finding roots of f(x) = 0.

    Works for scalar or vector-valued functions. If *fprime* is not given,
    the Jacobian is estimated via finite differences.

    Parameters
    ----------
    f : callable
        Function whose root is sought.
    x0 : float or ndarray
        Initial guess.
    fprime : callable, optional
        Derivative/Jacobian. Estimated numerically if omitted.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.
    h : float
        Step size for numerical differentiation.

    Returns
    -------
    DescriptiveResult
        ``value`` is the root; ``extra`` has iterations and convergence flag.
    """
    scalar = np.isscalar(x0)
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    n = len(x)
    converged = False
    for it in range(1, maxiter + 1):
        fx = np.atleast_1d(np.asarray(f(x[0] if scalar else x), dtype=float))
        if np.linalg.norm(fx) < tol:
            converged = True
            break
        if fprime is not None:
            J = np.atleast_2d(np.asarray(fprime(x[0] if scalar else x), dtype=float))
        else:
            J = np.zeros((len(fx), n))
            for j in range(n):
                xp = x.copy()
                xp[j] += h
                J[:, j] = (np.atleast_1d(np.asarray(f(xp[0] if scalar and n == 1 else xp), dtype=float)) - fx) / h
        if n == 1 and len(fx) == 1:
            if abs(J[0, 0]) < 1e-30:
                break
            x[0] -= fx[0] / J[0, 0]
        else:
            dx = np.linalg.solve(J, fx)
            x -= dx
    root = float(x[0]) if scalar else x
    return DescriptiveResult(
        name="Newton's Method",
        value=root,
        extra={"iterations": it, "converged": converged},
    )


newtm = newton_method
