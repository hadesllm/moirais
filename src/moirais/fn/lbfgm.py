# moirais.fn — function file (hadesllm/moirais)
"""L-BFGS optimizer."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._containers import DescriptiveResult


def lbfgs_optimize(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    *,
    m: int = 10,
    lr: float = 1.0,
    tol: float = 1e-8,
    maxiter: int = 200,
) -> DescriptiveResult:
    """Limited-memory BFGS quasi-Newton optimizer.

    Stores the last *m* gradient differences to approximate the inverse
    Hessian via two-loop recursion.

    Parameters
    ----------
    f : callable
        Objective function.
    grad : callable
        Gradient function.
    x0 : ndarray
        Initial point.
    m : int
        Memory size (number of stored correction pairs).
    lr : float
        Step size multiplier.
    tol : float
        Convergence tolerance on gradient norm.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    DescriptiveResult
        ``value`` is the final objective; ``extra`` has x and iteration count.
    """
    x = np.asarray(x0, dtype=float).copy()
    g = np.asarray(grad(x), dtype=float)
    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []
    rho_hist: list[float] = []
    converged = False
    for it in range(1, maxiter + 1):
        if np.linalg.norm(g) < tol:
            converged = True
            break
        q = g.copy()
        alphas = []
        for i in range(len(s_hist) - 1, -1, -1):
            a = rho_hist[i] * (s_hist[i] @ q)
            alphas.append(a)
            q -= a * y_hist[i]
        alphas.reverse()
        if s_hist:
            gamma = (s_hist[-1] @ y_hist[-1]) / (y_hist[-1] @ y_hist[-1] + 1e-30)
        else:
            gamma = 1.0
        r = gamma * q
        for i in range(len(s_hist)):
            b = rho_hist[i] * (y_hist[i] @ r)
            r += s_hist[i] * (alphas[i] - b)
        d = -r
        x_new = x + lr * d
        g_new = np.asarray(grad(x_new), dtype=float)
        s = x_new - x
        y = g_new - g
        sy = s @ y
        if sy > 1e-15:
            if len(s_hist) >= m:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(s)
            y_hist.append(y)
            rho_hist.append(1.0 / sy)
        x, g = x_new, g_new
    return DescriptiveResult(
        name="L-BFGS",
        value=float(f(x)),
        extra={"x": x, "iterations": it, "converged": converged},
    )


lbfgm = lbfgs_optimize
