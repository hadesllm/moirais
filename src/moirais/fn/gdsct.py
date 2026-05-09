# moirais.fn — function file (hadesllm/moirais)
"""Gradient descent with momentum."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._containers import DescriptiveResult


def gradient_descent(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    *,
    lr: float = 0.01,
    momentum: float = 0.9,
    tol: float = 1e-8,
    maxiter: int = 1000,
) -> DescriptiveResult:
    """Gradient descent with momentum.

    Parameters
    ----------
    f : callable
        Objective function f(x) -> scalar.
    grad : callable
        Gradient function grad(x) -> array.
    x0 : ndarray
        Initial point.
    lr : float
        Learning rate.
    momentum : float
        Momentum coefficient (0 = no momentum).
    tol : float
        Convergence tolerance on gradient norm.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    DescriptiveResult
        ``value`` is the final objective; ``extra`` has x, gradient norm,
        and iteration count.
    """
    x = np.asarray(x0, dtype=float).copy()
    v = np.zeros_like(x)
    converged = False
    for it in range(1, maxiter + 1):
        g = np.asarray(grad(x), dtype=float)
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            converged = True
            break
        v = momentum * v - lr * g
        x = x + v
    return DescriptiveResult(
        name="Gradient Descent",
        value=float(f(x)),
        extra={
            "x": x,
            "grad_norm": float(np.linalg.norm(np.asarray(grad(x)))),
            "iterations": it,
            "converged": converged,
        },
    )


gdsct = gradient_descent
