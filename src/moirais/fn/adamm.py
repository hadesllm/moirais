# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Adam optimizer (Kingma & Ba, 2015)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._containers import DescriptiveResult


def adam_optimize(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    *,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    tol: float = 1e-8,
    maxiter: int = 1000,
) -> DescriptiveResult:
    """Adam optimizer.

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
    beta1 : float
        Exponential decay for first moment.
    beta2 : float
        Exponential decay for second moment.
    eps : float
        Numerical stability constant.
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
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    converged = False
    for it in range(1, maxiter + 1):
        g = np.asarray(grad(x), dtype=float)
        if np.linalg.norm(g) < tol:
            converged = True
            break
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**it)
        v_hat = v / (1 - beta2**it)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return DescriptiveResult(
        name="Adam",
        value=float(f(x)),
        extra={"x": x, "iterations": it, "converged": converged},
    )


adamm = adam_optimize
