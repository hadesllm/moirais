# moirais.fn — function file (hadesllm/moirais)
"""RMSProp optimizer (Hinton, 2012)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._containers import DescriptiveResult


def rmsprop_optimize(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    *,
    lr: float = 0.001,
    decay: float = 0.9,
    eps: float = 1e-8,
    tol: float = 1e-8,
    maxiter: int = 1000,
) -> DescriptiveResult:
    """RMSProp adaptive learning rate optimizer.

    Uses an exponentially decaying average of squared gradients.

    Parameters
    ----------
    f : callable
        Objective function.
    grad : callable
        Gradient function.
    x0 : ndarray
        Initial point.
    lr : float
        Learning rate.
    decay : float
        Decay rate for running average of squared gradients.
    eps : float
        Numerical stability constant.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    DescriptiveResult
        ``value`` is the final objective; ``extra`` has x and iteration count.
    """
    x = np.asarray(x0, dtype=float).copy()
    v = np.zeros_like(x)
    converged = False
    for it in range(1, maxiter + 1):
        g = np.asarray(grad(x), dtype=float)
        if np.linalg.norm(g) < tol:
            converged = True
            break
        v = decay * v + (1 - decay) * g**2
        x -= lr * g / (np.sqrt(v) + eps)
    return DescriptiveResult(
        name="RMSProp",
        value=float(f(x)),
        extra={"x": x, "iterations": it, "converged": converged},
    )


rmspd = rmsprop_optimize
