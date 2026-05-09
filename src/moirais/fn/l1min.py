# moirais.fn — function file (hadesllm/moirais)
"""L1-minimization via ISTA."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You have power over your mind — not outside events. Realize this, and you will find strength. — Marcus Aurelius"


def l1_minimize(A, b, lambda_: float = 0.1, max_iter: int = 500, tol: float = 1e-6, **kwargs) -> DescriptiveResult:
    """L1-minimization via ISTA (Iterative Shrinkage-Thresholding).

    Solves: min_x  0.5*||Ax - b||^2 + lambda*||x||_1

    Parameters
    ----------
    A : array-like, shape (m, n)
        Measurement / design matrix.
    b : array-like, shape (m,)
        Observation vector.
    lambda_ : float
        L1 regularization weight (default 0.1).
    max_iter : int
        Maximum iterations (default 500).
    tol : float
        Convergence tolerance (default 1e-6).

    Returns
    -------
    DescriptiveResult
        ``value`` is final objective; ``extra`` has ``x`` (solution),
        ``residual``, ``sparsity`` (number of non-zeros), ``iterations``.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    n = A.shape[1]

    L = np.linalg.norm(A.T @ A, ord=2)
    step = 1.0 / L

    x = np.zeros(n)
    for it in range(max_iter):
        grad = A.T @ (A @ x - b)
        z = x - step * grad
        x_new = np.sign(z) * np.maximum(np.abs(z) - lambda_ * step, 0.0)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    residual = b - A @ x
    obj = 0.5 * np.dot(residual, residual) + lambda_ * np.sum(np.abs(x))
    sparsity = int(np.sum(np.abs(x) > 1e-10))

    return DescriptiveResult(
        name="l1_minimize",
        value=float(obj),
        extra={"x": x, "residual": residual, "sparsity": sparsity, "iterations": it + 1},
    )


l1min = l1_minimize


def cheatsheet() -> str:
    return "l1_minimize({}) -> L1-minimization via ISTA."
