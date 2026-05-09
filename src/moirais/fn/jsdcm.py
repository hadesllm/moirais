# moirais.fn — function file (hadesllm/moirais)
"""Joint/simultaneous sparse decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "This is the way."


def joint_sparse_decompose(
    X_list, lambda_: float = 0.1, max_iter: int = 200, tol: float = 1e-6, **kwargs
) -> DescriptiveResult:
    """Joint (simultaneous) sparse decomposition via group lasso across signals.

    Given multiple signals sharing the same support, solves:
    min_{C}  0.5 * sum_k ||X_k - c_k||^2 + lambda * sum_i ||C[i,:]||_2

    where C[i,:] is row i across all signals (mixed L2,1 norm).

    Parameters
    ----------
    X_list : list of array-like
        List of K signals, each of length n.
    lambda_ : float
        Joint sparsity penalty (default 0.1).
    max_iter : int
        Maximum iterations (default 200).
    tol : float
        Convergence tolerance (default 1e-6).

    Returns
    -------
    DescriptiveResult
        ``value`` is number of jointly active indices; ``extra`` has
        ``coefficients`` (n x K), ``active_indices``, ``row_norms``,
        ``iterations``.
    """
    signals = [np.asarray(s, dtype=float).ravel() for s in X_list]
    n = len(signals[0])
    K = len(signals)
    X = np.column_stack(signals)

    C = X.copy()
    step = 1.0

    for it in range(max_iter):
        grad = C - X
        Z = C - step * grad

        C_new = Z.copy()
        for i in range(n):
            row_norm = np.linalg.norm(Z[i, :])
            if row_norm > lambda_ * step:
                C_new[i, :] = Z[i, :] * (1.0 - lambda_ * step / row_norm)
            else:
                C_new[i, :] = 0.0

        if np.linalg.norm(C_new - C) < tol:
            C = C_new
            break
        C = C_new

    row_norms = np.linalg.norm(C, axis=1)
    active = [int(i) for i in range(n) if row_norms[i] > 1e-10]

    return DescriptiveResult(
        name="joint_sparse_decompose",
        value=len(active),
        extra={"coefficients": C, "active_indices": active, "row_norms": row_norms, "iterations": it + 1},
    )


jsdcm = joint_sparse_decompose


def cheatsheet() -> str:
    return "joint_sparse_decompose({}) -> Joint/simultaneous sparse decomposition."
