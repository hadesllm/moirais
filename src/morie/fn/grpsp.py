# morie.fn — function file (hadesllm/morie)
"""Group sparsity decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We are what they grow beyond."


def group_sparse_decompose(
    X, groups, lambda_: float = 0.1, max_iter: int = 200, tol: float = 1e-6, **kwargs
) -> DescriptiveResult:
    """Group sparsity decomposition via group lasso (proximal gradient).

    Solves: min_c  0.5*||X - D*c||^2 + lambda * sum_g ||c_g||_2
    where D = identity (analysis form) and groups define index sets.

    Parameters
    ----------
    X : array-like, shape (n,) or (n, p)
        Input signal or matrix.
    groups : list of list/array of int
        Group index sets. Each element is a list of indices belonging
        to one group.
    lambda_ : float
        Group sparsity penalty (default 0.1).
    max_iter : int
        Maximum iterations (default 200).
    tol : float
        Convergence tolerance (default 1e-6).

    Returns
    -------
    DescriptiveResult
        ``value`` is number of active groups; ``extra`` has ``coefficients``,
        ``active_groups``, ``group_norms``, ``iterations``.
    """
    X = np.asarray(X, dtype=float).ravel()
    n = len(X)

    c = X.copy()
    step = 1.0

    for it in range(max_iter):
        grad = c - X
        z = c - step * grad

        c_new = z.copy()
        for g_idx in groups:
            g = np.asarray(g_idx)
            norm_g = np.linalg.norm(z[g])
            if norm_g > lambda_ * step:
                c_new[g] = z[g] * (1.0 - lambda_ * step / norm_g)
            else:
                c_new[g] = 0.0

        if np.linalg.norm(c_new - c) < tol:
            c = c_new
            break
        c = c_new

    group_norms = [float(np.linalg.norm(c[np.asarray(g)])) for g in groups]
    active = [i for i, gn in enumerate(group_norms) if gn > 1e-10]

    return DescriptiveResult(
        name="group_sparse_decompose",
        value=len(active),
        extra={"coefficients": c, "active_groups": active, "group_norms": group_norms, "iterations": it + 1},
    )


grpsp = group_sparse_decompose


def cheatsheet() -> str:
    return "group_sparse_decompose({}) -> Group sparsity decomposition."
