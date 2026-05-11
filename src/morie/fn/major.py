# morie.fn — function file (hadesllm/morie)
"""SMACOF majorize step (Guttman transform). 'Shine, Kakyoin!' -- DIO, JoJo's Bizarre Adventure"""

from __future__ import annotations

from ._containers import DescriptiveResult


def majorize_step(X, D, W=None):
    """Single SMACOF iteration (Guttman transform).

    Parameters
    ----------
    X : array-like
        Current coordinate matrix (n x p).
    D : array-like
        Target distance matrix (n x n).
    W : array-like or None
        Weight matrix (n x n). Defaults to all ones.

    Returns
    -------
    DescriptiveResult
        value = updated coordinate matrix.
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    D = np.asarray(D, dtype=float)
    n = X.shape[0]
    if W is None:
        W = np.ones_like(D)
    else:
        W = np.asarray(W, dtype=float)

    D_hat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            D_hat[i, j] = d
            D_hat[j, i] = d

    B = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and D_hat[i, j] > 1e-12:
                B[i, j] = -W[i, j] * D[i, j] / D_hat[i, j]
        B[i, i] = -np.sum(B[i, :])

    V = np.diag(np.sum(W, axis=1)) - W * (1 - np.eye(n))
    V_diag = np.sum(W * (1 - np.eye(n)), axis=1)
    X_new = B @ X / np.maximum(V_diag[:, None], 1e-12)
    return DescriptiveResult(name="majorize_step", value=X_new, extra={"n": n})


major = majorize_step


def cheatsheet() -> str:
    return "majorize_step({}) -> SMACOF majorize step (Guttman transform). 'Shine, Kakyoin!' "
