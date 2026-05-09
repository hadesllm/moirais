"""Sum of squared distance errors for MDS. 'Ora ora ora!' -- Jotaro, JoJo's Bizarre Adventure"""

from __future__ import annotations

from ._containers import DescriptiveResult


def sse_mds(X, D_target):
    """Sum of squared errors between reconstructed and target distances.

    Parameters
    ----------
    X : array-like
        Coordinate matrix (n x p).
    D_target : array-like
        Target distance matrix (n x n).

    Returns
    -------
    DescriptiveResult
        value = SSE (float).
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    D_target = np.asarray(D_target, dtype=float)
    n = X.shape[0]
    sse = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d_hat = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            sse += (D_target[i, j] - d_hat) ** 2
    return DescriptiveResult(name="sse_mds", value=float(sse), extra={"n": n})


ssefn = sse_mds


def cheatsheet() -> str:
    return "sse_mds({}) -> Sum of squared distance errors for MDS. 'Ora ora ora!' -- Jo"
