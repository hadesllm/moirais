# morie.fn -- function file (hadesllm/morie)
"""MDS with missing data (weighted stress). 'Shoot Style.' -- Deku, My Hero Academia"""

from __future__ import annotations

from ._containers import DescriptiveResult


def mds_missing_data(D, W, n_dims=2, seed=42, max_iter=100):
    """Classical MDS with weighted stress for missing data.

    Parameters
    ----------
    D : array-like
        Distance matrix (NaN replaced by 0 or imputed).
    W : array-like
        Weight matrix (0 for missing, 1 for observed).
    n_dims : int
        Embedding dimensionality.
    seed : int
        Random seed.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    DescriptiveResult
        value = coordinate matrix, extra has final weighted stress.
    """
    import numpy as np

    D = np.asarray(D, dtype=float)
    W = np.asarray(W, dtype=float)
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_dims))

    for _ in range(max_iter):
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
        X_new = B @ X / n
        if np.max(np.abs(X_new - X)) < 1e-6:
            X = X_new
            break
        X = X_new

    stress = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            stress += W[i, j] * (D[i, j] - d) ** 2
    return DescriptiveResult(name="mds_missing_data", value=X, extra={"weighted_stress": float(stress)})


mdsmm = mds_missing_data


def cheatsheet() -> str:
    return "mds_missing_data({}) -> MDS with missing data (weighted stress). 'Shoot Style.' -- D"
