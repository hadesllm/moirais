# morie.fn — function file (hadesllm/morie)
"""Simulated annealing MDS. 'Muda muda muda!' -- DIO, JoJo's Bizarre Adventure"""

from __future__ import annotations

from ._containers import DescriptiveResult


def simulated_anneal_mds(D, n_dims=2, T0=1.0, cool=0.95, n_iter=500, seed=42):
    """Optimize MDS coordinates via simulated annealing.

    Parameters
    ----------
    D : array-like
        Target distance matrix (n x n).
    n_dims : int
        Embedding dimensionality.
    T0 : float
        Initial temperature.
    cool : float
        Cooling rate per iteration.
    n_iter : int
        Number of iterations.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        value = optimized coordinate matrix.
    """
    import numpy as np

    D = np.asarray(D, dtype=float)
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    X = rng.standard_normal((n, n_dims))

    def _stress(X_):
        s = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d_hat = np.sqrt(np.sum((X_[i] - X_[j]) ** 2))
                s += (D[i, j] - d_hat) ** 2
        return s

    best_X = X.copy()
    best_s = _stress(X)
    T = T0
    for _ in range(n_iter):
        X_new = X + rng.normal(0, T, X.shape)
        s_new = _stress(X_new)
        delta = s_new - best_s
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
            X = X_new
            if s_new < best_s:
                best_s = s_new
                best_X = X_new.copy()
        T *= cool
    return DescriptiveResult(name="simulated_anneal_mds", value=best_X, extra={"stress": best_s})


optsa = simulated_anneal_mds


def cheatsheet() -> str:
    return "simulated_anneal_mds({}) -> Simulated annealing MDS. 'Muda muda muda!' -- DIO, JoJo's Bi"
