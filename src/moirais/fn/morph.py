# moirais.fn — function file (hadesllm/moirais)
"""The whole is greater than the sum of its parts. — Aristotle"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import TsneRes


def tsne_reduce(
    data: pd.DataFrame | np.ndarray,
    *,
    n_dims: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 500,
    lr: float = 200.0,
    seed: int | None = 42,
) -> TsneRes:
    """t-SNE dimensionality reduction using gradient descent.

    Implements the symmetric SNE objective with Cauchy kernel in the
    low-dimensional space (van der Maaten & Hinton, 2008).

    Parameters
    ----------
    data : DataFrame or ndarray
        High-dimensional input (n x p). Non-numeric columns are dropped.
    n_dims : int
        Target dimensionality (usually 2 or 3).
    perplexity : float
        Effective number of neighbours (typically 5--50).
    n_iter : int
        Number of gradient descent iterations.
    lr : float
        Learning rate.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    TsneRes
        Embedding array of shape (n, n_dims).
    """
    if isinstance(data, pd.DataFrame):
        X = data.select_dtypes(include="number").to_numpy(dtype=np.float64)
    else:
        X = np.asarray(data, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations")

    rng = np.random.default_rng(seed)

    dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)

    def _p_row(di, target_entropy):
        lo, hi = 1e-10, 1e4
        for _ in range(50):
            sigma = (lo + hi) / 2
            pi = np.exp(-di / (2 * sigma**2))
            pi[pi < 1e-12] = 1e-12
            s = pi.sum()
            if s == 0:
                lo = sigma
                continue
            pi /= s
            ent = -np.sum(pi * np.log2(pi + 1e-15))
            if ent < target_entropy:
                lo = sigma
            else:
                hi = sigma
        return pi

    target_ent = np.log2(perplexity)
    P = np.zeros((n, n))
    for i in range(n):
        di = dists[i].copy()
        di[i] = np.inf
        P[i] = _p_row(di, target_ent)
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)

    Y = rng.normal(0, 1e-4, (n, n_dims))
    vel = np.zeros_like(Y)
    for it in range(n_iter):
        d_low = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
        Q = 1.0 / (1.0 + d_low)
        np.fill_diagonal(Q, 0)
        Q_sum = Q.sum()
        if Q_sum == 0:
            Q_sum = 1.0
        Q /= Q_sum
        Q = np.maximum(Q, 1e-12)

        PQ = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            w = PQ[i, :, None] * diff / (1 + d_low[i, :, None])
            grad[i] = 4 * np.sum(w, axis=0)
        vel = 0.8 * vel - lr * grad
        Y += vel

    return TsneRes(embedding=Y)


def cheatsheet() -> str:
    return "tsne_reduce({}) -> Dimensionality reduction via Barnes-Hut t-SNE. 'Free your mi"
