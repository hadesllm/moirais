"""t-SNE (t-distributed Stochastic Neighbour Embedding)."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import TsneRes


def _pairwise_sq_dist(X: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances."""
    sum_sq = np.sum(X**2, axis=1)
    return sum_sq[:, None] + sum_sq[None, :] - 2.0 * X @ X.T


def _compute_pij(X: np.ndarray, perplexity: float) -> np.ndarray:
    """Compute joint probability matrix P from high-dimensional data."""
    n = X.shape[0]
    D = _pairwise_sq_dist(X)
    target_H = np.log(perplexity)
    P = np.zeros((n, n))

    for i in range(n):
        lo, hi = 1e-20, 1e4
        beta = 1.0  # 1 / (2 * sigma_i^2)
        for _ in range(50):
            diffs = D[i].copy()
            diffs[i] = np.inf
            prow = np.exp(-diffs * beta)
            psum = np.sum(prow)
            if psum < 1e-30:
                psum = 1e-30
            prow /= psum
            H = -np.sum(prow * np.log(np.maximum(prow, 1e-30)))
            if abs(H - target_H) < 1e-5:
                break
            if target_H < H:
                lo = beta
                beta = (beta + hi) / 2.0 if hi < 1e4 else beta * 2.0
            else:
                hi = beta
                beta = (beta + lo) / 2.0
        P[i] = prow

    # Symmetrise
    P = (P + P.T) / (2.0 * n)
    P = np.maximum(P, 1e-12)
    return P


def tsne(
    data: np.ndarray,
    n_dims: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    lr: float = 200.0,
    momentum_init: float = 0.5,
    momentum_final: float = 0.8,
    seed: int | None = 42,
) -> TsneRes:
    """t-distributed Stochastic Neighbour Embedding.

    Simplified exact t-SNE (no Barnes-Hut tree) suitable for n < 5000.
    Minimises KL divergence between high-dimensional joint probabilities
    and Student-t affinities in the low-dimensional embedding.

    Parameters
    ----------
    data : ndarray (n, p)
        High-dimensional data matrix.
    n_dims : int
        Embedding dimensionality.
    perplexity : float
        Target perplexity (effective number of neighbours).
    n_iter : int
        Gradient descent iterations.
    lr : float
        Learning rate.
    momentum_init, momentum_final : float
        Momentum switches from *init* to *final* at iteration 250.
    seed : int, optional
        Random seed for initialisation.

    Returns
    -------
    TsneRes
        ``embedding`` (n x n_dims).

    References
    ----------
    van der Maaten, L. & Hinton, G. (2008). Visualizing data using t-SNE.
    *JMLR*, 9, 2579-2605.
    """
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    P = _compute_pij(X, perplexity)
    # Early exaggeration
    P *= 4.0

    Y = rng.standard_normal((n, n_dims)) * 1e-4
    dY = np.zeros_like(Y)
    gains = np.ones_like(Y)

    for it in range(n_iter):
        if it == 100:
            P /= 4.0  # stop early exaggeration

        # Student-t affinities
        Dsq = _pairwise_sq_dist(Y)
        Q_num = 1.0 / (1.0 + Dsq)
        np.fill_diagonal(Q_num, 0.0)
        Q_sum = np.sum(Q_num)
        Q = Q_num / max(Q_sum, 1e-12)
        Q = np.maximum(Q, 1e-12)

        # Gradient
        PQ = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4.0 * np.sum((PQ[i] * Q_num[i])[:, None] * diff, axis=0)

        # Adaptive gains
        gains = (gains + 0.2) * ((grad > 0) != (dY > 0)) + (gains * 0.8) * ((grad > 0) == (dY > 0))
        gains = np.maximum(gains, 0.01)

        mom = momentum_init if it < 250 else momentum_final
        dY = mom * dY - lr * gains * grad
        Y += dY
        Y -= Y.mean(axis=0)

    return TsneRes(embedding=Y)


def cheatsheet() -> str:
    return "_pairwise_sq_dist({}) -> t-SNE (t-distributed Stochastic Neighbour Embedding)."
