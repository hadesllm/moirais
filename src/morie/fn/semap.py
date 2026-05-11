# morie.fn — function file (hadesllm/morie)
"""Sammon mapping."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ._containers import DescriptiveResult


def sammon_mapping(
    data: np.ndarray,
    n_dims: int = 2,
    max_iter: int = 200,
    lr: float = 0.3,
    seed: int = 42,
) -> DescriptiveResult:
    """Sammon mapping: non-linear dimensionality reduction preserving distances.

    Minimises Sammon stress:

    .. math::

        E = \\frac{1}{\\sum d_{ij}} \\sum_{i<j}
            \\frac{(d_{ij} - d^*_{ij})^2}{d_{ij}}

    Parameters
    ----------
    data : ndarray (n, p)
        Data matrix.
    n_dims : int
        Target dimensionality.
    max_iter : int
        Maximum gradient descent iterations.
    lr : float
        Learning rate.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is the embedding (n, n_dims).
        ``extra`` has ``stress`` and ``n_iter``.
    """
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[0]
    D = squareform(pdist(X))
    D[D == 0] = 1e-12

    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((n, n_dims)) * 0.1

    c = np.sum(D[np.triu_indices(n, k=1)])

    for it in range(max_iter):
        Dy = squareform(pdist(Y))
        Dy[Dy == 0] = 1e-12

        diff = D - Dy
        ratio = diff / (D * Dy)

        grad = np.zeros_like(Y)
        for i in range(n):
            grad[i] = -2 * np.sum(ratio[i, :, None] * (Y[i] - Y), axis=0) / c

        Y -= lr * grad

    Dy_final = squareform(pdist(Y))
    Dy_final[Dy_final == 0] = 1e-12
    stress = np.sum((D - Dy_final) ** 2 / D) / c if c > 0 else 0.0

    return DescriptiveResult(
        name="SammonMapping",
        value=Y,
        extra={"stress": float(stress), "n_iter": max_iter},
    )


semap = sammon_mapping


def cheatsheet() -> str:
    return "sammon_mapping({}) -> Sammon mapping dimensionality reduction."
