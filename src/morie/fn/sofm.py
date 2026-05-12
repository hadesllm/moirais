"""Self-organizing map (Kohonen network)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The man who moves a mountain begins by carrying away small stones. -- Confucius"


def self_org_map(X, grid_size=(10, 10), n_iter=1000, lr_init=0.5, sigma_init=None, **kwargs) -> DescriptiveResult:
    """Self-organizing feature map (Kohonen, 1982).

    Competitive learning to map high-dimensional input onto a 2-D grid
    while preserving topological structure.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Input data.
    grid_size : tuple (rows, cols)
        SOM grid dimensions (default (10, 10)).
    n_iter : int
        Training iterations (default 1000).
    lr_init : float
        Initial learning rate (default 0.5).
    sigma_init : float or None
        Initial neighborhood radius. Default: max(grid_size) / 2.

    Returns
    -------
    DescriptiveResult

    References
    ----------
    Kohonen, T. (1982). Self-organized formation of topologically correct
        feature maps. *Biological Cybernetics*, 43(1), 59--69.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    rows, cols = grid_size
    rng = np.random.default_rng(kwargs.get("seed", 42))

    weights = rng.normal(0, 1, (rows, cols, p))
    if sigma_init is None:
        sigma_init = max(rows, cols) / 2.0

    grid_coords = np.array([[r, c] for r in range(rows) for c in range(cols)])

    for t in range(n_iter):
        lr = lr_init * np.exp(-t / n_iter)
        sigma = max(sigma_init * np.exp(-t / n_iter), 0.5)
        idx = rng.integers(0, n)
        x = X[idx]

        dists = np.sum((weights - x) ** 2, axis=2)
        bmu = np.unravel_index(np.argmin(dists), (rows, cols))
        bmu_coord = np.array([bmu[0], bmu[1]])

        for i, coord in enumerate(grid_coords):
            d = np.sum((coord - bmu_coord) ** 2)
            h = np.exp(-d / (2 * sigma**2))
            r, c = coord
            weights[r, c] += lr * h * (x - weights[r, c])

    bmu_indices = np.zeros(n, dtype=int)
    for i in range(n):
        dists = np.sum((weights.reshape(-1, p) - X[i]) ** 2, axis=1)
        bmu_indices[i] = np.argmin(dists)

    qe = np.mean([np.sqrt(np.sum((X[i] - weights.reshape(-1, p)[bmu_indices[i]]) ** 2)) for i in range(n)])

    return DescriptiveResult(
        name="self_org_map",
        value=float(qe),
        extra={
            "weights": weights,
            "bmu_indices": bmu_indices,
            "quantization_error": float(qe),
            "grid_size": grid_size,
            "n_iter": n_iter,
        },
    )


sofm = self_org_map


def cheatsheet() -> str:
    return "self_org_map({}) -> Self-organizing map (Kohonen network)."
