# moirais.fn — function file (hadesllm/moirais)
"""Lloyd-Max optimal scalar quantizer."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lloyd_max(
    x: np.ndarray,
    levels: int = 8,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> DescriptiveResult:
    """Lloyd-Max optimal scalar quantizer.

    Iteratively finds optimal decision boundaries and reconstruction
    levels that minimize MSE for the given data distribution.

    :param x: Input data (1-D).
    :param levels: Number of quantization levels (e.g. 8 for 3-bit).
    :param max_iter: Maximum Lloyd iterations.
    :param tol: Convergence tolerance on centroids.
    :return: DescriptiveResult with codebook and MSE.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    vmin, vmax = float(x.min()), float(x.max())
    centroids = np.linspace(vmin, vmax, levels)
    iteration = 0
    for iteration in range(max_iter):
        dists = np.abs(x[:, None] - centroids[None, :])
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array(
            [x[labels == c].mean() if np.any(labels == c) else centroids[c] for c in range(levels)]
        )
        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids
    labels = np.argmin(np.abs(x[:, None] - centroids[None, :]), axis=1)
    x_hat = centroids[labels]
    mse = float(np.mean((x - x_hat) ** 2))
    return DescriptiveResult(
        name="lloyd_max",
        value=mse,
        extra={
            "centroids": centroids,
            "labels": labels,
            "levels": levels,
            "iterations": iteration + 1,
            "mse": mse,
        },
    )


def cheatsheet() -> str:
    return "lloyd_max(x, levels) -> Lloyd-Max optimal scalar quantizer"


lloym = lloyd_max
