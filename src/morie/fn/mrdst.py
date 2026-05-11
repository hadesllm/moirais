# morie.fn — function file (hadesllm/morie)
"""Mahalanobis distance."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def mahalanobis_distance(
    x: np.ndarray,
    mean: np.ndarray | None = None,
    cov: np.ndarray | None = None,
) -> DescriptiveResult:
    """Mahalanobis distance for each row of x.

    Parameters
    ----------
    x : (n, p) or (p,) array
    mean : (p,) array or None (uses sample mean)
    cov : (p, p) array or None (uses sample covariance)

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n, p = x.shape

    if mean is None:
        mean = x.mean(axis=0)
    else:
        mean = np.asarray(mean, dtype=float)
    if cov is None:
        cov = np.cov(x, rowvar=False)
    else:
        cov = np.asarray(cov, dtype=float)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)

    inv_cov = np.linalg.inv(cov + 1e-10 * np.eye(p))
    diff = x - mean
    dists = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

    return DescriptiveResult(
        name="mahalanobis",
        value=float(dists.mean()),
        extra={"distances": dists.tolist(), "n": n, "p": p, "max": float(dists.max()), "min": float(dists.min())},
    )


mrdst = mahalanobis_distance


def cheatsheet() -> str:
    return "mahalanobis_distance({}) -> Mahalanobis distance."
