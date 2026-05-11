# morie.fn — function file (hadesllm/morie)
"""Cutting plane proximity per legislator."""

from __future__ import annotations

from ._containers import DescriptiveResult


def cutting_plane_proximity(X, normal, cutpoint) -> DescriptiveResult:
    """Signed distance from each legislator to a cutting plane.

    .. epigraph:: "Science, bitch!" -- Jesse, Breaking Bad
    """
    import numpy as np

    X = np.asarray(X, dtype=float)
    n = np.asarray(normal, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    proj = X @ n
    dist = proj - cutpoint
    return DescriptiveResult(
        name="cutting_plane_proximity",
        value=float(np.mean(np.abs(dist))),
        extra={
            "distances": dist,
            "mean_abs_dist": float(np.mean(np.abs(dist))),
            "n_legislators": X.shape[0],
        },
    )


ctprx = cutting_plane_proximity


def cheatsheet() -> str:
    return "cutting_plane_proximity({}) -> Cutting plane proximity per legislator."
