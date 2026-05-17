"""Spatial distance computation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def spatial_distance(x, y, metric: str = "euclidean") -> DescriptiveResult:
    """Compute spatial distance between two points.

    :param x: First point.
    :param y: Second point.
    :param metric: 'euclidean' or 'manhattan'.
    :return: DescriptiveResult with distance.

    .. epigraph:: No man ever steps in the same river twice. -- Heraclitus
    """
    import numpy as np

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if metric == "manhattan":
        d = float(np.sum(np.abs(x - y)))
    else:
        d = float(np.sqrt(np.sum((x - y) ** 2)))
    return DescriptiveResult(
        name="spatial_distance",
        value=d,
        extra={"metric": metric, "n_dims": len(x)},
    )


spdst = spatial_distance


def cheatsheet() -> str:
    return "spatial_distance({}) -> Spatial distance computation."
