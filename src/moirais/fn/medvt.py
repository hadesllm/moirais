# moirais.fn — function file (hadesllm/moirais)
"""Median voter theorem computation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def median_voter(ideal_points) -> DescriptiveResult:
    """Compute the median voter position.

    :param ideal_points: Array of voter ideal points (1-D).
    :return: DescriptiveResult with median position.

    .. epigraph:: "The spice must flow." -- Stilgar, Dune
    """
    import numpy as np

    pts = np.asarray(ideal_points, dtype=float).ravel()
    med = float(np.median(pts))
    return DescriptiveResult(
        name="median_voter",
        value=med,
        extra={"n_voters": len(pts), "median": med, "min": float(pts.min()), "max": float(pts.max())},
    )


medvt = median_voter


def cheatsheet() -> str:
    return "median_voter({}) -> Median voter theorem computation."
