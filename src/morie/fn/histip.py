# morie.fn -- function file (hadesllm/morie)
"""Histogram of ideal points."""

from __future__ import annotations

from ._containers import DescriptiveResult


def histogram_ideal_points(positions, bins: int = 30) -> DescriptiveResult:
    """Compute histogram bin edges and counts for ideal point distribution.

    :param positions: Array of ideal point estimates.
    :param bins: Number of histogram bins.
    :return: DescriptiveResult with bin data.

    .. epigraph:: "I'll take a potato chip... and eat it!" -- Light Yagami, Death Note
    """
    import numpy as np

    pos = np.asarray(positions, dtype=float).ravel()
    pos = pos[~np.isnan(pos)]
    counts, edges = np.histogram(pos, bins=bins)
    return DescriptiveResult(
        name="histogram_ideal_points",
        value=int(len(pos)),
        extra={"counts": counts.tolist(), "bin_edges": edges.tolist(), "bins": bins},
    )


histip = histogram_ideal_points


def cheatsheet() -> str:
    return "histogram_ideal_points({}) -> Histogram of ideal points."
