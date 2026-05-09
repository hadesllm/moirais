"""Weighted issue utility aggregation.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svwut(position, ideal=None, weights=None):
    """Weighted issue utility aggregation.

    Parameters
    ----------
    position : array-like
        Current position in issue space.
    ideal : array-like, optional
        Ideal point.
    weights : array-like, optional
        Issue weights.

    Returns
    -------
    DescriptiveResult
    """
    position = np.asarray(position, dtype=float)
    if ideal is None:
        ideal = np.zeros_like(position)
    else:
        ideal = np.asarray(ideal, dtype=float)
    if weights is None:
        weights = np.ones(len(position)) / len(position)
    else:
        weights = np.asarray(weights, dtype=float)
    dist_sq = float(np.sum(weights * (position - ideal) ** 2))
    stat = float(np.exp(-0.5 * dist_sq))
    return DescriptiveResult(
        name="svwut",
        value=stat,
        extra={"dist_sq": dist_sq},
    )


short = "svwut"
alias = "svwut"
quote = "The spice must flow. -- Paul Atreides"
svwut = svwut


def cheatsheet() -> str:
    return "svwut({}) -> Weighted issue utility aggregation."
